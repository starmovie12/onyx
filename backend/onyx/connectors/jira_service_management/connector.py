"""
Jira Service Management (JSM) Connector — standalone implementation.

This connector handles JSM projects independently of JiraConnector.
It implements the same abstract interfaces directly (no concrete inheritance)
in line with the Onyx contribution guideline:
  "Prefer composition and functional style over inheritance/OOP"
"""

import copy
import re
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Final

from jira import JIRA
from jira.resources import Issue
from typing_extensions import override

from onyx.access.models import ExternalAccess
from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.app_configs import JIRA_CONNECTOR_LABELS_TO_SKIP
from onyx.configs.app_configs import JIRA_SLIM_PAGE_SIZE
from onyx.configs.constants import DocumentSource
from onyx.connectors.cross_connector_utils.miscellaneous_utils import (
    is_atlassian_date_error,
)
from onyx.connectors.exceptions import (
    ConnectorValidationError,
    CredentialExpiredError,
    InsufficientPermissionsError,
    UnexpectedValidationError,
)
from onyx.connectors.interfaces import (
    CheckpointedConnectorWithPermSync,
    CheckpointOutput,
    GenerateSlimDocumentOutput,
    SecondsSinceUnixEpoch,
    SlimConnectorWithPermSync,
)
from onyx.connectors.jira.access import get_project_permissions
from onyx.connectors.jira.connector import (
    JiraConnectorCheckpoint,
    ONE_HOUR,
    _JIRA_FULL_PAGE_SIZE,
    _perform_jql_search,
    _is_cloud_client,
    make_checkpoint_callback,
    process_jira_issue,
)
from onyx.connectors.jira.utils import (
    best_effort_get_field_from_issue,
    build_jira_client,
    build_jira_url,
)
from onyx.connectors.models import (
    ConnectorFailure,
    ConnectorMissingCredentialError,
    Document,
    DocumentFailure,
    HierarchyNode,
    SlimDocument,
)
from onyx.db.enums import HierarchyNodeType
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.utils.logger import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Jira field name constants (mirrored from jira/connector.py)
# ---------------------------------------------------------------------------
_FIELD_REPORTER = "reporter"
_FIELD_ASSIGNEE = "assignee"
_FIELD_PRIORITY = "priority"
_FIELD_STATUS = "status"
_FIELD_RESOLUTION = "resolution"
_FIELD_LABELS = "labels"
_FIELD_KEY = "key"
_FIELD_CREATED = "created"
_FIELD_DUEDATE = "duedate"
_FIELD_ISSUETYPE = "issuetype"
_FIELD_PARENT = "parent"
_FIELD_ASSIGNEE_EMAIL = "assignee_email"
_FIELD_REPORTER_EMAIL = "reporter_email"
_FIELD_PROJECT = "project"
_FIELD_PROJECT_NAME = "project_name"
_FIELD_UPDATED = "updated"
_FIELD_RESOLUTION_DATE = "resolutiondate"
_FIELD_RESOLUTION_DATE_KEY = "resolution_date"

# ---------------------------------------------------------------------------
# SLA field-name patterns
# ---------------------------------------------------------------------------
_SLA_FIELD_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"time\s+to\s+first\s+response", "sla_time_to_first_response"),
    (r"time\s+to\s+resolution", "sla_time_to_resolution"),
    (r"time\s+to\s+close", "sla_time_to_close"),
    (r"time\s+to\s+respond", "sla_time_to_respond"),
    (r"time\s+to\s+approve", "sla_time_to_approve"),
    (r"satisfaction\s+rating", "jsm_satisfaction_rating"),
]

_COMPILED_SLA_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(pat, re.IGNORECASE), key) for pat, key in _SLA_FIELD_PATTERNS
]

_CUSTOMER_REQUEST_TYPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"customer\s+request\s+type", re.IGNORECASE
)

_META_REQUEST_TYPE: Final[str] = "jsm_request_type"
_META_SERVICE_DESK: Final[str] = "jsm_service_desk_id"
_BREACH_SUFFIX: Final[str] = "_breached"


# ---------------------------------------------------------------------------
# SLA value extraction helpers
# ---------------------------------------------------------------------------


def _extract_sla_display(sla_field_value: Any) -> tuple[str | None, bool]:
    """Return (human-readable SLA string | None, is_breached).

    JSM returns SLA data in two formats depending on the deployment type:

    **Cloud** – a nested dict with ongoingCycle / completedCycles.
    **Server / Data Center** – a plain string or simple dict.

    Returns (None, False) for any unrecognised or null value.
    """
    if sla_field_value is None:
        return None, False

    if isinstance(sla_field_value, str):
        return sla_field_value or None, False

    if not isinstance(sla_field_value, dict):
        return None, False

    if "text" in sla_field_value and "ongoingCycle" not in sla_field_value:
        text = sla_field_value.get("text") or None
        breached = bool(sla_field_value.get("breached", False))
        return text, breached

    ongoing = sla_field_value.get("ongoingCycle")
    if isinstance(ongoing, dict):
        remaining = ongoing.get("remainingTime", {})
        friendly = remaining.get("friendly") if isinstance(remaining, dict) else None
        breached = bool(ongoing.get("breached", False))
        if not friendly:
            friendly = "Breached" if breached else None
        return friendly, breached

    completed = sla_field_value.get("completedCycles")
    if isinstance(completed, list) and completed:
        last = completed[-1]
        if isinstance(last, dict):
            remaining = last.get("remainingTime", {})
            friendly = (
                remaining.get("friendly") if isinstance(remaining, dict) else None
            )
            breached = bool(last.get("breached", False))
            return friendly, breached

    return None, False


def _get_raw_field(issue: Issue, field_id: str) -> Any:
    """Safely retrieve a raw field value from an Issue object."""
    try:
        return getattr(issue.fields, field_id, None)
    except Exception:
        logger.debug(
            "Failed to read field %r from issue %r",
            field_id,
            getattr(issue, "key", "?"),
            exc_info=True,
        )
        return None


def _get_request_type(
    issue: Issue,
    request_type_field_id: str | None = None,
) -> str | None:
    """Extract the JSM request type name from an issue, if present."""
    try:
        rt = getattr(issue.fields, "requestType", None)
        if rt is not None:
            return getattr(rt, "name", None) or None

        if request_type_field_id:
            raw_fields: dict[str, Any] = (
                issue.raw.get("fields", {}) if isinstance(issue.raw, dict) else {}
            )
            fval = raw_fields.get(request_type_field_id)
            if isinstance(fval, dict):
                rt_dict = fval.get("requestType")
                if isinstance(rt_dict, dict):
                    name = rt_dict.get("name")
                    if name:
                        return str(name)
    except Exception:
        logger.debug(
            "Failed to extract request type from issue %r",
            getattr(issue, "key", "?"),
            exc_info=True,
        )
    return None


def _get_service_desk_id(issue: Issue) -> str | None:
    """Extract the numeric service desk ID from the issue, if present."""
    try:
        sd = getattr(issue.fields, "serviceDeskId", None)
        if sd is not None:
            return str(sd)
    except Exception:
        logger.debug(
            "Failed to extract serviceDeskId from issue %r",
            getattr(issue, "key", "?"),
            exc_info=True,
        )
    return None


# ---------------------------------------------------------------------------
# Connector class
# ---------------------------------------------------------------------------


class JiraServiceManagementConnector(
    CheckpointedConnectorWithPermSync[JiraConnectorCheckpoint],
    SlimConnectorWithPermSync,
):
    """Standalone connector for Jira Service Management (JSM) projects.

    Implements all indexing, pagination, hierarchy, and permission logic
    independently without inheriting from JiraConnector — in accordance with
    the Onyx guideline "Prefer composition and functional style over
    inheritance/OOP".

    JSM-specific additions: dynamic SLA field discovery and document enrichment.
    """

    # Maximum number of times to attempt field discovery before giving up.
    _MAX_SLA_DISCOVERY_ATTEMPTS: ClassVar[int] = 3

    def __init__(
        self,
        jira_base_url: str,
        project_key: str | None = None,
        comment_email_blacklist: list[str] | None = None,
        batch_size: int = INDEX_BATCH_SIZE,
        labels_to_skip: list[str] = JIRA_CONNECTOR_LABELS_TO_SKIP,
        jql_query: str | None = None,
        scoped_token: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.jira_base = jira_base_url.rstrip("/")
        self.jira_project = project_key
        self._comment_email_blacklist = comment_email_blacklist or []
        self.labels_to_skip = set(labels_to_skip)
        self.jql_query = jql_query
        self.scoped_token = scoped_token
        self._jira_client: JIRA | None = None
        self._project_permissions_cache: dict[str, ExternalAccess | None] = {}
        # ``None`` means discovery has not yet been attempted or is still
        # retrying after a transient failure; ``{}`` means discovery either
        # succeeded but found no SLA fields on this instance, OR was
        # permanently disabled after exhausting ``_MAX_SLA_DISCOVERY_ATTEMPTS``
        # consecutive API failures.  In both cases, ``_fields_discovered``
        # is set to ``True`` so future calls hit the fast-path immediately.
        self._sla_field_map: dict[str, str] | None = None
        self._sla_discovery_attempts: int = 0
        self._request_type_field_id: str | None = None
        self._fields_discovered: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def comment_email_blacklist(self) -> tuple:
        return tuple(email.strip() for email in self._comment_email_blacklist)

    @property
    def jira_client(self) -> JIRA:
        if self._jira_client is None:
            raise ConnectorMissingCredentialError("Jira")
        return self._jira_client

    @property
    def quoted_jira_project(self) -> str:
        if not self.jira_project:
            return ""
        return f'"{self.jira_project}"'

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self._jira_client = build_jira_client(
            credentials=credentials,
            jira_base=self.jira_base,
            scoped_token=self.scoped_token,
        )
        return None

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def _get_project_permissions(
        self, project_key: str, add_prefix: bool = False
    ) -> ExternalAccess | None:
        """Get project permissions with caching (JSM source)."""
        cache_key = f"{project_key}:{'prefixed' if add_prefix else 'unprefixed'}"
        if cache_key not in self._project_permissions_cache:
            self._project_permissions_cache[cache_key] = get_project_permissions(
                jira_client=self.jira_client,
                jira_project=project_key,
                add_prefix=add_prefix,
                source=DocumentSource.JIRA_SERVICE_MANAGEMENT,
            )
        return self._project_permissions_cache[cache_key]

    # ------------------------------------------------------------------
    # JQL helpers
    # ------------------------------------------------------------------

    def _get_jql_query(
        self, start: SecondsSinceUnixEpoch, end: SecondsSinceUnixEpoch
    ) -> str:
        start_date_str = datetime.fromtimestamp(start, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        end_date_str = datetime.fromtimestamp(end, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )

        time_jql = f"updated >= '{start_date_str}' AND updated <= '{end_date_str}'"

        if self.jql_query:
            return f"({self.jql_query}) AND {time_jql}"

        if self.jira_project:
            base_jql = f"project = {self.quoted_jira_project}"
            return f"{base_jql} AND {time_jql}"

        return time_jql

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_connector_settings(self) -> None:
        if self._jira_client is None:
            raise ConnectorMissingCredentialError("Jira")

        if self.jql_query:
            try:
                next(
                    iter(
                        _perform_jql_search(
                            jira_client=self.jira_client,
                            jql=self.jql_query,
                            start=0,
                            max_results=1,
                            all_issue_ids=[],
                        )
                    ),
                    None,
                )
            except Exception as e:
                self._handle_jira_connector_settings_error(e)

        elif self.jira_project:
            try:
                self.jira_client.project(self.jira_project)
            except Exception as e:
                self._handle_jira_connector_settings_error(e)
        else:
            try:
                self.jira_client.projects()
            except Exception as e:
                self._handle_jira_connector_settings_error(e)

    def _handle_jira_connector_settings_error(self, e: Exception) -> None:
        status_code = getattr(e, "status_code", None)
        logger.error(f"Jira API error during validation: {e}")

        if status_code == 401:
            raise CredentialExpiredError(
                "Jira credential appears to be expired or invalid (HTTP 401)."
            )
        elif status_code == 403:
            raise InsufficientPermissionsError(
                "Your Jira token does not have sufficient permissions for this configuration (HTTP 403)."
            )
        elif status_code == 429:
            raise ConnectorValidationError(
                "Validation failed due to Jira rate-limits being exceeded. Please try again later."
            )

        error_message = getattr(e, "text", None)
        if error_message is None:
            raise UnexpectedValidationError(
                f"Unexpected Jira error during validation: {e}"
            )

        raise ConnectorValidationError(
            f"Validation failed due to Jira error: {error_message}"
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @override
    def validate_checkpoint_json(self, checkpoint_json: str) -> JiraConnectorCheckpoint:
        return JiraConnectorCheckpoint.model_validate_json(checkpoint_json)

    @override
    def build_dummy_checkpoint(self) -> JiraConnectorCheckpoint:
        return JiraConnectorCheckpoint(
            has_more=True,
        )

    def update_checkpoint_for_next_run(
        self,
        checkpoint: JiraConnectorCheckpoint,
        current_offset: int,
        starting_offset: int,
        page_size: int,
    ) -> None:
        if _is_cloud_client(self.jira_client):
            checkpoint.has_more = (
                len(checkpoint.all_issue_ids) > 0 or not checkpoint.ids_done
            )
        else:
            checkpoint.offset = current_offset
            checkpoint.has_more = current_offset - starting_offset == page_size

    # ------------------------------------------------------------------
    # Hierarchy helpers
    # ------------------------------------------------------------------

    def _is_epic(self, issue: Issue) -> bool:
        issuetype = best_effort_get_field_from_issue(issue, _FIELD_ISSUETYPE)
        if issuetype is None:
            return False
        return issuetype.name.lower() == "epic"

    def _is_parent_epic(self, parent: Any) -> bool:
        parent_issuetype = (
            getattr(parent.fields, "issuetype", None)
            if hasattr(parent, "fields")
            else None
        )
        if parent_issuetype is None:
            return False
        return parent_issuetype.name.lower() == "epic"

    def _yield_project_hierarchy_node(
        self,
        project_key: str,
        project_name: str | None,
        seen_hierarchy_node_ids: set[str],
    ) -> Generator[HierarchyNode, None, None]:
        if project_key in seen_hierarchy_node_ids:
            return

        seen_hierarchy_node_ids.add(project_key)

        yield HierarchyNode(
            raw_node_id=project_key,
            raw_parent_id=None,
            display_name=project_name or project_key,
            link=f"{self.jira_base}/projects/{project_key}",
            node_type=HierarchyNodeType.PROJECT,
        )

    def _yield_epic_hierarchy_node(
        self,
        issue: Issue,
        project_key: str,
        seen_hierarchy_node_ids: set[str],
    ) -> Generator[HierarchyNode, None, None]:
        issue_key = issue.key
        if issue_key in seen_hierarchy_node_ids:
            return

        seen_hierarchy_node_ids.add(issue_key)

        yield HierarchyNode(
            raw_node_id=issue_key,
            raw_parent_id=project_key,
            display_name=f"{issue_key}: {issue.fields.summary}",
            link=build_jira_url(self.jira_base, issue_key),
            node_type=HierarchyNodeType.FOLDER,
        )

    def _yield_parent_hierarchy_node_if_epic(
        self,
        parent: Any,
        project_key: str,
        seen_hierarchy_node_ids: set[str],
    ) -> Generator[HierarchyNode, None, None]:
        parent_key = parent.key
        if parent_key in seen_hierarchy_node_ids:
            return

        if not self._is_parent_epic(parent):
            return

        seen_hierarchy_node_ids.add(parent_key)

        parent_summary = (
            getattr(parent.fields, "summary", None)
            if hasattr(parent, "fields")
            else None
        )
        display_name = (
            f"{parent_key}: {parent_summary}" if parent_summary else parent_key
        )

        yield HierarchyNode(
            raw_node_id=parent_key,
            raw_parent_id=project_key,
            display_name=display_name,
            link=build_jira_url(self.jira_base, parent_key),
            node_type=HierarchyNodeType.FOLDER,
        )

    def _get_parent_hierarchy_raw_node_id(self, issue: Issue, project_key: str) -> str:
        parent = best_effort_get_field_from_issue(issue, _FIELD_PARENT)
        if parent is None:
            return project_key

        if self._is_parent_epic(parent):
            return parent.key

        return project_key

    # ------------------------------------------------------------------
    # Core loading logic
    # ------------------------------------------------------------------

    def _load_from_checkpoint(
        self, jql: str, checkpoint: JiraConnectorCheckpoint, include_permissions: bool
    ) -> CheckpointOutput[JiraConnectorCheckpoint]:
        starting_offset = checkpoint.offset or 0
        current_offset = starting_offset
        new_checkpoint = copy.deepcopy(checkpoint)

        seen_hierarchy_node_ids = set(new_checkpoint.seen_hierarchy_node_ids)

        checkpoint_callback = make_checkpoint_callback(new_checkpoint)

        for issue in _perform_jql_search(
            jira_client=self.jira_client,
            jql=jql,
            start=current_offset,
            max_results=_JIRA_FULL_PAGE_SIZE,
            all_issue_ids=new_checkpoint.all_issue_ids,
            checkpoint_callback=checkpoint_callback,
            nextPageToken=new_checkpoint.cursor,
            ids_done=new_checkpoint.ids_done,
        ):
            issue_key = issue.key
            try:
                project = best_effort_get_field_from_issue(issue, _FIELD_PROJECT)
                project_key = project.key if project else None
                project_name = project.name if project else None

                if project_key:
                    yield from self._yield_project_hierarchy_node(
                        project_key, project_name, seen_hierarchy_node_ids
                    )

                    parent = best_effort_get_field_from_issue(issue, _FIELD_PARENT)
                    if parent:
                        yield from self._yield_parent_hierarchy_node_if_epic(
                            parent, project_key, seen_hierarchy_node_ids
                        )

                    if self._is_epic(issue):
                        yield from self._yield_epic_hierarchy_node(
                            issue, project_key, seen_hierarchy_node_ids
                        )

                parent_hierarchy_raw_node_id = (
                    self._get_parent_hierarchy_raw_node_id(issue, project_key)
                    if project_key
                    else None
                )

                if document := process_jira_issue(
                    jira_base_url=self.jira_base,
                    issue=issue,
                    comment_email_blacklist=self.comment_email_blacklist,
                    labels_to_skip=self.labels_to_skip,
                    parent_hierarchy_raw_node_id=parent_hierarchy_raw_node_id,
                    source=DocumentSource.JIRA_SERVICE_MANAGEMENT,
                ):
                    document = self._enrich_document(document, issue)

                    if include_permissions:
                        document.external_access = self._get_project_permissions(
                            project_key,
                            add_prefix=True,
                        )
                    yield document

            except Exception as e:
                yield ConnectorFailure(
                    failed_document=DocumentFailure(
                        document_id=issue_key,
                        document_link=build_jira_url(self.jira_base, issue_key),
                    ),
                    failure_message=f"Failed to process Jira issue: {str(e)}",
                    exception=e,
                )

            current_offset += 1

        new_checkpoint.seen_hierarchy_node_ids = list(seen_hierarchy_node_ids)

        self.update_checkpoint_for_next_run(
            new_checkpoint, current_offset, starting_offset, _JIRA_FULL_PAGE_SIZE
        )

        return new_checkpoint

    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: JiraConnectorCheckpoint,
    ) -> CheckpointOutput[JiraConnectorCheckpoint]:
        jql = self._get_jql_query(start, end)
        try:
            return self._load_from_checkpoint(
                jql, checkpoint, include_permissions=False
            )
        except Exception as e:
            if is_atlassian_date_error(e):
                jql = self._get_jql_query(start - ONE_HOUR, end)
                return self._load_from_checkpoint(
                    jql, checkpoint, include_permissions=False
                )
            raise e

    def load_from_checkpoint_with_perm_sync(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: JiraConnectorCheckpoint,
    ) -> CheckpointOutput[JiraConnectorCheckpoint]:
        jql = self._get_jql_query(start, end)
        try:
            return self._load_from_checkpoint(jql, checkpoint, include_permissions=True)
        except Exception as e:
            if is_atlassian_date_error(e):
                jql = self._get_jql_query(start - ONE_HOUR, end)
                return self._load_from_checkpoint(
                    jql, checkpoint, include_permissions=True
                )
            raise e

    def retrieve_all_slim_docs_perm_sync(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
        callback: IndexingHeartbeatInterface | None = None,  # noqa: ARG002
    ) -> GenerateSlimDocumentOutput:
        one_day = timedelta(hours=24).total_seconds()

        start = start or 0
        end = end or datetime.now().timestamp() + one_day

        jql = self._get_jql_query(start, end)
        checkpoint = self.build_dummy_checkpoint()
        checkpoint_callback = make_checkpoint_callback(checkpoint)
        prev_offset = 0
        current_offset = 0
        slim_doc_batch: list[SlimDocument | HierarchyNode] = []

        seen_hierarchy_node_ids: set[str] = set()

        while checkpoint.has_more:
            for issue in _perform_jql_search(
                jira_client=self.jira_client,
                jql=jql,
                start=current_offset,
                max_results=JIRA_SLIM_PAGE_SIZE,
                all_issue_ids=checkpoint.all_issue_ids,
                checkpoint_callback=checkpoint_callback,
                nextPageToken=checkpoint.cursor,
                ids_done=checkpoint.ids_done,
            ):
                project = best_effort_get_field_from_issue(issue, _FIELD_PROJECT)
                project_key = project.key if project else None
                project_name = project.name if project else None

                if not project_key:
                    continue

                for node in self._yield_project_hierarchy_node(
                    project_key, project_name, seen_hierarchy_node_ids
                ):
                    slim_doc_batch.append(node)

                parent = best_effort_get_field_from_issue(issue, _FIELD_PARENT)
                if parent:
                    for node in self._yield_parent_hierarchy_node_if_epic(
                        parent, project_key, seen_hierarchy_node_ids
                    ):
                        slim_doc_batch.append(node)

                if self._is_epic(issue):
                    for node in self._yield_epic_hierarchy_node(
                        issue, project_key, seen_hierarchy_node_ids
                    ):
                        slim_doc_batch.append(node)

                issue_key = best_effort_get_field_from_issue(issue, _FIELD_KEY)
                doc_id = build_jira_url(self.jira_base, issue_key)

                slim_doc_batch.append(
                    SlimDocument(
                        id=doc_id,
                        external_access=self._get_project_permissions(
                            project_key, add_prefix=False
                        ),
                        parent_hierarchy_raw_node_id=(
                            self._get_parent_hierarchy_raw_node_id(issue, project_key)
                            if project_key
                            else None
                        ),
                    )
                )
                current_offset += 1
                if len(slim_doc_batch) >= JIRA_SLIM_PAGE_SIZE:
                    yield slim_doc_batch
                    slim_doc_batch = []
            self.update_checkpoint_for_next_run(
                checkpoint, current_offset, prev_offset, JIRA_SLIM_PAGE_SIZE
            )
            prev_offset = current_offset

        if slim_doc_batch:
            yield slim_doc_batch

    # ------------------------------------------------------------------
    # Field discovery (SLA + request type)
    # ------------------------------------------------------------------

    def _ensure_fields_discovered(self) -> None:
        """Fetch the Jira field list once and populate SLA and request-type caches.

        Transient failures are retried up to _MAX_SLA_DISCOVERY_ATTEMPTS times;
        after that, an empty map is cached to prevent unbounded failing API calls.
        """
        if self._fields_discovered:
            return

        self._sla_discovery_attempts += 1

        try:
            all_fields: list[dict[str, Any]] = self.jira_client.fields()
        except Exception:
            if self._sla_discovery_attempts < self._MAX_SLA_DISCOVERY_ATTEMPTS:
                logger.warning(
                    "JSM field fetch failed (attempt %d/%d) — "
                    "SLA metadata will be omitted for this document. "
                    "Retrying on next document. "
                    "Check connector credentials / permissions.",
                    self._sla_discovery_attempts,
                    self._MAX_SLA_DISCOVERY_ATTEMPTS,
                    exc_info=True,
                )
                return
            else:
                logger.warning(
                    "JSM field fetch failed (attempt %d/%d) — "
                    "No more retries; SLA enrichment disabled for this run. "
                    "Check connector credentials / permissions.",
                    self._sla_discovery_attempts,
                    self._MAX_SLA_DISCOVERY_ATTEMPTS,
                    exc_info=True,
                )
                self._sla_field_map = {}
                self._fields_discovered = True
                return

        try:
            self._discover_sla_mapping(all_fields)
            self._discover_request_type_mapping(all_fields)
        finally:
            self._fields_discovered = True

    def _discover_sla_mapping(self, all_fields: list[dict[str, Any]]) -> None:
        """Populate ``_sla_field_map`` from a pre-fetched field list."""
        mapping: dict[str, str] = {}
        try:
            for field_meta in all_fields:
                field_id: str = field_meta.get("id", "")
                field_name: str = field_meta.get("name", "")
                if not field_id.startswith("customfield_"):
                    continue
                for pattern, canonical_key in _COMPILED_SLA_PATTERNS:
                    if pattern.search(field_name):
                        mapping[field_id] = canonical_key
                        logger.debug(
                            "JSM SLA field discovered: %r (%r) → %r",
                            field_id,
                            field_name,
                            canonical_key,
                        )
                        break
            self._sla_field_map = mapping
        except Exception:
            logger.warning(
                "JSM SLA field processing failed after caching %d field(s) — "
                "SLA metadata may be incomplete.",
                len(mapping),
                exc_info=True,
            )
            self._sla_field_map = mapping

    def _discover_request_type_mapping(self, all_fields: list[dict[str, Any]]) -> None:
        """Populate ``_request_type_field_id`` from a pre-fetched field list."""
        try:
            for field_meta in all_fields:
                field_id: str = field_meta.get("id", "")
                field_name: str = field_meta.get("name", "")
                if not field_id.startswith("customfield_"):
                    continue
                if _CUSTOMER_REQUEST_TYPE_PATTERN.search(field_name):
                    self._request_type_field_id = field_id
                    logger.debug(
                        "JSM request-type field discovered: %r (%r)",
                        field_id,
                        field_name,
                    )
                    break
        except Exception:
            logger.warning(
                "JSM request-type field processing failed — "
                "Server/DC request-type extraction will be skipped.",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Document enrichment
    # ------------------------------------------------------------------

    def _enrich_document(self, document: Document, issue: Issue) -> Document:
        """Attach JSM-specific metadata (SLA, request type, service desk ID)."""
        self._ensure_fields_discovered()
        try:
            self._attach_sla_metadata(document, issue)
        except Exception:
            logger.warning(
                "Failed to attach SLA metadata to %r", document.id, exc_info=True
            )
        try:
            self._attach_jsm_metadata(document, issue)
        except Exception:
            logger.warning(
                "Failed to attach JSM metadata to %r", document.id, exc_info=True
            )
        return document

    def _attach_sla_metadata(self, document: Document, issue: Issue) -> None:
        """Populate SLA-related keys in ``document.metadata``."""
        sla_field_map: dict[str, str] | None = self._sla_field_map
        if sla_field_map is None:
            logger.debug(
                "SLA field discovery not yet complete (transient failure); "
                "skipping SLA enrichment for %r.",
                document.id,
            )
            return
        if not sla_field_map:
            logger.debug(
                "SLA field map is empty for %r "
                "(either no SLA fields on this instance, or discovery "
                "permanently failed — check earlier WARNING logs); "
                "skipping SLA enrichment.",
                document.id,
            )
            return

        for field_id, canonical_key in sla_field_map.items():
            raw_value = _get_raw_field(issue, field_id)
            if raw_value is None:
                continue

            display_str, is_breached = _extract_sla_display(raw_value)
            if display_str is not None:
                document.metadata[canonical_key] = display_str
            if is_breached:
                document.metadata[canonical_key + _BREACH_SUFFIX] = "true"

    def _attach_jsm_metadata(self, document: Document, issue: Issue) -> None:
        """Populate non-SLA JSM-specific metadata keys."""
        request_type = _get_request_type(issue, self._request_type_field_id)
        if request_type:
            document.metadata[_META_REQUEST_TYPE] = request_type

        service_desk_id = _get_service_desk_id(issue)
        if service_desk_id:
            document.metadata[_META_SERVICE_DESK] = service_desk_id


if __name__ == "__main__":
    import os
    from onyx.utils.variable_functionality import global_version
    from tests.daily.connectors.utils import load_all_from_connector

    # For connector permission testing, set EE to true.
    global_version.set_ee()

    connector = JiraServiceManagementConnector(
        jira_base_url=os.environ["JSM_BASE_URL"],
        project_key=os.environ.get("JSM_PROJECT_KEY"),
        comment_email_blacklist=[],
    )

    connector.load_credentials(
        {
            "jira_user_email": os.environ["JSM_USER_EMAIL"],
            "jira_api_token": os.environ["JSM_API_TOKEN"],
        }
    )

    start = 0
    end = datetime.now().timestamp()

    for slim_doc in connector.retrieve_all_slim_docs_perm_sync(
        start=start,
        end=end,
    ):
        print(slim_doc)

    for doc in load_all_from_connector(
        connector=connector,
        start=start,
        end=end,
    ).documents:
        print(doc)
