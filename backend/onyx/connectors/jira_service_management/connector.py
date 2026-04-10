"""
Jira Service Management Connector.

Inherits from the standard Jira connector because JSM shares the same
Jira REST API.  The subclass adds three things that are unique to JSM:

1.  **Source tagging** – documents are tagged with
    ``DocumentSource.JIRA_SERVICE_MANAGEMENT`` instead of
    ``DocumentSource.JIRA``, which also makes EE permission groups use the
    correct ``jira_service_management_`` prefix automatically through the
    inherited ``_get_project_permissions`` / ``_source`` mechanism.

2.  **Dynamic SLA field discovery** – rather than hard-coding
    ``customfield_10010`` (which is assigned sequentially per Jira
    instance and differs across tenants), we call ``jira_client.fields()``
    once per connector lifetime, match the returned field names against a
    set of well-known JSM SLA name patterns, and cache the mapping.  This
    means the connector works correctly on *any* Jira installation without
    manual configuration.

3.  **SLA enrichment** – the ``_enrich_document`` hook (added to the base
    class for exactly this purpose) attaches human-readable SLA metadata —
    "Time to First Response", "Time to Resolution", breach flags, and
    remaining-time strings — to each indexed document.  Both Cloud (nested
    dict) and Server / Data Center (plain string) SLA payload formats are
    handled.
"""

from __future__ import annotations

import re
from typing import Any

from typing_extensions import override

from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.app_configs import JIRA_CONNECTOR_LABELS_TO_SKIP
from onyx.configs.constants import DocumentSource
from onyx.connectors.jira.connector import JiraConnector
from onyx.connectors.models import Document
from onyx.utils.logger import setup_logger

# jira.resources.Issue is only imported for type-checking; we keep the
# runtime dependency optional so that unit tests can mock freely.
try:
    from jira.resources import Issue
except ImportError:  # pragma: no cover
    Issue = Any  # type: ignore[assignment,misc]

logger = setup_logger()

# ---------------------------------------------------------------------------
# SLA field-name patterns
# ---------------------------------------------------------------------------
# These patterns are matched (case-insensitively) against the *name* field
# returned by the Jira ``/rest/api/2/field`` endpoint.  They are broad
# enough to catch common Jira-Cloud and Server naming conventions while
# remaining specific enough not to collide with ordinary custom fields.
_SLA_FIELD_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern,  canonical metadata key)
    (r"time\s+to\s+first\s+response", "sla_time_to_first_response"),
    (r"time\s+to\s+resolution", "sla_time_to_resolution"),
    (r"time\s+to\s+close", "sla_time_to_close"),
    (r"time\s+to\s+respond", "sla_time_to_respond"),
    (r"time\s+to\s+approve", "sla_time_to_approve"),
    (r"satisfaction\s+rating", "sla_satisfaction_rating"),
]

# Metadata key under which the request type is stored.
_META_REQUEST_TYPE = "jsm_request_type"
_META_SERVICE_DESK = "jsm_service_desk_id"

# SLA breach metadata suffix
_BREACH_SUFFIX = "_breached"


def _compile_sla_patterns() -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(pat, re.IGNORECASE), key) for pat, key in _SLA_FIELD_PATTERNS]


_COMPILED_SLA_PATTERNS = _compile_sla_patterns()


# ---------------------------------------------------------------------------
# SLA value extraction helpers
# ---------------------------------------------------------------------------


def _extract_sla_display(sla_field_value: Any) -> tuple[str | None, bool]:
    """Return (human-readable SLA string | None, is_breached).

    JSM returns SLA data in two formats depending on the deployment type:

    **Cloud** – a nested dict such as::

        {
            "ongoingCycle": {
                "remainingTime": {"friendly": "2h 30m", "millis": 9000000},
                "breached": false,
                "paused": false,
                ...
            },
            "completedCycles": [
                {"breached": false, "remainingTime": {"friendly": "1h"}, ...}
            ]
        }

    **Server / Data Center** – a plain string such as ``"2h 30m"`` or a
    simple dict ``{"text": "2h 30m", "breached": false}``.

    Returns ``(None, False)`` for any unrecognised or null value so that
    callers never raise on unexpected shapes.
    """
    if sla_field_value is None:
        return None, False

    # --- Plain string (Server DC simple case) ---
    if isinstance(sla_field_value, str):
        return sla_field_value or None, False

    if not isinstance(sla_field_value, dict):
        return None, False

    # --- Server/DC simple dict ---
    if "text" in sla_field_value and "ongoingCycle" not in sla_field_value:
        text = sla_field_value.get("text") or None
        breached = bool(sla_field_value.get("breached", False))
        return text, breached

    # --- Cloud nested dict ---
    ongoing = sla_field_value.get("ongoingCycle")
    if isinstance(ongoing, dict):
        remaining = ongoing.get("remainingTime", {})
        friendly = remaining.get("friendly") if isinstance(remaining, dict) else None
        breached = bool(ongoing.get("breached", False))
        if not friendly:
            friendly = "Breached" if breached else None
        return friendly, breached

    # Ticket already resolved — look at the last completed cycle.
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


def _get_raw_field(issue: Any, field_id: str) -> Any:
    """Safely retrieve a raw field value from an Issue object."""
    try:
        return getattr(issue.fields, field_id, None)
    except Exception:
        logger.debug(
            f"Failed to read field {field_id!r} from issue {getattr(issue, 'key', '?')!r}",
            exc_info=True,
        )
        return None


def _get_request_type(issue: Any) -> str | None:
    """Extract the JSM request type name from an issue, if present."""
    try:
        # Cloud: issue.fields.requestType  (added by JSM REST layer)
        rt = getattr(issue.fields, "requestType", None)
        if rt is not None:
            return getattr(rt, "name", None) or str(rt)
        # Server/DC: sometimes stored under customfield as a dict
        raw_fields: dict[str, Any] = (
            issue.raw.get("fields", {}) if isinstance(issue.raw, dict) else {}
        )
        for _fid, fval in raw_fields.items():
            if not _fid.startswith("customfield_"):
                continue
            if isinstance(fval, dict):
                rt = fval.get("requestType")
                if isinstance(rt, dict):
                    name = rt.get("name")
                    if name:
                        return str(name)
    except Exception:
        logger.debug(
            f"Failed to extract request type from issue {getattr(issue, 'key', '?')!r}",
            exc_info=True,
        )
    return None


def _get_service_desk_id(issue: Any) -> str | None:
    """Extract the numeric service desk ID from the issue, if present.

    Jira service desk IDs are numeric strings (e.g. ``"1"``, ``"2"``).
    If ``serviceDeskId`` is absent we return ``None`` rather than
    substituting the project key, which is a string of a different type
    and would produce semantically incorrect metadata.
    """
    try:
        sd = getattr(issue.fields, "serviceDeskId", None)
        if sd is not None:
            return str(sd)
    except Exception:
        logger.debug(
            f"Failed to extract serviceDeskId from issue {getattr(issue, 'key', '?')!r}",
            exc_info=True,
        )
    return None


# ---------------------------------------------------------------------------
# Connector class
# ---------------------------------------------------------------------------


class JiraServiceManagementConnector(JiraConnector):
    """Connector for Jira Service Management (JSM) projects.

    Reuses *all* indexing, pagination, ADF parsing, hierarchy,
    permission logic, and permission caching from the base
    ``JiraConnector``.

    The ``_source`` class attribute causes the inherited
    ``_get_project_permissions`` to automatically prefix EE permission
    group IDs with ``jira_service_management_`` instead of ``jira_``,
    with zero duplication.

    SLA metadata is attached via the ``_enrich_document`` hook that the
    base class calls for every successfully-processed issue.
    """

    _source: DocumentSource = DocumentSource.JIRA_SERVICE_MANAGEMENT

    # Maximum number of times we will attempt SLA field discovery before giving
    # up for the lifetime of this connector run.  This prevents a persistent
    # failure (e.g. a missing OAuth scope) from issuing one failing HTTP call
    # per issue when indexing large projects.
    _MAX_SLA_DISCOVERY_ATTEMPTS: int = 3

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
        super().__init__(
            jira_base_url=jira_base_url,
            project_key=project_key,
            comment_email_blacklist=comment_email_blacklist,
            batch_size=batch_size,
            labels_to_skip=labels_to_skip,
            jql_query=jql_query,
            scoped_token=scoped_token,
        )
        # Maps customfield_XXXXX → canonical metadata key (populated lazily).
        self._sla_field_map: dict[str, str] | None = None
        # Tracks how many times SLA field discovery has been attempted so that
        # persistent failures do not generate unbounded failing API calls.
        self._sla_discovery_attempts: int = 0

    # ------------------------------------------------------------------
    # SLA field discovery
    # ------------------------------------------------------------------

    def _discover_sla_fields(self) -> dict[str, str] | None:
        """Discover which ``customfield_*`` IDs correspond to SLA fields.

        Calls ``GET /rest/api/2/field`` on the first invocation and caches
        the result.  Transient failures are retried on subsequent calls up to
        ``_MAX_SLA_DISCOVERY_ATTEMPTS`` times; after that the method returns
        an empty dict for the remainder of the connector run to avoid
        hammering the Jira API on persistent failures (e.g. a missing OAuth
        scope on a 10,000-issue project).

        Returns:
            Mapping of ``{"customfield_XXXXX": "sla_<canonical_key>", ...}``
        """
        # Fast-path: already discovered (successfully or permanently failed).
        if self._sla_field_map is not None:
            return self._sla_field_map

        self._sla_discovery_attempts += 1
        mapping: dict[str, str] = {}
        try:
            all_fields: list[dict[str, Any]] = self.jira_client.fields()
            for field_meta in all_fields:
                field_id: str = field_meta.get("id", "")
                field_name: str = field_meta.get("name", "")
                if not field_id.startswith("customfield_"):
                    continue
                for pattern, canonical_key in _COMPILED_SLA_PATTERNS:
                    if pattern.search(field_name):
                        mapping[field_id] = canonical_key
                        logger.debug(
                            f"JSM SLA field discovered: {field_id!r} "
                            f"({field_name!r}) → {canonical_key!r}"
                        )
                        break  # one canonical key per field ID
            # Success — cache so we never call fields() again.
            self._sla_field_map = mapping
        except Exception:
            if self._sla_discovery_attempts < self._MAX_SLA_DISCOVERY_ATTEMPTS:
                logger.error(
                    f"JSM SLA field discovery failed (attempt "
                    f"{self._sla_discovery_attempts}/{self._MAX_SLA_DISCOVERY_ATTEMPTS}) — "
                    f"SLA metadata will be omitted for this document. "
                    f"Retrying on next document. "
                    f"Check connector credentials / permissions.",
                    exc_info=True,
                )
                # Do NOT cache on failure yet — allow retries up to the cap.
                return None
            else:
                logger.error(
                    f"JSM SLA field discovery failed (attempt "
                    f"{self._sla_discovery_attempts}/{self._MAX_SLA_DISCOVERY_ATTEMPTS}) — "
                    f"No more retries; SLA enrichment disabled for this run. "
                    f"Check connector credentials / permissions.",
                    exc_info=True,
                )
                # Max attempts reached — cache empty map so all future calls
                # hit the fast-path and make zero further API calls.
                self._sla_field_map = mapping

        return mapping

    # ------------------------------------------------------------------
    # _enrich_document hook
    # ------------------------------------------------------------------

    @override
    def _enrich_document(self, document: Document, issue: Issue) -> Document:
        """Attach JSM-specific metadata (SLA, request type) to a document.

        This method is called by the base class ``_load_from_checkpoint``
        immediately after ``process_jira_issue`` succeeds.  It mutates
        ``document.metadata`` in-place and returns the same object.

        Failures are logged at WARNING level and never propagated — a
        missing SLA field must never cause an otherwise-healthy document
        to be dropped.
        """
        try:
            self._attach_sla_metadata(document, issue)
        except Exception:
            logger.warning(
                f"Failed to attach SLA metadata to {document.id!r}",
                exc_info=True,
            )

        try:
            self._attach_jsm_metadata(document, issue)
        except Exception:
            logger.warning(
                f"Failed to attach JSM metadata to {document.id!r}",
                exc_info=True,
            )

        return document

    def _attach_sla_metadata(self, document: Document, issue: Issue) -> None:
        """Populate SLA-related keys in ``document.metadata``."""
        sla_field_map = self._discover_sla_fields()
        if sla_field_map is None:
            logger.debug(
                f"SLA field discovery not yet complete (transient failure); "
                f"skipping SLA enrichment for {document.id!r}."
            )
            return
        if not sla_field_map:
            logger.debug(
                f"SLA field map is empty; skipping SLA enrichment for {document.id!r}."
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
        request_type = _get_request_type(issue)
        if request_type:
            document.metadata[_META_REQUEST_TYPE] = request_type

        service_desk_id = _get_service_desk_id(issue)
        if service_desk_id:
            document.metadata[_META_SERVICE_DESK] = service_desk_id        # Fast-path: already discovered (successfully or permanently failed).
        if self._sla_field_map is not None:
            return self._sla_field_map

        self._sla_discovery_attempts += 1
        mapping: dict[str, str] = {}
        try:
            all_fields: list[dict[str, Any]] = self.jira_client.fields()
            for field_meta in all_fields:
                field_id: str = field_meta.get("id", "")
                field_name: str = field_meta.get("name", "")
                if not field_id.startswith("customfield_"):
                    continue
                for pattern, canonical_key in _COMPILED_SLA_PATTERNS:
                    if pattern.search(field_name):
                        mapping[field_id] = canonical_key
                        logger.debug(
                            f"JSM SLA field discovered: {field_id!r} "
                            f"({field_name!r}) → {canonical_key!r}"
                        )
                        break  # one canonical key per field ID
            # Success — cache so we never call fields() again.
            self._sla_field_map = mapping
        except Exception:
            if self._sla_discovery_attempts < self._MAX_SLA_DISCOVERY_ATTEMPTS:
                logger.error(
                    f"JSM SLA field discovery failed (attempt "
                    f"{self._sla_discovery_attempts}/{self._MAX_SLA_DISCOVERY_ATTEMPTS}) — "
                    f"SLA metadata will be omitted for this document. "
                    f"Retrying on next document. "
                    f"Check connector credentials / permissions.",
                    exc_info=True,
                )
                # Do NOT cache on failure yet — allow retries up to the cap.
                return None
            else:
                logger.error(
                    f"JSM SLA field discovery failed (attempt "
                    f"{self._sla_discovery_attempts}/{self._MAX_SLA_DISCOVERY_ATTEMPTS}) — "
                    f"No more retries; SLA enrichment disabled for this run. "
                    f"Check connector credentials / permissions.",
                    exc_info=True,
                )
                # Max attempts reached — cache empty map so all future calls
                # hit the fast-path and make zero further API calls.
                self._sla_field_map = {}

        return mapping

    # ------------------------------------------------------------------
    # _enrich_document hook
    # ------------------------------------------------------------------

    @override
    def _enrich_document(self, document: Document, issue: Issue) -> Document:
        """Attach JSM-specific metadata (SLA, request type) to a document.

        This method is called by the base class ``_load_from_checkpoint``
        immediately after ``process_jira_issue`` succeeds.  It mutates
        ``document.metadata`` in-place and returns the same object.

        Failures are logged at WARNING level and never propagated — a
        missing SLA field must never cause an otherwise-healthy document
        to be dropped.
        """
        try:
            self._attach_sla_metadata(document, issue)
        except Exception:
            logger.warning(
                f"Failed to attach SLA metadata to {document.id!r}",
                exc_info=True,
            )

        try:
            self._attach_jsm_metadata(document, issue)
        except Exception:
            logger.warning(
                f"Failed to attach JSM metadata to {document.id!r}",
                exc_info=True,
            )

        return document

    def _attach_sla_metadata(self, document: Document, issue: Issue) -> None:
        """Populate SLA-related keys in ``document.metadata``."""
        sla_field_map = self._discover_sla_fields()
        if sla_field_map is None:
            logger.debug(
                f"SLA field discovery not yet complete (transient failure); "
                f"skipping SLA enrichment for {document.id!r}."
            )
            return
        if not sla_field_map:
            logger.debug(
                f"SLA field map is empty; skipping SLA enrichment for {document.id!r}."
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
        request_type = _get_request_type(issue)
        if request_type:
            document.metadata[_META_REQUEST_TYPE] = request_type

        service_desk_id = _get_service_desk_id(issue)
        if service_desk_id:
            document.metadata[_META_SERVICE_DESK] = service_desk_id
