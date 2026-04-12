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
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from typing_extensions import override

from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.app_configs import JIRA_CONNECTOR_LABELS_TO_SKIP
from onyx.configs.constants import DocumentSource
from onyx.connectors.jira.connector import JiraConnector
from onyx.connectors.models import Document
from onyx.utils.logger import setup_logger

# Issue is only imported for type-checking.  At runtime the jira library is
# always available (it is a hard dependency of JiraConnector), but using
# TYPE_CHECKING keeps mypy happy and avoids a circular-import risk.
if TYPE_CHECKING:
    from jira.resources import Issue

logger = setup_logger()

# ---------------------------------------------------------------------------
# SLA field-name patterns
# ---------------------------------------------------------------------------
# These patterns are matched (case-insensitively) against the *name* field
# returned by the Jira ``/rest/api/2/field`` endpoint.  They are broad
# enough to catch common Jira-Cloud and Server naming conventions while
# remaining specific enough not to collide with ordinary custom fields.
_SLA_FIELD_PATTERNS: Final[list[tuple[str, str]]] = [
    # (regex pattern,  canonical metadata key)
    (r"time\s+to\s+first\s+response", "sla_time_to_first_response"),
    (r"time\s+to\s+resolution", "sla_time_to_resolution"),
    (r"time\s+to\s+close", "sla_time_to_close"),
    (r"time\s+to\s+respond", "sla_time_to_respond"),
    (r"time\s+to\s+approve", "sla_time_to_approve"),
    (r"satisfaction\s+rating", "sla_satisfaction_rating"),
]

# Pre-compiled versions of the above patterns.  Expressed as a plain list
# comprehension at module level — no hidden function call, no side-effectful
# wrapper function.
_COMPILED_SLA_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(pat, re.IGNORECASE), key) for pat, key in _SLA_FIELD_PATTERNS
]

# Pattern used to identify the "Customer Request Type" custom field on
# Server/Data Center deployments.
_CUSTOMER_REQUEST_TYPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"customer\s+request\s+type", re.IGNORECASE
)

# Metadata keys under which JSM-specific values are stored.
_META_REQUEST_TYPE: Final[str] = "jsm_request_type"
_META_SERVICE_DESK: Final[str] = "jsm_service_desk_id"

# Suffix appended to an SLA metadata key when the SLA has been breached.
_BREACH_SUFFIX: Final[str] = "_breached"


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


def _get_raw_field(issue: "Issue", field_id: str) -> Any:
    """Safely retrieve a raw field value from an Issue object.

    Returns ``None`` rather than raising if the field is absent or if the
    Issue object itself is in an unexpected state.  SLA metadata errors must
    never cause a document to be dropped.
    """
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
    issue: "Issue",
    request_type_field_id: str | None = None,
) -> str | None:
    """Extract the JSM request type name from an issue, if present.

    Args:
        issue: The Jira Issue object.
        request_type_field_id: The custom field ID (e.g. ``"customfield_10020"``)
            that holds the request type on Server/DC deployments.  Discovered
            once at startup by ``_discover_request_type_mapping`` and passed in
            here so we perform a targeted lookup rather than scanning every raw
            custom field — which could match unrelated fields whose stored value
            happens to contain a ``requestType`` dict.
    """
    try:
        # Cloud path: the JSM REST layer surfaces this as a first-class attribute.
        rt = getattr(issue.fields, "requestType", None)
        if rt is not None:
            # Return the name attribute when present; fall back to None rather than
            # stringifying the object (which could store a Python repr in metadata).
            return getattr(rt, "name", None) or None

        # Server/DC path: targeted lookup using the discovered field ID only.
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


def _get_service_desk_id(issue: "Issue") -> str | None:
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
            "Failed to extract serviceDeskId from issue %r",
            getattr(issue, "key", "?"),
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

    _source: ClassVar[DocumentSource] = DocumentSource.JIRA_SERVICE_MANAGEMENT

    # Maximum number of times we will attempt field discovery before giving
    # up for the lifetime of this connector run.  This prevents a persistent
    # failure (e.g. a missing OAuth scope) from issuing one failing HTTP call
    # per issue when indexing large projects.
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
        # ``None`` means discovery has not yet been attempted or is still
        # retrying after a transient failure; ``{}`` means discovery succeeded
        # but found no SLA fields on this instance.
        self._sla_field_map: dict[str, str] | None = None

        # Tracks how many times field discovery has been attempted so that
        # persistent failures do not generate unbounded failing API calls.
        self._sla_discovery_attempts: int = 0

        # The customfield_* ID whose value holds the "Customer Request Type"
        # on Server/DC deployments.  Populated as a side-output of
        # ``_ensure_fields_discovered``; ``None`` until discovery runs or
        # when the field does not exist on this instance.
        self._request_type_field_id: str | None = None

        # Set to True once the discovery phase is fully complete (successfully
        # or after permanently exhausting retries).  Used as the fast-path
        # sentinel in ``_ensure_fields_discovered`` so that ``_sla_field_map``
        # carries only its semantic meaning and does not double as a state flag.
        self._fields_discovered: bool = False

    # ------------------------------------------------------------------
    # Field discovery
    # ------------------------------------------------------------------

    def _ensure_fields_discovered(self) -> None:
        """Fetch the Jira field list once and populate SLA and request-type caches.

        Calls ``GET /rest/api/2/field`` on the first invocation and caches
        the result.  Transient failures are retried on subsequent calls up to
        ``_MAX_SLA_DISCOVERY_ATTEMPTS`` times; after that the method caches an
        empty map so future calls hit the fast-path and make zero further API
        calls.

        Both SLA and request-type discovery are delegated to isolated helper
        methods (``_discover_sla_mapping`` and ``_discover_request_type_mapping``)
        so that a processing error in one cannot corrupt the cached state of
        the other.

        Callers read ``self._sla_field_map`` and ``self._request_type_field_id``
        directly after this call rather than using a return value.
        """
        # Fast-path: already discovered (successfully or permanently failed).
        # Using a dedicated boolean rather than testing ``_sla_field_map is not
        # None`` prevents the permanent-failure branch from inadvertently
        # bypassing ``_discover_request_type_mapping`` just because it sets
        # ``_sla_field_map = {}`` before returning.
        if self._fields_discovered:
            return

        self._sla_discovery_attempts += 1

        # Fetch the field list once; both helpers share it.
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
                # Do NOT cache on failure yet — allow retries up to the cap.
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
                # Max attempts reached — cache empty map and mark discovery
                # complete so all future calls hit the fast-path with zero
                # further API calls.
                self._sla_field_map = {}
                self._fields_discovered = True
                return

        # Each helper has its own failure scope — one cannot corrupt the other.
        self._discover_sla_mapping(all_fields)
        self._discover_request_type_mapping(all_fields)
        self._fields_discovered = True

    def _discover_sla_mapping(self, all_fields: list[dict[str, Any]]) -> None:
        """Populate ``_sla_field_map`` from a pre-fetched field list.

        Isolated from request-type discovery so that a processing error here
        does not affect ``_request_type_field_id`` state.
        """
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
                        break  # one canonical key per field ID
            self._sla_field_map = mapping
        except Exception:
            logger.warning(
                "JSM SLA field processing failed after caching %d field(s) — "
                "SLA metadata may be incomplete.",
                len(mapping),
                exc_info=True,
            )
            # Cache the partial result so we don't retry the processing loop
            # on every subsequent document.
            self._sla_field_map = mapping

    def _discover_request_type_mapping(self, all_fields: list[dict[str, Any]]) -> None:
        """Populate ``_request_type_field_id`` from a pre-fetched field list.

        Isolated from SLA discovery so that a processing error here does not
        affect ``_sla_field_map`` state.  Only the first matching field is
        used; Jira instances should have exactly one "Customer Request Type"
        field.
        """
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
    # _enrich_document hook
    # ------------------------------------------------------------------

    @override
    def _enrich_document(self, document: Document, issue: "Issue") -> Document:
        """Attach JSM-specific metadata (SLA, request type) to a document.

        This method is called by the base class ``_load_from_checkpoint``
        immediately after ``process_jira_issue`` succeeds.  It mutates
        ``document.metadata`` in-place and returns the same object.

        Field discovery is performed exactly once per document here and the
        cached state is shared between the two metadata helpers, preventing
        the retry budget from being consumed twice per document during
        transient API failures.

        Failures are logged at WARNING level and never propagated — a
        missing SLA field must never cause an otherwise-healthy document
        to be dropped.
        """
        # Single discovery call per document; both helpers read cached state.
        self._ensure_fields_discovered()

        try:
            self._attach_sla_metadata(document, issue)
        except Exception:
            logger.warning(
                "Failed to attach SLA metadata to %r",
                document.id,
                exc_info=True,
            )

        try:
            self._attach_jsm_metadata(document, issue)
        except Exception:
            logger.warning(
                "Failed to attach JSM metadata to %r",
                document.id,
                exc_info=True,
            )

        return document

    def _attach_sla_metadata(self, document: Document, issue: "Issue") -> None:
        """Populate SLA-related keys in ``document.metadata``.

        Reads ``self._sla_field_map`` which is guaranteed to be populated
        (or left as ``None`` on a transient failure) by the
        ``_ensure_fields_discovered`` call in ``_enrich_document``.
        """
        # Discovery already called by _enrich_document; use cached result.
        sla_field_map: dict[str, str] | None = self._sla_field_map
        if sla_field_map is None:
            # Transient failure during this document's discovery attempt.
            logger.debug(
                "SLA field discovery not yet complete (transient failure); "
                "skipping SLA enrichment for %r.",
                document.id,
            )
            return
        if not sla_field_map:
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

    def _attach_jsm_metadata(self, document: Document, issue: "Issue") -> None:
        """Populate non-SLA JSM-specific metadata keys.

        ``_request_type_field_id`` is already populated (or left as ``None``)
        by the ``_ensure_fields_discovered`` call in ``_enrich_document``, so
        no further discovery is needed here.
        """
        request_type = _get_request_type(issue, self._request_type_field_id)
        if request_type:
            document.metadata[_META_REQUEST_TYPE] = request_type

        service_desk_id = _get_service_desk_id(issue)
        if service_desk_id:
            document.metadata[_META_SERVICE_DESK] = service_desk_id
