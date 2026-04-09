"""
Unit tests for the Jira Service Management connector.

Coverage targets
----------------
* Source tagging (all documents tagged with JIRA_SERVICE_MANAGEMENT)
* Dynamic SLA field discovery — success, partial match, API failure, caching
* SLA value extraction — Cloud nested dict, Server plain-string, breach flags,
  completed cycles, None / unknown shapes
* Document enrichment via _enrich_document hook
* JSM metadata (request type, service desk ID)
* doc_sync URL validation (P1 fix)
* Interface smoke tests (instantiation, load_credentials)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from onyx.configs.constants import DocumentSource
from onyx.connectors.jira_service_management.connector import _extract_sla_display
from onyx.connectors.jira_service_management.connector import _get_raw_field
from onyx.connectors.jira_service_management.connector import _get_request_type
from onyx.connectors.jira_service_management.connector import _get_service_desk_id
from onyx.connectors.jira_service_management.connector import (
    JiraServiceManagementConnector,
)
from onyx.connectors.models import Document
from onyx.connectors.models import TextSection
from tests.unit.onyx.connectors.jira_service_management.conftest import make_mock_issue


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_doc(doc_id: str = "https://example.atlassian.net/browse/HELP-1") -> Document:
    return Document(
        id=doc_id,
        sections=[TextSection(link=doc_id, text="some content")],
        source=DocumentSource.JIRA,  # deliberately wrong — enrichment must fix it
        semantic_identifier="HELP-1: Need help",
        title="HELP-1 Need help",
        metadata={},
    )


def _make_field_meta(field_id: str, name: str) -> dict[str, str]:
    return {"id": field_id, "name": name, "schema": {"type": "any"}}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Instantiation and source attribute
# ──────────────────────────────────────────────────────────────────────────────


class TestInstantiation:
    def test_source_attribute_is_jsm(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        assert jsm_connector._source is DocumentSource.JIRA_SERVICE_MANAGEMENT

    def test_sla_field_map_starts_as_none(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        assert jsm_connector._sla_field_map is None

    def test_inherits_from_jira_connector(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        from onyx.connectors.jira.connector import JiraConnector

        assert isinstance(jsm_connector, JiraConnector)

    def test_project_key_stored(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        assert jsm_connector.jira_project == "HELP"

    def test_base_url_stored(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        assert jsm_connector.jira_base == "https://example.atlassian.net"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dynamic SLA field discovery
# ──────────────────────────────────────────────────────────────────────────────


class TestSLAFieldDiscovery:
    def test_discovers_time_to_first_response(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = [
            _make_field_meta("customfield_10020", "Time to first response"),
            _make_field_meta("summary", "Summary"),  # non-custom — must be ignored
        ]
        result = jsm_connector._discover_sla_fields()
        assert result == {"customfield_10020": "sla_time_to_first_response"}

    def test_discovers_time_to_resolution(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = [
            _make_field_meta("customfield_10030", "Time to resolution"),
        ]
        result = jsm_connector._discover_sla_fields()
        assert "customfield_10030" in result
        assert result["customfield_10030"] == "sla_time_to_resolution"

    def test_discovery_is_case_insensitive(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = [
            _make_field_meta("customfield_10050", "TIME TO FIRST RESPONSE"),
        ]
        result = jsm_connector._discover_sla_fields()
        assert "customfield_10050" in result

    def test_non_customfield_ids_are_ignored(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = [
            _make_field_meta("summary", "Time to first response"),
            _make_field_meta("description", "Time to resolution"),
        ]
        result = jsm_connector._discover_sla_fields()
        assert result == {}

    def test_empty_fields_returns_empty_map(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = []
        assert jsm_connector._discover_sla_fields() == {}

    def test_api_failure_returns_empty_map_and_does_not_raise(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.side_effect = RuntimeError("API down")
        result = jsm_connector._discover_sla_fields()
        assert result == {}

    def test_discovery_cached_after_first_call(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = [
            _make_field_meta("customfield_10020", "Time to first response"),
        ]
        first = jsm_connector._discover_sla_fields()
        second = jsm_connector._discover_sla_fields()
        # Should only call the API once
        mock_jira_client.fields.assert_called_once()
        assert first is second  # same dict object returned from cache

    def test_discovers_multiple_sla_fields(
        self, jsm_connector: JiraServiceManagementConnector, mock_jira_client: MagicMock
    ) -> None:
        mock_jira_client.fields.return_value = [
            _make_field_meta("customfield_10020", "Time to first response"),
            _make_field_meta("customfield_10030", "Time to resolution"),
            _make_field_meta("customfield_10040", "Time to close"),
        ]
        result = jsm_connector._discover_sla_fields()
        assert len(result) == 3
        assert result["customfield_10020"] == "sla_time_to_first_response"
        assert result["customfield_10030"] == "sla_time_to_resolution"
        assert result["customfield_10040"] == "sla_time_to_close"


# ──────────────────────────────────────────────────────────────────────────────
# 3. SLA value extraction (_extract_sla_display)
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractSLADisplay:
    def test_none_returns_none_not_breached(self) -> None:
        display, breached = _extract_sla_display(None)
        assert display is None
        assert breached is False

    def test_plain_string_returns_as_is(self) -> None:
        display, breached = _extract_sla_display("2h 30m")
        assert display == "2h 30m"
        assert breached is False

    def test_empty_string_returns_none(self) -> None:
        display, breached = _extract_sla_display("")
        assert display is None
        assert breached is False

    def test_server_dc_simple_dict(self) -> None:
        sla = {"text": "1h 15m", "breached": False}
        display, breached = _extract_sla_display(sla)
        assert display == "1h 15m"
        assert breached is False

    def test_server_dc_breached_dict(self) -> None:
        sla = {"text": "Breached", "breached": True}
        display, breached = _extract_sla_display(sla)
        assert display == "Breached"
        assert breached is True

    def test_cloud_ongoing_cycle_not_breached(self) -> None:
        sla = {
            "ongoingCycle": {
                "remainingTime": {"friendly": "3h 0m", "millis": 10800000},
                "breached": False,
                "paused": False,
            }
        }
        display, breached = _extract_sla_display(sla)
        assert display == "3h 0m"
        assert breached is False

    def test_cloud_ongoing_cycle_breached(self) -> None:
        sla = {
            "ongoingCycle": {
                "remainingTime": {"friendly": "", "millis": -3600000},
                "breached": True,
            }
        }
        display, breached = _extract_sla_display(sla)
        # friendly is empty so falls back to "Breached"
        assert display == "Breached"
        assert breached is True

    def test_cloud_completed_cycle(self) -> None:
        sla = {
            "completedCycles": [
                {
                    "remainingTime": {"friendly": "0h 30m"},
                    "breached": False,
                },
            ]
        }
        display, breached = _extract_sla_display(sla)
        assert display == "0h 30m"
        assert breached is False

    def test_cloud_completed_cycle_breached(self) -> None:
        sla = {
            "completedCycles": [
                {
                    "remainingTime": {"friendly": ""},
                    "breached": True,
                },
            ]
        }
        display, breached = _extract_sla_display(sla)
        assert breached is True

    def test_unknown_type_returns_none(self) -> None:
        display, breached = _extract_sla_display(12345)  # type: ignore[arg-type]
        assert display is None
        assert breached is False

    def test_empty_dict_returns_none(self) -> None:
        display, breached = _extract_sla_display({})
        assert display is None
        assert breached is False


# ──────────────────────────────────────────────────────────────────────────────
# 4. Document enrichment — _enrich_document hook
# ──────────────────────────────────────────────────────────────────────────────


class TestEnrichDocument:
    def test_source_always_set_to_jsm(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {}  # skip discovery
        doc = _make_doc()
        issue = make_mock_issue()
        result = jsm_connector._enrich_document(doc, issue)
        assert result.source is DocumentSource.JIRA_SERVICE_MANAGEMENT

    def test_sla_metadata_attached_when_field_present(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {
            "customfield_10020": "sla_time_to_first_response",
        }
        sla_value = {"text": "2h 0m", "breached": False}
        issue = make_mock_issue(extra_fields={"customfield_10020": sla_value})
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        assert result.metadata.get("sla_time_to_first_response") == "2h 0m"

    def test_breach_flag_attached_when_breached(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {
            "customfield_10020": "sla_time_to_first_response",
        }
        sla_value = {"text": "Breached", "breached": True}
        issue = make_mock_issue(extra_fields={"customfield_10020": sla_value})
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        assert result.metadata.get("sla_time_to_first_response_breached") == "true"

    def test_no_breach_flag_when_not_breached(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {
            "customfield_10020": "sla_time_to_first_response",
        }
        sla_value = {"text": "1h", "breached": False}
        issue = make_mock_issue(extra_fields={"customfield_10020": sla_value})
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        assert "sla_time_to_first_response_breached" not in result.metadata

    def test_multiple_sla_fields_all_attached(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {
            "customfield_10020": "sla_time_to_first_response",
            "customfield_10030": "sla_time_to_resolution",
        }
        issue = make_mock_issue(
            extra_fields={
                "customfield_10020": {"text": "2h", "breached": False},
                "customfield_10030": {"text": "4h", "breached": False},
            }
        )
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        assert "sla_time_to_first_response" in result.metadata
        assert "sla_time_to_resolution" in result.metadata

    def test_missing_sla_field_on_issue_does_not_add_metadata(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {
            "customfield_10020": "sla_time_to_first_response",
        }
        issue = make_mock_issue()  # customfield_10020 not set
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        assert "sla_time_to_first_response" not in result.metadata

    def test_sla_error_does_not_drop_document(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        """A bad SLA value must never cause the document to be dropped."""
        jsm_connector._sla_field_map = {
            "customfield_10020": "sla_time_to_first_response",
        }
        broken_sla = (
            object()
        )  # not a dict or str — will fail inside _extract_sla_display
        issue = make_mock_issue(extra_fields={"customfield_10020": broken_sla})
        doc = _make_doc()
        # Must not raise
        result = jsm_connector._enrich_document(doc, issue)
        assert result is not None
        assert result.source is DocumentSource.JIRA_SERVICE_MANAGEMENT

    def test_empty_sla_map_skips_sla_enrichment(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {}
        issue = make_mock_issue()
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        # No SLA keys added, but source still fixed
        sla_keys = [k for k in result.metadata if k.startswith("sla_")]
        assert sla_keys == []
        assert result.source is DocumentSource.JIRA_SERVICE_MANAGEMENT

    def test_returns_same_document_object(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {}
        doc = _make_doc()
        issue = make_mock_issue()
        result = jsm_connector._enrich_document(doc, issue)
        assert result is doc


# ──────────────────────────────────────────────────────────────────────────────
# 5. JSM metadata helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestJSMMetadataHelpers:
    def test_get_raw_field_returns_value(self) -> None:
        issue = make_mock_issue(extra_fields={"customfield_10020": "2h"})
        assert _get_raw_field(issue, "customfield_10020") == "2h"

    def test_get_raw_field_missing_returns_none(self) -> None:
        issue = make_mock_issue()
        assert _get_raw_field(issue, "customfield_99999") is None

    def test_get_request_type_from_field(self) -> None:
        issue = make_mock_issue()
        rt = MagicMock()
        rt.name = "IT Support"
        issue.fields.requestType = rt
        assert _get_request_type(issue) == "IT Support"

    def test_get_request_type_missing_returns_none(self) -> None:
        issue = make_mock_issue()
        issue.fields.requestType = None
        assert _get_request_type(issue) is None

    def test_get_service_desk_id_from_project_key(self) -> None:
        issue = make_mock_issue(project_key="HELP")
        result = _get_service_desk_id(issue)
        assert result == "HELP"

    def test_jsm_metadata_attached_to_document(
        self, jsm_connector: JiraServiceManagementConnector
    ) -> None:
        jsm_connector._sla_field_map = {}
        issue = make_mock_issue(project_key="SD")
        rt = MagicMock()
        rt.name = "Password Reset"
        issue.fields.requestType = rt
        doc = _make_doc()
        result = jsm_connector._enrich_document(doc, issue)
        assert result.metadata.get("jsm_request_type") == "Password Reset"
        assert result.metadata.get("jsm_service_desk_id") == "SD"


# ──────────────────────────────────────────────────────────────────────────────
# 6. doc_sync URL validation (P1 fix)
# ──────────────────────────────────────────────────────────────────────────────


class TestDocSyncURLValidation:
    def _call_validate(self, config: dict) -> None:
        from ee.onyx.external_permissions.jira_service_management.doc_sync import (
            _validate_jsm_config,
        )

        _validate_jsm_config(config)

    def test_valid_https_url_passes(self) -> None:
        self._call_validate({"jira_base_url": "https://example.atlassian.net"})

    def test_valid_http_url_passes(self) -> None:
        self._call_validate({"jira_base_url": "http://jira.internal.corp"})

    def test_missing_key_raises(self) -> None:
        from onyx.connectors.exceptions import ConnectorValidationError

        with pytest.raises(ConnectorValidationError, match="jira_base_url"):
            self._call_validate({})

    def test_empty_string_raises(self) -> None:
        from onyx.connectors.exceptions import ConnectorValidationError

        with pytest.raises(ConnectorValidationError, match="non-empty"):
            self._call_validate({"jira_base_url": ""})

    def test_whitespace_only_raises(self) -> None:
        from onyx.connectors.exceptions import ConnectorValidationError

        with pytest.raises(ConnectorValidationError, match="non-empty"):
            self._call_validate({"jira_base_url": "   "})

    def test_url_without_scheme_raises(self) -> None:
        from onyx.connectors.exceptions import ConnectorValidationError

        with pytest.raises(ConnectorValidationError, match="http"):
            self._call_validate({"jira_base_url": "example.atlassian.net"})

    def test_non_string_value_raises(self) -> None:
        from onyx.connectors.exceptions import ConnectorValidationError

        with pytest.raises(ConnectorValidationError):
            self._call_validate({"jira_base_url": 12345})


# ──────────────────────────────────────────────────────────────────────────────
# 7. Cloud SLA format — extended edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestCloudSLAEdgeCases:
    def test_ongoing_cycle_missing_remainingTime_friendly(self) -> None:
        sla = {
            "ongoingCycle": {
                "remainingTime": {},  # no "friendly" key
                "breached": False,
            }
        }
        display, breached = _extract_sla_display(sla)
        assert display is None
        assert breached is False

    def test_ongoing_cycle_remainingTime_not_dict(self) -> None:
        sla = {
            "ongoingCycle": {
                "remainingTime": "1h",  # Server style inside Cloud wrapper
                "breached": False,
            }
        }
        display, breached = _extract_sla_display(sla)
        # remainingTime is not a dict so friendly cannot be extracted
        assert display is None

    def test_multiple_completed_cycles_uses_last(self) -> None:
        sla = {
            "completedCycles": [
                {"remainingTime": {"friendly": "old"}, "breached": False},
                {"remainingTime": {"friendly": "newest"}, "breached": True},
            ]
        }
        display, breached = _extract_sla_display(sla)
        assert display == "newest"
        assert breached is True

    def test_empty_completedCycles_list(self) -> None:
        sla = {"completedCycles": []}
        display, breached = _extract_sla_display(sla)
        assert display is None
        assert breached is False
