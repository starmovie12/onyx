"""
Shared fixtures for Jira Service Management unit tests.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from jira import JIRA
from jira.resources import Issue

from onyx.connectors.jira_service_management.connector import (
    JiraServiceManagementConnector,
)
from onyx.connectors.jira.utils import JIRA_SERVER_API_VERSION


# ---------------------------------------------------------------------------
# Basic value fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsm_base_url() -> str:
    return "https://example.atlassian.net"


@pytest.fixture
def jsm_project_key() -> str:
    return "HELP"


# ---------------------------------------------------------------------------
# Mock Jira client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_jira_client() -> MagicMock:
    mock = MagicMock(spec=JIRA)
    mock._options = {"rest_api_version": JIRA_SERVER_API_VERSION}
    mock.search_issues = MagicMock(return_value=[])
    mock.project = MagicMock()
    mock.projects = MagicMock(return_value=[])
    mock.fields = MagicMock(return_value=[])
    return mock


# ---------------------------------------------------------------------------
# Connector fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def jsm_connector(
    jsm_base_url: str,
    jsm_project_key: str,
    mock_jira_client: MagicMock,
) -> Generator[JiraServiceManagementConnector, None, None]:
    connector = JiraServiceManagementConnector(
        jira_base_url=jsm_base_url,
        project_key=jsm_project_key,
        comment_email_blacklist=[],
        labels_to_skip=[],
    )
    connector._jira_client = mock_jira_client
    connector._jira_client.client_info = MagicMock(return_value=jsm_base_url)
    with patch("onyx.connectors.jira.connector._JIRA_FULL_PAGE_SIZE", 2):
        yield connector


# ---------------------------------------------------------------------------
# Issue factory
# ---------------------------------------------------------------------------


def make_mock_issue(
    key: str = "HELP-1",
    summary: str = "Need help",
    description: str = "Issue description",
    updated: str = "2024-01-01T12:00:00.000+0000",
    project_key: str = "HELP",
    project_name: str = "Help Desk",
    issuetype_name: str = "Service Request",
    labels: list[str] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a lightweight mock ``Issue`` object."""
    issue = MagicMock(spec=Issue)
    issue.key = key
    issue.fields = MagicMock()
    issue.fields.summary = summary
    issue.fields.description = description
    issue.fields.updated = updated
    issue.fields.labels = labels or []
    issue.fields.created = updated
    issue.fields.duedate = None
    issue.fields.resolutiondate = None

    # issuetype
    issue.fields.issuetype = MagicMock()
    issue.fields.issuetype.name = issuetype_name

    # project
    issue.fields.project = MagicMock()
    issue.fields.project.key = project_key
    issue.fields.project.name = project_name

    # reporter / assignee — minimal
    issue.fields.reporter = MagicMock()
    issue.fields.reporter.displayName = "Reporter Name"
    issue.fields.reporter.emailAddress = "reporter@example.com"

    issue.fields.assignee = None
    issue.fields.priority = None
    issue.fields.status = MagicMock()
    issue.fields.status.name = "Open"
    issue.fields.resolution = None
    issue.fields.parent = None

    # raw dict (used by extract_text_from_adf)
    issue.raw = {
        "fields": {
            "description": description,
            **(extra_fields or {}),
        }
    }

    # Apply any extra_fields as attributes too
    if extra_fields:
        for fname, fval in extra_fields.items():
            setattr(issue.fields, fname, fval)

    return issue
