"""
Jira Service Management Connector.

Inherits from the standard Jira connector since JSM shares the same
Jira REST API. The only material difference is that indexed documents
are tagged with DocumentSource.JIRA_SERVICE_MANAGEMENT instead of
DocumentSource.JIRA, so that JSM content is distinguishable from
regular Jira content in search results and permission sync.
"""

from typing_extensions import override

from onyx.configs.constants import DocumentSource
from onyx.connectors.interfaces import CheckpointOutput
from onyx.connectors.jira.connector import JiraConnector
from onyx.connectors.jira.connector import JiraConnectorCheckpoint
from onyx.connectors.models import Document


class JiraServiceManagementConnector(JiraConnector):
    """
    Connector for Jira Service Management (JSM) projects.

    Reuses all indexing, pagination, ADF parsing, hierarchy, permission
    logic, and permission caching from the standard Jira connector.

    The ``_source`` class attribute causes the inherited
    ``_get_project_permissions`` to automatically prefix EE permission
    group IDs with ``jira_service_management_`` instead of ``jira_``,
    with no caching logic duplication required.
    """

    # Overriding the class-level attribute is the only change needed for
    # correct EE permission group prefixing (see JiraConnector._get_project_permissions).
    _source: DocumentSource = DocumentSource.JIRA_SERVICE_MANAGEMENT

    @override
    def _load_from_checkpoint(
        self,
        jql: str,
        checkpoint: JiraConnectorCheckpoint,
        include_permissions: bool,
    ) -> CheckpointOutput[JiraConnectorCheckpoint]:
        """Wrap the parent generator, re-tagging Documents with the JSM source."""
        gen = super()._load_from_checkpoint(jql, checkpoint, include_permissions)
        try:
            while True:
                item = next(gen)
                if isinstance(item, Document):
                    item.source = DocumentSource.JIRA_SERVICE_MANAGEMENT
                yield item
        except StopIteration as e:
            return e.value  # the updated checkpoint
