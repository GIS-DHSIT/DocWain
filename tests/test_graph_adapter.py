from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.docwain_intel.graph_adapter import (
    CypherGraphAdapter,
    GraphEdge,
    GraphNode,
    _try_connect_memgraph,
    _try_connect_neo4j,
    get_graph_adapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_driver():
    """Return a mock neo4j driver whose session().run() yields record-like objects."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


def _set_run_result(session, rows: list[dict]):
    """Configure session.run() to return record mocks with .data()."""
    records = []
    for row in rows:
        rec = MagicMock()
        rec.data.return_value = row
        records.append(rec)
    session.run.return_value = records


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestGraphNode:
    def test_create_minimal(self):
        node = GraphNode(node_id="n1", node_type="Entity")
        assert node.node_id == "n1"
        assert node.node_type == "Entity"
        assert node.properties == {}

    def test_create_with_properties(self):
        node = GraphNode(node_id="d1", node_type="Document", properties={"title": "Resume"})
        assert node.properties["title"] == "Resume"


class TestGraphEdge:
    def test_create_minimal(self):
        edge = GraphEdge(source_id="a", target_id="b", edge_type="RELATES_TO")
        assert edge.source_id == "a"
        assert edge.edge_type == "RELATES_TO"
        assert edge.properties == {}

    def test_create_with_properties(self):
        edge = GraphEdge(
            source_id="a", target_id="b", edge_type="MENTIONED_IN",
            properties={"confidence": 0.95},
        )
        assert edge.properties["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Adapter unit tests (mocked driver)
# ---------------------------------------------------------------------------

class TestCypherGraphAdapter:
    def setup_method(self):
        self.driver, self.session = _make_mock_driver()
        self.adapter = CypherGraphAdapter(self.driver)

    # -- upsert_node ---------------------------------------------------------

    def test_upsert_node_calls_session_run(self):
        _set_run_result(self.session, [])
        node = GraphNode(node_id="e1", node_type="Entity", properties={"text": "Alice"})
        self.adapter.upsert_node(node)
        self.session.run.assert_called_once()
        call_args = self.session.run.call_args
        assert "MERGE" in call_args[1].get("query", "") or "MERGE" in (call_args[0][0] if call_args[0] else "")

    # -- upsert_edge ---------------------------------------------------------

    def test_upsert_edge_calls_session_run(self):
        _set_run_result(self.session, [])
        edge = GraphEdge(source_id="e1", target_id="d1", edge_type="MENTIONED_IN")
        self.adapter.upsert_edge(edge)
        self.session.run.assert_called_once()
        query = self.session.run.call_args[0][0]
        assert "MERGE" in query
        assert "MENTIONED_IN" in query

    # -- get_entity_facts ----------------------------------------------------

    def test_get_entity_facts_returns_list(self):
        fake = [{"entity_id": "e1", "target_props": {"skill": "Python"}, "target_labels": ["Entity"]}]
        _set_run_result(self.session, fake)
        result = self.adapter.get_entity_facts("Alice", "sub1", "prof1")
        assert result == fake

    def test_get_entity_facts_filters_by_profile(self):
        _set_run_result(self.session, [])
        self.adapter.get_entity_facts("Alice", "sub1", "prof1")
        query = self.session.run.call_args[0][0]
        assert "subscription_id" in query
        assert "profile_id" in query

    # -- get_related_entities ------------------------------------------------

    def test_get_related_entities_returns_list(self):
        fake = [{"node_id": "e2", "text": "Bob", "rel_type": "RELATES_TO", "props": {}}]
        _set_run_result(self.session, fake)
        result = self.adapter.get_related_entities("Alice", "sub1", "prof1")
        assert result == fake

    def test_get_related_entities_filters_by_profile(self):
        _set_run_result(self.session, [])
        self.adapter.get_related_entities("Alice", "sub1", "prof1")
        query = self.session.run.call_args[0][0]
        assert "subscription_id" in query
        assert "profile_id" in query

    # -- get_document_entities -----------------------------------------------

    def test_get_document_entities_returns_list(self):
        fake = [{"node_id": "e1", "text": "Alice", "props": {}}]
        _set_run_result(self.session, fake)
        result = self.adapter.get_document_entities("doc1", "sub1", "prof1")
        assert result == fake

    def test_get_document_entities_filters_by_profile(self):
        _set_run_result(self.session, [])
        self.adapter.get_document_entities("doc1", "sub1", "prof1")
        query = self.session.run.call_args[0][0]
        assert "subscription_id" in query
        assert "profile_id" in query

    # -- get_all_entities ----------------------------------------------------

    def test_get_all_entities(self):
        fake = [{"node_id": "e1", "text": "X", "props": {}}]
        _set_run_result(self.session, fake)
        result = self.adapter.get_all_entities("sub1", "prof1")
        assert result == fake

    # -- counts --------------------------------------------------------------

    def test_count_entities(self):
        _set_run_result(self.session, [{"cnt": 42}])
        assert self.adapter.count_entities("sub1", "prof1") == 42

    def test_count_facts(self):
        _set_run_result(self.session, [{"cnt": 7}])
        assert self.adapter.count_facts("sub1", "prof1") == 7

    def test_count_entities_empty(self):
        _set_run_result(self.session, [])
        assert self.adapter.count_entities("sub1", "prof1") == 0

    # -- ensure_schema -------------------------------------------------------

    def test_ensure_schema_runs_without_error(self):
        _set_run_result(self.session, [])
        self.adapter.ensure_schema()
        # 3 labels x 2 indexes = 6 calls
        assert self.session.run.call_count == 6

    # -- close ---------------------------------------------------------------

    def test_close_calls_driver_close(self):
        self.adapter.close()
        self.driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# Singleton / connection tests
# ---------------------------------------------------------------------------

class TestGetGraphAdapter:
    def setup_method(self):
        # Reset the module-level singleton before each test
        import src.docwain_intel.graph_adapter as mod
        mod._adapter_instance = None

    @patch("src.docwain_intel.graph_adapter._try_connect_neo4j")
    @patch("src.docwain_intel.graph_adapter._try_connect_memgraph")
    def test_returns_adapter_when_neo4j_available(self, mock_mg, mock_neo):
        mock_neo.return_value = MagicMock()
        adapter = get_graph_adapter()
        assert adapter is not None
        assert isinstance(adapter, CypherGraphAdapter)
        mock_mg.assert_not_called()

    @patch("src.docwain_intel.graph_adapter._try_connect_neo4j")
    @patch("src.docwain_intel.graph_adapter._try_connect_memgraph")
    def test_falls_back_to_memgraph(self, mock_mg, mock_neo):
        mock_neo.return_value = None
        mock_mg.return_value = MagicMock()
        adapter = get_graph_adapter()
        assert adapter is not None

    @patch("src.docwain_intel.graph_adapter._try_connect_neo4j")
    @patch("src.docwain_intel.graph_adapter._try_connect_memgraph")
    def test_returns_none_when_no_db(self, mock_mg, mock_neo):
        mock_neo.return_value = None
        mock_mg.return_value = None
        assert get_graph_adapter() is None

    @patch("src.docwain_intel.graph_adapter._try_connect_neo4j")
    def test_singleton_returns_same_instance(self, mock_neo):
        mock_neo.return_value = MagicMock()
        a1 = get_graph_adapter()
        a2 = get_graph_adapter()
        assert a1 is a2


class TestConnectionHelpers:
    @patch("src.docwain_intel.graph_adapter.neo4j", None)
    def test_try_connect_neo4j_no_driver_lib(self):
        assert _try_connect_neo4j() is None

    @patch("src.docwain_intel.graph_adapter.neo4j", None)
    def test_try_connect_memgraph_no_driver_lib(self):
        assert _try_connect_memgraph() is None
