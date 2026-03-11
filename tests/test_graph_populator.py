"""Tests for graph_populator.populate_graph()."""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from src.docwain_intel.graph_adapter import CypherGraphAdapter, GraphEdge, GraphNode
from src.docwain_intel.graph_populator import populate_graph
from src.docwain_intel.models import (
    EntitySpan,
    ExtractionResult,
    FactTriple,
    SemanticUnit,
    StructuredDocument,
    UnitType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter() -> MagicMock:
    """Return a mock CypherGraphAdapter."""
    adapter = MagicMock(spec=CypherGraphAdapter)
    return adapter


def _make_entity(entity_id: str, unit_id: str, text: str = "Alice", label: str = "PERSON") -> EntitySpan:
    return EntitySpan(
        entity_id=entity_id,
        text=text,
        normalized=text.lower(),
        label=label,
        unit_id=unit_id,
        char_start=0,
        char_end=len(text),
        confidence=0.95,
        source="spacy",
    )


def _make_unit(unit_id: str, text: str = "Some text", page: int = 1) -> SemanticUnit:
    return SemanticUnit(
        unit_id=unit_id,
        unit_type=UnitType.PARAGRAPH,
        text=text,
        page_start=page,
        page_end=page,
    )


def _make_fact(fact_id: str, subject_id: str, object_id: str, predicate: str = "WORKS_AT") -> FactTriple:
    return FactTriple(
        fact_id=fact_id,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        unit_id="u1",
        raw_text=f"{subject_id} {predicate} {object_id}",
        confidence=0.9,
    )


def _default_kwargs(
    adapter: MagicMock,
    entities: list[EntitySpan] | None = None,
    facts: list[FactTriple] | None = None,
    units: list[SemanticUnit] | None = None,
) -> dict:
    if entities is None:
        entities = [_make_entity("e1", "u1"), _make_entity("e2", "u1", text="Acme", label="ORG")]
    if facts is None:
        facts = [_make_fact("f1", "e1", "e2")]
    if units is None:
        units = [_make_unit("u1"), _make_unit("u2", text="Another chunk", page=2)]

    return dict(
        adapter=adapter,
        extraction=ExtractionResult(
            document_id="doc-1",
            entities=entities,
            facts=facts,
        ),
        structured_doc=StructuredDocument(
            document_id="doc-1",
            units=units,
            unit_count=len(units),
        ),
        document_id="doc-1",
        subscription_id="sub-1",
        profile_id="prof-1",
        fingerprint_tags=["resume", "technical"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPopulateGraph:
    """Core graph population tests."""

    def test_creates_document_node(self):
        adapter = _make_adapter()
        populate_graph(**_default_kwargs(adapter))

        # Find the Document upsert call
        doc_calls = [
            c for c in adapter.upsert_node.call_args_list
            if c[0][0].node_type == "Document"
        ]
        assert len(doc_calls) == 1
        node: GraphNode = doc_calls[0][0][0]
        assert node.node_id == "doc-1"
        assert node.properties["filename"] == "doc-1"
        assert node.properties["fingerprint_tags"] == ["resume", "technical"]

    def test_creates_entity_nodes(self):
        adapter = _make_adapter()
        entities = [
            _make_entity("e1", "u1", "Alice", "PERSON"),
            _make_entity("e2", "u1", "Acme", "ORG"),
            _make_entity("e3", "u1", "Python", "SKILL"),
        ]
        populate_graph(**_default_kwargs(adapter, entities=entities))

        entity_calls = [
            c for c in adapter.upsert_node.call_args_list
            if c[0][0].node_type == "Entity"
        ]
        assert len(entity_calls) == 3
        texts = {c[0][0].properties["text"] for c in entity_calls}
        assert texts == {"Alice", "Acme", "Python"}

    def test_creates_chunk_nodes(self):
        adapter = _make_adapter()
        units = [_make_unit("u1"), _make_unit("u2"), _make_unit("u3")]
        populate_graph(**_default_kwargs(adapter, units=units))

        chunk_calls = [
            c for c in adapter.upsert_node.call_args_list
            if c[0][0].node_type == "Chunk"
        ]
        assert len(chunk_calls) == 3
        for cc in chunk_calls:
            node: GraphNode = cc[0][0]
            assert "text_hash" in node.properties
            assert len(node.properties["text_hash"]) == 12

    def test_creates_appears_in_edges(self):
        adapter = _make_adapter()
        populate_graph(**_default_kwargs(adapter))

        appears_in = [
            c for c in adapter.upsert_edge.call_args_list
            if c[0][0].edge_type == "APPEARS_IN"
        ]
        # 2 entities both in unit u1
        assert len(appears_in) == 2
        edge: GraphEdge = appears_in[0][0][0]
        assert "char_start" in edge.properties
        assert "char_end" in edge.properties
        assert "confidence" in edge.properties

    def test_creates_belongs_to_edges(self):
        adapter = _make_adapter()
        units = [_make_unit("u1"), _make_unit("u2")]
        populate_graph(**_default_kwargs(adapter, units=units))

        belongs_to = [
            c for c in adapter.upsert_edge.call_args_list
            if c[0][0].edge_type == "BELONGS_TO"
        ]
        assert len(belongs_to) == 2
        for bt in belongs_to:
            assert bt[0][0].target_id == "doc-1"

    def test_creates_relates_to_edges(self):
        adapter = _make_adapter()
        entities = [_make_entity("e1", "u1"), _make_entity("e2", "u1", text="Acme")]
        facts = [
            _make_fact("f1", "e1", "e2", "WORKS_AT"),
            _make_fact("f2", "e2", "e1", "EMPLOYS"),
        ]
        populate_graph(**_default_kwargs(adapter, entities=entities, facts=facts))

        relates_to = [
            c for c in adapter.upsert_edge.call_args_list
            if c[0][0].edge_type == "RELATES_TO"
        ]
        assert len(relates_to) == 2
        predicates = {c[0][0].properties["predicate"] for c in relates_to}
        assert predicates == {"WORKS_AT", "EMPLOYS"}

    def test_creates_mentioned_in_edges_with_frequency(self):
        adapter = _make_adapter()
        # Same entity_id appears twice (in two different units)
        entities = [
            _make_entity("e1", "u1", "Alice"),
            _make_entity("e1", "u2", "Alice"),
            _make_entity("e2", "u1", "Bob"),
        ]
        units = [_make_unit("u1"), _make_unit("u2")]
        populate_graph(**_default_kwargs(adapter, entities=entities, units=units))

        mentioned_in = [
            c for c in adapter.upsert_edge.call_args_list
            if c[0][0].edge_type == "MENTIONED_IN"
        ]
        # 2 unique entity ids: e1 and e2
        assert len(mentioned_in) == 2
        freq_map = {c[0][0].source_id: c[0][0].properties["frequency"] for c in mentioned_in}
        assert freq_map["e1"] == 2
        assert freq_map["e2"] == 1

    def test_profile_isolation_all_nodes_have_scope(self):
        adapter = _make_adapter()
        populate_graph(**_default_kwargs(adapter))

        for call_obj in adapter.upsert_node.call_args_list:
            node: GraphNode = call_obj[0][0]
            assert node.properties["subscription_id"] == "sub-1", \
                f"{node.node_type} missing subscription_id"
            assert node.properties["profile_id"] == "prof-1", \
                f"{node.node_type} missing profile_id"

    def test_empty_extraction_no_errors(self):
        adapter = _make_adapter()
        populate_graph(**_default_kwargs(adapter, entities=[], facts=[], units=[]))

        # Only 1 node call (the Document node)
        assert adapter.upsert_node.call_count == 1
        assert adapter.upsert_node.call_args[0][0].node_type == "Document"
        # No edges at all
        assert adapter.upsert_edge.call_count == 0

    def test_fact_with_non_entity_object_skipped(self):
        """FactTriple where object_id is not in entity set should not create RELATES_TO."""
        adapter = _make_adapter()
        entities = [_make_entity("e1", "u1")]
        facts = [_make_fact("f1", "e1", "unknown-id")]
        populate_graph(**_default_kwargs(adapter, entities=entities, facts=facts))

        relates_to = [
            c for c in adapter.upsert_edge.call_args_list
            if c[0][0].edge_type == "RELATES_TO"
        ]
        assert len(relates_to) == 0

    def test_fact_with_no_object_id_skipped(self):
        """FactTriple with object_id=None should not create RELATES_TO."""
        adapter = _make_adapter()
        entities = [_make_entity("e1", "u1")]
        fact = FactTriple(
            fact_id="f1", subject_id="e1", predicate="HAS_SKILL",
            object_id=None, object_value="Python",
            unit_id="u1", raw_text="Alice knows Python",
        )
        populate_graph(**_default_kwargs(adapter, entities=entities, facts=[fact]))

        relates_to = [
            c for c in adapter.upsert_edge.call_args_list
            if c[0][0].edge_type == "RELATES_TO"
        ]
        assert len(relates_to) == 0
