# Response Intelligence Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified intelligence layer that produces GPT-level response quality through algorithmic response synthesis, graph-backed knowledge, and closed-loop quality validation.

**Architecture:** Six new components wrap around the existing RAG pipeline: Graph DB Adapter, Query Analyzer, Evidence Organizer, Rendering Spec Generator, Constrained LLM Prompter, and Output Quality Engine. A Conversation Graph maintains session state. All components live in `src/docwain_intel/` and never modify existing `src/rag_v3/` files.

**Tech Stack:** Python 3.10, Pydantic, spaCy (en_core_web_lg), neo4j driver, NetworkX (in-memory conversation graph), existing Qdrant client

---

## Task 1: Graph DB Adapter (Neo4j/Memgraph Cypher)

**Files:**
- Create: `src/docwain_intel/graph_adapter.py`
- Test: `tests/test_graph_adapter.py`

**Context:** This adapter speaks Cypher to Neo4j or Memgraph interchangeably. It manages the graph schema (Entity, Document, Chunk nodes; APPEARS_IN, BELONGS_TO, RELATES_TO, MENTIONED_IN edges). All queries enforce profile isolation via `subscription_id` + `profile_id` filters.

**Step 1: Write the failing tests**

```python
"""tests/test_graph_adapter.py"""
import pytest
from unittest.mock import MagicMock, patch

from src.docwain_intel.graph_adapter import (
    CypherGraphAdapter,
    GraphNode,
    GraphEdge,
    get_graph_adapter,
)


class TestGraphNodeModel:
    def test_entity_node_creation(self):
        node = GraphNode(
            node_id="ent_abc123",
            node_type="Entity",
            properties={"text": "John Smith", "label": "PERSON", "normalized": "john smith"},
        )
        assert node.node_id == "ent_abc123"
        assert node.node_type == "Entity"
        assert node.properties["label"] == "PERSON"

    def test_document_node_creation(self):
        node = GraphNode(
            node_id="doc_001",
            node_type="Document",
            properties={"filename": "resume.pdf", "subscription_id": "sub1", "profile_id": "prof1"},
        )
        assert node.node_type == "Document"


class TestGraphEdgeModel:
    def test_relates_to_edge(self):
        edge = GraphEdge(
            source_id="ent_abc",
            target_id="ent_def",
            edge_type="RELATES_TO",
            properties={"predicate": "works_at", "confidence": 0.9},
        )
        assert edge.edge_type == "RELATES_TO"
        assert edge.properties["predicate"] == "works_at"


class TestCypherGraphAdapter:
    def test_upsert_entity_node(self):
        adapter = CypherGraphAdapter(driver=MagicMock())
        adapter.upsert_node(GraphNode(
            node_id="ent_1", node_type="Entity",
            properties={"text": "John", "label": "PERSON",
                         "subscription_id": "s1", "profile_id": "p1"},
        ))
        adapter._driver.session.assert_called()

    def test_upsert_edge(self):
        adapter = CypherGraphAdapter(driver=MagicMock())
        adapter.upsert_edge(GraphEdge(
            source_id="ent_1", target_id="doc_1",
            edge_type="MENTIONED_IN", properties={"frequency": 3},
        ))
        adapter._driver.session.assert_called()

    def test_query_entity_facts(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = [
            {"predicate": "works_at", "object_value": "Google", "confidence": 0.9}
        ]
        adapter = CypherGraphAdapter(driver=mock_driver)
        facts = adapter.get_entity_facts(
            entity_text="John", subscription_id="s1", profile_id="p1",
        )
        assert isinstance(facts, list)

    def test_profile_isolation_in_queries(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []
        adapter = CypherGraphAdapter(driver=mock_driver)
        adapter.get_entity_facts(entity_text="John", subscription_id="s1", profile_id="p1")
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "subscription_id" in query
        assert "profile_id" in query

    def test_get_related_entities(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []
        adapter = CypherGraphAdapter(driver=mock_driver)
        related = adapter.get_related_entities(
            entity_text="John", subscription_id="s1", profile_id="p1",
        )
        assert isinstance(related, list)

    def test_get_document_entities(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []
        adapter = CypherGraphAdapter(driver=mock_driver)
        entities = adapter.get_document_entities(
            document_id="doc1", subscription_id="s1", profile_id="p1",
        )
        assert isinstance(entities, list)


class TestGetGraphAdapter:
    @patch("src.docwain_intel.graph_adapter._try_connect_neo4j")
    def test_returns_adapter_when_neo4j_available(self, mock_connect):
        mock_connect.return_value = MagicMock()
        adapter = get_graph_adapter()
        assert adapter is not None

    @patch("src.docwain_intel.graph_adapter._try_connect_neo4j", return_value=None)
    @patch("src.docwain_intel.graph_adapter._try_connect_memgraph", return_value=None)
    def test_returns_none_when_no_db(self, mock_mem, mock_neo):
        adapter = get_graph_adapter()
        assert adapter is None
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_graph_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the graph adapter**

```python
"""src/docwain_intel/graph_adapter.py — Cypher graph adapter for Neo4j/Memgraph."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_adapter_lock = threading.Lock()
_adapter_instance: Optional[CypherGraphAdapter] = None


class GraphNode(BaseModel):
    node_id: str
    node_type: str  # Entity, Document, Chunk
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str  # APPEARS_IN, BELONGS_TO, RELATES_TO, MENTIONED_IN
    properties: Dict[str, Any] = Field(default_factory=dict)


def _try_connect_neo4j() -> Any:
    """Try to connect to Neo4j. Returns driver or None."""
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
        return driver
    except Exception as exc:
        logger.debug("Neo4j not available: %s", exc)
        return None


def _try_connect_memgraph() -> Any:
    """Try to connect to Memgraph. Returns driver or None."""
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7688")
        driver = GraphDatabase.driver(uri)
        driver.verify_connectivity()
        logger.info("Connected to Memgraph at %s", uri)
        return driver
    except Exception as exc:
        logger.debug("Memgraph not available: %s", exc)
        return None


class CypherGraphAdapter:
    """Speaks Cypher to Neo4j or Memgraph interchangeably."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def ensure_schema(self) -> None:
        """Create indexes and constraints for the graph schema."""
        statements = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.node_id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.subscription_id, e.profile_id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.subscription_id, d.profile_id)",
        ]
        for stmt in statements:
            try:
                self._run(stmt)
            except Exception as exc:
                logger.debug("Schema statement skipped: %s", exc)

    def upsert_node(self, node: GraphNode) -> None:
        query = (
            f"MERGE (n:{node.node_type} {{node_id: $node_id}}) "
            f"SET n += $props"
        )
        props = {**node.properties, "node_id": node.node_id}
        self._run(query, {"node_id": node.node_id, "props": props})

    def upsert_edge(self, edge: GraphEdge) -> None:
        query = (
            f"MATCH (a {{node_id: $src}}), (b {{node_id: $tgt}}) "
            f"MERGE (a)-[r:{edge.edge_type}]->(b) "
            f"SET r += $props"
        )
        self._run(query, {
            "src": edge.source_id,
            "tgt": edge.target_id,
            "props": edge.properties,
        })

    def get_entity_facts(
        self, *, entity_text: str, subscription_id: str, profile_id: str,
    ) -> List[Dict[str, Any]]:
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[r:RELATES_TO]->(t) "
            "WHERE toLower(e.text) CONTAINS toLower($text) OR toLower(e.normalized) CONTAINS toLower($text) "
            "RETURN e.text AS subject, r.predicate AS predicate, "
            "COALESCE(t.text, r.object_value) AS value, r.confidence AS confidence"
        )
        return self._run(query, {"text": entity_text, "sub": subscription_id, "prof": profile_id})

    def get_related_entities(
        self, *, entity_text: str, subscription_id: str, profile_id: str,
    ) -> List[Dict[str, Any]]:
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[r]-(related:Entity) "
            "WHERE toLower(e.text) CONTAINS toLower($text) "
            "RETURN related.text AS text, related.label AS label, "
            "type(r) AS relationship, r.predicate AS predicate"
        )
        return self._run(query, {"text": entity_text, "sub": subscription_id, "prof": profile_id})

    def get_document_entities(
        self, *, document_id: str, subscription_id: str, profile_id: str,
    ) -> List[Dict[str, Any]]:
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[:MENTIONED_IN]->(d:Document {node_id: $doc_id}) "
            "RETURN e.node_id AS entity_id, e.text AS text, e.label AS label, "
            "e.normalized AS normalized, e.confidence AS confidence"
        )
        return self._run(query, {"doc_id": document_id, "sub": subscription_id, "prof": profile_id})

    def get_all_entities(
        self, *, subscription_id: str, profile_id: str,
    ) -> List[Dict[str, Any]]:
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof}) "
            "RETURN e.node_id AS entity_id, e.text AS text, e.label AS label, "
            "e.normalized AS normalized"
        )
        return self._run(query, {"sub": subscription_id, "prof": profile_id})

    def count_entities(self, *, subscription_id: str, profile_id: str) -> int:
        result = self._run(
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof}) RETURN count(e) AS cnt",
            {"sub": subscription_id, "prof": profile_id},
        )
        return result[0]["cnt"] if result else 0

    def count_facts(self, *, subscription_id: str, profile_id: str) -> int:
        result = self._run(
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})-[r:RELATES_TO]->() "
            "RETURN count(r) AS cnt",
            {"sub": subscription_id, "prof": profile_id},
        )
        return result[0]["cnt"] if result else 0


def get_graph_adapter() -> Optional[CypherGraphAdapter]:
    """Get or create the graph adapter singleton. Tries Neo4j first, then Memgraph."""
    global _adapter_instance
    if _adapter_instance is not None:
        return _adapter_instance
    with _adapter_lock:
        if _adapter_instance is not None:
            return _adapter_instance
        driver = _try_connect_neo4j()
        if not driver:
            driver = _try_connect_memgraph()
        if driver:
            _adapter_instance = CypherGraphAdapter(driver=driver)
            _adapter_instance.ensure_schema()
            return _adapter_instance
        logger.warning("No graph database available (Neo4j/Memgraph)")
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_graph_adapter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/docwain_intel/graph_adapter.py tests/test_graph_adapter.py
git commit -m "feat(docwain-intel): add Cypher graph adapter for Neo4j/Memgraph"
```

---

## Task 2: Graph Population (Wire Intel Pipeline to Graph)

**Files:**
- Create: `src/docwain_intel/graph_populator.py`
- Modify: `src/docwain_intel/intel_pipeline.py:131-146`
- Test: `tests/test_graph_populator.py`

**Context:** After the intel pipeline verifies extraction (stage 5), populate the graph DB with Entity nodes, Document nodes, Chunk nodes, and all edges. This replaces the stub stage 6 in `intel_pipeline.py`.

**Step 1: Write the failing tests**

```python
"""tests/test_graph_populator.py"""
import pytest
from unittest.mock import MagicMock, call

from src.docwain_intel.graph_populator import populate_graph
from src.docwain_intel.models import (
    EntitySpan, FactTriple, StructuredDocument, SemanticUnit, UnitType,
    ExtractionResult, DocumentFingerprint,
)


def _make_extraction():
    entities = [
        EntitySpan(entity_id="ent_1", text="John Smith", normalized="john smith",
                    label="PERSON", unit_id="u1", confidence=0.9, source="spacy"),
        EntitySpan(entity_id="ent_2", text="Google", normalized="google",
                    label="ORG", unit_id="u1", confidence=0.85, source="spacy"),
    ]
    facts = [
        FactTriple(fact_id="f1", subject_id="ent_1", predicate="works_at",
                    object_id="ent_2", object_value="Google", unit_id="u1",
                    raw_text="John works at Google", confidence=0.8,
                    extraction_method="textacy_svo"),
    ]
    return ExtractionResult(document_id="doc1", entities=entities, facts=facts)


def _make_structured_doc():
    units = [
        SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH,
                      text="John Smith works at Google.", page_start=1, page_end=1),
    ]
    return StructuredDocument(document_id="doc1", units=units, unit_count=1, total_chars=30)


class TestPopulateGraph:
    def test_creates_entity_nodes(self):
        adapter = MagicMock()
        extraction = _make_extraction()
        doc = _make_structured_doc()
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=["resume"],
        )
        node_calls = [c for c in adapter.upsert_node.call_args_list
                       if c[0][0].node_type == "Entity"]
        assert len(node_calls) == 2

    def test_creates_document_node(self):
        adapter = MagicMock()
        extraction = _make_extraction()
        doc = _make_structured_doc()
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=["resume"],
        )
        node_calls = [c for c in adapter.upsert_node.call_args_list
                       if c[0][0].node_type == "Document"]
        assert len(node_calls) == 1

    def test_creates_chunk_nodes(self):
        adapter = MagicMock()
        extraction = _make_extraction()
        doc = _make_structured_doc()
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=[],
        )
        node_calls = [c for c in adapter.upsert_node.call_args_list
                       if c[0][0].node_type == "Chunk"]
        assert len(node_calls) >= 1

    def test_creates_relates_to_edges(self):
        adapter = MagicMock()
        extraction = _make_extraction()
        doc = _make_structured_doc()
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=[],
        )
        edge_calls = [c for c in adapter.upsert_edge.call_args_list
                       if c[0][0].edge_type == "RELATES_TO"]
        assert len(edge_calls) >= 1

    def test_creates_mentioned_in_edges(self):
        adapter = MagicMock()
        extraction = _make_extraction()
        doc = _make_structured_doc()
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=[],
        )
        edge_calls = [c for c in adapter.upsert_edge.call_args_list
                       if c[0][0].edge_type == "MENTIONED_IN"]
        assert len(edge_calls) >= 1

    def test_profile_isolation_on_nodes(self):
        adapter = MagicMock()
        extraction = _make_extraction()
        doc = _make_structured_doc()
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=[],
        )
        for c in adapter.upsert_node.call_args_list:
            node = c[0][0]
            assert node.properties.get("subscription_id") == "s1"
            assert node.properties.get("profile_id") == "p1"

    def test_empty_extraction_no_error(self):
        adapter = MagicMock()
        extraction = ExtractionResult(document_id="doc1")
        doc = StructuredDocument(document_id="doc1")
        populate_graph(
            adapter=adapter, extraction=extraction, structured_doc=doc,
            document_id="doc1", subscription_id="s1", profile_id="p1",
            fingerprint_tags=[],
        )
        # Should not raise
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_graph_populator.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement graph populator**

Create `src/docwain_intel/graph_populator.py` — a function `populate_graph()` that takes adapter, extraction result, structured doc, and tenant IDs, and creates all nodes and edges in the graph. Then wire it into `intel_pipeline.py` stage 6.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_graph_populator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/docwain_intel/graph_populator.py src/docwain_intel/intel_pipeline.py tests/test_graph_populator.py
git commit -m "feat(docwain-intel): add graph population from intel pipeline"
```

---

## Task 3: Query Analyzer (Algorithmic Query Geometry)

**Files:**
- Create: `src/docwain_intel/query_analyzer.py`
- Test: `tests/test_query_analyzer.py`

**Context:** Replaces template-based intent matching with algorithmic query geometry derivation. Uses spaCy NLP features to compute: intent type, expected entity count, granularity level, temporal ordering, comparison flag. All computed — no lookup tables or domain-specific branches.

**Step 1: Write the failing tests**

Tests should cover:
- Single-entity query → entity_count=1, granularity derived from specificity
- Multi-entity query → entity_count=2+, comparison flag
- "When did..." → temporal_ordering=True
- Short specific query → high granularity (concise answer expected)
- Long vague query → low granularity (comprehensive answer expected)
- Question word geometry: "who" → entity-centric, "how" → process-centric
- Empty/conversational query → minimal geometry
- Numeric constraint detection
- Works with existing QueryAnalysis from query_router.py (takes it as input, enriches it)

At least 12 tests covering these scenarios.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_query_analyzer.py -v`

**Step 3: Implement query analyzer**

Create `src/docwain_intel/query_analyzer.py` with:
- `QueryGeometry` Pydantic model with computed fields
- `analyze_query(query: str, analysis: QueryAnalysis) -> QueryGeometry` function
- All derivation via spaCy doc features (POS, dep, lemma, NER), no hardcoded templates

**Step 4: Run tests, commit**

---

## Task 4: Evidence Organizer

**Files:**
- Create: `src/docwain_intel/evidence_organizer.py`
- Test: `tests/test_evidence_organizer.py`

**Context:** Takes retrieved chunks + graph facts and restructures evidence before the LLM sees it. Groups by entity, deduplicates using FactTriples, detects gaps, threads provenance.

**Step 1: Write the failing tests**

Tests should cover:
- Chunks with shared entity_ids get grouped together
- Duplicate facts (same subject+predicate+value) deduplicated, richest version kept
- Gap detection: query mentions entity X, no evidence about X → gap flagged
- Provenance preserved: each evidence piece retains source document, page, section
- Empty input → empty organized output, no error
- Mixed evidence: some from graph facts, some from vector chunks → merged coherently
- Ordering within groups: by relevance score descending
- Large input (50+ chunks) completes without timeout

At least 10 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_evidence_organizer.py -v`

**Step 3: Implement evidence organizer**

Create `src/docwain_intel/evidence_organizer.py` with:
- `OrganizedEvidence` Pydantic model with entity_groups, gaps, provenance_map
- `organize_evidence(chunks, facts, query_entities) -> OrganizedEvidence` function
- Entity grouping via entity_id matching from Qdrant payload
- Fact deduplication via (subject_id, predicate, object_value) key
- Gap detection by comparing query entities against evidence entities

**Step 4: Run tests, commit**

---

## Task 5: Rendering Spec Generator

**Files:**
- Create: `src/docwain_intel/rendering_spec.py`
- Test: `tests/test_rendering_spec.py`

**Context:** Algorithmically computes the output format specification from data shape + query geometry. No fixed templates — the spec is derived fresh for every query using three-axis analysis (data shape, query geometry, structural inference).

**Step 1: Write the failing tests**

Tests should cover:
- Single entity + 4+ fields in evidence → card-like layout spec
- 3+ entities with shared fields → tabular layout spec
- Chronological evidence → temporal ordering in spec
- Single fact requested (high specificity) → minimal/single-value spec
- Prose-only evidence → narrative layout spec
- Comparison query geometry → side-by-side spec
- Aggregation query → count/summary spec
- Detail level derived from specificity score (high → concise, low → comprehensive)
- Spec includes: layout_mode, field_ordering, grouping_strategy, detail_level
- Edge case: no evidence → no-results spec

At least 10 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rendering_spec.py -v`

**Step 3: Implement rendering spec generator**

Create `src/docwain_intel/rendering_spec.py` with:
- `RenderingSpec` Pydantic model with layout_mode, field_ordering, grouping_strategy, detail_level, use_headers, use_bold_values, use_table
- `generate_spec(query_geometry, organized_evidence) -> RenderingSpec` function
- Three-axis analysis: data_shape_analysis(), query_geometry_analysis(), structural_inference()
- All rules composable — not mutually exclusive categories

**Step 4: Run tests, commit**

---

## Task 6: Constrained LLM Prompter

**Files:**
- Create: `src/docwain_intel/constrained_prompter.py`
- Test: `tests/test_constrained_prompter.py`

**Context:** Translates the rendering spec + organized evidence into a constrained LLM prompt. The LLM fills content, not format. The prompt skeleton is generated dynamically from the spec — not from a template library.

**Step 1: Write the failing tests**

Tests should cover:
- Tabular spec → prompt contains table skeleton with headers from evidence fields
- Card spec → prompt contains field labels to fill
- Narrative spec → prompt requests structured paragraphs
- Gap markers injected: "No evidence found for [field]"
- Evidence pre-organized in prompt: grouped by entity, not raw dump
- Provenance instruction: "Cite source document and page"
- LLM is told NOT to add preambles, greetings, or meta-commentary
- Single-value spec → prompt requests one-line answer
- Prompt length stays within token budget (configurable max)

At least 8 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_constrained_prompter.py -v`

**Step 3: Implement constrained prompter**

Create `src/docwain_intel/constrained_prompter.py` with:
- `ConstrainedPrompt` Pydantic model with system_prompt, user_prompt, max_tokens
- `build_prompt(spec, organized_evidence, query) -> ConstrainedPrompt` function
- Dynamic skeleton generation from RenderingSpec fields
- Evidence formatting respects organization (groups, gaps, provenance)

**Step 4: Run tests, commit**

---

## Task 7: Output Quality Engine

**Files:**
- Create: `src/docwain_intel/quality_engine.py`
- Test: `tests/test_quality_engine.py`

**Context:** Closed-loop validation of LLM output against the rendering spec. Three layers: structural conformance (shape matches spec), content integrity (claims backed by facts), cleanliness (no LLM artifacts).

**Step 1: Write the failing tests**

Tests should cover:
- Structural conformance: spec says table, output is prose → restructured to table
- Structural conformance: spec says single value, output has preamble → stripped to value
- Content integrity: claim "John works at Google" verified against FactTriple → passes
- Content integrity: claim "John works at Meta" with no backing fact → flagged
- Cleanliness: "Based on the provided documents..." → stripped
- Cleanliness: "Here are the findings:" → stripped
- Output completeness: spec lists 4 fields, output has 3 → gap noted
- Pass-through: well-formed output matching spec → returned unchanged
- Entity names verified against EntitySpan data
- Numeric claims verified against evidence

At least 10 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_quality_engine.py -v`

**Step 3: Implement quality engine**

Create `src/docwain_intel/quality_engine.py` with:
- `QualityResult` Pydantic model with cleaned_text, issues_found, claims_verified, claims_unverified
- `validate_output(llm_output, spec, organized_evidence, extraction) -> QualityResult` function
- Three validation passes: structural_conformance(), content_integrity(), cleanliness()
- Meta-commentary detection via sentence semantics (sentences about answering vs the answer)
- Restructuring engine: parse output into semantic blocks, re-render per spec

**Step 4: Run tests, commit**

---

## Task 8: Conversation Graph

**Files:**
- Create: `src/docwain_intel/conversation_graph.py`
- Test: `tests/test_conversation_graph.py`

**Context:** In-memory session state tracking. Maintains entities discussed, facts disclosed, questions asked. Supports pronoun resolution, progressive disclosure, and ambiguity resolution.

**Step 1: Write the failing tests**

Tests should cover:
- Add turn with entities → entities tracked
- Pronoun resolution: "his" → resolves to most recent PERSON entity
- "What about her skills?" → resolves "her" + carries entity context
- Progressive disclosure: fact disclosed in turn 1 → marked as disclosed in turn 2
- "Anything else?" → returns only undisclosed facts
- Ambiguity resolution: recent topic weighted higher
- Empty conversation → no resolution possible
- Multi-entity conversation: tracks multiple entities across turns
- Turn decay: older entities have lower weight than recent ones
- Session reset clears all state

At least 10 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_conversation_graph.py -v`

**Step 3: Implement conversation graph**

Create `src/docwain_intel/conversation_graph.py` with:
- `ConversationGraph` class with add_turn(), resolve_reference(), get_undisclosed_facts(), get_context_entities()
- Internal NetworkX digraph or simple dict-based graph
- Turn tracking with entity salience (recency + frequency)
- Pronoun resolution using entity label matching (PERSON for "he/she/his/her", ORG for "it/they")

**Step 4: Run tests, commit**

---

## Task 9: Intelligence Orchestrator (Wire Everything Together)

**Files:**
- Create: `src/docwain_intel/intelligence.py`
- Modify: `src/docwain_intel/integration.py`
- Modify: `src/docwain_intel/__init__.py`
- Test: `tests/test_intelligence_orchestrator.py`

**Context:** The orchestrator ties all components into a single query-time flow. It replaces the existing `route_and_assemble()` in `integration.py` with the full intelligence pipeline: conversation graph → query analyzer → router → (graph path | LLM path) → quality engine → response.

**Step 1: Write the failing tests**

Tests should cover:
- GRAPH_DIRECT route: simple fact query → graph lookup → response, no LLM
- LLM_GENERATION route: complex query → evidence organized → spec generated → LLM prompted → quality validated
- Conversation context carried across turns (mock multi-turn)
- Fallback: graph DB unavailable → falls back to vector-only path
- Fallback: LLM unavailable → falls back to graph-direct structured answer
- Feature flag: disabled → returns None (existing behavior preserved)
- Profile isolation: subscription_id + profile_id passed through entire chain
- Timing: each stage timed and logged

At least 8 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_intelligence_orchestrator.py -v`

**Step 3: Implement orchestrator**

Create `src/docwain_intel/intelligence.py` with:
- `IntelligenceEngine` class that holds references to graph adapter, conversation graph
- `process_query(query, subscription_id, profile_id, session_id, chunks, facts) -> IntelligentResponse` method
- Route-based dispatch: GRAPH_DIRECT → graph adapter, LLM_GENERATION → full pipeline
- Update integration.py `route_and_assemble()` to use IntelligenceEngine when available
- Update `__init__.py` exports

**Step 4: Run tests, commit**

---

## Task 10: Qdrant Payload Enrichment

**Files:**
- Create: `src/docwain_intel/payload_enricher.py`
- Test: `tests/test_payload_enricher.py`

**Context:** Enriches Qdrant chunk payloads with entity IDs, fingerprint tags, and semantic unit type from the intel pipeline. This enables entity-filtered vector search without hitting the graph.

**Step 1: Write the failing tests**

Tests should cover:
- Chunk payload gets entity_ids list added
- Chunk payload gets fingerprint_tags added
- Chunk payload gets unit_type added
- Entity IDs matched by text overlap between chunk text and entity spans
- Multiple entities in one chunk → all IDs included
- No entities found → empty list (not missing key)
- Existing payload fields preserved (not overwritten)
- Batch enrichment processes multiple chunks

At least 8 tests.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_payload_enricher.py -v`

**Step 3: Implement payload enricher**

Create `src/docwain_intel/payload_enricher.py` with:
- `enrich_payload(chunk_payload, extraction, fingerprint) -> Dict` function
- `batch_enrich(payloads, extraction, fingerprint) -> List[Dict]` function
- Entity matching: check if entity.text appears in chunk text (case-insensitive)
- Additive: never removes existing payload fields

**Step 4: Run tests, commit**

---

## Execution Order

Tasks 1-2 (graph infra) must run first. Tasks 3-8 are largely independent and can run in any order. Task 9 depends on all others. Task 10 is independent.

Recommended: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 10 → 9

---

## Integration Constraints

- All new files in `src/docwain_intel/` — no modifications to `src/rag_v3/`
- Feature-flagged behind `DOCWAIN_INTEL_V2=1` environment variable
- Graceful degradation: if Neo4j/Memgraph unavailable, falls back to current behavior
- No new external service dependencies required to run (graph DB is optional)
- All algorithms generic — no domain-specific code paths, no hardcoded templates
