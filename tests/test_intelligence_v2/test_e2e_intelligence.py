"""End-to-end integration test for the intelligence_v2 pipeline.

Verifies the full flow: DocumentAnalyzer analyses a document, persists
intelligence to MongoDB, writes entities to the KG, and the IntentAnalyzer
can match a query back to the analysed document via answerable_topics.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Realistic LLM analysis response for an invoice document
# ---------------------------------------------------------------------------

_ANALYZER_LLM_RESPONSE: Dict[str, Any] = {
    "document_type": "invoice",
    "language": "en",
    "summary": "Invoice from Acme Corp for \u00a345,230 of IT services.",
    "section_summaries": {"Line Items": "5 items totaling \u00a345,230"},
    "entities": [
        {"type": "ORGANIZATION", "value": "Acme Corp", "role": "vendor"},
        {"type": "AMOUNT", "value": "\u00a345,230.00", "role": "total_due"},
    ],
    "facts": [
        {"claim": "Total is \u00a345,230.00", "evidence": "Section: Line Items, Page: 1"}
    ],
    "relationships": [
        {"from": "Acme Corp", "relation": "INVOICED", "to": "DHS", "context": "IT services"}
    ],
    "answerable_topics": ["invoice total amount", "vendor details", "line item breakdown"],
}

# ---------------------------------------------------------------------------
# Intent LLM response (used by IntentAnalyzer for the query step)
# ---------------------------------------------------------------------------

_INTENT_LLM_RESPONSE: Dict[str, Any] = {
    "task_type": "lookup",
    "complexity": "simple",
    "resolved_query": "What is the total on the Acme invoice?",
    "output_format": "prose",
    "relevant_documents": [],
    "cross_profile": False,
    "entities": ["Acme", "invoice", "total"],
}


# ---------------------------------------------------------------------------
# Simple dict-backed MongoDB fake that captures update_one calls
# ---------------------------------------------------------------------------


class _FakeMongoDB:
    """Minimal MongoDB stand-in that records update_one calls."""

    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.update_one_calls: List[Tuple[Dict, Dict]] = []

    def update_one(self, filter_doc: Dict, update_doc: Dict, **kwargs: Any) -> None:
        doc_id = filter_doc.get("document_id", "unknown")
        self.update_one_calls.append((filter_doc, update_doc))

        # Apply $set to our in-memory store
        existing = self.documents.setdefault(doc_id, {})
        set_fields = update_doc.get("$set", {})
        existing.update(set_fields)
        existing["document_id"] = doc_id

    def get(self, document_id: str) -> Dict[str, Any]:
        return self.documents.get(document_id, {})


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestFullIntelligencePipeline:
    """End-to-end: analyze document -> verify MongoDB -> verify KG -> verify
    topic matching at query time."""

    def test_full_intelligence_pipeline(self) -> None:
        # ---- 1. Setup mocks ------------------------------------------------

        # LLM mock for DocumentAnalyzer (uses generate_with_metadata)
        analyzer_llm = MagicMock()
        analyzer_llm.generate_with_metadata.return_value = (
            json.dumps(_ANALYZER_LLM_RESPONSE),
            {},
        )

        # Neo4j mock: run_query returns [] (we just verify it's called)
        neo4j = MagicMock()
        neo4j.run_query.return_value = []

        # MongoDB: dict-backed fake
        mongodb = _FakeMongoDB()

        # ---- 2. Run DocumentAnalyzer ----------------------------------------

        from src.intelligence_v2.analyzer import DocumentAnalyzer

        analyzer = DocumentAnalyzer(
            llm_gateway=analyzer_llm,
            neo4j_store=neo4j,
            mongodb=mongodb,
        )

        result = analyzer.analyze(
            document_id="doc_inv_001",
            extracted={
                "full_text": "Acme Corp invoice. Total: \u00a345,230.00",
                "sections": [],
            },
            subscription_id="sub_1",
            profile_id="prof_1",
            filename="acme_invoice.pdf",
            doc_type="invoice",
        )

        # ---- 3. Verify MongoDB got intelligence -----------------------------

        assert len(mongodb.update_one_calls) == 1, "Expected exactly one update_one call"

        stored = mongodb.get("doc_inv_001")

        # intelligence_ready flag
        assert stored.get("intelligence_ready") is True

        # summary mentions Acme
        assert "Acme" in stored.get("document_summary", ""), (
            f"Expected 'Acme' in summary, got: {stored.get('document_summary')}"
        )

        # answerable_topics is a non-empty list
        topics = stored.get("doc_intent_tags", [])
        assert isinstance(topics, list) and len(topics) > 0, (
            f"Expected non-empty answerable_topics, got: {topics}"
        )

        # entities list is non-empty
        entities = stored.get("key_entities", [])
        assert isinstance(entities, list) and len(entities) > 0, (
            f"Expected non-empty entities, got: {entities}"
        )

        # intelligence dict is present and has expected keys
        intel = stored.get("intelligence", {})
        assert intel.get("summary"), "intelligence.summary should be non-empty"
        assert intel.get("answerable_topics"), "intelligence.answerable_topics should be non-empty"

        # ---- 4. Verify KG was called ----------------------------------------

        assert neo4j.run_query.called, "Expected neo4j.run_query to be called"

        # Collect all Cypher queries issued
        cypher_queries = [
            call_args[0][0]
            for call_args in neo4j.run_query.call_args_list
        ]

        # At least one query should create Entity nodes
        entity_queries = [q for q in cypher_queries if "Entity" in q]
        assert len(entity_queries) > 0, (
            f"Expected at least one Cypher query mentioning 'Entity', got queries: {cypher_queries}"
        )

        # ---- 5. Verify topic matching works at query time -------------------

        # Build doc_intelligence list from what MongoDB captured (mimics what
        # CoreAgent would fetch from MongoDB before calling IntentAnalyzer)
        doc_intelligence = [
            {
                "document_id": "doc_inv_001",
                "intelligence": intel,
            }
        ]

        # LLM mock for IntentAnalyzer (uses generate)
        intent_llm = MagicMock()
        intent_llm.generate.return_value = json.dumps(_INTENT_LLM_RESPONSE)

        from src.agent.intent import IntentAnalyzer

        intent_analyzer = IntentAnalyzer(llm_gateway=intent_llm)

        understanding = intent_analyzer.analyze(
            query="What is the total on the Acme invoice?",
            subscription_id="sub_1",
            profile_id="prof_1",
            doc_intelligence=doc_intelligence,
            conversation_history=None,
        )

        # The document should appear in relevant_documents via topic matching
        matched_doc_ids = [
            doc.get("document_id") for doc in understanding.relevant_documents
        ]
        assert "doc_inv_001" in matched_doc_ids, (
            f"Expected doc_inv_001 in relevant_documents, got: {understanding.relevant_documents}"
        )

        # ---- 6. Verify the analyzer result dict is well-formed --------------

        assert result["document_id"] == "doc_inv_001"
        assert isinstance(result["intelligence"], dict)
        assert "kg_stats" in result
        assert result["kg_stats"]["entities"] == 2
        assert result["kg_stats"]["facts"] == 1
