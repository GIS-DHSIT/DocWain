"""Integration tests for the adaptive ReasoningEngine.

Tests fast path, full path, no-evidence handling, and metadata consistency.
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.intelligence.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mock_point(text, score, source_name, page="1", chunk_id="c1", doc_id="d1"):
    point = MagicMock()
    point.score = score
    point.payload = {
        "content": text,
        "canonical_text": text,
        "source_name": source_name,
        "page": page,
        "section_title": "Details",
        "chunk_id": chunk_id,
        "document_id": doc_id,
    }
    point.id = chunk_id
    return point


def _mock_query_response(points):
    """Create a mock Qdrant query_points response."""
    resp = MagicMock()
    resp.points = points
    return resp


def _build_engine(qdrant_points, llm_responses):
    """Build a ReasoningEngine with mocked dependencies."""
    llm = MagicMock()
    response_iter = iter(llm_responses)

    def _gen_meta(prompt, **kwargs):
        try:
            text = next(response_iter)
        except StopIteration:
            text = "fallback [SOURCE-1]"
        return text, {"response": text}

    llm.generate_with_metadata = _gen_meta
    llm.generate = lambda p, **kw: _gen_meta(p, **kw)[0]

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.array([[0.1] * 1024]))

    qdrant = MagicMock()
    qdrant.query_points = MagicMock(return_value=_mock_query_response(qdrant_points))

    return ReasoningEngine(
        llm_client=llm,
        qdrant_client=qdrant,
        embedder=embedder,
        collection_name="test_col",
        subscription_id="sub1",
        profile_id="prof1",
    )


# ---------------------------------------------------------------------------
# UNDERSTAND response builder (replaces old THINK/REASON)
# ---------------------------------------------------------------------------

def _understand_json(primary_intent="comparison", complexity="complex"):
    return json.dumps({
        "primary_intent": primary_intent,
        "sub_intents": [],
        "entities": ["X", "Y"],
        "output_format": "table" if primary_intent == "comparison" else "prose",
        "complexity": complexity,
        "needs_clarification": False,
        "clarification_question": None,
        "resolved_query": "compare X and Y",
        "thinking_required": complexity == "complex",
        "domain_hints": {},
    })


_FILTER_PATCH = "src.api.vector_store.build_qdrant_filter"


# ===========================================================================
# TestFastPathIntegration
# ===========================================================================

class TestFastPathIntegration:
    """Fast path: RETRIEVE -> GENERATE (trivially simple queries)."""

    @patch(_FILTER_PATCH, return_value=None)
    def test_simple_factual_query_fast_path(self, _mock_filter):
        """Simple 'What is X's salary?' query takes fast path, returns answer with sources."""
        points = [
            _build_mock_point("John's salary is $120,000 per year.", 0.92, "hr_report.pdf", chunk_id="c1"),
            _build_mock_point("John joined in 2019.", 0.85, "hr_report.pdf", chunk_id="c2"),
            _build_mock_point("Compensation details for staff.", 0.80, "hr_report.pdf", chunk_id="c3"),
        ]
        llm_responses = [
            # GENERATE (trivial path - single LLM call)
            "John's salary is $120,000 per year [SOURCE-1]. He joined in 2019 [SOURCE-2].",
        ]
        engine = _build_engine(points, llm_responses)
        result = engine.answer("What is John's salary?")

        assert result["context_found"] is True
        assert result["metadata"]["fast_path"] is True
        assert len(result["sources"]) > 0
        assert "salary" in result["response"].lower() or "$120,000" in result["response"]

    @patch(_FILTER_PATCH, return_value=None)
    def test_fast_path_returns_citations(self, _mock_filter):
        """Fast path response contains [SOURCE-N] citations."""
        points = [
            _build_mock_point("Revenue was $5M in Q1.", 0.90, "finance.pdf", chunk_id="c1"),
            _build_mock_point("Revenue grew 10% YoY.", 0.88, "finance.pdf", chunk_id="c2"),
            _build_mock_point("Q1 summary report data.", 0.82, "finance.pdf", chunk_id="c3"),
        ]
        llm_responses = [
            # GENERATE
            "Revenue was $5M in Q1 [SOURCE-1], growing 10% year-over-year [SOURCE-2].",
        ]
        engine = _build_engine(points, llm_responses)
        result = engine.answer("What was Q1 revenue?")

        assert "[SOURCE-" in result["response"]
        assert result["metadata"]["fast_path"] is True


# ===========================================================================
# TestFullPathIntegration
# ===========================================================================

class TestFullPathIntegration:
    """Full path: UNDERSTAND -> RETRIEVE -> GENERATE -> VERIFY (complex queries)."""

    @patch(_FILTER_PATCH, return_value=None)
    def test_comparison_query_full_path(self, _mock_filter):
        """'Compare X and Y' triggers full path with intent=comparison."""
        points = [
            _build_mock_point("Product X costs $100 and has 10 features.", 0.88, "catalog.pdf", chunk_id="c1"),
            _build_mock_point("Product Y costs $150 and has 15 features.", 0.85, "catalog.pdf", chunk_id="c2"),
            _build_mock_point("Both products released in 2025.", 0.78, "catalog.pdf", chunk_id="c3"),
        ]
        llm_responses = [
            # UNDERSTAND
            _understand_json(primary_intent="comparison", complexity="complex"),
            # GENERATE
            "Product X costs $100 with 10 features [SOURCE-1], while Product Y costs $150 with 15 features [SOURCE-2].",
        ]
        engine = _build_engine(points, llm_responses)
        result = engine.answer(
            "Compare Product X and Product Y in terms of cost and features",
            task_type="compare",
        )

        assert result["context_found"] is True
        assert result["metadata"]["fast_path"] is False
        assert result["metadata"]["intent"] == "comparison"
        assert len(result["sources"]) > 0


# ===========================================================================
# TestNoEvidenceHandling
# ===========================================================================

class TestNoEvidenceHandling:
    """When Qdrant returns empty results, engine reports context_found=False."""

    @patch(_FILTER_PATCH, return_value=None)
    def test_no_evidence_returns_not_found(self, _mock_filter):
        """Empty Qdrant results lead to context_found=False."""
        engine = _build_engine(qdrant_points=[], llm_responses=[])
        result = engine.answer("What is the meaning of life?")

        assert result["context_found"] is False
        assert result["grounded"] is False
        assert result["sources"] == []
        assert result["metadata"]["evidence_count"] == 0


# ===========================================================================
# TestMetadataConsistency
# ===========================================================================

_REQUIRED_METADATA_KEYS = {
    "engine", "fast_path", "intent", "complexity",
    "evidence_count", "confidence", "verification", "timing_ms",
}


class TestMetadataConsistency:
    """Verify metadata structure for both fast and full paths."""

    @patch(_FILTER_PATCH, return_value=None)
    def test_fast_path_metadata_has_required_fields(self, _mock_filter):
        """Fast path metadata includes all required keys."""
        points = [
            _build_mock_point("The policy states X.", 0.91, "policy.pdf", chunk_id="c1"),
            _build_mock_point("Additional policy info.", 0.87, "policy.pdf", chunk_id="c2"),
            _build_mock_point("More policy details.", 0.83, "policy.pdf", chunk_id="c3"),
        ]
        llm_responses = [
            "The policy states X [SOURCE-1] with additional context [SOURCE-2].",
        ]
        engine = _build_engine(points, llm_responses)
        result = engine.answer("What does the policy say?")

        meta = result["metadata"]
        missing = _REQUIRED_METADATA_KEYS - set(meta.keys())
        assert not missing, f"Fast path metadata missing keys: {missing}"
        assert meta["engine"] == "reasoning_engine"
        assert meta["fast_path"] is True
        assert isinstance(meta["timing_ms"], dict)
        assert "total" in meta["timing_ms"]

    @patch(_FILTER_PATCH, return_value=None)
    def test_full_path_metadata_has_required_fields(self, _mock_filter):
        """Full path metadata includes all required keys."""
        points = [
            _build_mock_point("Department A budget is $1M.", 0.88, "budget.pdf", chunk_id="c1"),
            _build_mock_point("Department B budget is $2M.", 0.84, "budget.pdf", chunk_id="c2"),
            _build_mock_point("Overall budget summary.", 0.79, "budget.pdf", chunk_id="c3"),
        ]
        llm_responses = [
            # UNDERSTAND
            _understand_json(primary_intent="comparison", complexity="moderate"),
            # GENERATE
            "Department A has $1M [SOURCE-1] vs Department B with $2M [SOURCE-2].",
        ]
        engine = _build_engine(points, llm_responses)
        result = engine.answer(
            "Compare the budgets of Department A and Department B across all categories",
            task_type="compare",
        )

        meta = result["metadata"]
        missing = _REQUIRED_METADATA_KEYS - set(meta.keys())
        assert not missing, f"Full path metadata missing keys: {missing}"
        assert meta["engine"] == "reasoning_engine"
        assert meta["fast_path"] is False
        assert isinstance(meta["timing_ms"], dict)
        assert "total" in meta["timing_ms"]
        assert isinstance(meta["confidence"], float)
