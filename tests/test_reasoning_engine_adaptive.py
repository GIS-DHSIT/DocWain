"""
Tests for the adaptive trivial/intelligent path logic in ReasoningEngine.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.intelligence.understand import QueryUnderstanding, UnderstandResult, DomainHints, SubIntent
from src.intelligence.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# Trivial detection tests (replaces _is_simple_query)
# ---------------------------------------------------------------------------

class TestIsSimpleQuery:
    def test_simple_query_detected(self):
        """Short factual queries with no history are trivially simple."""
        assert QueryUnderstanding.is_trivially_simple("What is the revenue?", []) is True
        assert QueryUnderstanding.is_trivially_simple("Who is the CEO?", []) is True

    def test_comparison_not_simple(self):
        """Queries containing complexity signals are not trivially simple."""
        assert QueryUnderstanding.is_trivially_simple("Compare A and B", []) is False
        assert QueryUnderstanding.is_trivially_simple("Summarize all findings", []) is False
        assert QueryUnderstanding.is_trivially_simple("Rank all candidates", []) is False

    def test_long_query_not_simple(self):
        """Queries with 7+ words are not trivially simple."""
        long_query = " ".join(["word"] * 10)
        assert QueryUnderstanding.is_trivially_simple(long_query, []) is False

    def test_extract_task_short_is_simple(self):
        """Short extract queries without history are trivially simple."""
        assert QueryUnderstanding.is_trivially_simple("List the skills", []) is True

    def test_extract_task_long_not_simple(self):
        """Long queries are NOT trivially simple regardless of content."""
        long_query = " ".join(["word"] * 10)
        assert QueryUnderstanding.is_trivially_simple(long_query, []) is False

    def test_with_history_not_simple(self):
        """Queries with conversation history are not trivially simple."""
        assert QueryUnderstanding.is_trivially_simple(
            "yes", [{"role": "user", "content": "prior question"}]
        ) is False


# ---------------------------------------------------------------------------
# Helpers for engine-level tests
# ---------------------------------------------------------------------------

def _make_mock_point(text: str, score: float, source_name: str = "doc.pdf"):
    """Create a mock Qdrant search result point."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "content": text,
        "canonical_text": text,
        "source_name": source_name,
        "page": "1",
        "section_title": "Overview",
        "chunk_id": f"chunk-{hash(text) % 10000}",
        "document_id": "doc-001",
    }
    point.id = f"id-{hash(text) % 10000}"
    return point


def _mock_query_points_response():
    """Create mock Qdrant query_points response."""
    response = MagicMock()
    response.points = [
        _make_mock_point("Revenue was $10M in 2025.", 0.92, "report.pdf"),
        _make_mock_point("The company was founded in 2010.", 0.88, "report.pdf"),
        _make_mock_point("CEO is John Smith.", 0.85, "report.pdf"),
        _make_mock_point("Headquarters in New York.", 0.80, "report.pdf"),
    ]
    return response


def _build_engine():
    """Build a ReasoningEngine with fully mocked dependencies."""
    llm_client = MagicMock()
    llm_client.generate_with_metadata = MagicMock(
        return_value=("Generated answer with [SOURCE-1] citation.", {})
    )

    embedder = MagicMock()
    # Return 2D array for encode (batch interface)
    embedder.encode = MagicMock(return_value=np.array([[0.1] * 1024]))

    qdrant_client = MagicMock()
    qdrant_client.query_points = MagicMock(return_value=_mock_query_points_response())

    engine = ReasoningEngine(
        llm_client=llm_client,
        qdrant_client=qdrant_client,
        embedder=embedder,
        collection_name="test_collection",
        subscription_id="sub-001",
        profile_id="prof-001",
    )
    return engine, llm_client, qdrant_client, embedder


# ---------------------------------------------------------------------------
# Engine-level adaptive path tests
# ---------------------------------------------------------------------------

class TestFastPathMetadata:
    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_fast_path_metadata(self, _mock_filter):
        """Trivially simple query triggers fast path."""
        engine, llm_client, _, _ = _build_engine()

        result = engine.answer("What is the revenue?")

        assert result["metadata"]["fast_path"] is True
        assert result["context_found"] is True
        assert result["metadata"]["engine"] == "reasoning_engine"
        assert result["metadata"]["complexity"] == "simple"
        assert result["metadata"]["verification"]["skipped"] is True

    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_fast_path_skips_verify_high_confidence(self, _mock_filter):
        """Fast path always skips VERIFY (verification is only on intelligent path)."""
        engine, llm_client, _, _ = _build_engine()

        llm_client.generate_with_metadata = MagicMock(
            return_value=("Revenue is $10M [SOURCE-1]. Founded in 2010 [SOURCE-2]. CEO is John [SOURCE-3].", {})
        )

        result = engine.answer("What is the revenue?")

        assert result["metadata"]["fast_path"] is True
        assert result["metadata"]["verification"]["skipped"] is True
        assert result["metadata"]["verification"]["ok"] is True


class TestFullPathMetadata:
    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_full_path_metadata(self, _mock_filter):
        """Complex query triggers intelligent path with fast_path=False."""
        engine, llm_client, _, _ = _build_engine()

        # UNDERSTAND call returns structured JSON
        understand_response = '{"primary_intent":"comparison","sub_intents":[],"entities":["A","B"],"output_format":"table","complexity":"complex","needs_clarification":false,"clarification_question":null,"resolved_query":"Compare revenue across departments","thinking_required":true,"domain_hints":{}}'
        # GENERATE call returns answer
        generate_response = "| Department | Revenue |\n|---|---|\n| A | $5M [SOURCE-1] |\n| B | $3M [SOURCE-2] |"

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (understand_response, {})
            return (generate_response, {})

        llm_client.generate_with_metadata = MagicMock(side_effect=side_effect)

        long_query = "Compare the revenue figures across all departments and summarize the key differences between them"
        result = engine.answer(long_query, task_type="compare")

        assert result["metadata"]["fast_path"] is False
        assert result["context_found"] is True
        assert result["metadata"]["engine"] == "reasoning_engine"
        assert "verification" in result["metadata"]

    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_full_path_with_clarification(self, _mock_filter):
        """UNDERSTAND returning needs_clarification returns clarification response."""
        engine, llm_client, _, _ = _build_engine()

        understand_response = '{"primary_intent":"extract","sub_intents":[],"entities":[],"output_format":"prose","complexity":"simple","needs_clarification":true,"clarification_question":"Which quarter do you mean?","resolved_query":"","thinking_required":false,"domain_hints":{}}'

        llm_client.generate_with_metadata = MagicMock(
            return_value=(understand_response, {})
        )

        result = engine.answer("What was the revenue for that period?")

        assert result["metadata"]["needs_clarification"] is True
        assert "Which quarter" in result["response"]
