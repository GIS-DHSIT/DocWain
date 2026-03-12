"""
Tests for the adaptive fast/full path logic in ReasoningEngine.
"""
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.intelligence.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# _is_simple_query classification tests
# ---------------------------------------------------------------------------

class TestIsSimpleQuery:
    def test_simple_query_detected(self):
        """Short factual queries are classified as simple."""
        assert ReasoningEngine._is_simple_query("What is the revenue?") is True
        assert ReasoningEngine._is_simple_query("Who is the CEO?") is True
        assert ReasoningEngine._is_simple_query("When was it founded?") is True

    def test_comparison_not_simple(self):
        """Comparison task_type forces full path regardless of query length."""
        assert ReasoningEngine._is_simple_query("Compare A and B", task_type="compare") is False
        assert ReasoningEngine._is_simple_query("short", task_type="summarize") is False
        assert ReasoningEngine._is_simple_query("short", task_type="rank") is False
        assert ReasoningEngine._is_simple_query("short", task_type="generate") is False
        assert ReasoningEngine._is_simple_query("short", task_type="reasoning") is False

    def test_long_query_not_simple(self):
        """Queries with 20+ words are not simple (no task_type)."""
        long_query = " ".join(["word"] * 25)
        assert ReasoningEngine._is_simple_query(long_query) is False

    def test_extract_task_short_is_simple(self):
        """Extract/list/qa task types are simple when query is short."""
        assert ReasoningEngine._is_simple_query("List the top skills", task_type="extract") is True
        assert ReasoningEngine._is_simple_query("List the top skills", task_type="list") is True
        assert ReasoningEngine._is_simple_query("What is X?", task_type="qa") is True

    def test_extract_task_long_not_simple(self):
        """Extract/list/qa task types are NOT simple when query is long."""
        long_query = " ".join(["word"] * 25)
        assert ReasoningEngine._is_simple_query(long_query, task_type="extract") is False


# ---------------------------------------------------------------------------
# Helpers for engine-level tests
# ---------------------------------------------------------------------------

def _make_mock_point(text: str, score: float, source_name: str = "doc.pdf"):
    """Create a mock Qdrant search result point."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "content": text,
        "source_name": source_name,
        "page": "1",
        "section_title": "Overview",
        "chunk_id": f"chunk-{hash(text) % 10000}",
        "document_id": "doc-001",
    }
    point.id = f"id-{hash(text) % 10000}"
    return point


def _build_engine():
    """Build a ReasoningEngine with fully mocked dependencies."""
    llm_client = MagicMock()
    llm_client.generate_with_metadata = MagicMock(
        return_value=("Generated answer with [SOURCE-1] citation.", {})
    )

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.array([[0.1] * 1024]))

    qdrant_client = MagicMock()
    qdrant_client.search = MagicMock(return_value=[
        _make_mock_point("Revenue was $10M in 2025.", 0.92, "report.pdf"),
        _make_mock_point("The company was founded in 2010.", 0.88, "report.pdf"),
        _make_mock_point("CEO is John Smith.", 0.85, "report.pdf"),
        _make_mock_point("Headquarters in New York.", 0.80, "report.pdf"),
    ])

    thinker = MagicMock()
    # THINK returns a valid JSON search plan
    think_response = '{"intent":"factual","complexity":"simple","actions":[{"query":"test","strategy":"semantic"}],"key_entities":[],"reasoning":"test"}'
    # REASON returns a valid JSON assessment
    reason_response = '{"sufficient":true,"confidence":0.9,"gaps":[],"contradictions":[],"key_findings":["found data"],"reasoning":"ok"}'
    # VERIFY returns a valid JSON verification
    verify_response = '{"ok":true,"unsupported_claims":[],"reasoning":"all good"}'

    thinker.generate_with_metadata = MagicMock(
        side_effect=[(think_response, {}), (reason_response, {}), (verify_response, {})]
    )

    engine = ReasoningEngine(
        llm_client=llm_client,
        thinking_client=thinker,
        qdrant_client=qdrant_client,
        embedder=embedder,
        collection_name="test_collection",
        subscription_id="sub-001",
        profile_id="prof-001",
    )
    return engine, llm_client, thinker, qdrant_client, embedder


# ---------------------------------------------------------------------------
# Engine-level adaptive path tests
# ---------------------------------------------------------------------------

class TestFastPathMetadata:
    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_fast_path_metadata(self, _mock_filter):
        """Simple query triggers fast path with fast_path=True in metadata."""
        engine, llm_client, thinker, _, _ = _build_engine()

        # For fast path, thinker is only called for VERIFY (no THINK/REASON)
        # Reset thinker side_effect for fast path: only verify may be called
        thinker.generate_with_metadata = MagicMock(
            return_value=('{"ok":true,"unsupported_claims":[],"reasoning":"verified"}', {})
        )

        result = engine.answer("What is the revenue?")

        assert result["metadata"]["fast_path"] is True
        assert result["context_found"] is True
        assert result["metadata"]["engine"] == "reasoning_engine"
        assert result["metadata"]["complexity"] == "simple"
        assert "verification" in result["metadata"]
        assert "skipped" in result["metadata"]["verification"]

    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_fast_path_skips_verify_high_confidence(self, _mock_filter):
        """Fast path skips VERIFY when evidence has high scores and 3+ citations."""
        engine, llm_client, thinker, _, _ = _build_engine()

        # Thinker should not be called at all in fast path if verify is skipped
        thinker.generate_with_metadata = MagicMock(
            side_effect=AssertionError("thinker should not be called when verify is skipped")
        )

        result = engine.answer("What is the revenue?")

        # 4 mock points with scores >= 0.8, so verify should be skipped
        assert result["metadata"]["fast_path"] is True
        assert result["metadata"]["verification"]["skipped"] is True
        assert result["metadata"]["verification"]["ok"] is True


class TestFullPathMetadata:
    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_full_path_metadata(self, _mock_filter):
        """Complex query triggers full path with fast_path=False in metadata."""
        engine, llm_client, thinker, _, _ = _build_engine()

        # Full path calls: THINK, REASON, VERIFY (3 thinker calls)
        think_resp = '{"intent":"comparison","complexity":"complex","actions":[{"query":"compare A B","strategy":"semantic"}],"key_entities":["A","B"],"reasoning":"comparing"}'
        reason_resp = '{"sufficient":true,"confidence":0.9,"gaps":[],"contradictions":[],"key_findings":["found"],"reasoning":"ok"}'
        verify_resp = '{"ok":true,"unsupported_claims":[],"reasoning":"verified"}'
        thinker.generate_with_metadata = MagicMock(
            side_effect=[(think_resp, {}), (reason_resp, {}), (verify_resp, {})]
        )

        long_query = "Compare the revenue figures across all departments and summarize the key differences between them"
        result = engine.answer(long_query, task_type="compare")

        assert result["metadata"]["fast_path"] is False
        assert result["context_found"] is True
        assert result["metadata"]["engine"] == "reasoning_engine"
        assert "verification" in result["metadata"]
        assert "skipped" in result["metadata"]["verification"]

    @patch("src.api.vector_store.build_qdrant_filter", return_value=None)
    def test_full_path_skips_verify_high_confidence(self, _mock_filter):
        """Full path skips VERIFY when assessment confidence >= 0.8 and 3+ citations."""
        engine, llm_client, thinker, _, _ = _build_engine()

        # High confidence assessment should cause verify skip
        think_resp = '{"intent":"comparison","complexity":"complex","actions":[{"query":"compare","strategy":"semantic"}],"key_entities":[],"reasoning":"plan"}'
        reason_resp = '{"sufficient":true,"confidence":0.85,"gaps":[],"contradictions":[],"key_findings":["found"],"reasoning":"high confidence"}'
        thinker.generate_with_metadata = MagicMock(
            side_effect=[(think_resp, {}), (reason_resp, {})]
        )

        result = engine.answer("Compare A and B in detail across multiple dimensions", task_type="compare")

        assert result["metadata"]["fast_path"] is False
        assert result["metadata"]["verification"]["skipped"] is True
        assert result["metadata"]["verification"]["ok"] is True
