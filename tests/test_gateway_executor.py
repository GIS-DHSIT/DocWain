"""Unit tests for the Screening Gateway executor.

Tests: screening execution (category, batch), audit log, and response builder.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gateway.unified_executor import (
    ScreeningExecutor,
    _build_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async function in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_executor() -> ScreeningExecutor:
    return ScreeningExecutor()


# ---------------------------------------------------------------------------
# TestScreeningExecutor
# ---------------------------------------------------------------------------


class TestScreeningExecutor:
    """Tests for the ScreeningExecutor dispatcher routing."""

    def test_category_requires_doc_ids(self):
        """Category screening without doc_ids returns missing_doc_ids error."""
        executor = _make_executor()
        result = _run(executor.execute_screening(categories=["integrity"]))
        assert result["status"] == "error"
        assert result["error"]["code"] == "missing_doc_ids"

    def test_batch_requires_profile_ids(self):
        """Batch screening without profile_ids returns missing_profile_ids error."""
        executor = _make_executor()
        result = _run(executor.execute_screening(categories=["run"]))
        assert result["status"] == "error"
        assert result["error"]["code"] == "missing_profile_ids"

    def test_category_routes_to_screening_category(self):
        """Named categories route to _execute_screening_category."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"risk_level": "LOW", "overall_score_0_100": 80.0}
        mock_engine.run_all.return_value = mock_result
        executor._screening_engine = mock_engine

        result = _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1"],
        ))
        assert result["status"] == "success"
        assert result["action"] == "screen:all"
        assert len(result["documents"]) == 1

    def test_internal_error_caught(self):
        """Unexpected exceptions from engine are caught per-doc."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_engine.run_all.side_effect = RuntimeError("Boom")
        executor._screening_engine = mock_engine

        result = _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1"],
        ))
        # Per-doc failure → partial status
        assert result["documents"][0]["status"] == "failed"

    def test_correlation_id_passthrough(self):
        """Custom correlation_id is included in response."""
        executor = _make_executor()
        result = _run(executor.execute_screening(
            categories=["integrity"],
            correlation_id="test-corr-123",
        ))
        assert result["correlation_id"] == "test-corr-123"

    def test_all_category_without_doc_ids(self):
        """'all' category still requires doc_ids."""
        executor = _make_executor()
        result = _run(executor.execute_screening(categories=["all"]))
        assert result["status"] == "error"
        assert result["error"]["code"] == "missing_doc_ids"


# ---------------------------------------------------------------------------
# TestScreeningExecution
# ---------------------------------------------------------------------------


class TestScreeningExecution:
    """Tests for screening backend execution details."""

    def test_screening_category_success(self):
        """Category screening with doc_ids succeeds."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"risk_level": "LOW", "overall_score_0_100": 80.0}
        mock_engine.run_all.return_value = mock_result
        executor._screening_engine = mock_engine

        result = _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1"],
        ))
        assert result["status"] == "success"
        assert len(result["documents"]) == 1
        assert result["documents"][0]["status"] == "success"

    def test_screening_category_partial_failure(self):
        """When one doc fails in category screening, status is partial."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"risk_level": "LOW"}
        mock_engine.run_all.side_effect = [mock_result, ValueError("Doc not found")]
        executor._screening_engine = mock_engine

        result = _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1", "doc2"],
        ))
        assert result["status"] == "partial"
        assert result["documents"][0]["status"] == "success"
        assert result["documents"][1]["status"] == "failed"

    def test_named_category_uses_run_category(self):
        """Named category (not 'all') calls engine.run_category."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_engine.run_category.return_value = []
        executor._screening_engine = mock_engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["integrity"],
                doc_ids=["doc1"],
            ))
        assert result["status"] == "success"
        mock_engine.run_category.assert_called_once()

    def test_multiple_doc_ids(self):
        """Multiple doc_ids each produce a document result entry."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"risk_level": "LOW"}
        mock_engine.run_all.return_value = mock_result
        executor._screening_engine = mock_engine

        result = _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1", "doc2", "doc3"],
        ))
        assert result["status"] == "success"
        assert len(result["documents"]) == 3
        assert result["metadata"]["documents_processed"] == 3

    def test_multiple_categories_per_doc(self):
        """Multiple categories run per document, results grouped by category."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_engine.run_category.return_value = []
        executor._screening_engine = mock_engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["security", "integrity"],
                doc_ids=["doc1"],
            ))

        assert result["status"] == "success"
        assert result["action"] == "screen:security,integrity"
        doc = result["documents"][0]
        assert "categories" in doc
        assert "security" in doc["categories"]
        assert "integrity" in doc["categories"]
        assert doc["categories"]["security"]["status"] == "success"
        assert doc["categories"]["integrity"]["status"] == "success"
        assert result["metadata"]["categories"] == ["security", "integrity"]

    def test_multiple_categories_partial_failure(self):
        """When one category fails, doc status is partial."""
        executor = _make_executor()
        mock_engine = MagicMock()
        # security succeeds, integrity fails
        mock_engine.run_category.side_effect = [[], RuntimeError("Integrity check failed")]
        executor._screening_engine = mock_engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["security", "integrity"],
                doc_ids=["doc1"],
            ))

        assert result["status"] == "partial"
        doc = result["documents"][0]
        assert doc["status"] == "partial"
        assert doc["categories"]["security"]["status"] == "success"
        assert doc["categories"]["integrity"]["status"] == "failed"

    def test_single_category_flat_response(self):
        """Single category returns flat doc result (backward compat)."""
        executor = _make_executor()
        mock_engine = MagicMock()
        mock_engine.run_category.return_value = []
        executor._screening_engine = mock_engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["integrity"],
                doc_ids=["doc1"],
            ))

        doc = result["documents"][0]
        # Single category: flat result, no "categories" key
        assert "result" in doc
        assert "categories" not in doc


# ---------------------------------------------------------------------------
# TestAuditLog
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Tests for MongoDB audit log persistence."""

    def test_audit_persists_on_success(self):
        """Successful execution persists to actions collection."""
        executor = _make_executor()
        mock_collection = MagicMock()
        executor._actions_collection = mock_collection
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"risk_level": "LOW"}
        mock_engine.run_all.return_value = mock_result
        executor._screening_engine = mock_engine

        _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1"],
        ))
        mock_collection.insert_one.assert_called_once()
        doc = mock_collection.insert_one.call_args[0][0]
        assert doc["action"] == "screen:all"
        assert doc["status"] == "success"

    def test_audit_persists_on_error(self):
        """Failed execution also persists to actions collection."""
        executor = _make_executor()
        mock_collection = MagicMock()
        executor._actions_collection = mock_collection

        _run(executor.execute_screening(categories=["integrity"]))
        mock_collection.insert_one.assert_called_once()
        doc = mock_collection.insert_one.call_args[0][0]
        assert doc["status"] == "error"

    def test_audit_input_summary_strips_text(self):
        """Large text values in input are replaced with length only."""
        summary = ScreeningExecutor._summarize_input({
            "text": "A" * 200,
            "target_lang": "fr",
            "items": [1, 2, 3],
        })
        assert "text_length" in summary
        assert summary["text_length"] == 200
        assert "text" not in summary
        assert summary["target_lang"] == "fr"
        assert summary["items_count"] == 3

    def test_audit_mongodb_down_resilient(self):
        """MongoDB failures don't break execution."""
        executor = _make_executor()
        mock_collection = MagicMock()
        mock_collection.insert_one.side_effect = Exception("MongoDB down")
        executor._actions_collection = mock_collection
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"risk_level": "LOW"}
        mock_engine.run_all.return_value = mock_result
        executor._screening_engine = mock_engine

        result = _run(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc1"],
        ))
        assert result["status"] == "success"  # Execution still succeeds


# ---------------------------------------------------------------------------
# TestBuildResponse
# ---------------------------------------------------------------------------


class TestBuildResponse:
    """Tests for _build_response helper."""

    def test_response_shape(self):
        """_build_response returns all required fields."""
        resp = _build_response(
            status="success",
            action="screen:integrity",
            correlation_id="cid-1",
            start_time=time.time() - 0.1,
        )
        assert resp["status"] == "success"
        assert resp["action"] == "screen:integrity"
        assert resp["correlation_id"] == "cid-1"
        assert resp["timestamp"]  # ISO format
        assert resp["duration_ms"] >= 0
        assert resp["sources"] == []
        assert resp["warnings"] == []

    def test_response_duration(self):
        """duration_ms accurately reflects elapsed time."""
        start = time.time() - 0.5
        resp = _build_response(
            status="success",
            action="test",
            correlation_id="c",
            start_time=start,
        )
        assert resp["duration_ms"] >= 400  # At least 400ms


# ---------------------------------------------------------------------------
# TestMultiCategoryScreening — end-to-end accuracy
# ---------------------------------------------------------------------------


class TestMultiCategoryScreening:
    """Verify documents are screened accurately when multiple categories are sent."""

    def _make_engine_mock(self, category_results: Dict[str, Any]):
        """Build a mock engine that returns different results per category.

        category_results maps category name → list of ToolResult-like mocks.
        """
        engine = MagicMock()

        def _run_category(cat, doc_id, **kwargs):
            key = cat.lower().replace("_", "-")
            if key not in category_results:
                raise ValueError(f"Unknown category: {cat}")
            return category_results[key]

        engine.run_category.side_effect = _run_category
        return engine

    def test_each_category_result_isolated_per_doc(self):
        """Each category's result is attached to the correct key per document."""
        executor = _make_executor()

        security_result = [MagicMock(to_dict=MagicMock(return_value={"tool_name": "pii_sensitivity", "score_0_1": 0.8}))]
        integrity_result = [MagicMock(to_dict=MagicMock(return_value={"tool_name": "integrity_hash", "score_0_1": 0.1}))]

        engine = self._make_engine_mock({
            "security": security_result,
            "integrity": integrity_result,
        })
        executor._screening_engine = engine

        with patch("src.screening.helpers.format_results") as mock_format:
            def _fmt(doc_id, results, eng=None):
                return {"doc_id": doc_id, "results": [r.to_dict() for r in results]}
            mock_format.side_effect = _fmt

            result = _run(executor.execute_screening(
                categories=["security", "integrity"],
                doc_ids=["doc1"],
            ))

        assert result["status"] == "success"
        doc = result["documents"][0]
        assert doc["categories"]["security"]["status"] == "success"
        assert doc["categories"]["integrity"]["status"] == "success"
        # Verify actual results are category-specific
        sec_results = doc["categories"]["security"]["result"]["results"]
        int_results = doc["categories"]["integrity"]["result"]["results"]
        assert sec_results[0]["tool_name"] == "pii_sensitivity"
        assert int_results[0]["tool_name"] == "integrity_hash"

    def test_multi_category_multi_doc(self):
        """Multiple categories × multiple documents all produce results."""
        executor = _make_executor()
        engine = MagicMock()
        engine.run_category.return_value = []
        executor._screening_engine = engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "x", "results": []}
            result = _run(executor.execute_screening(
                categories=["security", "quality", "language"],
                doc_ids=["doc1", "doc2"],
            ))

        assert result["status"] == "success"
        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert set(doc["categories"].keys()) == {"security", "quality", "language"}
        # 3 categories × 2 docs = 6 run_category calls
        assert engine.run_category.call_count == 6

    def test_all_in_multi_category_calls_run_all(self):
        """'all' within a multi-category list calls engine.run_all for that slot."""
        executor = _make_executor()
        engine = MagicMock()
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"risk_level": "LOW", "overall_score_0_100": 20.0}
        engine.run_all.return_value = mock_report
        engine.run_category.return_value = []
        executor._screening_engine = engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["all", "security"],
                doc_ids=["doc1"],
            ))

        doc = result["documents"][0]
        assert "all" in doc["categories"]
        assert "security" in doc["categories"]
        engine.run_all.assert_called_once()
        engine.run_category.assert_called_once()

    def test_one_category_fails_others_succeed(self):
        """Per-category failure doesn't block other categories on the same doc."""
        executor = _make_executor()
        engine = MagicMock()

        call_count = {"n": 0}
        def _side_effect(cat, doc_id, **kwargs):
            call_count["n"] += 1
            if cat == "integrity":
                raise RuntimeError("integrity tool crashed")
            return []

        engine.run_category.side_effect = _side_effect
        executor._screening_engine = engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["security", "integrity", "quality"],
                doc_ids=["doc1"],
            ))

        doc = result["documents"][0]
        assert doc["status"] == "partial"
        assert doc["categories"]["security"]["status"] == "success"
        assert doc["categories"]["integrity"]["status"] == "failed"
        assert "integrity tool crashed" in doc["categories"]["integrity"]["error"]
        assert doc["categories"]["quality"]["status"] == "success"
        # All 3 categories were attempted
        assert call_count["n"] == 3

    def test_metadata_includes_categories_list(self):
        """Response metadata contains the categories that were requested."""
        executor = _make_executor()
        engine = MagicMock()
        engine.run_category.return_value = []
        executor._screening_engine = engine

        with patch("src.screening.helpers.format_results") as mock_format:
            mock_format.return_value = {"doc_id": "doc1", "results": []}
            result = _run(executor.execute_screening(
                categories=["security", "compliance"],
                doc_ids=["doc1"],
            ))

        assert result["metadata"]["categories"] == ["security", "compliance"]
        assert result["action"] == "screen:security,compliance"
