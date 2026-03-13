"""Tests for document_ids filtering in UnifiedRetriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import FieldCondition, MatchAny, MatchValue

from src.retrieval.retriever import UnifiedRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever() -> tuple[UnifiedRetriever, MagicMock, MagicMock]:
    """Return a retriever with mocked qdrant client and embedder."""
    qdrant = MagicMock()
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1] * 1024]

    # Default: query_points returns empty result
    result_obj = MagicMock()
    result_obj.points = []
    qdrant.query_points.return_value = result_obj
    qdrant.scroll.return_value = ([], None)

    retriever = UnifiedRetriever(qdrant_client=qdrant, embedder=embedder)
    return retriever, qdrant, embedder


def _extract_filter_conditions(qdrant_mock: MagicMock) -> List[FieldCondition]:
    """Extract the must conditions from the query_filter passed to query_points."""
    qdrant_mock.query_points.assert_called_once()
    call_kwargs = qdrant_mock.query_points.call_args
    qfilter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
    assert qfilter is not None, "query_filter not passed to query_points"
    return list(qfilter.must)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDocumentIdsFilter:
    """Verify that document_ids are correctly wired into the Qdrant filter."""

    @patch("src.retrieval.retriever.build_collection_name", return_value="col_sub1")
    def test_retriever_filters_by_document_ids(self, _mock_col):
        """When document_ids provided, Qdrant search filter should include them."""
        retriever, qdrant, _ = _make_retriever()

        retriever.retrieve(
            query="What is the policy?",
            subscription_id="sub_1",
            profile_ids=["prof_1"],
            document_ids=["doc_123", "doc_456"],
        )

        conditions = _extract_filter_conditions(qdrant)

        # Should have 3 conditions: subscription_id, profile_id, document_id
        assert len(conditions) == 3

        doc_cond = conditions[2]
        assert doc_cond.key == "document_id"
        assert isinstance(doc_cond.match, MatchAny)
        assert set(doc_cond.match.any) == {"doc_123", "doc_456"}

    @patch("src.retrieval.retriever.build_collection_name", return_value="col_sub1")
    def test_retriever_no_document_filter_when_none(self, _mock_col):
        """When document_ids is None, no document_id condition should be present."""
        retriever, qdrant, _ = _make_retriever()

        retriever.retrieve(
            query="What is the policy?",
            subscription_id="sub_1",
            profile_ids=["prof_1"],
            document_ids=None,
        )

        conditions = _extract_filter_conditions(qdrant)

        # Only subscription_id + profile_id
        assert len(conditions) == 2
        keys = {c.key for c in conditions}
        assert keys == {"subscription_id", "profile_id"}
        assert all(not c.key == "document_id" for c in conditions)

    @patch("src.retrieval.retriever.build_collection_name", return_value="col_sub1")
    def test_retriever_single_document_id_uses_match_value(self, _mock_col):
        """A single document_id should use MatchValue for efficiency."""
        retriever, qdrant, _ = _make_retriever()

        retriever.retrieve(
            query="What is the policy?",
            subscription_id="sub_1",
            profile_ids=["prof_1"],
            document_ids=["doc_only"],
        )

        conditions = _extract_filter_conditions(qdrant)
        assert len(conditions) == 3

        doc_cond = conditions[2]
        assert doc_cond.key == "document_id"
        assert isinstance(doc_cond.match, MatchValue)
        assert doc_cond.match.value == "doc_only"

    @patch("src.retrieval.retriever.build_collection_name", return_value="col_sub1")
    def test_retriever_empty_document_ids_no_filter(self, _mock_col):
        """An empty document_ids list should not add a document_id condition."""
        retriever, qdrant, _ = _make_retriever()

        retriever.retrieve(
            query="What is the policy?",
            subscription_id="sub_1",
            profile_ids=["prof_1"],
            document_ids=[],
        )

        conditions = _extract_filter_conditions(qdrant)
        assert len(conditions) == 2
        keys = {c.key for c in conditions}
        assert "document_id" not in keys


class TestBuildFilterUnit:
    """Direct unit tests on _build_filter for clarity."""

    def test_build_filter_with_multiple_doc_ids(self):
        retriever, _, _ = _make_retriever()
        f = retriever._build_filter("sub_1", "prof_1", document_ids=["a", "b", "c"])
        doc_conds = [c for c in f.must if c.key == "document_id"]
        assert len(doc_conds) == 1
        assert isinstance(doc_conds[0].match, MatchAny)
        assert doc_conds[0].match.any == ["a", "b", "c"]

    def test_build_filter_without_doc_ids(self):
        retriever, _, _ = _make_retriever()
        f = retriever._build_filter("sub_1", "prof_1", document_ids=None)
        doc_conds = [c for c in f.must if c.key == "document_id"]
        assert len(doc_conds) == 0

    def test_build_filter_single_doc_id(self):
        retriever, _, _ = _make_retriever()
        f = retriever._build_filter("sub_1", "prof_1", document_ids=["only"])
        doc_conds = [c for c in f.must if c.key == "document_id"]
        assert len(doc_conds) == 1
        assert isinstance(doc_conds[0].match, MatchValue)
        assert doc_conds[0].match.value == "only"
