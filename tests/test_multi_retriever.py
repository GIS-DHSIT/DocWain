"""Tests for multi-strategy retrieval orchestrator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag_v3.multi_retriever import (
    _dispatch_subquery,
    retrieve_decomposed,
    rrf_fuse,
)
from src.rag_v3.query_decomposer import DecomposedQuery, SubQuery
from src.rag_v3.types import Chunk, ChunkSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(cid: str, score: float = 0.5, text: str = "some text") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        score=score,
        source=ChunkSource(document_name="doc.pdf"),
    )


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    """Tests for rrf_fuse()."""

    def test_rrf_fusion_basic(self):
        """c2 ranks highest when top in list_b, 2nd in list_a; 4 unique chunks total."""
        c1 = _chunk("c1", 0.9)
        c2 = _chunk("c2", 0.8)
        c3 = _chunk("c3", 0.7)
        c4 = _chunk("c4", 0.6)

        list_a = [c1, c2, c3]  # c1 rank 0, c2 rank 1, c3 rank 2
        list_b = [c2, c4, c1]  # c2 rank 0, c4 rank 1, c1 rank 2

        fused = rrf_fuse([list_a, list_b], k=60, top_n=10)

        ids = [c.id for c in fused]
        assert len(set(ids)) == 4  # 4 unique chunks

        # c2 appears at rank 1 in list_a (1/(60+1+1)) and rank 0 in list_b (1/(60+0+1))
        # c1 appears at rank 0 in list_a (1/(60+0+1)) and rank 2 in list_b (1/(60+2+1))
        # c2 RRF = 1/62 + 1/61 > c1 RRF = 1/61 + 1/63
        assert ids[0] == "c2", f"Expected c2 to be ranked first, got {ids[0]}"

    def test_rrf_deduplicates_by_id(self):
        """Same chunk in 3 lists should produce exactly 1 result."""
        c = _chunk("dup", 0.9)
        c_low = _chunk("dup", 0.3)

        fused = rrf_fuse([[c], [c_low], [c]])

        assert len(fused) == 1
        assert fused[0].id == "dup"
        # RRF score should be sum of three 1/(60+0+1) = 3/61
        expected_rrf = 3.0 / 61.0
        assert abs(fused[0].score - expected_rrf) < 1e-9

    def test_rrf_empty_lists(self):
        assert rrf_fuse([]) == []

    def test_rrf_single_list(self):
        c1 = _chunk("a", 0.9)
        c2 = _chunk("b", 0.5)
        fused = rrf_fuse([[c1, c2]], top_n=10)
        assert len(fused) == 2
        assert fused[0].id == "a"  # rank 0 -> higher RRF score

    def test_rrf_top_n_limits(self):
        chunks = [_chunk(f"c{i}", 0.5) for i in range(30)]
        fused = rrf_fuse([chunks], top_n=5)
        assert len(fused) == 5


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_MODULE = "src.rag_v3.multi_retriever"


class TestDispatch:
    """Tests for _dispatch_subquery() routing logic."""

    @patch(f"src.rag_v3.retrieve.retrieve_entity_scoped")
    def test_entity_scope_dispatches_entity_scoped(self, mock_entity):
        """When entity_scope is set, retrieve_entity_scoped is called."""
        expected = [_chunk("e1")]
        mock_entity.return_value = expected

        sq = SubQuery(text="Alice skills", entity_scope="Alice")
        result = _dispatch_subquery(
            sq,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )

        mock_entity.assert_called_once()
        call_args = mock_entity.call_args
        assert call_args[0][1] == "Alice"  # entity_name positional arg
        assert result == expected

    @patch(f"src.rag_v3.retrieve.retrieve_section_filtered")
    def test_section_focus_dispatches_section_filtered(self, mock_section):
        """When section_focus is set (no entity_scope), retrieve_section_filtered is called."""
        expected = [_chunk("s1")]
        mock_section.return_value = expected

        sq = SubQuery(text="education details", section_focus="education")
        result = _dispatch_subquery(
            sq,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )

        mock_section.assert_called_once()
        call_kwargs = mock_section.call_args
        assert call_kwargs[1].get("section_kind") == "education" or call_kwargs.kwargs.get("section_kind") == "education"
        assert result == expected

    @patch(f"src.rag_v3.retrieve.retrieve_chunks")
    def test_default_fallback_to_retrieve_chunks(self, mock_chunks):
        """No entity_scope, no section_focus -> retrieve_chunks (default)."""
        expected = [_chunk("d1")]
        mock_chunks.return_value = expected

        sq = SubQuery(text="general question")
        result = _dispatch_subquery(
            sq,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )

        mock_chunks.assert_called_once()
        assert result == expected

    @patch(f"src.rag_v3.retrieve.retrieve_chunks")
    @patch(f"src.rag_v3.retrieve.retrieve_entity_scoped")
    def test_entity_scope_falls_back_on_empty(self, mock_entity, mock_chunks):
        """entity_scope returns [], should fall through to retrieve_chunks."""
        mock_entity.return_value = []
        fallback = [_chunk("fb1")]
        mock_chunks.return_value = fallback

        sq = SubQuery(text="Bob experience", entity_scope="Bob")
        result = _dispatch_subquery(
            sq,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )

        mock_entity.assert_called_once()
        mock_chunks.assert_called_once()
        assert result == fallback


# ---------------------------------------------------------------------------
# retrieve_decomposed (integration of dispatch + fusion)
# ---------------------------------------------------------------------------

class TestRetrieveDecomposed:
    """Tests for retrieve_decomposed()."""

    @patch(f"{_MODULE}._dispatch_subquery")
    def test_single_subquery_no_fusion(self, mock_dispatch):
        """Single SubQuery goes directly to _dispatch_subquery, no fusion."""
        expected = [_chunk("x1"), _chunk("x2")]
        mock_dispatch.return_value = expected

        decomposed = DecomposedQuery(
            original="Alice skills",
            sub_queries=[SubQuery(text="Alice skills", entity_scope="Alice")],
        )

        result = retrieve_decomposed(
            decomposed,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )

        mock_dispatch.assert_called_once()
        assert result == expected

    @patch(f"{_MODULE}._dispatch_subquery")
    def test_retrieve_for_decomposed_query(self, mock_dispatch):
        """Multiple sub-queries trigger dispatch for each, results are RRF-fused."""
        c1 = _chunk("c1", 0.9)
        c2 = _chunk("c2", 0.8)
        c3 = _chunk("c3", 0.7)

        # First sub-query returns c1, c2; second returns c2, c3
        mock_dispatch.side_effect = [
            [c1, c2],
            [c2, c3],
        ]

        decomposed = DecomposedQuery(
            original="Compare Alice and Bob skills",
            sub_queries=[
                SubQuery(text="Alice skills", entity_scope="Alice"),
                SubQuery(text="Bob skills", entity_scope="Bob"),
            ],
            fusion_strategy="rrf",
            intent="compare",
        )

        result = retrieve_decomposed(
            decomposed,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )

        assert mock_dispatch.call_count == 2
        result_ids = {c.id for c in result}
        assert result_ids == {"c1", "c2", "c3"}
        # c2 appears in both lists -> highest RRF score
        assert result[0].id == "c2"

    @patch(f"{_MODULE}._dispatch_subquery")
    def test_empty_subqueries(self, mock_dispatch):
        """No sub-queries -> empty result."""
        decomposed = DecomposedQuery(original="", sub_queries=[])
        result = retrieve_decomposed(
            decomposed,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )
        assert result == []
        mock_dispatch.assert_not_called()

    @patch(f"{_MODULE}._dispatch_subquery")
    def test_all_subqueries_fail(self, mock_dispatch):
        """When all sub-query dispatches raise, return empty."""
        mock_dispatch.side_effect = RuntimeError("boom")

        decomposed = DecomposedQuery(
            original="test",
            sub_queries=[
                SubQuery(text="a"),
                SubQuery(text="b"),
            ],
        )
        result = retrieve_decomposed(
            decomposed,
            collection="coll",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=MagicMock(),
            qdrant_client=MagicMock(),
        )
        assert result == []
