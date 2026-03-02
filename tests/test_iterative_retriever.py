"""Tests for iterative multi-hop retrieval."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.rag_v3.types import Chunk, ChunkSource
from src.rag_v3.query_decomposer import SubQuery, DecomposedQuery


def _chunk(text: str, score: float = 0.7, doc: str = "d.pdf") -> Chunk:
    return Chunk(
        id=f"c_{hash(text) % 9999}",
        text=text,
        score=score,
        source=ChunkSource(document_name=doc),
    )


class TestIterativeRetrieval:

    def test_sufficient_evidence_stops_early(self):
        """If first hop has sufficient evidence, no second hop occurs."""
        from src.rag_v3.iterative_retriever import iterative_retrieve

        good_chunks = [
            _chunk("Prudhvi has AWS cert and 8yr cloud exp", 0.85, "prudhvi.pdf"),
        ]

        call_count = {"n": 0}
        def mock_retrieve(*args, **kwargs):
            call_count["n"] += 1
            return good_chunks

        with patch("src.rag_v3.iterative_retriever.retrieve_decomposed", side_effect=mock_retrieve):
            dq = DecomposedQuery(
                original="Prudhvi cloud experience",
                sub_queries=[SubQuery(text="Prudhvi cloud experience", entity_scope="Prudhvi")],
            )
            result = iterative_retrieve(
                dq, collection="c", subscription_id="s", profile_id="p",
                entity_hints=["Prudhvi"],
            )

        assert len(result.chunks) >= 1
        assert result.hops_used == 1

    def test_insufficient_triggers_second_hop(self):
        """If first hop misses an entity, second hop should target it."""
        from src.rag_v3.iterative_retriever import iterative_retrieve, IterativeResult
        from src.rag_v3.evidence_evaluator import EvidenceSufficiency

        hop1_chunks = [_chunk("Prudhvi has AWS skills", 0.8, "prudhvi.pdf")]
        hop2_chunks = [_chunk("Ajay is Azure certified", 0.75, "ajay.pdf")]

        hop = {"n": 0}
        def mock_retrieve(dq, *args, **kwargs):
            hop["n"] += 1
            if hop["n"] == 1:
                return hop1_chunks
            return hop2_chunks

        eval_call = {"n": 0}
        def mock_evaluate(query, chunks, entity_hints=None, **kwargs):
            eval_call["n"] += 1
            all_text = " ".join(c.text for c in chunks).lower()
            has_prudhvi = "prudhvi" in all_text
            has_ajay = "ajay" in all_text
            missing = []
            if not has_prudhvi:
                missing.append("Prudhvi")
            if not has_ajay:
                missing.append("Ajay")
            both_found = has_prudhvi and has_ajay
            suff = EvidenceSufficiency.__new__(EvidenceSufficiency)
            suff.coverage_score = 1.0 if both_found else 0.5
            suff.relevance_score = 0.8
            suff.diversity_score = 0.3
            suff.overall_score = 0.9 if both_found else 0.4
            suff.is_sufficient = both_found
            suff.missing_entities = missing
            return suff

        with patch("src.rag_v3.iterative_retriever.retrieve_decomposed", side_effect=mock_retrieve), \
             patch("src.rag_v3.iterative_retriever.evaluate_evidence", side_effect=mock_evaluate):
            dq = DecomposedQuery(
                original="Compare Prudhvi and Ajay",
                sub_queries=[
                    SubQuery(text="Prudhvi", entity_scope="Prudhvi"),
                    SubQuery(text="Ajay", entity_scope="Ajay"),
                ],
                intent="compare",
            )
            result = iterative_retrieve(
                dq, collection="c", subscription_id="s", profile_id="p",
                entity_hints=["Prudhvi", "Ajay"],
                max_hops=3,
            )

        assert result.hops_used >= 2
        all_text = " ".join(c.text for c in result.chunks).lower()
        assert "prudhvi" in all_text
        assert "ajay" in all_text

    def test_max_hops_respected(self):
        """Should not exceed max_hops even if evidence is always insufficient."""
        from src.rag_v3.iterative_retriever import iterative_retrieve

        def mock_retrieve(dq, *args, **kwargs):
            return []  # Always empty

        with patch("src.rag_v3.iterative_retriever.retrieve_decomposed", side_effect=mock_retrieve):
            dq = DecomposedQuery(
                original="test",
                sub_queries=[SubQuery(text="test")],
            )
            result = iterative_retrieve(
                dq, collection="c", subscription_id="s", profile_id="p",
                max_hops=3,
            )

        assert result.hops_used <= 3

    def test_stalled_retrieval_stops(self):
        """If no new chunks are found on a hop, iteration should stop."""
        from src.rag_v3.iterative_retriever import iterative_retrieve

        same_chunks = [_chunk("Only this chunk", 0.5, "doc.pdf")]

        call_count = {"n": 0}
        def mock_retrieve(dq, *args, **kwargs):
            call_count["n"] += 1
            return same_chunks

        with patch("src.rag_v3.iterative_retriever.retrieve_decomposed", side_effect=mock_retrieve):
            dq = DecomposedQuery(
                original="Find info about Unknown Person",
                sub_queries=[SubQuery(text="Unknown Person info", entity_scope="Unknown Person")],
            )
            result = iterative_retrieve(
                dq, collection="c", subscription_id="s", profile_id="p",
                entity_hints=["Unknown Person"],
                max_hops=5,
            )

        # Should stop early because same chunks returned each time (no new chunks added)
        assert result.hops_used <= 2
        assert call_count["n"] <= 2

    def test_per_hop_chunks_tracked(self):
        """per_hop_chunks should record what each hop returned."""
        from src.rag_v3.iterative_retriever import iterative_retrieve

        hop_data = [
            [_chunk("Hop 1 data", 0.8, "a.pdf")],
            [_chunk("Hop 2 data", 0.7, "b.pdf")],
        ]

        hop_idx = {"n": 0}
        def mock_retrieve(dq, *args, **kwargs):
            idx = hop_idx["n"]
            hop_idx["n"] += 1
            if idx < len(hop_data):
                return hop_data[idx]
            return []

        with patch("src.rag_v3.iterative_retriever.retrieve_decomposed", side_effect=mock_retrieve):
            dq = DecomposedQuery(
                original="Compare A and B",
                sub_queries=[
                    SubQuery(text="A info", entity_scope="A"),
                    SubQuery(text="B info", entity_scope="B"),
                ],
            )
            result = iterative_retrieve(
                dq, collection="c", subscription_id="s", profile_id="p",
                entity_hints=["Hop 1", "Hop 2"],
                max_hops=3,
            )

        assert len(result.per_hop_chunks) >= 1
        assert len(result.per_hop_chunks[0]) == 1


class TestMergeChunks:

    def test_dedup_by_id_keeps_best_score(self):
        from src.rag_v3.iterative_retriever import _merge_chunks
        c1 = _chunk("same text", 0.5, "d.pdf")
        c2 = Chunk(id=c1.id, text="same text updated", score=0.9,
                    source=ChunkSource(document_name="d.pdf"))
        merged = _merge_chunks([c1], [c2])
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_merge_preserves_unique(self):
        from src.rag_v3.iterative_retriever import _merge_chunks
        c1 = _chunk("text A", 0.8, "a.pdf")
        c2 = _chunk("text B", 0.7, "b.pdf")
        merged = _merge_chunks([c1], [c2])
        assert len(merged) == 2

    def test_merge_sorted_by_score_desc(self):
        from src.rag_v3.iterative_retriever import _merge_chunks
        c1 = _chunk("low score", 0.3, "a.pdf")
        c2 = _chunk("high score", 0.9, "b.pdf")
        merged = _merge_chunks([c1], [c2])
        assert merged[0].score >= merged[-1].score


class TestBuildGapQueries:

    def test_generates_per_entity_queries(self):
        from src.rag_v3.iterative_retriever import _build_gap_queries
        gaps = _build_gap_queries(["Alice", "Bob"], "cloud experience")
        assert len(gaps) == 2
        assert gaps[0].entity_scope == "Alice"
        assert gaps[1].entity_scope == "Bob"
        assert "Alice" in gaps[0].text
        assert "Bob" in gaps[1].text

    def test_empty_missing_returns_empty(self):
        from src.rag_v3.iterative_retriever import _build_gap_queries
        gaps = _build_gap_queries([], "some query")
        assert gaps == []
