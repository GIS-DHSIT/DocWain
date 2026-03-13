"""Tests for the unified retriever, reranker, and context builder."""

import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from src.retrieval.retriever import EvidenceChunk, RetrievalResult, UnifiedRetriever
from src.retrieval.reranker import rerank_chunks
from src.retrieval.context_builder import build_context


# ---------------------------------------------------------------------------
# EvidenceChunk tests
# ---------------------------------------------------------------------------

class TestEvidenceChunk:
    def test_creation_and_fields(self):
        chunk = EvidenceChunk(
            text="Some text",
            source_name="doc.pdf",
            document_id="doc-1",
            profile_id="prof-1",
            section="Introduction",
            page_start=1,
            page_end=2,
            score=0.95,
            chunk_id="chunk-001",
        )
        assert chunk.text == "Some text"
        assert chunk.source_name == "doc.pdf"
        assert chunk.document_id == "doc-1"
        assert chunk.profile_id == "prof-1"
        assert chunk.section == "Introduction"
        assert chunk.page_start == 1
        assert chunk.page_end == 2
        assert chunk.score == 0.95
        assert chunk.chunk_id == "chunk-001"
        assert chunk.chunk_type == "text"
        assert chunk.profile_name == ""

    def test_defaults(self):
        chunk = EvidenceChunk(
            text="t", source_name="s", document_id="d", profile_id="p",
            section="sec", page_start=0, page_end=0, score=0.5, chunk_id="c",
        )
        assert chunk.chunk_type == "text"
        assert chunk.profile_name == ""

    def test_custom_chunk_type(self):
        chunk = EvidenceChunk(
            text="t", source_name="s", document_id="d", profile_id="p",
            section="sec", page_start=0, page_end=0, score=0.5, chunk_id="c",
            chunk_type="table",
        )
        assert chunk.chunk_type == "table"


# ---------------------------------------------------------------------------
# RetrievalResult tests
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_structure(self):
        c1 = EvidenceChunk(
            text="a", source_name="s", document_id="d1", profile_id="p1",
            section="", page_start=0, page_end=0, score=0.9, chunk_id="c1",
        )
        result = RetrievalResult(chunks=[c1], profiles_searched=["p1"], total_found=1)
        assert len(result.chunks) == 1
        assert result.profiles_searched == ["p1"]
        assert result.total_found == 1

    def test_empty_result(self):
        result = RetrievalResult(chunks=[], profiles_searched=[], total_found=0)
        assert result.chunks == []
        assert result.total_found == 0


# ---------------------------------------------------------------------------
# UnifiedRetriever tests
# ---------------------------------------------------------------------------

def _make_mock_point(payload, score=0.9):
    """Create a mock Qdrant scored point."""
    pt = MagicMock()
    pt.score = score
    pt.payload = payload
    return pt


def _sample_payload():
    return {
        "canonical_text": "The quick brown fox",
        "embedding_text": "quick brown fox",
        "document_id": "doc-123",
        "profile_id": "prof-abc",
        "source_file": "report.pdf",
        "chunk": {"id": "chunk-42", "type": "text", "index": 3},
        "section": {"title": "Summary", "path": ["Chapter 1", "Summary"]},
        "provenance": {"page_start": 5, "page_end": 6, "source_file": "report.pdf"},
    }


class TestUnifiedRetriever:
    def test_init(self):
        client = MagicMock()
        embedder = MagicMock()
        retriever = UnifiedRetriever(client, embedder)
        assert retriever.qdrant_client is client
        assert retriever.embedder is embedder

    def test_requires_subscription_id(self):
        client = MagicMock()
        embedder = MagicMock()
        retriever = UnifiedRetriever(client, embedder)
        with pytest.raises(ValueError, match="subscription_id"):
            retriever.retrieve("query", subscription_id="", profile_ids=["p1"])

    def test_retrieve_single_profile(self):
        client = MagicMock()
        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_point = _make_mock_point(_sample_payload(), score=0.85)
        query_result = MagicMock()
        query_result.points = [mock_point]
        client.query_points.return_value = query_result

        retriever = UnifiedRetriever(client, embedder)
        result = retriever.retrieve("test query", subscription_id="sub-1", profile_ids=["prof-abc"])

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) >= 1
        assert result.chunks[0].document_id == "doc-123"
        assert result.chunks[0].source_name == "report.pdf"
        assert result.chunks[0].section == "Summary"
        assert result.chunks[0].page_start == 5
        assert result.chunks[0].chunk_id == "chunk-42"
        assert result.profiles_searched == ["prof-abc"]

    def test_retrieve_multiple_profiles(self):
        client = MagicMock()
        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2]]

        payload1 = _sample_payload()
        payload1["profile_id"] = "p1"
        payload2 = _sample_payload()
        payload2["profile_id"] = "p2"
        payload2["document_id"] = "doc-456"

        qr1 = MagicMock()
        qr1.points = [_make_mock_point(payload1, 0.9)]
        qr2 = MagicMock()
        qr2.points = [_make_mock_point(payload2, 0.8)]
        client.query_points.side_effect = [qr1, qr2]

        retriever = UnifiedRetriever(client, embedder)
        result = retriever.retrieve("query", subscription_id="sub-1", profile_ids=["p1", "p2"])

        assert len(result.chunks) == 2
        assert set(result.profiles_searched) == {"p1", "p2"}

    def test_point_to_chunk_conversion(self):
        payload = _sample_payload()
        mock_point = _make_mock_point(payload, score=0.77)

        chunk = UnifiedRetriever._point_to_chunk(mock_point, "prof-abc")
        assert chunk.text == "The quick brown fox"
        assert chunk.score == 0.77
        assert chunk.chunk_type == "text"
        assert chunk.page_end == 6


# ---------------------------------------------------------------------------
# rerank_chunks tests
# ---------------------------------------------------------------------------

def _make_chunks(n):
    return [
        EvidenceChunk(
            text=f"chunk text number {i}",
            source_name=f"doc{i}.pdf",
            document_id=f"d{i}",
            profile_id="p1",
            section="sec",
            page_start=i,
            page_end=i,
            score=0.5 + i * 0.05,
            chunk_id=f"c{i}",
        )
        for i in range(n)
    ]


class TestReranker:
    def test_empty_input(self):
        assert rerank_chunks("query", []) == []

    def test_preserves_chunks(self):
        chunks = _make_chunks(5)
        result = rerank_chunks("query", chunks, top_k=10)
        assert len(result) == 5

    def test_respects_top_k(self):
        chunks = _make_chunks(10)
        result = rerank_chunks("query", chunks, top_k=3)
        assert len(result) == 3

    def test_sorted_by_score(self):
        chunks = _make_chunks(5)
        result = rerank_chunks("some query text", chunks, top_k=5)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_with_cross_encoder(self):
        chunks = _make_chunks(4)
        ce = MagicMock()
        ce.predict.return_value = [0.9, 0.1, 0.5, 0.7]

        result = rerank_chunks("query", chunks, top_k=4, cross_encoder=ce)
        assert len(result) == 4
        ce.predict.assert_called_once()
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_empty_input(self):
        evidence, doc_ctx = build_context([], {})
        assert evidence == []
        assert doc_ctx == {}

    def test_builds_numbered_evidence(self):
        chunks = _make_chunks(3)
        evidence, doc_ctx = build_context(chunks, {})
        assert len(evidence) == 3
        assert evidence[0]["source_index"] == 1
        assert evidence[1]["source_index"] == 2
        assert evidence[2]["source_index"] == 3
        assert "text" in evidence[0]
        assert "source_name" in evidence[0]
        assert "score" in evidence[0]
        assert "document_id" in evidence[0]
        assert "profile_id" in evidence[0]
        assert "chunk_id" in evidence[0]

    def test_aggregates_doc_intelligence(self):
        chunks = [
            EvidenceChunk(
                text="text1", source_name="a.pdf", document_id="d1", profile_id="p1",
                section="sec", page_start=1, page_end=1, score=0.9, chunk_id="c1",
            ),
            EvidenceChunk(
                text="text2", source_name="b.pdf", document_id="d2", profile_id="p1",
                section="sec", page_start=2, page_end=2, score=0.8, chunk_id="c2",
            ),
        ]
        doc_intel = {
            "d1": {"summary": "Doc one summary", "entities": ["Alice", "Bob"], "key_facts": ["fact1"]},
            "d2": {"summary": "Doc two summary", "entities": ["Bob", "Carol"], "key_facts": ["fact2"]},
        }
        evidence, doc_ctx = build_context(chunks, doc_intel)
        assert len(evidence) == 2
        assert "summary" in doc_ctx
        assert "Alice" in doc_ctx["entities"]
        assert "Bob" in doc_ctx["entities"]
        assert "Carol" in doc_ctx["entities"]
        # Deduplicated
        assert doc_ctx["entities"].count("Bob") == 1
        assert "fact1" in doc_ctx["key_facts"]
        assert "fact2" in doc_ctx["key_facts"]
