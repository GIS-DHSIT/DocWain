from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.security.response_sanitizer import sanitize_user_payload
from src.services.retrieval.confidence import RetrievalConfidenceScorer
from src.services.retrieval.context_builder import ContextAssembler
from src.services.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
from src.services.retrieval.reranker import Reranker


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


class FakeResult:
    def __init__(self, points: List[FakePoint]):
        self.points = points


class FakeClient:
    def __init__(self, points: List[FakePoint]):
        self._points = points

    def query_points(self, **kwargs):  # noqa: ANN003
        return FakeResult(self._points)


class FakeEmbedder:
    def encode(self, text: str, convert_to_numpy: bool = True, normalize_embeddings: bool = True):  # noqa: ANN001
        import numpy as np
        return np.ones(4, dtype=float)


def test_hybrid_scoring_uses_lexical_signal():
    points = [
        FakePoint(id="1", score=0.5, payload={"text": "alpha beta gamma", "source_file": "doc1.pdf"}),
        FakePoint(id="2", score=0.5, payload={"text": "delta epsilon zeta", "source_file": "doc2.pdf"}),
    ]
    retriever = HybridRetriever(
        client=FakeClient(points),
        embedder=FakeEmbedder(),
        config=HybridRetrieverConfig(topk_dense=2, hybrid_alpha=0.5),
    )
    candidates = retriever.retrieve(
        collection_name="collection",
        query="alpha",
        profile_id="profile",
        filters={},
        explicit_hints={},
    )
    assert len(candidates) == 2
    assert candidates[0].lexical_score >= candidates[1].lexical_score
    assert candidates[0].score >= candidates[1].score


def test_reranker_ordering_stability():
    class Candidate:
        def __init__(self, text: str, score: float):
            self.text = text
            self.score = score

    items = [Candidate("A", 0.5), Candidate("B", 0.5), Candidate("C", 0.5)]
    reranker = Reranker(cross_encoder=None, llm_client=None)
    reranked = reranker.rerank(query="test", candidates=items, top_k=3)
    assert [c.text for c in reranked] == ["A", "B", "C"]


def test_context_deduplication():
    assembler = ContextAssembler(max_tokens=200, dedup_threshold=0.8)
    chunks = [
        {"text": "This is a sample paragraph about policies.", "score": 0.9, "metadata": {"source_file": "a.pdf"}},
        {"text": "This is a sample paragraph about policies.", "score": 0.8, "metadata": {"source_file": "a.pdf"}},
    ]
    result = assembler.build(chunks)
    assert len(result.selected_chunks) == 1


def test_refusal_when_confidence_low():
    scorer = RetrievalConfidenceScorer()
    assert scorer.should_refuse(0.2, 0.62)
    assert not scorer.should_refuse(0.8, 0.62)


def test_no_internal_ids_in_sanitized_output():
    assembler = ContextAssembler(max_tokens=200, dedup_threshold=0.8)
    chunks = [
        {
            "text": "Sample text",
            "score": 0.9,
            "metadata": {
                "source_file": "/tmp/contract.pdf",
                "chunk_id": "abc123",
                "document_id": "deadbeefdeadbeefdeadbeef",
            },
        }
    ]
    result = assembler.build(chunks)
    payload = {"response": "answer", "sources": result.sources}
    sanitized = sanitize_user_payload(payload)
    assert "chunk_id" not in str(sanitized).lower()
    assert "document_id" not in str(sanitized).lower()
