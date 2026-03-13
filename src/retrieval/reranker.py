"""Ensemble reranking for retrieved evidence chunks."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from src.retrieval.retriever import EvidenceChunk


def _keyword_f1(query: str, text: str) -> float:
    """Token-level F1 overlap between query and text."""
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = q_tokens & t_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(q_tokens)
    recall = len(overlap) / len(t_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def rerank_chunks(
    query: str,
    chunks: List[EvidenceChunk],
    top_k: int = 15,
    cross_encoder=None,
) -> List[EvidenceChunk]:
    """Ensemble rerank: dense score + keyword F1 + optional cross-encoder.

    Weights:
        Without cross-encoder: 0.7 * dense + 0.3 * keyword_f1
        With cross-encoder:    0.5 * dense + 0.2 * keyword_f1 + 0.3 * ce
    """
    if not chunks:
        return []

    dense_scores = _normalize([c.score for c in chunks])
    kw_scores = _normalize([_keyword_f1(query, c.text) for c in chunks])

    if cross_encoder is not None:
        pairs = [[query, c.text] for c in chunks]
        raw_ce = cross_encoder.predict(pairs)
        ce_scores = _normalize(list(raw_ce))
        combined = [
            0.5 * d + 0.2 * k + 0.3 * ce
            for d, k, ce in zip(dense_scores, kw_scores, ce_scores)
        ]
    else:
        combined = [
            0.7 * d + 0.3 * k
            for d, k in zip(dense_scores, kw_scores)
        ]

    # Create new chunks with updated scores
    scored = []
    for chunk, new_score in zip(chunks, combined):
        scored.append(replace(chunk, score=new_score))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
