from __future__ import annotations

import logging
from typing import Any, List, Sequence

from .types import Chunk

logger = logging.getLogger(__name__)


def rerank_chunks(
    *,
    query: str,
    chunks: List[Chunk],
    cross_encoder: Any = None,
    top_k: int = 8,
    correlation_id: str | None = None,
) -> List[Chunk]:
    if not chunks:
        return []

    ordered = list(chunks)
    if cross_encoder is not None:
        try:
            pairs = [[query, chunk.text] for chunk in ordered]
            scores = None
            if hasattr(cross_encoder, "predict"):
                scores = cross_encoder.predict(pairs)
            elif callable(cross_encoder):
                scores = cross_encoder(pairs)
            flat = _normalize_scores(scores, len(ordered))
            if flat:
                scored = list(zip(ordered, flat))
                scored.sort(key=lambda item: -item[1])
                reranked = []
                for chunk, score in scored:
                    chunk.score = float(score)
                    reranked.append(chunk)
                logger.info(
                    "RAG v3 rerank used cross-encoder",
                    extra={"stage": "rerank", "correlation_id": correlation_id, "candidates": len(reranked)},
                )
                return reranked[:top_k]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RAG v3 rerank failed; falling back to score: %s",
                exc,
                extra={"stage": "rerank", "correlation_id": correlation_id},
            )

    ordered.sort(key=lambda c: (-c.score, c.id))
    return ordered[:top_k]


def rerank(
    *,
    query: str,
    chunks: List[Chunk],
    cross_encoder: Any = None,
    top_k: int = 8,
    correlation_id: str | None = None,
) -> List[Chunk]:
    return rerank_chunks(
        query=query,
        chunks=chunks,
        cross_encoder=cross_encoder,
        top_k=top_k,
        correlation_id=correlation_id,
    )


def _normalize_scores(scores: Any, expected: int) -> List[float]:
    if scores is None:
        return []
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        return [float(scores)]
    if isinstance(scores, Sequence):
        if len(scores) == expected and all(isinstance(x, (int, float)) for x in scores):
            return [float(x) for x in scores]
        if len(scores) == 1 and isinstance(scores[0], Sequence):
            inner = scores[0]
            if len(inner) == expected:
                return [float(x) for x in inner]
    return []
