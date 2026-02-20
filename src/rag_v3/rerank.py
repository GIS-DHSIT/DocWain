from __future__ import annotations

import logging
from typing import Any, List, Sequence

from .types import Chunk

logger = logging.getLogger(__name__)

# Minimum rerank score threshold - chunks below this are considered irrelevant
MIN_RERANK_SCORE = 0.15  # Cross-encoder scores are typically -10 to 10, normalized
MIN_HIGH_QUALITY = 3  # Minimum chunks to keep even if below threshold


def rerank_chunks(
    *,
    query: str,
    chunks: List[Chunk],
    cross_encoder: Any = None,
    top_k: int = 8,
    correlation_id: str | None = None,
    min_score: float = MIN_RERANK_SCORE,
) -> List[Chunk]:
    if not chunks:
        return []

    ordered = list(chunks)
    if cross_encoder is not None:
        result = _try_cross_encoder(cross_encoder, query, ordered, min_score, top_k, correlation_id)
        if result is not None:
            return result
        # GPU attempt failed — try CPU fallback
        cpu_encoder = _move_to_cpu(cross_encoder)
        if cpu_encoder is not None:
            result = _try_cross_encoder(cpu_encoder, query, ordered, min_score, top_k, correlation_id)
            if result is not None:
                return result

    ordered.sort(key=lambda c: (-c.score, c.id))
    return ordered[:top_k]


def rerank(
    *,
    query: str,
    chunks: List[Chunk],
    cross_encoder: Any = None,
    top_k: int = 8,
    correlation_id: str | None = None,
    min_score: float = MIN_RERANK_SCORE,
) -> List[Chunk]:
    return rerank_chunks(
        query=query,
        chunks=chunks,
        cross_encoder=cross_encoder,
        top_k=top_k,
        correlation_id=correlation_id,
        min_score=min_score,
    )


def _try_cross_encoder(
    encoder: Any,
    query: str,
    ordered: List[Chunk],
    min_score: float,
    top_k: int,
    correlation_id: str | None,
) -> List[Chunk] | None:
    """Run cross-encoder scoring. Returns reranked list or None on failure."""
    try:
        pairs = [[query, chunk.text] for chunk in ordered]
        scores = None
        if hasattr(encoder, "predict"):
            scores = encoder.predict(pairs)
        elif callable(encoder):
            scores = encoder(pairs)
        flat = _normalize_scores(scores, len(ordered))
        if not flat:
            return None
        scored = list(zip(ordered, flat))
        scored.sort(key=lambda item: -item[1])

        reranked = []
        for i, (chunk, score) in enumerate(scored):
            chunk.score = float(score)
            if score >= min_score or i < MIN_HIGH_QUALITY:
                reranked.append(chunk)

        filtered_count = len(scored) - len(reranked)
        if filtered_count > 0:
            logger.info(
                "RAG v3 rerank filtered %d low-score chunks (threshold=%.2f)",
                filtered_count,
                min_score,
                extra={"stage": "rerank_filter", "correlation_id": correlation_id},
            )
        logger.info(
            "RAG v3 rerank used cross-encoder",
            extra={
                "stage": "rerank",
                "correlation_id": correlation_id,
                "candidates": len(reranked),
                "filtered": filtered_count,
            },
        )
        return reranked[:top_k]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 rerank cross-encoder failed: %s",
            exc,
            extra={"stage": "rerank", "correlation_id": correlation_id},
        )
        return None


def _move_to_cpu(encoder: Any) -> Any:
    """Attempt to move a cross-encoder model to CPU for CUDA OOM fallback."""
    try:
        import torch
        if hasattr(encoder, "model") and hasattr(encoder.model, "to"):
            encoder.model.to("cpu")
            # Update internal device tracking — CrossEncoder.device is read-only
            # so we set the private attribute directly.
            try:
                encoder._target_device = torch.device("cpu")
            except Exception:  # noqa: BLE001
                pass
            logger.info("RAG v3 rerank moved cross-encoder to CPU for fallback")
            return encoder
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG v3 rerank CPU fallback failed: %s", exc)
    return None


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
