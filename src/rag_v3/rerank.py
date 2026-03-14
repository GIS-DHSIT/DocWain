from __future__ import annotations

import concurrent.futures
from src.utils.logging_utils import get_logger
import os
import re
from typing import Any, List, Sequence

from .types import Chunk

logger = get_logger(__name__)

# Minimum rerank score threshold - chunks below this are considered irrelevant
MIN_RERANK_SCORE = 0.20  # Raised from 0.15: reduces low-relevance noise in context
MIN_HIGH_QUALITY = 3  # Minimum chunks to keep even if below threshold

# Domain-adaptive minimum scores — structured domains need higher precision
_DOMAIN_MIN_SCORES = {
    "medical": 0.30,
    "invoice": 0.30,
    "legal": 0.25,
    "hr": 0.25,
    "policy": 0.25,
}

def get_domain_min_score(domain_hint: str | None = None) -> float:
    """Return domain-adaptive minimum rerank score."""
    if domain_hint:
        return _DOMAIN_MIN_SCORES.get(domain_hint.lower(), MIN_RERANK_SCORE)
    return MIN_RERANK_SCORE

# Configurable timeout — cross-encoder on CPU needs 2-4s for 8-14 pairs.
# Fail fast (6s) to avoid blocking the response; fallback to bi-encoder scores.
try:
    from src.api.config import Config as _Cfg
    _RERANK_TIMEOUT_S = getattr(getattr(_Cfg, "Reranker", None), "TIMEOUT_S", 6.0)
except Exception:
    _RERANK_TIMEOUT_S = float(os.getenv("RERANKER_TIMEOUT_S", "6.0"))

def _tag_rerank_confidence(chunks: List[Chunk], min_score: float) -> None:
    """Tag chunks with rerank confidence signal.

    When all returned chunks are below min_score, tags them with
    'rerank_low_confidence' in meta so the pipeline can adjust.
    """
    if not chunks:
        return
    above_threshold = sum(1 for c in chunks if c.score >= min_score)
    if above_threshold == 0:
        for c in chunks:
            meta = getattr(c, "meta", None)
            if meta is None:
                meta = {}
                c.meta = meta
            if isinstance(meta, dict):
                meta["rerank_low_confidence"] = True

def _infer_domain_from_chunks(chunks: List[Chunk]) -> str | None:
    """Infer dominant domain from chunk metadata."""
    from collections import Counter
    domains: Counter = Counter()
    for chunk in chunks[:10]:  # Sample first 10
        meta = getattr(chunk, "meta", None) or {}
        dom = meta.get("doc_domain") or meta.get("domain") or ""
        if dom:
            domains[dom] += 1
    if domains:
        best, count = domains.most_common(1)[0]
        # Use plurality (most common) with minimum 30% instead of 50% majority.
        # This ensures mixed-domain profiles (3 medical + 2 HR + 1 generic)
        # still get domain-adaptive thresholds applied.
        sampled = min(len(chunks), 10)
        if count >= max(2, sampled * 0.3):
            return best
    return None

def rerank_chunks(
    *,
    query: str,
    chunks: List[Chunk],
    cross_encoder: Any = None,
    top_k: int = 8,
    correlation_id: str | None = None,
    min_score: float = MIN_RERANK_SCORE,
    entity_hints: List[str] | None = None,
) -> List[Chunk]:
    if not chunks:
        return []

    # Infer domain from chunk metadata if not explicitly provided
    if not entity_hints:
        entity_hints = None  # normalize

    # Infer domain-adaptive min_score from chunk metadata when no explicit hint
    if min_score == MIN_RERANK_SCORE and chunks:
        _inferred = _infer_domain_from_chunks(chunks)
        if _inferred:
            _domain_min = _DOMAIN_MIN_SCORES.get(_inferred)
            if _domain_min and _domain_min > min_score:
                min_score = _domain_min
                logger.debug("Rerank inferred domain=%s, raised min_score=%.2f", _inferred, min_score)

    ordered = list(chunks)

    # Entity boost applied BEFORE cross-encoder for fallback path only.
    # For cross-encoder path, entity boost is applied AFTER scoring (inside _try_cross_encoder).
    if cross_encoder is not None:
        result = _try_cross_encoder(cross_encoder, query, ordered, min_score, top_k, correlation_id, entity_hints)
        if result is not None:
            return result

    # Fallback path: apply entity boost to bi-encoder scores
    if entity_hints:
        _apply_entity_boost(ordered, entity_hints)

    # Fallback: sort by vector score, apply same minimum quality gate as primary path.
    # Use the caller's min_score (domain-aware) instead of hardcoded 0.3.
    _fallback_min = max(min_score, MIN_RERANK_SCORE)
    ordered = [c for c in ordered if c.score >= _fallback_min] or ordered[:MIN_HIGH_QUALITY]
    ordered.sort(key=lambda c: (-c.score, c.id))

    # Apply lightweight diversity penalty to fallback results — the cross-encoder
    # normally penalizes 3rd+ chunks from the same doc+section, but this is lost
    # on timeout.  Cap per doc+section to prevent duplicate evidence.
    # Only apply when multiple doc+section groups exist (otherwise all chunks are
    # from the same source and we need them all).
    _seen_sections: dict[tuple[str, str], int] = {}
    for _c in ordered[:top_k]:
        _meta = getattr(_c, "meta", None) or {}
        _doc = _meta.get("source_name") or _meta.get("document_name") or ""
        _sec = _meta.get("section_kind") or _meta.get("chunk_type") or ""
        _key = (_doc, _sec)
        _seen_sections[_key] = _seen_sections.get(_key, 0) + 1

    # Only diversify when we have 3+ groups (multi-source) — single-source
    # results should keep all chunks for completeness
    if len(_seen_sections) >= 3:
        _n_groups = len(_seen_sections)
        _seen_sections.clear()
        # Adaptive diversity penalty consistent with cross-encoder path
        _fb_penalty = min(0.05, 0.02 + 0.005 * max(0, _n_groups - 1))
        for _c in ordered:
            _meta = getattr(_c, "meta", None) or {}
            _doc = _meta.get("source_name") or _meta.get("document_name") or ""
            _sec = _meta.get("section_kind") or _meta.get("chunk_type") or ""
            _key = (_doc, _sec)
            _count = _seen_sections.get(_key, 0)
            if _count >= 2:
                _c.score = _c.score * (1.0 - _fb_penalty * (_count - 1))
            _seen_sections[_key] = _count + 1
        ordered.sort(key=lambda c: (-c.score, c.id))
        _diverse = ordered[:top_k]
        _tag_rerank_confidence(_diverse, min_score)
        return _diverse

    _result = ordered[:top_k]
    _tag_rerank_confidence(_result, min_score)
    return _result

def rerank(
    *,
    query: str,
    chunks: List[Chunk],
    cross_encoder: Any = None,
    top_k: int = 8,
    correlation_id: str | None = None,
    min_score: float = MIN_RERANK_SCORE,
    entity_hints: List[str] | None = None,
) -> List[Chunk]:
    return rerank_chunks(
        query=query,
        chunks=chunks,
        cross_encoder=cross_encoder,
        top_k=top_k,
        correlation_id=correlation_id,
        min_score=min_score,
        entity_hints=entity_hints,
    )

_ENTITY_BOOST = 0.10  # Score boost for chunks containing conversation entities

def _apply_entity_boost(chunks: List[Chunk], entity_hints: List[str]) -> None:
    """Boost scores of chunks that mention known entities (persons, orgs from conversation).

    Applied in-place before cross-encoder scoring. The boost is additive so
    entity-relevant chunks float to the top even if their vector score is
    marginally lower.

    For multi-entity queries, boost is proportional to how many entities
    a chunk mentions (important for comparison queries needing all entities).
    """
    hints_lower = [h.lower() for h in entity_hints if h and len(h) >= 2]
    if not hints_lower:
        return

    is_multi_entity = len(hints_lower) >= 2

    for chunk in chunks:
        text_lower = chunk.text.lower()
        matched_count = 0
        for hint in hints_lower:
            if re.search(r'\b' + re.escape(hint) + r'\b', text_lower):
                matched_count += 1

        if matched_count > 0:
            if is_multi_entity:
                # Proportional boost: chunks mentioning more entities get higher boost
                # A chunk mentioning 2/3 entities gets 2/3 of the max boost
                proportion = matched_count / len(hints_lower)
                boost = _ENTITY_BOOST * (0.5 + 0.5 * proportion)
            else:
                boost = _ENTITY_BOOST

            # Use multiplicative boost for low-scoring chunks so entity-relevant
            # chunks below threshold still get meaningful lift. High-scoring chunks
            # get additive boost (they're already above threshold).
            if chunk.score < 0.25 and chunk.score > 0.0:
                # Multiplicative: 0.15 * 1.5 = 0.225 (significant lift)
                chunk.score = min(1.0, chunk.score * (1.0 + boost * 5))
            else:
                chunk.score = min(1.0, chunk.score + boost)

def _try_cross_encoder(
    encoder: Any,
    query: str,
    ordered: List[Chunk],
    min_score: float,
    top_k: int,
    correlation_id: str | None,
    entity_hints: List[str] | None = None,
) -> List[Chunk] | None:
    """Run cross-encoder scoring with hard timeout. Returns reranked list or None on failure."""
    # Filter out chunks with empty/whitespace-only text — cross-encoder
    # crashes with "index out of range in self" on empty inputs.
    valid_chunks = [c for c in ordered if c.text and c.text.strip()]
    if not valid_chunks:
        return None
    if len(valid_chunks) < len(ordered):
        logger.debug(
            "Rerank: filtered %d empty chunks before cross-encoder",
            len(ordered) - len(valid_chunks),
        )
        ordered = valid_chunks

    def _score():
        pairs = [[query, chunk.text] for chunk in ordered]
        if hasattr(encoder, "predict"):
            return encoder.predict(pairs, show_progress_bar=False)
        elif callable(encoder):
            return encoder(pairs)
        return None

    try:
        # Hard timeout — cross-encoder on CPU can stall under GPU contention
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_score)
            try:
                scores = future.result(timeout=_RERANK_TIMEOUT_S)
            except (TimeoutError, concurrent.futures.TimeoutError):
                future.cancel()
                logger.warning(
                    "RAG v3 rerank cross-encoder timed out (%.1fs) — skipping | cid=%s",
                    _RERANK_TIMEOUT_S, correlation_id,
                )
                return None

        flat = _normalize_scores(scores, len(ordered))
        if not flat:
            return None
        scored = list(zip(ordered, flat))
        scored.sort(key=lambda item: -item[1])

        # Write cross-encoder scores to chunks first
        for chunk, ce_score in scored:
            chunk.score = float(ce_score)

        # Apply entity boost BEFORE diversity penalty so entity-relevant chunks
        # from repeated doc+sections still get full boost value
        if entity_hints:
            _apply_entity_boost([c for c, _ in scored], entity_hints)

        # Apply diversity penalty: reduce score for subsequent chunks from
        # same document+section to promote diverse evidence coverage.
        # Adaptive: penalty scales with number of unique sources (more sources
        # available → stronger penalty to promote diversity).
        _seen_doc_sections: dict = {}  # (doc_name, section) → count
        _unique_sources = len({
            (getattr(c, "meta", None) or {}).get("source_name")
            or (getattr(c, "meta", None) or {}).get("document_name") or ""
            for c, _ in scored
        })
        # Adaptive penalty: 3% base for single-source, up to 8% for 5+ sources
        _diversity_penalty = min(0.08, 0.03 + 0.01 * max(0, _unique_sources - 1))
        # Score clustering: if top scores are tightly clustered, reduce penalty
        # (all chunks equally relevant → diversity less important than completeness)
        _top_scores = [s for _, s in scored[:min(8, len(scored))]]
        if len(_top_scores) >= 3:
            _score_spread = max(_top_scores) - min(_top_scores)
            if _score_spread < 0.10:
                _diversity_penalty *= 0.5  # halve penalty for tight clusters
        for i, (chunk, _ce_score) in enumerate(scored):
            meta = getattr(chunk, "meta", None) or {}
            _doc = meta.get("source_name") or meta.get("document_name") or ""
            _sec = meta.get("section_kind") or meta.get("chunk_type") or ""
            _key = (_doc, _sec)
            _count = _seen_doc_sections.get(_key, 0)
            # Penalize 3rd+ chunk from same doc+section
            if _count >= 2:
                chunk.score = chunk.score * (1.0 - _diversity_penalty * (_count - 1))
            _seen_doc_sections[_key] = _count + 1

        # Now filter by min_score after entity boost has been applied
        reranked = []
        for i, (chunk, _) in enumerate(scored):
            if chunk.score >= min_score or i < MIN_HIGH_QUALITY:
                reranked.append(chunk)
        reranked.sort(key=lambda c: (-c.score, c.id))

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
        _final = reranked[:top_k]
        _tag_rerank_confidence(_final, min_score)
        return _final
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 rerank cross-encoder failed: %s",
            exc,
            extra={"stage": "rerank", "correlation_id": correlation_id},
        )
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
