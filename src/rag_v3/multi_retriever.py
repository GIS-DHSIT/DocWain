"""Multi-strategy retrieval orchestrator.

Dispatches sub-queries to the appropriate retrieval strategy
(entity-scoped, section-filtered, or standard vector search) and
fuses results via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from .query_decomposer import DecomposedQuery, SubQuery
from .types import Chunk

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    ranked_lists: List[List[Chunk]],
    *,
    k: int = 60,
    top_n: int = 20,
) -> List[Chunk]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    For each chunk present in any list, the fused score is:
        score(chunk) = sum(1 / (k + rank_i + 1))  across all lists

    Deduplicates by ``chunk.id`` and keeps the version with the highest
    *original* score.  The returned chunks carry the RRF fused score.
    """
    if not ranked_lists:
        return []

    # Accumulate RRF scores and track best original-score version per id.
    rrf_scores: Dict[str, float] = {}
    best_chunk: Dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            cid = chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

            existing = best_chunk.get(cid)
            if existing is None or chunk.score > existing.score:
                best_chunk[cid] = chunk

    # Build result list with RRF scores replacing chunk.score.
    results: List[Chunk] = []
    for cid, chunk in best_chunk.items():
        # Create a copy with the fused score to avoid mutating callers' data.
        fused = chunk.model_copy(update={"score": rrf_scores[cid]})
        results.append(fused)

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_n]

# ---------------------------------------------------------------------------
# Sub-query dispatch
# ---------------------------------------------------------------------------

def _dispatch_subquery(
    sq: SubQuery,
    collection: str,
    subscription_id: str,
    profile_id: str,
    *,
    embedder: Any,
    qdrant_client: Any,
    top_k: int = 20,
    correlation_id: str = "",
) -> List[Chunk]:
    """Route a single sub-query to the best retrieval strategy.

    Strategy selection:
    1. ``entity_scope`` set -> ``retrieve_entity_scoped``
    2. ``section_focus`` set -> ``retrieve_section_filtered``
    3. Default -> ``retrieve_chunks``

    Each specialised path falls back to ``retrieve_chunks`` on empty results.
    """
    # Lazy imports to avoid circular dependencies.
    from src.rag_v3.retrieve import (  # noqa: C0415
        retrieve_chunks,
        retrieve_entity_scoped,
        retrieve_section_filtered,
    )

    # --- Strategy 1: entity-scoped ---
    if sq.entity_scope:
        try:
            chunks = retrieve_entity_scoped(
                sq.text,
                sq.entity_scope,
                collection,
                subscription_id,
                profile_id,
                top_k=top_k,
                embedder=embedder,
                qdrant_client=qdrant_client,
                correlation_id=correlation_id,
            )
            if chunks:
                logger.debug(
                    "Entity-scoped retrieval returned %d chunks for entity=%s",
                    len(chunks),
                    sq.entity_scope,
                    extra={"correlation_id": correlation_id},
                )
                return chunks
        except Exception:  # noqa: BLE001
            logger.warning(
                "Entity-scoped retrieval failed for entity=%s; falling back",
                sq.entity_scope,
                exc_info=True,
                extra={"correlation_id": correlation_id},
            )

    # --- Strategy 2: section-filtered ---
    if sq.section_focus:
        try:
            chunks = retrieve_section_filtered(
                sq.text,
                collection,
                subscription_id,
                profile_id,
                section_kind=sq.section_focus,
                doc_domain=sq.doc_type_filter,
                top_k=top_k,
                embedder=embedder,
                qdrant_client=qdrant_client,
                correlation_id=correlation_id,
            )
            if chunks:
                logger.debug(
                    "Section-filtered retrieval returned %d chunks for section=%s",
                    len(chunks),
                    sq.section_focus,
                    extra={"correlation_id": correlation_id},
                )
                return chunks
        except Exception:  # noqa: BLE001
            logger.warning(
                "Section-filtered retrieval failed for section=%s; falling back",
                sq.section_focus,
                exc_info=True,
                extra={"correlation_id": correlation_id},
            )

    # --- Strategy 3: standard vector search (default / fallback) ---
    try:
        from src.api.vector_store import build_collection_name  # noqa: C0415

        chunks = retrieve_chunks(
            query=sq.text,
            subscription_id=subscription_id,
            profile_id=profile_id,
            qdrant_client=qdrant_client,
            embedder=embedder,
            top_k=top_k,
            correlation_id=correlation_id,
        )
        return chunks
    except Exception:  # noqa: BLE001
        logger.warning(
            "Default retrieve_chunks failed for sub-query: %s",
            sq.text[:80],
            exc_info=True,
            extra={"correlation_id": correlation_id},
        )
        return []

# ---------------------------------------------------------------------------
# Orchestrator entry-point
# ---------------------------------------------------------------------------

def retrieve_decomposed(
    decomposed: DecomposedQuery,
    collection: str,
    subscription_id: str,
    profile_id: str,
    *,
    embedder: Any,
    qdrant_client: Any,
    top_k: int = 20,
    correlation_id: str = "",
) -> List[Chunk]:
    """Retrieve chunks for a decomposed query.

    * Single sub-query: direct dispatch (no fusion overhead).
    * Multiple sub-queries: dispatch each, then fuse via ``rrf_fuse``.
    """
    subs = decomposed.sub_queries
    if not subs:
        return []

    # Fast path -- single sub-query, no fusion needed.
    if len(subs) == 1:
        return _dispatch_subquery(
            subs[0],
            collection,
            subscription_id,
            profile_id,
            embedder=embedder,
            qdrant_client=qdrant_client,
            top_k=top_k,
            correlation_id=correlation_id,
        )

    # Multiple sub-queries -> dispatch in parallel and fuse.
    ranked_lists: List[List[Chunk]] = []
    max_workers = min(len(subs), 4)  # Cap at 4 to avoid Qdrant connection exhaustion
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _dispatch_subquery,
                sq,
                collection,
                subscription_id,
                profile_id,
                embedder=embedder,
                qdrant_client=qdrant_client,
                top_k=top_k,
                correlation_id=correlation_id,
            ): sq
            for sq in subs
        }
        for future in as_completed(futures):
            sq = futures[future]
            try:
                result = future.result()
                if result:
                    ranked_lists.append(result)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Sub-query dispatch failed: %s",
                    sq.text[:80],
                    exc_info=True,
                    extra={"correlation_id": correlation_id},
                )

    if not ranked_lists:
        return []

    if len(ranked_lists) == 1:
        return ranked_lists[0][:top_k]

    return rrf_fuse(ranked_lists, top_n=top_k)
