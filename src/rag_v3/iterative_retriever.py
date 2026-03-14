"""Iterative (multi-hop) retrieval: retrieve -> evaluate -> re-retrieve for missing evidence."""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.rag_v3.evidence_evaluator import EvidenceSufficiency, evaluate_evidence
from src.rag_v3.multi_retriever import retrieve_decomposed
from src.rag_v3.query_decomposer import DecomposedQuery, SubQuery
from src.rag_v3.types import Chunk

logger = get_logger(__name__)

@dataclass
class IterativeResult:
    """Result of iterative retrieval across multiple hops."""
    chunks: List[Chunk]
    hops_used: int
    sufficiency: EvidenceSufficiency
    per_hop_chunks: List[List[Chunk]] = field(default_factory=list)

def _build_gap_queries(
    missing_entities: List[str],
    original_query: str,
) -> List[SubQuery]:
    """Generate targeted sub-queries to fill evidence gaps."""
    gap_queries = []
    for entity in missing_entities:
        gap_queries.append(SubQuery(
            text=f"{entity} {original_query}",
            entity_scope=entity,
            importance=1.2,
        ))
    return gap_queries

def _merge_chunks(existing: List[Chunk], new: List[Chunk]) -> List[Chunk]:
    """Merge new chunks into existing set, dedup by id, keep best score."""
    by_id: Dict[str, Chunk] = {c.id: c for c in existing}
    for c in new:
        if c.id not in by_id or c.score > by_id[c.id].score:
            by_id[c.id] = c
    return sorted(by_id.values(), key=lambda c: -(c.score or 0))

def iterative_retrieve(
    decomposed: DecomposedQuery,
    collection: str,
    subscription_id: str,
    profile_id: str,
    *,
    entity_hints: Optional[List[str]] = None,
    max_hops: int = 3,
    sufficiency_threshold: float = 0.35,
    embedder=None,
    qdrant_client=None,
    correlation_id: str = "",
) -> IterativeResult:
    """Execute iterative retrieval with evidence gap detection.

    Hop 1: Retrieve for all sub-queries in the decomposed query.
    Hop 2+: If evidence is insufficient (missing entities, low coverage),
            generate targeted follow-up queries and retrieve again.
    Stops when: evidence is sufficient, max_hops reached, or no new chunks found.
    """
    all_chunks: List[Chunk] = []
    per_hop: List[List[Chunk]] = []
    current_dq = decomposed

    for hop in range(1, max_hops + 1):
        new_chunks = retrieve_decomposed(
            current_dq,
            collection=collection,
            subscription_id=subscription_id,
            profile_id=profile_id,
            embedder=embedder,
            qdrant_client=qdrant_client,
            correlation_id=correlation_id,
        )
        per_hop.append(new_chunks)
        prev_count = len(all_chunks)
        all_chunks = _merge_chunks(all_chunks, new_chunks)

        sufficiency = evaluate_evidence(
            decomposed.original,
            all_chunks,
            entity_hints=entity_hints,
            sufficiency_threshold=sufficiency_threshold,
        )

        logger.debug(
            "Hop %d: %d new chunks, %d total, sufficiency=%.2f, missing=%s",
            hop, len(new_chunks), len(all_chunks),
            sufficiency.overall_score, sufficiency.missing_entities,
        )

        if sufficiency.is_sufficient:
            return IterativeResult(
                chunks=all_chunks, hops_used=hop,
                sufficiency=sufficiency, per_hop_chunks=per_hop,
            )

        if len(all_chunks) == prev_count:
            return IterativeResult(
                chunks=all_chunks, hops_used=hop,
                sufficiency=sufficiency, per_hop_chunks=per_hop,
            )

        gap_queries = _build_gap_queries(
            sufficiency.missing_entities, decomposed.original,
        )
        if not gap_queries:
            break

        current_dq = DecomposedQuery(
            original=decomposed.original,
            sub_queries=gap_queries,
            fusion_strategy="rrf",
            intent=decomposed.intent,
        )

    final_sufficiency = evaluate_evidence(
        decomposed.original, all_chunks,
        entity_hints=entity_hints,
        sufficiency_threshold=sufficiency_threshold,
    )
    return IterativeResult(
        chunks=all_chunks, hops_used=min(hop, max_hops),
        sufficiency=final_sufficiency, per_hop_chunks=per_hop,
    )
