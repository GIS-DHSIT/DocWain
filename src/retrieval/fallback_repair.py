from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging_utils import get_logger

from .evidence_constraints import EvidenceConstraints, EvidenceRequirements
from .hybrid_ranker import HybridRanker
from .retrieval_quality import RetrievalQualityResult

logger = get_logger(__name__)


@dataclass
class FallbackResult:
    chunks: List[Any]
    used_fallback: bool
    reason: str
    attempts: List[Dict[str, Any]]
    quality: RetrievalQualityResult


class FallbackRepair:
    """Bounded retrieval repair with optional single-pass rewrite."""

    def __init__(
        self,
        *,
        max_attempts: int = 1,
        rewrite_enabled: bool = True,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.rewrite_enabled = rewrite_enabled
        self.time_fn = time_fn or time.time
        self.constraints = EvidenceConstraints()
        self.ranker = HybridRanker()

    def repair(
        self,
        *,
        query: str,
        chunks: List[Any],
        requirements: EvidenceRequirements,
        quality: RetrievalQualityResult,
        retrieve_fn: Optional[Callable[[str, int], List[Any]]] = None,
        top_k: int = 50,
    ) -> FallbackResult:
        logger.debug("repair: chunks=%d, quality_score=%.3f, is_low=%s", len(chunks), quality.score, quality.is_low)
        attempts: List[Dict[str, Any]] = []
        if not chunks:
            return FallbackResult(chunks, False, "no_chunks", attempts, quality)

        if not quality.is_low:
            logger.debug("repair: quality ok, skipping fallback")
            return FallbackResult(chunks, False, "quality_ok", attempts, quality)

        # Attempt 1: relax evidence gating within existing candidates
        relaxed = self.ranker.rank(query, chunks, requirements, relax_evidence=True)
        relaxed_quality = quality
        relaxed_quality = RetrievalQualityResult(
            score=quality.score,
            breakdown=quality.breakdown,
            is_low=quality.is_low,
            is_high=quality.is_high,
            elapsed_ms=quality.elapsed_ms,
        )
        attempts.append({"label": "relax_evidence", "hits": len(relaxed)})

        if not self.constraints.evaluate(relaxed, requirements, min_ratio=0.5).satisfied:
            if retrieve_fn and self.rewrite_enabled and self.max_attempts > 0:
                rewritten = self._rewrite_query(query, requirements)
                retrieved = retrieve_fn(rewritten, max(top_k, int(top_k * 1.2)))
                attempts.append({"label": "rewrite_retrieve", "query": rewritten, "hits": len(retrieved)})
                if retrieved:
                    ranked = self.ranker.rank(rewritten, retrieved, requirements)
                    logger.debug("repair: rewrite_retrieve returned %d chunks", len(ranked))
                    return FallbackResult(ranked, True, "rewrite_retrieve", attempts, quality)
        logger.debug("repair: relax_evidence returned %d chunks", len(relaxed))
        return FallbackResult(relaxed, True, "relax_evidence", attempts, relaxed_quality)

    @staticmethod
    def _rewrite_query(query: str, requirements: EvidenceRequirements) -> str:
        tokens = [query]
        if requirements.requires_number:
            tokens.append("total number amount")
        if requirements.requires_date_range:
            tokens.append("date range period")
        if requirements.required_keywords:
            tokens.append(" ".join(requirements.required_keywords[:4]))
        if requirements.required_entities:
            tokens.append(" ".join(requirements.required_entities[:3]))
        return " ".join(dict.fromkeys(" ".join(tokens).split()))
