from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils.payload_utils import get_source_name

from .evidence_constraints import EvidenceConstraints, EvidenceRequirements

logger = get_logger(__name__)


@dataclass
class RetrievalQualityResult:
    score: float
    breakdown: Dict[str, float]
    is_low: bool
    is_high: bool
    elapsed_ms: float


class RetrievalQualityScorer:
    """Fast guardrail scoring for retrieval quality."""

    def __init__(
        self,
        *,
        threshold_low: float = 0.45,
        threshold_high: float = 0.75,
        budget_ms: int = 40,
        time_fn: Optional[callable] = None,
    ) -> None:
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.budget_ms = budget_ms
        self.time_fn = time_fn or time.time
        self.constraints = EvidenceConstraints()

    def evaluate(
        self,
        query: str,
        chunks: List[Any],
        requirements: EvidenceRequirements,
    ) -> RetrievalQualityResult:
        logger.debug("evaluate: chunks=%d", len(chunks))
        start = self.time_fn()
        if not chunks:
            return RetrievalQualityResult(0.0, {"overlap": 0.0, "evidence": 0.0, "gap": 0.0, "diversity": 0.0}, True, False, 0.0)

        overlap = self._overlap_score(query, chunks)
        evidence = self.constraints.evaluate(chunks, requirements, min_ratio=0.5).coverage_ratio
        gap = self._score_gap(chunks)
        diversity = self._section_diversity(chunks)

        score = (0.35 * overlap) + (0.35 * evidence) + (0.2 * gap) + (0.1 * diversity)
        score = max(0.0, min(1.0, score))
        elapsed = (self.time_fn() - start) * 1000
        breakdown = {
            "overlap": round(overlap, 4),
            "evidence": round(evidence, 4),
            "gap": round(gap, 4),
            "diversity": round(diversity, 4),
        }
        result = RetrievalQualityResult(
            score=round(score, 4),
            breakdown=breakdown,
            is_low=score < self.threshold_low,
            is_high=score >= self.threshold_high,
            elapsed_ms=round(elapsed, 2),
        )
        logger.debug("evaluate: score=%.4f, is_low=%s, is_high=%s, elapsed_ms=%.2f", result.score, result.is_low, result.is_high, result.elapsed_ms)
        return result

    @staticmethod
    def _overlap_score(query: str, chunks: List[Any]) -> float:
        tokens = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
        if not tokens:
            return 0.0
        scores = []
        for chunk in chunks[:5]:
            text = getattr(chunk, "text", None) or ""
            chunk_tokens = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
            if not chunk_tokens:
                scores.append(0.0)
                continue
            overlap = len(tokens & chunk_tokens) / max(len(tokens), 1)
            scores.append(overlap)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def _score_gap(chunks: List[Any]) -> float:
        scores = [float(getattr(chunk, "score", 0.0)) for chunk in chunks[:5]]
        if not scores:
            return 0.0
        best = scores[0]
        avg = sum(scores) / max(len(scores), 1)
        if best <= 0:
            return 0.0
        gap = max(0.0, (best - avg) / max(best, 1e-6))
        return min(gap, 1.0)

    @staticmethod
    def _section_diversity(chunks: List[Any]) -> float:
        keys = set()
        for chunk in chunks[:8]:
            meta = getattr(chunk, "metadata", {}) or {}
            doc_id = str(meta.get("document_id") or meta.get("doc_id") or get_source_name(meta) or "")
            section = str(meta.get("section_path") or meta.get("section_title") or meta.get("section") or "")
            keys.add((doc_id, section))
        total = min(len(chunks), 8)
        if total <= 0:
            return 0.0
        return min(1.0, len(keys) / total)
