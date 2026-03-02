from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ConfidenceResult:
    confidence: float
    breakdown: Dict[str, float]


class RetrievalConfidenceScorer:
    """Compute confidence from rerank scores, score gap, and coverage."""

    def score(self, reranked_chunks: List[Any], context_sources: List[Dict[str, Any]]) -> ConfidenceResult:
        if not reranked_chunks:
            return ConfidenceResult(0.0, {"best_score": 0.0, "score_gap": 0.0, "coverage": 0.0})

        scores = [float(getattr(c, "score", 0.0)) for c in reranked_chunks if c is not None]
        if not scores:
            return ConfidenceResult(0.0, {"best_score": 0.0, "score_gap": 0.0, "coverage": 0.0})

        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            norm_scores = [1.0 for _ in scores]
        else:
            span = max_score - min_score
            norm_scores = [(s - min_score) / span for s in scores]

        best_score = norm_scores[0]
        top_k = norm_scores[: min(len(norm_scores), 5)]
        avg_top = sum(top_k) / max(len(top_k), 1)
        score_gap = max(0.0, best_score - avg_top)

        distinct_sections = set()
        for src in context_sources or []:
            section = src.get("section") or ""
            doc = src.get("source_name") or ""
            distinct_sections.add((doc, section))
        coverage = min(1.0, len(distinct_sections) / max(1, len(context_sources) or 1))

        confidence = (0.5 * best_score) + (0.25 * score_gap) + (0.25 * coverage)
        confidence = max(0.0, min(1.0, confidence))

        return ConfidenceResult(
            confidence=round(confidence, 4),
            breakdown={
                "best_score": round(best_score, 4),
                "score_gap": round(score_gap, 4),
                "coverage": round(coverage, 4),
            },
        )

    @staticmethod
    def should_refuse(confidence: float, min_confidence: float) -> bool:
        return float(confidence) < float(min_confidence)
