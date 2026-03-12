"""
Adaptive threshold tuning for dynamic retrieval quality optimization.

Implements query-aware and profile-aware threshold adjustment to optimize
recall-precision trade-off based on query characteristics and document profiles.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for adaptive thresholding."""

    base_threshold: float = 0.20  # Default similarity threshold
    min_threshold: float = 0.10  # Minimum allowed threshold
    max_threshold: float = 0.50  # Maximum allowed threshold
    ambiguous_query_threshold: float = 0.15  # Lower for vague queries
    specific_query_threshold: float = 0.25  # Higher for specific queries
    low_coverage_threshold: float = 0.12  # Lower threshold if no results
    min_results_target: int = 5  # Target minimum results
    coverage_boost_factor: float = 0.85  # Multiply threshold when below min_results

@dataclass(frozen=True)
class ThresholdAdjustment:
    """Result of threshold adjustment."""

    original_threshold: float
    adjusted_threshold: float
    adjustment_reason: str
    adjustment_factor: float

class AdaptiveThresholdTuner:
    """Dynamically adjusts retrieval thresholds based on query and context."""

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()

    def compute_adaptive_threshold(
        self,
        query: str,
        *,
        profile_context: Optional[Dict[str, Any]] = None,
        retrieval_stats: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
        document_count: Optional[int] = None,
    ) -> ThresholdAdjustment:
        """Compute adaptive threshold based on multiple signals."""
        base = self.config.base_threshold
        adjustment_factor = 1.0
        reasons = []

        # Signal 1: Query specificity
        specificity = self._compute_query_specificity(query)
        if specificity < 0.3:  # Vague query
            adjustment_factor *= self._map_specificity_to_factor(specificity)
            reasons.append(
                f"vague_query(specificity={specificity:.2f})",
            )
        elif specificity > 0.7:  # Very specific query
            adjustment_factor *= self._map_specificity_to_factor(specificity)
            reasons.append(
                f"specific_query(specificity={specificity:.2f})",
            )

        # Signal 2: Query intent
        intent_factor = self._get_intent_threshold_factor(query_intent)
        if intent_factor != 1.0:
            adjustment_factor *= intent_factor
            reasons.append(f"intent_based({query_intent})")

        # Signal 3: Profile coverage
        if profile_context:
            coverage_factor = self._compute_coverage_factor(profile_context)
            if coverage_factor != 1.0:
                adjustment_factor *= coverage_factor
                reasons.append(f"profile_coverage({coverage_factor:.2f})")

            # Signal 4: Document diversity
            diversity_factor = self._compute_diversity_factor(profile_context)
            if diversity_factor != 1.0:
                adjustment_factor *= diversity_factor
                reasons.append(f"diversity({diversity_factor:.2f})")

        # Signal 5: Retrieval performance
        if retrieval_stats:
            perf_factor = self._compute_performance_factor(retrieval_stats)
            if perf_factor != 1.0:
                adjustment_factor *= perf_factor
                reasons.append(f"retrieval_perf({perf_factor:.2f})")

        # Signal 6: Document collection size
        if document_count:
            size_factor = self._compute_size_factor(document_count)
            if size_factor != 1.0:
                adjustment_factor *= size_factor
                reasons.append(f"collection_size({size_factor:.2f})")

        # Apply adjustment with bounds
        adjusted = base * adjustment_factor
        adjusted = max(self.config.min_threshold, min(self.config.max_threshold, adjusted))

        return ThresholdAdjustment(
            original_threshold=base,
            adjusted_threshold=adjusted,
            adjustment_reason="; ".join(reasons) or "no_adjustment",
            adjustment_factor=adjustment_factor,
        )

    def _compute_query_specificity(self, query: str) -> float:
        """Estimate query specificity (0=vague, 1=very specific)."""
        if not query or len(query.strip()) < 5:
            return 0.1

        # Heuristics for specificity
        query_lower = query.lower()
        specificity_signals = 0.0
        total_signals = 0.0

        # Signal: Query length (normalized)
        length_score = min(1.0, len(query.split()) / 10.0)
        specificity_signals += length_score * 0.3
        total_signals += 0.3

        # Signal: Named entities (capitalized words)
        entities = len(re.findall(r"\b[A-Z][a-z]+", query))
        entity_score = min(1.0, entities / 3.0)
        specificity_signals += entity_score * 0.2
        total_signals += 0.2

        # Signal: Technical/domain terms (hyphenated, underscored, etc.)
        tech_terms = len(re.findall(r"[a-z]+[-_][a-z]+|[A-Z]{2,}", query))
        tech_score = min(1.0, tech_terms / 2.0)
        specificity_signals += tech_score * 0.25
        total_signals += 0.25

        # Signal: Question words reduce specificity
        question_words = [
            "what",
            "how",
            "why",
            "when",
            "where",
            "can",
            "could",
        ]
        has_question = any(qw in query_lower for qw in question_words)
        specificity_signals += (0.0 if has_question else 1.0) * 0.25
        total_signals += 0.25

        return specificity_signals / total_signals if total_signals > 0 else 0.5

    def _map_specificity_to_factor(self, specificity: float) -> float:
        """Map query specificity to threshold adjustment factor."""
        if specificity < 0.3:  # Vague
            # Lower threshold to get more results
            return 0.75  # 25% reduction
        elif specificity > 0.7:  # Specific
            # Increase threshold to get more relevant results
            return 1.25  # 25% increase
        else:
            return 1.0

    def _get_intent_threshold_factor(self, query_intent: Optional[str]) -> float:
        """Get threshold adjustment factor based on query intent."""
        intent_factors = {
            "summary": 0.9,  # Broader retrieval for summaries
            "detail": 1.1,  # Stricter for detailed questions
            "comparison": 0.95,  # Balanced for comparisons
            "verification": 1.2,  # Strict for verification
            "list": 0.85,  # Broader for list queries
            "navigation": 1.0,  # Normal for navigation
        }
        return intent_factors.get(query_intent, 1.0)

    def _compute_coverage_factor(self, profile_context: Dict[str, Any]) -> float:
        """Compute factor based on profile document coverage."""
        coverage_ratio = profile_context.get("coverage_ratio", 1.0)

        if coverage_ratio > 0.8:
            return 1.0  # Good coverage, use normal threshold
        elif coverage_ratio > 0.5:
            return 0.9  # Moderate coverage, lower threshold slightly
        elif coverage_ratio > 0.2:
            return 0.8  # Low coverage, lower threshold more
        else:
            return 0.7  # Very low coverage, significantly lower threshold

        return 1.0

    def _compute_diversity_factor(self, profile_context: Dict[str, Any]) -> float:
        """Compute factor based on document domain diversity."""
        doc_domains = profile_context.get("document_domains", set())
        unique_domains = len(doc_domains)

        if unique_domains >= 5:
            return 1.0  # High diversity, normal threshold
        elif unique_domains >= 3:
            return 0.95  # Moderate diversity
        elif unique_domains >= 1:
            return 0.9  # Low diversity, slightly lower threshold
        else:
            return 0.85  # Single domain, lower threshold

        return 1.0

    def _compute_performance_factor(
        self,
        retrieval_stats: Dict[str, Any],
    ) -> float:
        """Compute factor based on retrieval performance."""
        result_count = retrieval_stats.get("result_count", 0)
        min_target = self.config.min_results_target

        if result_count >= min_target * 2:
            return 1.0  # Plenty of results, normal threshold
        elif result_count >= min_target:
            return 0.95  # Adequate results
        elif result_count > 0:
            return 0.9  # Low results, reduce threshold slightly
        else:
            return 0.85  # No results, reduce threshold significantly

        return 1.0

    def _compute_size_factor(self, document_count: int) -> float:
        """Compute factor based on collection size."""
        if document_count > 1000:
            return 1.0  # Large collection, normal threshold
        elif document_count > 100:
            return 0.95  # Medium collection
        elif document_count > 10:
            return 0.9  # Small collection, slightly lower threshold
        else:
            return 0.85  # Very small collection, lower threshold

        return 1.0

    def adjust_threshold_for_coverage(
        self,
        current_threshold: float,
        result_count: int,
    ) -> float:
        """Adjust threshold down if we have insufficient results."""
        if result_count >= self.config.min_results_target:
            return current_threshold  # Sufficient results

        # Progressively lower threshold
        adjusted = current_threshold * self.config.coverage_boost_factor
        adjusted = max(self.config.min_threshold, adjusted)

        if adjusted < current_threshold:
            logger.info(
                "Lowering threshold due to low coverage: %.3f -> %.3f (results: %d)",
                current_threshold,
                adjusted,
                result_count,
            )

        return adjusted

    def compute_threshold_for_expansion(
        self,
        base_threshold: float,
        fallback_reason: str,
    ) -> float:
        """Compute threshold for expansion queries."""
        # Lower threshold for fallback/expansion queries
        fallback_factors = {
            "low_coverage": 0.8,  # Domain filter removed
            "no_results": 0.7,  # More aggressive expansion
            "low_confidence": 0.75,  # Confidence too low
            "diversity_boost": 0.85,  # Expand for diversity
        }

        factor = fallback_factors.get(fallback_reason, 0.8)
        return base_threshold * factor

    def get_threshold_statistics(
        self,
        thresholds: List[float],
    ) -> Dict[str, float]:
        """Compute statistics for a set of thresholds."""
        if not thresholds:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        import statistics

        sorted_thresholds = sorted(thresholds)
        return {
            "min": min(thresholds),
            "max": max(thresholds),
            "mean": statistics.mean(thresholds),
            "median": statistics.median(sorted_thresholds),
            "std": statistics.stdev(thresholds) if len(thresholds) > 1 else 0.0,
            "count": len(thresholds),
        }

__all__ = ["AdaptiveThresholdTuner", "ThresholdConfig", "ThresholdAdjustment"]

