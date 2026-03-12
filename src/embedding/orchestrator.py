"""
Enhanced embedding and vectorization orchestrator.

Coordinates all embedding improvements including caching, reranking,
adaptive thresholds, and quality evaluation for efficient and accurate responses.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from src.embedding.cache_manager import EmbeddingCacheManager
from src.embedding.reranker import RetrievalReranker, RerankingConfig
from src.embedding.threshold_tuner import AdaptiveThresholdTuner, ThresholdConfig
from src.embedding.quality_evaluator import SemanticChunkQualityEvaluator

logger = get_logger(__name__)

class EnhancedEmbeddingOrchestrator:
    """
    Orchestrates embedding pipeline with caching, reranking, and quality optimization.

    Integrates:
    - EmbeddingCacheManager for efficient query vectorization
    - RetrievalReranker for dual-model ensemble scoring
    - AdaptiveThresholdTuner for dynamic threshold adjustment
    - SemanticChunkQualityEvaluator for data quality
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        cross_encoder_model: Optional[Any] = None,
        reranking_config: Optional[RerankingConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        self.cache_manager = EmbeddingCacheManager(redis_client=redis_client)
        self.reranker = RetrievalReranker(
            config=reranking_config,
            cross_encoder_model=cross_encoder_model,
        )
        self.threshold_tuner = AdaptiveThresholdTuner(config=threshold_config)
        self.quality_evaluator = SemanticChunkQualityEvaluator()
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "reranks_applied": 0,
            "thresholds_adjusted": 0,
            "quality_filters_applied": 0,
        }

    def embed_query_efficient(
        self,
        query: str,
        embedder: Any,
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """Embed query with caching and fallback."""
        # Try cache first
        cached = self.cache_manager.get_cached_embedding(
            query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            model_name=model_name,
        )

        if cached:
            self.metrics["cache_hits"] += 1
            logger.debug("Query embedding from cache")
            return cached

        # Compute embedding
        self.metrics["cache_misses"] += 1
        vector = embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # Cache for future use
        self.cache_manager.cache_embedding(
            query,
            [float(v) for v in vector],
            subscription_id=subscription_id,
            profile_id=profile_id,
            model_name=model_name,
        )

        return [float(v) for v in vector]

    def embed_batch_efficient(
        self,
        texts: List[str],
        embedder: Any,
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Embed multiple texts efficiently with caching."""
        # Get cached embeddings and identify uncached
        embeddings, uncached_indices = self.cache_manager.get_batch_embeddings(
            texts,
            subscription_id=subscription_id,
            profile_id=profile_id,
            model_name=model_name,
        )

        # Batch compute uncached
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            computed = embedder.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                batch_size=batch_size,
            )

            # Convert to float lists
            computed_vectors = [
                [float(v) for v in vec] for vec in computed
            ]

            # Cache computed embeddings
            self.cache_manager.cache_batch_embeddings(
                uncached_texts,
                computed_vectors,
                subscription_id=subscription_id,
                profile_id=profile_id,
                model_name=model_name,
            )

            # Fill in results
            for idx, computed_idx in enumerate(uncached_indices):
                embeddings[computed_idx] = computed_vectors[idx]

        return embeddings

    def compute_adaptive_threshold(
        self,
        query: str,
        *,
        profile_context: Optional[Dict[str, Any]] = None,
        retrieval_stats: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
        document_count: Optional[int] = None,
    ) -> float:
        """Compute adaptive threshold with multiple signals."""
        adjustment = self.threshold_tuner.compute_adaptive_threshold(
            query,
            profile_context=profile_context,
            retrieval_stats=retrieval_stats,
            query_intent=query_intent,
            document_count=document_count,
        )

        if adjustment.adjustment_factor != 1.0:
            self.metrics["thresholds_adjusted"] += 1
            logger.info(
                "Adaptive threshold: %.3f -> %.3f (%s)",
                adjustment.original_threshold,
                adjustment.adjusted_threshold,
                adjustment.adjustment_reason,
            )

        return adjustment.adjusted_threshold

    def adjust_threshold_for_coverage(
        self,
        current_threshold: float,
        result_count: int,
    ) -> float:
        """Adjust threshold if retrieval results are insufficient."""
        adjusted = self.threshold_tuner.adjust_threshold_for_coverage(
            current_threshold,
            result_count,
        )

        if adjusted < current_threshold:
            self.metrics["thresholds_adjusted"] += 1

        return adjusted

    def rerank_retrieved_chunks(
        self,
        query: str,
        chunks: List[Any],
        *,
        profile_context: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
    ) -> List[Any]:
        """Rerank chunks using ensemble scoring and optional cross-encoder."""
        if not chunks or len(chunks) < 2:
            return chunks

        reranked = self.reranker.rerank_chunks(
            query,
            chunks,
            profile_context=profile_context,
            query_intent=query_intent,
        )

        self.metrics["reranks_applied"] += 1
        logger.info("Reranked %d chunks", len(reranked))

        return reranked

    def evaluate_chunk_quality(
        self,
        chunk_id: str,
        chunk_text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate chunk quality metrics."""
        metrics = self.quality_evaluator.evaluate_chunk_quality(
            chunk_text,
            chunk_id,
            context=context,
        )

        return {
            "chunk_id": metrics.chunk_id,
            "quality_grade": metrics.quality_grade,
            "overall_quality": metrics.overall_quality,
            "length_score": metrics.length_score,
            "completeness_score": metrics.completeness_score,
            "coherence_score": metrics.coherence_score,
            "density_score": metrics.information_density_score,
            "readability_score": metrics.readability_score,
            "issues": metrics.issues,
        }

    def filter_low_quality_chunks(
        self,
        chunks: List[tuple[str, str]],
        min_quality_grade: str = "C",
    ) -> tuple[List[str], List[str]]:
        """Filter chunks by quality grade."""
        kept_ids, filtered_ids, _ = self.quality_evaluator.filter_low_quality_chunks(
            chunks,
            min_quality_grade=min_quality_grade,
        )

        self.metrics["quality_filters_applied"] += len(filtered_ids)
        if filtered_ids:
            logger.info(
                "Filtered %d low-quality chunks (grade < %s)",
                len(filtered_ids),
                min_quality_grade,
            )

        return kept_ids, filtered_ids

    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        cache_stats = self.cache_manager.get_cache_stats()

        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = (
            self.metrics["cache_hits"] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "cache": {
                **cache_stats,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "hits": self.metrics["cache_hits"],
                "misses": self.metrics["cache_misses"],
            },
            "reranking": {
                "total_reranks": self.metrics["reranks_applied"],
            },
            "thresholding": {
                "total_adjustments": self.metrics["thresholds_adjusted"],
            },
            "quality": {
                "total_filters": self.metrics["quality_filters_applied"],
            },
        }

    def clear_profile_cache(
        self,
        subscription_id: str,
        profile_id: str,
    ) -> int:
        """Clear cache for a profile."""
        cleared = self.cache_manager.clear_profile_cache(
            subscription_id,
            profile_id,
        )
        return cleared

__all__ = ["EnhancedEmbeddingOrchestrator"]

