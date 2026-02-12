"""
Unit tests for enhanced embedding system components.
"""

import pytest
from unittest.mock import Mock

from src.embedding.cache_manager import EmbeddingCacheManager
from src.embedding.reranker import RetrievalReranker, RerankingConfig
from src.embedding.threshold_tuner import AdaptiveThresholdTuner
from src.embedding.quality_evaluator import SemanticChunkQualityEvaluator


class TestEmbeddingCacheManager:
    """Test cache manager functionality."""

    def test_cache_hit_and_miss(self):
        """Test caching and retrieval."""
        cache = EmbeddingCacheManager(redis_client=None)
        text = "test query"
        vector = [0.1, 0.2, 0.3]

        # Should miss on first call
        assert cache.get_cached_embedding(text) is None

        # Cache it
        cache.cache_embedding(text, vector)

        # Should hit on second call
        cached = cache.get_cached_embedding(text)
        assert cached == vector

    def test_batch_embeddings(self):
        """Test batch caching."""
        cache = EmbeddingCacheManager(redis_client=None)
        texts = ["q1", "q2", "q3"]
        vectors = [[0.1], [0.2], [0.3]]

        # Cache batch
        cache.cache_batch_embeddings(texts, vectors)

        # Check all are cached
        embeddings, uncached = cache.get_batch_embeddings(texts)
        assert uncached == []


class TestAdaptiveThresholdTuner:
    """Test threshold tuning."""

    def test_query_specificity(self):
        """Test query specificity analysis."""
        tuner = AdaptiveThresholdTuner()

        vague = "stuff"
        specific = "List your Python and Java technical skills for backend development"

        vague_spec = tuner._compute_query_specificity(vague)
        specific_spec = tuner._compute_query_specificity(specific)

        assert specific_spec > vague_spec

    def test_adaptive_threshold(self):
        """Test threshold adjustment."""
        tuner = AdaptiveThresholdTuner()

        adjustment = tuner.compute_adaptive_threshold(
            "What are your recent projects?",
            document_count=100,
            query_intent="experience",
        )

        assert 0.1 <= adjustment.adjusted_threshold <= 0.5


class TestSemanticQualityEvaluator:
    """Test quality evaluation."""

    def test_quality_scoring(self):
        """Test chunk quality scoring."""
        evaluator = SemanticChunkQualityEvaluator()

        good_text = ("John Doe has extensive experience in software engineering, "
                     "with 10 years of Python development. He holds a Bachelor's "
                     "degree in Computer Science from MIT and has published multiple "
                     "research papers on distributed systems.")
        bad_text = "a"

        good_metrics = evaluator.evaluate_chunk_quality(good_text, "1")
        bad_metrics = evaluator.evaluate_chunk_quality(bad_text, "2")

        assert good_metrics.overall_quality > bad_metrics.overall_quality

    def test_quality_filtering(self):
        """Test filtering by quality grade."""
        evaluator = SemanticChunkQualityEvaluator()

        good_text = ("John Doe has extensive experience in software engineering, "
                     "with 10 years of Python development. He holds a Bachelor's "
                     "degree in Computer Science from MIT and has published multiple "
                     "research papers on distributed systems.")

        chunks = [
            (good_text, "1"),
            ("x", "2"),
        ]

        kept, filtered, metrics = evaluator.filter_low_quality_chunks(chunks, min_quality_grade="C")

        assert len(kept) > 0
        assert len(filtered) >= 0


class TestRetrievalReranker:
    """Test reranking functionality."""

    def test_reranker_initialization(self):
        """Test reranker creation."""
        config = RerankingConfig(use_cross_encoder=False)
        reranker = RetrievalReranker(config=config)
        assert reranker is not None

    def test_sparse_scoring(self):
        """Test BM25-like scoring."""
        reranker = RetrievalReranker()

        chunks = []
        for text in ["Python skills", "Java programming", "Cooking recipes"]:
            chunk = Mock()
            chunk.chunk_id = text
            chunk.text = text
            chunks.append(chunk)

        scores = reranker._compute_sparse_scores("Python skills", chunks)
        assert scores[chunks[0].chunk_id] > scores[chunks[2].chunk_id]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

