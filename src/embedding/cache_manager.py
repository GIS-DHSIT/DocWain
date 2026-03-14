"""
Embedding cache manager for efficient query vectorization and batch processing.

Implements Redis-backed caching with LRU eviction, batch embedding optimization,
and profile-aware cache hints for improved query-time performance.
"""

from __future__ import annotations

import hashlib
import json
from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

class EmbeddingCacheManager:
    """Manages embedding cache for queries and batch vectors with TTL and profile hints."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        cache_ttl_seconds: int = 86400,  # 24 hours
        batch_threshold: int = 10,
        profile_partition: bool = True,
    ):
        self.redis_client = redis_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self.batch_threshold = batch_threshold
        self.profile_partition = profile_partition
        self._local_cache: Dict[str, Tuple[List[float], float]] = {}
        self._local_cache_max_size = 1000

    def _cache_key(
        self,
        text: str,
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Generate cache key with optional profile partitioning for better locality."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        model_part = model_name or "default"
        if self.profile_partition and profile_id and subscription_id:
            return f"embed:{subscription_id}:{profile_id}:{model_part}:{text_hash}"
        elif subscription_id:
            return f"embed:{subscription_id}:{model_part}:{text_hash}"
        return f"embed:{model_part}:{text_hash}"

    def get_cached_embedding(
        self,
        text: str,
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[List[float]]:
        """Retrieve cached embedding with local cache fallback."""
        cache_key = self._cache_key(
            text,
            subscription_id=subscription_id,
            profile_id=profile_id,
            model_name=model_name,
        )

        # Check local cache first
        if cache_key in self._local_cache:
            vector, timestamp = self._local_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl_seconds:
                logger.debug("Local cache hit for embedding: %s", cache_key[:50])
                return vector
            del self._local_cache[cache_key]

        # Check Redis cache if available
        if not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                vector = json.loads(cached)
                # Refresh TTL
                self.redis_client.expire(cache_key, self.cache_ttl_seconds)
                logger.debug("Redis cache hit for embedding: %s", cache_key[:50])
                # Populate local cache
                self._populate_local_cache(cache_key, vector)
                return vector
        except Exception as exc:
            logger.warning("Cache retrieval failed: %s", exc)

        return None

    def cache_embedding(
        self,
        text: str,
        vector: List[float],
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Store embedding in both local and Redis cache."""
        cache_key = self._cache_key(
            text,
            subscription_id=subscription_id,
            profile_id=profile_id,
            model_name=model_name,
        )

        # Store in local cache
        self._populate_local_cache(cache_key, vector)

        # Store in Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl_seconds,
                    json.dumps([float(v) for v in vector]),
                )
                logger.debug("Cached embedding: %s", cache_key[:50])
            except Exception as exc:
                logger.warning("Cache storage failed: %s", exc)

    def _populate_local_cache(self, cache_key: str, vector: List[float]) -> None:
        """Store in local cache with size limit."""
        if len(self._local_cache) >= self._local_cache_max_size:
            # Simple FIFO eviction
            oldest_key = min(
                self._local_cache.items(),
                key=lambda x: x[1][1],  # Sort by timestamp
            )[0]
            del self._local_cache[oldest_key]

        self._local_cache[cache_key] = (vector, time.time())

    def get_batch_embeddings(
        self,
        texts: List[str],
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Retrieve multiple embeddings with cache hits, returning uncached indices."""
        embeddings = [None] * len(texts)
        uncached_indices = []

        for idx, text in enumerate(texts):
            cached = self.get_cached_embedding(
                text,
                subscription_id=subscription_id,
                profile_id=profile_id,
                model_name=model_name,
            )
            if cached is not None:
                embeddings[idx] = cached
            else:
                uncached_indices.append(idx)

        hit_rate = 1.0 - (len(uncached_indices) / max(1, len(texts)))
        logger.info(
            "Batch embedding cache hit rate: %.1f%% (%d/%d)",
            hit_rate * 100,
            len(texts) - len(uncached_indices),
            len(texts),
        )

        return embeddings, uncached_indices

    def cache_batch_embeddings(
        self,
        texts: List[str],
        vectors: List[List[float]],
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Cache multiple embeddings efficiently using pipeline."""
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors length mismatch")

        if not self.redis_client:
            for text, vector in zip(texts, vectors):
                self.cache_embedding(
                    text,
                    vector,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    model_name=model_name,
                )
            return

        try:
            pipeline = self.redis_client.pipeline()
            for text, vector in zip(texts, vectors):
                cache_key = self._cache_key(
                    text,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    model_name=model_name,
                )
                pipeline.setex(
                    cache_key,
                    self.cache_ttl_seconds,
                    json.dumps([float(v) for v in vector]),
                )
                self._populate_local_cache(cache_key, vector)

            pipeline.execute()
            logger.info("Cached %d embeddings via pipeline", len(texts))
        except Exception as exc:
            logger.warning("Batch cache storage failed: %s", exc)

    def clear_profile_cache(
        self,
        subscription_id: str,
        profile_id: str,
        model_name: Optional[str] = None,
    ) -> int:
        """Clear all cached embeddings for a profile."""
        if not self.redis_client:
            cleared = sum(
                1
                for k in list(self._local_cache.keys())
                if f"embed:{subscription_id}:{profile_id}" in k
            )
            self._local_cache = {
                k: v
                for k, v in self._local_cache.items()
                if f"embed:{subscription_id}:{profile_id}" not in k
            }
            return cleared

        try:
            model_part = model_name or "default"
            pattern = f"embed:{subscription_id}:{profile_id}:{model_part}:*"
            cursor = 0
            cleared = 0
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=1000,
                )
                if keys:
                    cleared += self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Cleared %d cached embeddings for profile %s", cleared, profile_id)
            return cleared
        except Exception as exc:
            logger.warning("Cache clear failed: %s", exc)
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "local_cache_size": len(self._local_cache),
            "local_cache_max_size": self._local_cache_max_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "redis_available": self.redis_client is not None,
            "profile_partitioning": self.profile_partition,
        }

__all__ = ["EmbeddingCacheManager"]

