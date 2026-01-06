"""
Redis Cache Manager for Enterprise RAG System
Provides multi-level caching with graceful degradation
"""

import json
import hashlib
import logging
import pickle
import time
from typing import Any, Optional, List, Dict, Tuple
from functools import wraps
import redis
from api.config import Config

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Enterprise-grade Redis cache manager with:
    - Multi-level caching (answers, embeddings, chunks)
    - TTL management
    - Graceful degradation
    - Cache invalidation
    - Metrics tracking
    """

    def __init__(self):
        self.client = None
        self.enabled = False
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'sets': 0
        }
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Redis client with error handling"""
        try:
            self.client = redis.Redis(
                host=Config.Redis.HOST,
                port=Config.Redis.PORT,
                username=Config.Redis.USERNAME or None,
                password=Config.Redis.PASSWORD or None,
                db=Config.Redis.DB,
                ssl=Config.Redis.SSL,
                decode_responses=False,  # Handle binary data
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            self.client.ping()
            self.enabled = True
            logger.info("Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Caching disabled.")
            self.client = None
            self.enabled = False

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key with namespace"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)

        # Hash for consistent key length
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"docwain:{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None

        try:
            value = self.client.get(key)
            if value is not None:
                self.metrics['hits'] += 1
                logger.debug(f"Cache HIT: {key}")
                return pickle.loads(value)
            else:
                self.metrics['misses'] += 1
                logger.debug(f"Cache MISS: {key}")
                return None

        except Exception as e:
            self.metrics['errors'] += 1
            logger.warning(f"Cache GET error for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        if not self.enabled:
            return False

        try:
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
            self.metrics['sets'] += 1
            logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
            return True

        except Exception as e:
            self.metrics['errors'] += 1
            logger.warning(f"Cache SET error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled:
            return False

        try:
            self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True

        except Exception as e:
            logger.warning(f"Cache DELETE error for {key}: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.enabled:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching {pattern}")
                return deleted
            return 0

        except Exception as e:
            logger.warning(f"Cache DELETE_PATTERN error for {pattern}: {e}")
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        total_requests = self.metrics['hits'] + self.metrics['misses']
        hit_rate = (self.metrics['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            'enabled': self.enabled,
            'hits': self.metrics['hits'],
            'misses': self.metrics['misses'],
            'errors': self.metrics['errors'],
            'sets': self.metrics['sets'],
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests
        }

    def clear_all(self) -> bool:
        """Clear all docwain cache keys (use with caution!)"""
        if not self.enabled:
            return False

        try:
            deleted = self.delete_pattern("docwain:*")
            logger.warning(f"Cleared ALL cache: {deleted} keys deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# Global cache instance
_CACHE = None


def get_cache() -> RedisCacheManager:
    """Get or create global cache instance"""
    global _CACHE
    if _CACHE is None:
        _CACHE = RedisCacheManager()
    return _CACHE


class CacheStrategies:
    """Predefined caching strategies for different data types"""

    # TTL values (in seconds)
    ANSWER_TTL = 3600  # 1 hour for answers
    EMBEDDING_TTL = 86400  # 24 hours for embeddings
    CHUNKS_TTL = 7200  # 2 hours for retrieved chunks
    COLLECTION_INFO_TTL = 300  # 5 minutes for collection metadata

    @staticmethod
    def cache_answer(
            query: str,
            subscription_id: str,
            profile_id: str,
            model_name: str
    ) -> Tuple[str, int]:
        """Generate cache key for answers"""
        cache = get_cache()
        key = cache._generate_key(
            "answer",
            subscription_id,
            profile_id,
            model_name,
            query
        )
        return key, CacheStrategies.ANSWER_TTL

    @staticmethod
    def cache_embedding(text: str, model_name: str) -> Tuple[str, int]:
        """Generate cache key for embeddings"""
        cache = get_cache()
        # Use hash of text to avoid key size issues
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = cache._generate_key("embedding", model_name, text_hash)
        return key, CacheStrategies.EMBEDDING_TTL

    @staticmethod
    def cache_chunks(
            query: str,
            collection_name: str,
            profile_id: str,
            top_k: int
    ) -> Tuple[str, int]:
        """Generate cache key for retrieved chunks"""
        cache = get_cache()
        key = cache._generate_key(
            "chunks",
            collection_name,
            profile_id,
            str(top_k),
            query
        )
        return key, CacheStrategies.CHUNKS_TTL

    @staticmethod
    def cache_collection_info(collection_name: str) -> Tuple[str, int]:
        """Generate cache key for collection metadata"""
        cache = get_cache()
        key = cache._generate_key("collection", collection_name)
        return key, CacheStrategies.COLLECTION_INFO_TTL


def cache_result(strategy_func):
    """
    Decorator for caching function results

    Usage:
        @cache_result(CacheStrategies.cache_answer)
        def answer_question(query, subscription_id, profile_id, model_name):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()

            if not cache.enabled:
                return func(*args, **kwargs)

            # Generate cache key using strategy
            try:
                cache_key, ttl = strategy_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Cache key generation failed: {e}")
                return func(*args, **kwargs)

            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class CacheInvalidation:
    """Handles cache invalidation when data changes"""

    @staticmethod
    def invalidate_document(subscription_id: str, document_id: str):
        """Invalidate cache when a document is updated/deleted"""
        cache = get_cache()

        # Clear all answer caches for this subscription
        pattern = f"docwain:answer:{subscription_id}:*"
        deleted = cache.delete_pattern(pattern)

        # Clear chunk caches for this subscription
        pattern = f"docwain:chunks:{subscription_id}:*"
        deleted += cache.delete_pattern(pattern)

        logger.info(f"Invalidated {deleted} cache entries for document {document_id}")

    @staticmethod
    def invalidate_profile(subscription_id: str, profile_id: str):
        """Invalidate cache when a profile is updated"""
        cache = get_cache()

        # Clear answer caches for this profile
        pattern = f"docwain:answer:{subscription_id}:{profile_id}:*"
        deleted = cache.delete_pattern(pattern)

        logger.info(f"Invalidated {deleted} cache entries for profile {profile_id}")

    @staticmethod
    def invalidate_subscription(subscription_id: str):
        """Invalidate all cache for a subscription"""
        cache = get_cache()

        patterns = [
            f"docwain:answer:{subscription_id}:*",
            f"docwain:chunks:{subscription_id}:*",
            f"docwain:collection:{subscription_id}:*"
        ]

        total_deleted = 0
        for pattern in patterns:
            total_deleted += cache.delete_pattern(pattern)

        logger.info(f"Invalidated {total_deleted} cache entries for subscription {subscription_id}")


def cache_embeddings_batch(texts: List[str], model_name: str, encode_func):
    """
    Cache-aware batch embedding encoding

    Args:
        texts: List of texts to encode
        model_name: Name of embedding model
        encode_func: Function to encode texts (model.encode)

    Returns:
        List of embeddings
    """
    cache = get_cache()

    if not cache.enabled:
        return encode_func(texts)

    embeddings = []
    texts_to_encode = []
    text_indices = []

    # Check cache for each text
    for idx, text in enumerate(texts):
        cache_key, _ = CacheStrategies.cache_embedding(text, model_name)
        cached = cache.get(cache_key)

        if cached is not None:
            embeddings.append(cached)
        else:
            embeddings.append(None)
            texts_to_encode.append(text)
            text_indices.append(idx)

    # Encode uncached texts
    if texts_to_encode:
        logger.info(f"Encoding {len(texts_to_encode)}/{len(texts)} texts (rest from cache)")
        new_embeddings = encode_func(texts_to_encode)

        # Cache new embeddings and fill results
        for i, (text, embedding) in enumerate(zip(texts_to_encode, new_embeddings)):
            cache_key, ttl = CacheStrategies.cache_embedding(text, model_name)
            cache.set(cache_key, embedding, ttl)
            embeddings[text_indices[i]] = embedding

    return embeddings


# Health check function
def check_cache_health() -> Dict[str, Any]:
    """Check cache health and return status"""
    cache = get_cache()

    if not cache.enabled:
        return {
            'status': 'disabled',
            'message': 'Redis cache is not enabled'
        }

    try:
        # Test connectivity
        start = time.time()
        cache.client.ping()
        latency = (time.time() - start) * 1000  # ms

        # Get info
        info = cache.client.info('stats')

        return {
            'status': 'healthy',
            'latency_ms': round(latency, 2),
            'connected_clients': info.get('connected_clients', 0),
            'total_commands_processed': info.get('total_commands_processed', 0),
            'metrics': cache.get_metrics()
        }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }