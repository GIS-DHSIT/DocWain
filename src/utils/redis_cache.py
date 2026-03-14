import json
from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Optional

logger = get_logger(__name__)

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "errors": self.errors}

class RedisJsonCache:
    def __init__(self, redis_client: Any, default_ttl: int = 600):
        self.redis = redis_client
        self.default_ttl = int(default_ttl)
        self.metrics: Dict[str, CacheMetrics] = {}

    def _metric(self, feature: str) -> CacheMetrics:
        return self.metrics.setdefault(feature, CacheMetrics())

    def get_json(self, key: str, *, feature: str) -> Optional[Any]:
        metric = self._metric(feature)
        if not self.redis:
            metric.misses += 1
            return None
        try:
            payload = self.redis.get(key)
            if not payload:
                metric.misses += 1
                return None
            metric.hits += 1
            return json.loads(payload)
        except Exception as exc:  # noqa: BLE001
            metric.errors += 1
            logger.debug("Redis get failed for %s: %s", key, exc)
            return None

    def set_json(self, key: str, value: Any, *, feature: str, ttl: Optional[int] = None) -> None:
        if not self.redis:
            return
        try:
            payload = json.dumps(value)
            ttl_val = int(ttl) if ttl is not None else self.default_ttl
            self.redis.setex(key, max(1, ttl_val), payload)
        except Exception as exc:  # noqa: BLE001
            self._metric(feature).errors += 1
            logger.debug("Redis set failed for %s: %s", key, exc)

def hash_query(value: str) -> str:
    normalized = " ".join((value or "").strip().lower().split())
    return sha256(normalized.encode("utf-8")).hexdigest()

def stamp_cache_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    stamped = dict(payload)
    stamped.setdefault("cached_at", time.time())
    return stamped
