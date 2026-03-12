from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.api.dw_newron import get_redis_client

logger = get_logger(__name__)

_MEMORY_LOCKS: dict[str, float] = {}
_MEMORY_LOCK = threading.Lock()

@dataclass(frozen=True)
class IdempotencyLock:
    key: str
    acquired: bool
    backend: str

def _lock_key(stage: str, subscription_id: Optional[str], document_id: str) -> str:
    scope = subscription_id or "unknown"
    return f"docwain:lock:{stage}:{scope}:{document_id}"

def acquire_lock(
    *,
    stage: str,
    document_id: str,
    subscription_id: Optional[str],
    ttl_seconds: int = 900,
) -> IdempotencyLock:
    key = _lock_key(stage, subscription_id, document_id)
    redis_client = None
    try:
        redis_client = get_redis_client()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis unavailable for idempotency: %s", exc)
        redis_client = None

    if redis_client:
        try:
            acquired = bool(redis_client.set(key, "1", nx=True, ex=max(1, ttl_seconds)))
            return IdempotencyLock(key=key, acquired=acquired, backend="redis")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis lock failed for %s: %s", key, exc)

    now = time.time()
    expiry = now + ttl_seconds
    with _MEMORY_LOCK:
        existing = _MEMORY_LOCKS.get(key)
        if existing and existing > now:
            return IdempotencyLock(key=key, acquired=False, backend="memory")
        _MEMORY_LOCKS[key] = expiry
        return IdempotencyLock(key=key, acquired=True, backend="memory")

def release_lock(lock: IdempotencyLock) -> None:
    if not lock or not lock.key:
        return
    if lock.backend == "redis":
        try:
            redis_client = get_redis_client()
            if redis_client:
                redis_client.delete(lock.key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to release redis lock %s: %s", lock.key, exc)
        return
    with _MEMORY_LOCK:
        _MEMORY_LOCKS.pop(lock.key, None)

__all__ = ["acquire_lock", "release_lock", "IdempotencyLock"]
