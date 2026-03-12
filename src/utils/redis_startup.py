from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Iterable, List

from src.api.config import Config

logger = get_logger(__name__)

def clear_unsafe_keys(redis_client: Any, patterns: Iterable[str]) -> Dict[str, Any]:
    if not redis_client:
        return {"cleared": 0, "patterns": []}
    patterns = [p for p in patterns if p]
    if not patterns:
        return {"cleared": 0, "patterns": []}

    cleared = 0
    errors: List[str] = []
    scan_count = int(getattr(Config.Redis, "CLEAR_SCAN_COUNT", 200))
    max_keys = int(getattr(Config.Redis, "CLEAR_MAX_KEYS", 5000))

    for pattern in patterns:
        cursor = 0
        while True:
            try:
                cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=scan_count)
                if keys:
                    redis_client.delete(*keys)
                    cleared += len(keys)
                if cleared >= max_keys:
                    logger.warning("Unsafe Redis key clear reached max_keys=%s", max_keys)
                    return {"cleared": cleared, "patterns": patterns, "errors": errors}
                if cursor == 0:
                    break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{pattern}: {exc}")
                break
    return {"cleared": cleared, "patterns": patterns, "errors": errors}

def parse_unsafe_patterns(raw: str) -> List[str]:
    if not raw:
        return []
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    return tokens

__all__ = ["clear_unsafe_keys", "parse_unsafe_patterns"]
