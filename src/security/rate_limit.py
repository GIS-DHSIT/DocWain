import time
from collections import defaultdict

_BUCKET = defaultdict(list)

def rate_limit(key: str, limit: int = 10, window: int = 60):
    now = time.time()
    _BUCKET[key] = [t for t in _BUCKET[key] if now - t < window]

    if len(_BUCKET[key]) >= limit:
        raise Exception("Rate limit exceeded")

    _BUCKET[key].append(now)
