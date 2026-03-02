from src.utils.redis_cache import RedisJsonCache


class DummyRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


def test_redis_json_cache_hit_miss():
    cache = RedisJsonCache(DummyRedis(), default_ttl=30)
    key = "k1"

    assert cache.get_json(key, feature="kgprobe") is None
    cache.set_json(key, {"value": 1}, feature="kgprobe")
    result = cache.get_json(key, feature="kgprobe")
    assert result["value"] == 1
    metrics = cache.metrics["kgprobe"].as_dict()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
