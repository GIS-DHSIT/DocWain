from typing import Any, Dict, List

from src.api.enhanced_retrieval import GraphGuidedRetriever, KGProbeResult
from src.utils.redis_cache import RedisJsonCache


class DummyRedis:
    def __init__(self):
        self.store: Dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value


class FakeNeo4jStore:
    def __init__(self, hits: List[Dict[str, Any]]):
        self._hits = hits

    def probe_entities(self, *, entity_ids: List[str], limit: int = 20, timeout_ms: int | None = None):
        _ = entity_ids
        _ = limit
        _ = timeout_ms
        return list(self._hits)


def test_graph_guided_probe_caches_results():
    redis_cache = RedisJsonCache(DummyRedis(), default_ttl=60)
    hits = [
        {"document_id": "doc-1", "section_path": "A", "hits": 3},
        {"document_id": "doc-2", "section_path": "B", "hits": 1},
    ]
    kg = GraphGuidedRetriever(
        neo4j_store=FakeNeo4jStore(hits),
        cache=redis_cache,
        probe_limit=10,
        probe_timeout_ms=50,
        cache_ttl_seconds=60,
    )
    result = kg.probe(tenant="t1", collection="c1", query="Find doc-1")
    assert result.document_ids[0] == "doc-1"

    cached = kg.get_cached_probe(tenant="t1", collection="c1", query="Find doc-1")
    assert cached is not None
    assert cached.source == "cache"
    assert "doc-1" in cached.document_ids


def test_graph_guided_probe_handles_missing_neo4j():
    redis_cache = RedisJsonCache(DummyRedis(), default_ttl=60)
    kg = GraphGuidedRetriever(
        neo4j_store=None,
        cache=redis_cache,
    )
    result = kg.probe(tenant="t2", collection="c2", query="Nothing here")
    assert result.document_ids == []
    assert result.section_paths == []


def test_graph_guided_boosts_scores():
    probe = KGProbeResult(document_ids=["doc-1"], section_paths=["section-a"], hits={"doc-1": 2})
    chunks = [
        {"score": 0.5, "metadata": {"document_id": "doc-1", "section_path": "section-a"}},
        {"score": 0.6, "metadata": {"document_id": "doc-2", "section_path": "other"}},
    ]
    boosted = GraphGuidedRetriever.apply_boosts(chunks, probe)
    assert boosted[0]["metadata"].get("kg_boost") is True
