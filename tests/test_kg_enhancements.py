from src.kg.retrieval import GraphAugmenter
from src.agent.graph_worker import GraphWorker
from src.kg.retrieval import GraphHints


class FakeStore:
    def __init__(self, match_rows=None, expand_rows=None, two_hop_rows=None, skill_rows=None):
        self.match_rows = match_rows or []
        self.expand_rows = expand_rows or []
        self.two_hop_rows = two_hop_rows or []
        self.skill_rows = skill_rows or []

    def run_query(self, query, params=None):
        if "collect(DISTINCT d.doc_id)" in query:
            return self.match_rows
        if "HAS_SKILL" in query:
            return self.skill_rows
        if "r2.chunk_id" in query:
            return self.expand_rows
        if "MATCH (d1:Document" in query:
            return self.two_hop_rows
        return []


class ExplodingStore:
    def run_query(self, query, params=None):
        raise RuntimeError("Neo4j down")


def test_entity_disambiguation_prefers_doc_intersection():
    store = FakeStore(
        match_rows=[
            {
                "entity_id": "PERSON::david smith",
                "name": "David Smith",
                "type": "PERSON",
                "doc_ids": ["doc-acme", "doc-other"],
            },
            {
                "entity_id": "ORGANIZATION::acme corp",
                "name": "Acme Corp",
                "type": "ORGANIZATION",
                "doc_ids": ["doc-acme"],
            },
        ],
        expand_rows=[
            {
                "entity_id": "ORGANIZATION::acme corp",
                "name": "Acme Corp",
                "type": "ORGANIZATION",
                "chunk_id": "chunk-1",
                "doc_id": "doc-acme",
                "doc_name": "Acme Resume",
                "relation": "MENTIONS",
            }
        ],
    )
    augmenter = GraphAugmenter(neo4j_store=store, enabled=True)
    hints = augmenter.augment("What is David Smith's role at Acme Corp?", "sub", "prof")
    assert hints.doc_ids == ["doc-acme"]
    assert hints.graph_snippets


def test_graph_worker_multi_hop_skill_candidates():
    store = FakeStore(
        match_rows=[
            {
                "entity_id": "SKILL::supply chain",
                "name": "supply chain",
                "type": "SKILL",
                "doc_ids": ["doc-1", "doc-2"],
            }
        ],
        skill_rows=[
            {"person": "Alex Doe", "doc_id": "doc-1", "chunk_id": "c-1", "confidence": 0.9},
            {"person": "Jamie Roe", "doc_id": "doc-2", "chunk_id": "c-2", "confidence": 0.85},
        ],
    )
    augmenter = GraphAugmenter(neo4j_store=store, enabled=True)
    worker = GraphWorker(augmenter, store)
    result = worker.run("top candidates with supply chain experience", "sub", "prof")
    assert {p["name"] for p in result.candidate_persons} == {"Alex Doe", "Jamie Roe"}
    assert sorted(result.candidate_doc_ids) == ["doc-1", "doc-2"]


def test_graph_snippets_require_chunk_provenance():
    store = FakeStore(
        match_rows=[
            {
                "entity_id": "SKILL::python",
                "name": "python",
                "type": "SKILL",
                "doc_ids": ["doc-1"],
            }
        ],
        expand_rows=[
            {
                "entity_id": "SKILL::python",
                "name": "python",
                "type": "SKILL",
                "chunk_id": None,
                "doc_id": "doc-1",
                "doc_name": "Resume",
                "relation": "MENTIONS",
            }
        ],
    )
    augmenter = GraphAugmenter(neo4j_store=store, enabled=True)
    hints = augmenter.augment("python experience", "sub", "prof")
    assert hints.graph_snippets == []


def test_graph_fallback_when_store_unavailable():
    augmenter = GraphAugmenter(neo4j_store=ExplodingStore(), enabled=True)
    hints = augmenter.augment("any query", "sub", "prof")
    assert isinstance(hints, GraphHints)
    assert not hints.entities_in_query
