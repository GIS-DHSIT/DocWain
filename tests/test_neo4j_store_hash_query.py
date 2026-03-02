from typing import Any, Dict, Iterable, List, Optional

from src.kg.neo4j_store import Neo4jStore


class DummyResult:
    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self._records = records or []

    def __iter__(self):
        return iter(self._records)


class DummySession:
    def __init__(self):
        self.last_query = None
        self.last_params = None
        self.run_calls = 0

    def run(self, query: str, **params):
        self.last_query = query
        self.last_params = params
        self.run_calls += 1
        return DummyResult([
            {"chunk_id": "c1", "chunk_hash": "h1"},
            {"chunk_id": "c2", "chunk_hash": None},
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyDriver:
    def __init__(self, session: DummySession):
        self._session = session

    def session(self, database: Optional[str] = None):
        return self._session

    def close(self):
        return None


def test_fetch_existing_hashes_uses_optional_match():
    session = DummySession()
    store = Neo4jStore(driver=DummyDriver(session))
    result = store.fetch_existing_hashes(["c1", "c2"])

    assert session.run_calls == 1
    assert "OPTIONAL MATCH" in session.last_query
    assert "UNWIND" in session.last_query
    assert result["c1"] == "h1"
    assert result["c2"] is None
