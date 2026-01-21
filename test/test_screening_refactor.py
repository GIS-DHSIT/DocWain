import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.screening import api as screening_api
from src.screening import storage_adapter
from bson import ObjectId


def test_models_route_unique():
    matches = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) == "/api/models" and "GET" in getattr(route, "methods", set())
    ]
    assert len(matches) == 1


def test_screening_health_and_document(monkeypatch):
    fake_engine = SimpleNamespace(
        config=SimpleNamespace(enabled_tools=["t1"]),
        evaluate=lambda **kwargs: {"doc_id": kwargs.get("doc_id"), "status": "ok"},
    )
    app.dependency_overrides[screening_api.get_screening_engine] = lambda: fake_engine
    client = TestClient(app)
    try:
        resp = client.get("/api/screening/health")
        assert resp.status_code == 200
        resp = client.post("/api/screening/document", json={"text": "hello world", "doc_id": "doc-1"})
        assert resp.status_code == 200
        assert resp.json().get("doc_id") == "doc-1"
    finally:
        app.dependency_overrides.pop(screening_api.get_screening_engine, None)


def test_screening_run_by_profile(monkeypatch):
    fake_docs = [{"_id": "docA", "doc_type": "TEST"}, {"_id": "docB", "doc_type": "TEST"}]

    class FakeCursor(list):
        def limit(self, n):
            return FakeCursor(self[:n])

    class FakeCollection:
        def __init__(self, docs):
            self._docs = docs

        def find(self, query, projection=None):
            return FakeCursor(self._docs)

    monkeypatch.setattr(screening_api, "_get_documents_collection", lambda: FakeCollection(fake_docs))
    monkeypatch.setattr(
        screening_api,
        "_screen_document_task",
        lambda task: {"doc_id": task["doc_id"], "results": {"all": {"doc_id": task["doc_id"]}}, "errors": []},
    )
    monkeypatch.setattr(
        screening_api, "_run_parallel_screening", lambda tasks: [screening_api._screen_document_task(task) for task in tasks]
    )

    client = TestClient(app)
    resp = client.post("/api/screening/run", json={"profile_ids": ["profile123"], "categories": ["all"]})
    assert resp.status_code == 200
    payload = resp.json()
    profile = payload["profiles"][0]
    assert profile["profile_id"] == "profile123"
    assert profile["summary"]["processed"] == 2
    assert profile["documents"][0]["results"]["all"]["doc_id"] == "docA"


def test_storage_adapter_handles_collection_truthiness(monkeypatch):
    class FakeCollection:
        def __bool__(self):
            raise NotImplementedError("PyMongo Collection truthiness not supported")

        def find_one(self, query):
            return None

    monkeypatch.setattr(storage_adapter, "_get_document_collection", lambda: FakeCollection())
    with pytest.raises(ValueError):
        storage_adapter.get_document_metadata("docX")


def test_storage_adapter_uses_object_id_query(monkeypatch):
    doc_id = "507f1f77bcf86cd799439011"
    record = {"_id": ObjectId(doc_id), "text": "hello", "subscriptionId": "sub-1"}
    queries = []

    class FakeCollection:
        def find_one(self, query):
            queries.append(query)
            if query == {"_id": ObjectId(doc_id)}:
                return record
            return None

    monkeypatch.setattr(storage_adapter, "_get_document_collection", lambda: FakeCollection())
    assert storage_adapter.get_document_metadata(doc_id) == record
    assert queries[0] == {"_id": ObjectId(doc_id)}


def test_subscription_id_extracted(monkeypatch):
    record = {"_id": "doc-1", "subscription_id": "sub-123"}

    class FakeCollection:
        def find_one(self, query):
            if "$or" in query:
                return record
            return None

    monkeypatch.setattr(storage_adapter, "_get_document_collection", lambda: FakeCollection())
    assert storage_adapter.get_document_subscription_id("doc-1") == "sub-123"


def test_non_legality_rejects_region_field(monkeypatch):
    def fake_run_parallel(tasks):
        return [
            {"doc_id": task["doc_id"], "status": "succeeded", "result": {"doc_id": task["doc_id"]}, "errors": []}
            for task in tasks
        ]

    def fake_persist(run_id, endpoint, options, doc_entries):
        return {"persisted": True, "persisted_count": len(doc_entries)}

    monkeypatch.setattr(screening_api, "_run_parallel_doc_tasks", fake_run_parallel)
    monkeypatch.setattr(screening_api, "_persist_screening_reports", fake_persist)

    client = TestClient(app)
    resp = client.post("/api/screening/integrity", json={"doc_ids": ["doc-123"], "region": "EU"})
    assert resp.status_code == 422


def test_legality_accepts_region_field(monkeypatch):
    def fake_run_parallel(tasks):
        return [
            {"doc_id": task["doc_id"], "status": "succeeded", "result": {"doc_id": task["doc_id"]}, "errors": []}
            for task in tasks
        ]

    def fake_persist(run_id, endpoint, options, doc_entries):
        return {"persisted": True, "persisted_count": len(doc_entries)}

    monkeypatch.setattr(screening_api, "_run_parallel_doc_tasks", fake_run_parallel)
    monkeypatch.setattr(screening_api, "_persist_screening_reports", fake_persist)

    client = TestClient(app)
    resp = client.post(
        "/api/screening/legality",
        json={"doc_ids": ["doc-123"], "region": "EU", "jurisdiction": "DE"},
    )
    assert resp.status_code == 200


def test_forbidden_term_absent():
    root = Path(__file__).resolve().parents[1]
    forbidden = "guard" + "rails"
    result = subprocess.run(
        ["rg", forbidden, str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, f"Forbidden substring '{forbidden}' found:\n{result.stdout}"


def test_screening_rejects_placeholder_doc_ids():
    client = TestClient(app)
    resp = client.post("/api/screening/integrity", json={"doc_ids": ["string"]})
    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error"]["code"] == "invalid_doc_ids"


def test_screening_processes_multiple_doc_ids(monkeypatch):
    def fake_run_parallel(tasks):
        return [
            {"doc_id": task["doc_id"], "status": "succeeded", "result": {"doc_id": task["doc_id"]}, "errors": []}
            for task in tasks
        ]

    def fake_persist(run_id, endpoint, options, doc_entries):
        return {"persisted": True, "persisted_count": len(doc_entries)}

    monkeypatch.setattr(screening_api, "_run_parallel_doc_tasks", fake_run_parallel)
    monkeypatch.setattr(screening_api, "_persist_screening_reports", fake_persist)

    client = TestClient(app)
    resp = client.post("/api/screening/integrity", json={"doc_ids": ["doc-123", "doc-456"]})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["summary"]["processed"] == 2
    assert {doc["doc_id"] for doc in payload["documents"]} == {"doc-123", "doc-456"}


def test_persist_screening_reports_upserts(monkeypatch):
    class FakeBulkResult:
        def __init__(self, upserted_count=0, modified_count=0):
            self.upserted_count = upserted_count
            self.modified_count = modified_count

    class FakeCollection:
        def __init__(self):
            self.ops = None

        def bulk_write(self, ops, ordered=False):
            self.ops = ops
            return FakeBulkResult(upserted_count=len(ops), modified_count=0)

    fake_collection = FakeCollection()
    monkeypatch.setattr(screening_api, "_get_screening_collection", lambda: fake_collection)

    doc_entries = [
        {"doc_id": "doc-1", "status": "succeeded", "result": {"ok": True}, "errors": []},
        {"doc_id": "doc-2", "status": "failed", "result": None, "errors": ["boom"]},
    ]
    result = screening_api._persist_screening_reports(
        "run-1",
        "integrity",
        {"doc_type": None, "internet_enabled": None, "region": None, "jurisdiction": None},
        doc_entries,
    )
    assert result["persisted"] is True
    assert result["persisted_count"] == 2
    assert fake_collection.ops is not None
    assert len(fake_collection.ops) == 2
    first_op = fake_collection.ops[0]
    assert getattr(first_op, "_filter", {}) == {"doc_id": "doc-1", "endpoint": "integrity", "run_id": "run-1"}
    assert "$set" in getattr(first_op, "_doc", {})


def test_qdrant_collection_missing_is_friendly():
    from src.api import dw_newron

    class FakeCollections:
        def __init__(self, names):
            self.collections = [SimpleNamespace(name=name) for name in names]

    class FakeQdrantClient:
        def __init__(self, names):
            self._names = set(names)

        def get_collection(self, collection_name):
            if collection_name not in self._names:
                raise Exception("Not found: Collection")
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3))))

        def get_collections(self):
            return FakeCollections(list(self._names))

    retriever = dw_newron.QdrantRetriever(FakeQdrantClient(["alpha"]), SimpleNamespace())
    with pytest.raises(dw_newron.QdrantCollectionNotFoundError) as exc:
        retriever.ensure_collection_exists("missing")
    assert "qdrant_collection_not_found" in str(exc.value)
    assert "alpha" in str(exc.value)


def test_collection_resolution_requires_input():
    from src.api import dw_newron

    with pytest.raises(ValueError):
        dw_newron._resolve_collection_name()
