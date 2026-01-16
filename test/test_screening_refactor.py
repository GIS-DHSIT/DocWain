import subprocess
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.docwain_api import app
from src.screening import api as screening_api
from src.screening import storage_adapter


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
    assert storage_adapter.get_document_metadata("docX") == {}


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
