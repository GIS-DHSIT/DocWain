import logging
from types import SimpleNamespace

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance

from src.api import document_status
from src.api.vector_store import QdrantVectorStore


def _get_path(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(doc, path, value):
    parts = path.split(".")
    cur = doc
    for part in parts[:-1]:
        if part not in cur:
            cur[part] = {}
        elif not isinstance(cur[part], dict):
            raise RuntimeError("PathNotViable")
        cur = cur[part]
    cur[parts[-1]] = value


def _unset_path(doc, path):
    parts = path.split(".")
    cur = doc
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


class FakeCollection:
    def __init__(self, doc=None):
        self.doc = doc or {}

    def find_one_and_update(self, _filter, update, upsert=False, return_document=None):  # noqa: ANN001
        if not self.doc and upsert:
            self.doc = {}
            for key, value in (update.get("$setOnInsert") or {}).items():
                _set_path(self.doc, key, value)
        self._apply_update(update)
        return self.doc

    def update_many(self, filt, update):  # noqa: ANN001
        modified = 0
        for path, value in (filt or {}).items():
            if value is None and _get_path(self.doc, path) is None:
                self._apply_update(update)
                modified = 1
                break
        return SimpleNamespace(modified_count=modified)

    def _apply_update(self, update):
        for key, value in (update.get("$set") or {}).items():
            _set_path(self.doc, key, value)
        for key in (update.get("$unset") or {}).keys():
            _unset_path(self.doc, key)


def test_mongo_error_update_handles_null_error(monkeypatch):
    doc = {"_id": "doc-1", "embedding": {"error": None}}
    fake = FakeCollection(doc)
    monkeypatch.setattr(document_status, "get_documents_collection", lambda: fake)

    error_payload = {
        "stage": "embedding",
        "message": "boom",
        "code": "TEST",
        "details": {"reason": "unit"},
        "at": 123.0,
        "run_id": "run-1",
    }

    document_status.update_stage(
        "doc-1",
        "embedding",
        {"status": "FAILED", "completed_at": 123.0, "error": error_payload},
    )

    stored = fake.doc.get("embedding", {}).get("error")
    assert isinstance(stored, dict)
    assert stored.get("message") == "boom"
    assert stored.get("stage") == "embedding"


def test_error_normalization_unsets_nulls():
    doc = {"_id": "doc-1", "error": None, "embedding": {"error": None}, "cleanup": {"error": None}}
    fake = FakeCollection(doc)
    result = document_status.normalize_error_fields(collection=fake)

    assert "error" not in doc
    assert "error" not in doc.get("embedding", {})
    assert "error" not in doc.get("cleanup", {})
    assert result.get("updated", 0) >= 1


class FakeQdrantConflict:
    def __init__(self, size=4, distance=Distance.COSINE):
        self.payload_schema = {}
        self.size = size
        self.distance = distance

    def get_collections(self):  # noqa: ANN001
        return SimpleNamespace(collections=[])

    def get_collection(self, collection_name):  # noqa: ANN001
        _ = collection_name
        return SimpleNamespace(
            payload_schema=self.payload_schema,
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=self.size, distance=self.distance))),
        )

    def create_collection(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        raise UnexpectedResponse(409, "Conflict", b"exists", httpx.Headers({}))

    def create_payload_index(self, collection_name, field_name, field_schema):  # noqa: ANN001
        _ = (collection_name, field_schema)
        self.payload_schema[field_name] = {"data_type": "keyword"}


def test_qdrant_create_collection_conflict_is_idempotent():
    store = QdrantVectorStore(client=FakeQdrantConflict())
    store.ensure_collection("sub-1", 4)


def test_safe_update_stage_logs_original_error(monkeypatch, caplog):
    from src.api import embedding_service

    def boom(*args, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("mongo write failed")

    monkeypatch.setattr(embedding_service, "update_stage", boom)
    cause = ValueError("qdrant schema mismatch")

    with caplog.at_level(logging.ERROR):
        embedding_service._safe_update_stage(
            "doc-1",
            "embedding",
            {"status": "FAILED", "error": {"message": "x"}},
            cause=cause,
        )

    assert "mongo write failed" in caplog.text
    assert "qdrant schema mismatch" in caplog.text
