import pickle

import pytest

from src.api import embedding_service
from src.api.blob_store import BlobInfo
from src.api.statuses import STATUS_EMBEDDING_COMPLETED, STATUS_TRAINING_COMPLETED


class FakeBlobStore:
    def __init__(self, payload):
        self.prefix = ""
        self.payload = payload
        self.delete_calls = 0

    def try_acquire_lease(self, blob_name, lease_duration=60):
        return "lease-id"

    def download_blob(self, blob_name, lease=None):
        return self.payload

    def delete_blob(self, blob_name, lease=None):
        self.delete_calls += 1
        return True

    def release_lease(self, blob_name, lease_id):
        return True


def _noop(*_args, **_kwargs):
    return None


def _setup_common_monkeypatch(monkeypatch, document_fields_sink=None):
    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda *_args, **_kwargs: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda *_args, **_kwargs: "prof-1")
    monkeypatch.setattr(embedding_service, "get_document_record", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(embedding_service, "update_stage", _noop)
    if document_fields_sink is None:
        monkeypatch.setattr(embedding_service, "update_document_fields", _noop)
    else:
        def _capture(_document_id, fields):
            document_fields_sink.append(fields)
            return None

        monkeypatch.setattr(embedding_service, "update_document_fields", _capture)
    monkeypatch.setattr(embedding_service, "_count_qdrant_points", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        embedding_service,
        "_verify_post_upsert_count",
        lambda **kwargs: (int(kwargs.get("expected_chunks", 0)), True),
    )
    monkeypatch.setattr(
        embedding_service,
        "_prepare_extracted_docs",
        lambda **_kwargs: ({"document": {"texts": ["chunk-1", "chunk-2"]}}, 2, [], None),
    )


def test_blob_deleted_after_successful_embed(monkeypatch):
    payload = pickle.dumps({"texts": ["chunk-1", "chunk-2"]}, protocol=pickle.HIGHEST_PROTOCOL)
    store = FakeBlobStore(payload)
    blob = BlobInfo(name="doc-1.pkl", metadata={"document_id": "doc-1"})

    _setup_common_monkeypatch(monkeypatch)
    monkeypatch.setattr(
        embedding_service,
        "train_on_document",
        lambda *_args, **_kwargs: {"chunks": 2, "points_saved": 2},
    )

    result = embedding_service._process_blob(store=store, blob=blob, subscription_id=None, profile_id=None, doc_type=None)
    assert result["status"] == "COMPLETED"
    assert store.delete_calls == 1


def test_blob_not_deleted_on_qdrant_failure(monkeypatch):
    payload = pickle.dumps({"texts": ["chunk-1", "chunk-2"]}, protocol=pickle.HIGHEST_PROTOCOL)
    store = FakeBlobStore(payload)
    blob = BlobInfo(name="doc-2.pkl", metadata={"document_id": "doc-2"})

    _setup_common_monkeypatch(monkeypatch)
    monkeypatch.setattr(
        embedding_service,
        "train_on_document",
        lambda *_args, **_kwargs: {"chunks": 2, "points_saved": 1},
    )

    result = embedding_service._process_blob(store=store, blob=blob, subscription_id=None, profile_id=None, doc_type=None)
    assert result["status"] == "FAILED"
    assert store.delete_calls == 0


def test_blob_not_deleted_when_post_upsert_not_verified(monkeypatch):
    payload = pickle.dumps({"texts": ["chunk-1", "chunk-2"]}, protocol=pickle.HIGHEST_PROTOCOL)
    store = FakeBlobStore(payload)
    blob = BlobInfo(name="doc-4.pkl", metadata={"document_id": "doc-4"})

    _setup_common_monkeypatch(monkeypatch)
    monkeypatch.setattr(
        embedding_service,
        "_verify_post_upsert_count",
        lambda **_kwargs: (1, False),
    )
    monkeypatch.setattr(
        embedding_service,
        "train_on_document",
        lambda *_args, **_kwargs: {"chunks": 2, "points_saved": 2},
    )

    result = embedding_service._process_blob(store=store, blob=blob, subscription_id=None, profile_id=None, doc_type=None)
    assert result["status"] == "COMPLETED"
    assert store.delete_calls == 0


def test_training_status_updated_on_success(monkeypatch):
    payload = pickle.dumps({"texts": ["chunk-1", "chunk-2"]}, protocol=pickle.HIGHEST_PROTOCOL)
    store = FakeBlobStore(payload)
    blob = BlobInfo(name="doc-3.pkl", metadata={"document_id": "doc-3"})
    status_updates = []

    _setup_common_monkeypatch(monkeypatch, document_fields_sink=status_updates)
    monkeypatch.setattr(
        embedding_service,
        "train_on_document",
        lambda *_args, **_kwargs: {"chunks": 2, "points_saved": 2},
    )

    result = embedding_service._process_blob(store=store, blob=blob, subscription_id=None, profile_id=None, doc_type=None)
    assert result["status"] == "COMPLETED"
    assert status_updates, "Expected at least one status update"
    latest = status_updates[-1]
    assert latest["status"] == STATUS_TRAINING_COMPLETED
    assert latest["embedding_status"] == STATUS_EMBEDDING_COMPLETED
    assert "trained_at" in latest
