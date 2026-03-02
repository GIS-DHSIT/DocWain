from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import embedding_service
from src.api.blob_store import BlobConfigurationError
from src.api.statuses import STATUS_SCREENING_COMPLETED, STATUS_TRAINING_COMPLETED, STATUS_TRAINING_FAILED
from src.main import app


def test_embed_batch_partial_failure_returns_200(monkeypatch):
    calls = []

    def fake_update_document_fields(document_id, fields):
        calls.append(document_id)

    def fake_process_local_document(**kwargs):
        doc_id = kwargs["document_id"]
        if doc_id == "doc2":
            raise RuntimeError("cannot copy out of meta tensor")
        embedding_service._set_document_status(doc_id, STATUS_TRAINING_COMPLETED)
        return {
            "blob_name": f"{doc_id}.pkl",
            "document_id": doc_id,
            "status": "COMPLETED",
            "chunks_count": 2,
            "points_upserted": 2,
            "error": None,
            "failed_reason": None,
        }

    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: False)
    monkeypatch.setattr(embedding_service, "_fetch_document_ids_by_filters", lambda **_kwargs: [])
    monkeypatch.setattr(embedding_service, "_process_local_document", fake_process_local_document)
    monkeypatch.setattr(embedding_service, "update_document_fields", fake_update_document_fields)

    client = TestClient(app)
    response = client.post("/api/documents/embed", json={"document_ids": ["doc1", "doc2", "doc3"]})

    assert response.status_code == 200
    payload = response.json()
    docs = payload.get("documents", [])
    assert [doc.get("status") for doc in docs] == ["COMPLETED", "FAILED", "COMPLETED"]
    assert payload.get("overall_status") == "PARTIAL"
    assert set(calls) == {"doc1", "doc2", "doc3"}


def test_embed_all_fail_returns_200(monkeypatch):
    def fake_process_local_document(**kwargs):
        doc_id = kwargs["document_id"]
        embedding_service._set_document_status(doc_id, STATUS_TRAINING_FAILED)
        return {
            "blob_name": f"{doc_id}.pkl",
            "document_id": doc_id,
            "status": "FAILED",
            "chunks_count": 0,
            "points_upserted": 0,
            "error": "training_failed",
            "failed_reason": "training_failed",
        }

    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: False)
    monkeypatch.setattr(embedding_service, "_fetch_document_ids_by_filters", lambda **_kwargs: [])
    monkeypatch.setattr(embedding_service, "_process_local_document", fake_process_local_document)
    monkeypatch.setattr(embedding_service, "update_document_fields", lambda *_args, **_kwargs: None)

    client = TestClient(app)
    response = client.post("/api/documents/embed", json={"document_ids": ["doc-a", "doc-b"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("overall_status") == "FAILED"
    assert payload.get("documents_failed") == 2


def test_embed_global_failure_returns_500(monkeypatch):
    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: True)
    monkeypatch.setattr(embedding_service, "_build_blob_store", lambda: (_ for _ in ()).throw(BlobConfigurationError("boom")))

    client = TestClient(app)
    response = client.post("/api/documents/embed", json={"document_ids": ["doc-a"]})

    assert response.status_code == 500


def test_cleanup_failure_does_not_fail(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "get_document_record",
        lambda _doc_id: {"status": STATUS_SCREENING_COMPLETED, "subscription_id": "sub", "profile_id": "prof"},
    )
    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda *_args, **_kwargs: "sub")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda *_args, **_kwargs: "prof")
    monkeypatch.setattr(embedding_service, "acquire_lock", lambda **_kwargs: SimpleNamespace(acquired=True))
    monkeypatch.setattr(embedding_service, "release_lock", lambda _lock: None)
    monkeypatch.setattr(embedding_service, "load_extracted_pickle", lambda _doc_id: {"file.txt": "text"})
    monkeypatch.setattr(
        embedding_service,
        "_prepare_extracted_docs",
        lambda **_kwargs: ({"file.txt": "text"}, 1, [], None),
    )
    monkeypatch.setattr(
        embedding_service,
        "train_on_document",
        lambda *_args, **_kwargs: {"chunks": 1, "points_saved": 1, "dropped_chunks": 0, "coverage_ratio": 1.0},
    )
    monkeypatch.setattr(embedding_service, "_verify_post_upsert_count", lambda **_kwargs: (1, True))
    # delete_extracted_pickle is no longer imported in embedding_service (pickle is preserved)
    # so we skip monkeypatching it — the cleanup path no longer calls it
    monkeypatch.setattr(embedding_service, "update_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding_service, "update_document_fields", lambda *_args, **_kwargs: None)

    result = embedding_service._process_local_document(
        document_id="doc-cleanup",
        subscription_id="sub",
        profile_id="prof",
        doc_type=None,
        embed_request_id="req-1",
    )

    assert result["status"] == "COMPLETED"
