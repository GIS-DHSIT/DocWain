from pathlib import Path

from fastapi.testclient import TestClient

from src.main import app
from src.api import extraction_service, screening_service, embedding_service
import src.screening.api as screening_api


class StatusStore:
    def __init__(self):
        self.docs = {}

    def update(self, doc_id, fields):
        doc = self.docs.get(doc_id, {"document_id": doc_id})
        doc.update(fields)
        self.docs[doc_id] = doc
        return doc

    def get(self, doc_id):
        return self.docs.get(doc_id, {})


def test_extract_endpoint_sets_status_and_pickle(tmp_path, monkeypatch):
    store = StatusStore()
    doc_id = "doc-123"
    doc_data = {
        "_id": doc_id,
        "status": "UNDER_REVIEW",
        "type": "LOCAL",
        "name": "file.pdf",
        "profile": "prof-1",
        "subscriptionId": "sub-1",
    }
    conn_data = {"locations": ["local/file.pdf"]}

    monkeypatch.setenv("DOCUMENT_CONTENT_DIR", str(tmp_path))
    monkeypatch.setattr(extraction_service, "extract_document_info", lambda: {doc_id: {"dataDict": doc_data, "connDict": conn_data}})
    monkeypatch.setattr(extraction_service, "resolve_subscription_id", lambda _doc_id, _candidate=None: "sub-1")
    monkeypatch.setattr(extraction_service, "get_subscription_pii_setting", lambda _sid: False)
    monkeypatch.setattr(extraction_service, "get_azure_docs", lambda _key, **_kwargs: b"hello")
    monkeypatch.setattr(extraction_service, "fileProcessor", lambda _bytes, _name: {"file.pdf": "hello world"})
    monkeypatch.setattr(extraction_service, "update_pii_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        extraction_service,
        "save_extracted_pickle",
        lambda doc_id, _obj: {"path": str(tmp_path / f"{doc_id}.pkl"), "sha256": "hash"},
    )
    monkeypatch.setattr(extraction_service, "update_extraction_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(extraction_service, "update_document_fields", lambda doc_id, fields: store.update(doc_id, fields))

    client = TestClient(app)
    response = client.get("/api/extract")
    assert response.status_code == 200

    payload = response.json()
    result = payload["message"]["results"]["successful"][0]
    assert result["document_id"] == doc_id
    assert result["status"] == "EXTRACTION_COMPLETED"

    pickle_path = Path(result["pickle_path"])
    pickle_path.write_bytes(b"pickle")
    assert pickle_path.exists()
    assert store.get(doc_id)["status"] == "EXTRACTION_COMPLETED"


def test_screening_security_filters_status_and_updates(monkeypatch):
    store = StatusStore()
    doc_ok = "doc-ok"
    doc_skip = "doc-skip"
    store.update(doc_ok, {"status": "EXTRACTION_COMPLETED"})
    store.update(doc_skip, {"status": "UNDER_REVIEW"})

    monkeypatch.setattr(screening_service, "get_document_record", lambda doc_id: store.get(doc_id))
    monkeypatch.setattr(screening_service, "update_document_fields", lambda doc_id, fields: store.update(doc_id, fields))
    monkeypatch.setattr(screening_service, "update_security_screening", lambda *_args, **_kwargs: None)

    captured = {}

    def fake_run_parallel_doc_tasks(tasks):
        captured["tasks"] = tasks
        return [
            {
                "doc_id": task["doc_id"],
                "status": "succeeded",
                "result": {"risk_level": "LOW"},
                "errors": [],
                "warnings": [],
                "subscription_id": "sub-1",
                "duration_seconds": 0.01,
            }
            for task in tasks
        ]

    monkeypatch.setattr(screening_api, "_run_parallel_doc_tasks", fake_run_parallel_doc_tasks)

    client = TestClient(app)
    response = client.post("/api/screening/security", json={"doc_ids": [doc_ok, doc_skip]})
    assert response.status_code == 200

    payload = response.json()
    doc_statuses = {doc["doc_id"]: doc["status"] for doc in payload["documents"]}
    assert doc_statuses[doc_ok] == "succeeded"
    assert doc_statuses[doc_skip] == "skipped"
    assert store.get(doc_ok)["status"] == "SCREENING_COMPLETED"
    assert store.get(doc_skip)["status"] == "UNDER_REVIEW"
    assert len(captured.get("tasks", [])) == 1


def test_documents_embed_requires_screening_completed(monkeypatch):
    store = StatusStore()
    doc_ready = "doc-ready"
    doc_blocked = "doc-blocked"
    store.update(doc_ready, {"status": "SCREENING_COMPLETED", "subscription_id": "sub-1", "profile_id": "prof-1"})
    store.update(doc_blocked, {"status": "EXTRACTION_COMPLETED"})

    monkeypatch.setattr(embedding_service, "get_document_record", lambda doc_id: store.get(doc_id))
    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda _doc_id, _candidate=None: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda _doc_id, _candidate=None: "prof-1")
    monkeypatch.setattr(
        embedding_service,
        "load_extracted_pickle",
        lambda _doc_id: {"file.pdf": "This is a complete sentence. " * 5},
    )
    monkeypatch.setattr(embedding_service, "delete_extracted_pickle", lambda _doc_id: True)
    monkeypatch.setattr(embedding_service, "update_document_fields", lambda doc_id, fields: store.update(doc_id, fields))
    monkeypatch.setattr(embedding_service, "update_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: False)
    monkeypatch.setattr(embedding_service, "_fetch_document_ids_by_filters", lambda **_kwargs: [])

    calls = []

    def fake_train_on_document(content, subscription_id, profile_id, doc_id, file_name):
        calls.append((doc_id, file_name))
        return {"chunks": 1, "points_saved": 1, "dropped_chunks": 0}

    monkeypatch.setattr(embedding_service, "train_on_document", fake_train_on_document)

    client = TestClient(app)

    ready_response = client.post("/api/documents/embed", json={"document_id": doc_ready})
    assert ready_response.status_code == 200
    ready_payload = ready_response.json()
    assert ready_payload["documents_processed"] == 1
    assert ready_payload["documents_succeeded"] == 1
    assert ready_payload["documents_failed"] == 0
    assert ready_payload["total_chunks"] == 1
    assert ready_payload["total_points_upserted"] == 1
    assert store.get(doc_ready)["status"] == "TRAINING_COMPLETED"

    blocked_response = client.post("/api/documents/embed", json={"document_id": doc_blocked})
    assert blocked_response.status_code == 200
    blocked_payload = blocked_response.json()
    assert blocked_payload["documents_processed"] == 0
    assert blocked_payload["documents_succeeded"] == 0
    assert blocked_payload["documents_failed"] == 0
    assert store.get(doc_blocked)["status"] == "EXTRACTION_COMPLETED"

    assert calls


def test_documents_embed_empty_extraction_sets_training_failed(monkeypatch):
    store = StatusStore()
    doc_ready = "doc-empty"
    store.update(doc_ready, {"status": "SCREENING_COMPLETED", "subscription_id": "sub-1", "profile_id": "prof-1"})

    monkeypatch.setattr(embedding_service, "get_document_record", lambda doc_id: store.get(doc_id))
    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda _doc_id, _candidate=None: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda _doc_id, _candidate=None: "prof-1")
    monkeypatch.setattr(embedding_service, "load_extracted_pickle", lambda _doc_id: {"file.pdf": ""})
    monkeypatch.setattr(embedding_service, "delete_extracted_pickle", lambda _doc_id: True)
    monkeypatch.setattr(embedding_service, "update_document_fields", lambda doc_id, fields: store.update(doc_id, fields))
    monkeypatch.setattr(embedding_service, "update_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: False)
    monkeypatch.setattr(embedding_service, "_fetch_document_ids_by_filters", lambda **_kwargs: [])
    monkeypatch.setattr(
        embedding_service,
        "_prepare_extracted_docs",
        lambda **_kwargs: (None, None, [], "empty_extraction"),
    )

    client = TestClient(app)
    response = client.post("/api/documents/embed", json={"document_id": doc_ready})
    assert response.status_code == 200
    payload = response.json()

    assert payload["documents_processed"] == 1
    assert payload["documents_succeeded"] == 0
    assert payload["documents_failed"] == 1
    assert payload["failure_reasons"].get("empty_extraction") == 1

    latest = store.get(doc_ready)
    assert latest["status"] == "TRAINING_FAILED"
    assert latest.get("error_summary") == "empty_extraction"


def test_documents_embed_accepts_array_and_filters(monkeypatch):
    selected_ids = ["doc-a", "doc-b"]

    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: False)
    monkeypatch.setattr(embedding_service, "_fetch_document_ids_by_filters", lambda **_kwargs: list(selected_ids))

    processed = []

    def fake_process_local_document(**kwargs):
        doc_id = kwargs["document_id"]
        processed.append(doc_id)
        return {
            "document_id": doc_id,
            "status": "COMPLETED",
            "chunks_count": 1,
            "points_upserted": 1,
            "error": None,
        }

    monkeypatch.setattr(embedding_service, "_process_local_document", fake_process_local_document)

    client = TestClient(app)
    response = client.post(
        "/api/documents/embed",
        json={"document_ids": ["doc-1"], "profile_id": "prof-1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents_processed"] == 3
    assert payload["documents_succeeded"] == 3
    assert payload["documents_failed"] == 0
    assert payload["total_chunks"] == 3
    assert payload["total_points_upserted"] == 3
    assert set(processed) == {"doc-1", "doc-a", "doc-b"}
