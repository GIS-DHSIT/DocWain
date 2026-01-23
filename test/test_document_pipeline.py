import numpy as np
import pytest
from types import SimpleNamespace

from azure.core.exceptions import ResourceNotFoundError

from src.api import blob_content_store
from src.api.pipeline_models import ExtractedDocument
from src.api import dataHandler
from src.screening import storage_adapter


class FakeBlobClient:
    def __init__(self, storage, name):
        self._storage = storage
        self._name = name

    def upload_blob(self, data, overwrite=False, metadata=None, content_settings=None):
        if not overwrite and self._name in self._storage:
            raise ValueError("Blob already exists")
        self._storage[self._name] = {
            "data": data,
            "metadata": metadata or {},
        }

    def get_blob_properties(self):
        if self._name not in self._storage:
            raise ResourceNotFoundError("missing")
        record = self._storage[self._name]
        return SimpleNamespace(etag="etag", metadata=record.get("metadata", {}))

    def download_blob(self):
        if self._name not in self._storage:
            raise ResourceNotFoundError("missing")
        data = self._storage[self._name]["data"]
        return SimpleNamespace(readall=lambda: data)

    def delete_blob(self):
        if self._name not in self._storage:
            raise ResourceNotFoundError("missing")
        del self._storage[self._name]


class FakeContainerClient:
    def __init__(self, storage):
        self._storage = storage

    def get_blob_client(self, blob_name):
        return FakeBlobClient(self._storage, blob_name)


def test_blob_pickle_roundtrip(monkeypatch):
    storage = {}
    container = FakeContainerClient(storage)
    monkeypatch.setattr(blob_content_store, "get_blob_client", lambda: container)

    extracted = ExtractedDocument(full_text="hello", sections=[], tables=[], figures=[], chunk_candidates=[])
    result = blob_content_store.save_extracted_pickle("doc-1", extracted)
    assert result["blob_name"] == "doc-1.pkl"
    assert "doc-1.pkl" in storage

    loaded = blob_content_store.load_extracted_pickle("doc-1")
    assert isinstance(loaded, ExtractedDocument)
    assert loaded.full_text == "hello"

    assert blob_content_store.delete_extracted_pickle("doc-1") is True
    assert blob_content_store.delete_extracted_pickle("doc-1") is False


def test_storage_adapter_prefers_blob_text(monkeypatch):
    extracted = ExtractedDocument(full_text="from blob", sections=[], tables=[], figures=[], chunk_candidates=[])
    monkeypatch.setattr(storage_adapter, "load_extracted_pickle", lambda doc_id: extracted)
    monkeypatch.setattr(storage_adapter, "_get_document_record", lambda doc_id: {"_id": doc_id})

    assert storage_adapter.get_document_text("doc-1") == "from blob"


def test_chunk_coverage_fallback(monkeypatch):
    captured = {}

    class DummyModel:
        def encode(self, chunks, convert_to_numpy=True, normalize_embeddings=True):
            return np.zeros((len(chunks), 3))

    def fake_save_embeddings(payload, subscription_id, profile_id, doctag, source_filename, batch_size=100):
        captured["texts"] = payload["texts"]
        return {"status": "success", "points_saved": len(payload["texts"])}

    monkeypatch.setattr(dataHandler, "get_model", lambda: DummyModel())
    monkeypatch.setattr(dataHandler, "save_embeddings_to_qdrant", fake_save_embeddings)
    monkeypatch.setattr(dataHandler, "build_sparse_vectors", lambda texts: [{"indices": [0], "values": [1.0]}] * len(texts))
    monkeypatch.setattr(dataHandler, "compute_section_summaries", lambda chunks, meta, extracted=None: [""] * len(chunks))
    monkeypatch.setattr(dataHandler.Config.Model, "EMBEDDING_DIM", 3)

    extracted = ExtractedDocument(
        full_text="a" * 2000,
        sections=[],
        tables=[],
        figures=[],
        chunk_candidates=[dataHandler.ChunkCandidate(text="short", page=None, section_title="S", section_id="s")],
    )

    result = dataHandler.train_on_document(extracted, "sub-1", "profile-1", "doc-1", "file.pdf")
    assert result["chunks"] > 1
    assert len(captured.get("texts", [])) == result["chunks"]
    assert len("".join(captured["texts"])) >= len(extracted.full_text)


def test_save_embeddings_rejects_default_subscription():
    with pytest.raises(ValueError):
        dataHandler.save_embeddings_to_qdrant(
            {"embeddings": [[0.0, 0.0]], "texts": ["text"]},
            "default",
            "profile-1",
            "doc-1",
            "file.txt",
        )


def test_process_pipeline_blocks_on_security(monkeypatch):
    extracted = ExtractedDocument(full_text="hello", sections=[], tables=[], figures=[], chunk_candidates=[])
    monkeypatch.setattr(dataHandler, "fileProcessor", lambda content, file: {"file": extracted})
    monkeypatch.setattr(dataHandler, "save_extracted_pickle", lambda doc_id, obj: {"blob_name": f"{doc_id}.pkl"})
    monkeypatch.setattr(dataHandler, "update_extraction_metadata", lambda *a, **k: None)
    monkeypatch.setattr(dataHandler, "resolve_subscription_id", lambda doc_id, provided=None: "sub-1")
    monkeypatch.setattr(dataHandler, "resolve_profile_id", lambda doc_id, provided=None: "profile-1")
    monkeypatch.setattr(dataHandler, "run_security_screening", lambda doc_id: {"risk_level": "HIGH"})
    monkeypatch.setattr(dataHandler, "update_security_screening", lambda *a, **k: None)

    called = {"deleted": False, "embedded": False}

    def fake_delete(doc_id):
        called["deleted"] = True
        return True

    def fake_train(*args, **kwargs):
        called["embedded"] = True
        return {"status": "success", "points_saved": 1, "chunks": 1, "dropped_chunks": 0}

    monkeypatch.setattr(dataHandler, "delete_extracted_pickle", fake_delete)
    monkeypatch.setattr(dataHandler, "train_on_document", fake_train)

    result = dataHandler.process_document_pipeline("doc-1", b"bytes", "file.pdf")
    assert result["security"]["status"] == "failed"
    assert result["embedding"]["status"] == "skipped"
    assert called["embedded"] is False
    assert called["deleted"] is False


def test_process_pipeline_deletes_on_success(monkeypatch):
    extracted = ExtractedDocument(full_text="hello", sections=[], tables=[], figures=[], chunk_candidates=[])
    monkeypatch.setattr(dataHandler, "fileProcessor", lambda content, file: {"file": extracted})
    monkeypatch.setattr(dataHandler, "save_extracted_pickle", lambda doc_id, obj: {"blob_name": f"{doc_id}.pkl"})
    monkeypatch.setattr(dataHandler, "update_extraction_metadata", lambda *a, **k: None)
    monkeypatch.setattr(dataHandler, "resolve_subscription_id", lambda doc_id, provided=None: "sub-1")
    monkeypatch.setattr(dataHandler, "resolve_profile_id", lambda doc_id, provided=None: "profile-1")
    monkeypatch.setattr(dataHandler, "run_security_screening", lambda doc_id: {"risk_level": "LOW"})
    monkeypatch.setattr(dataHandler, "update_security_screening", lambda *a, **k: None)
    monkeypatch.setattr(dataHandler, "get_subscription_pii_setting", lambda sub: False)
    monkeypatch.setattr(dataHandler, "mask_document_content", lambda docs: (docs, 0, False, []))
    monkeypatch.setattr(dataHandler, "update_pii_stats", lambda *a, **k: None)

    monkeypatch.setattr(
        dataHandler,
        "train_on_document",
        lambda *args, **kwargs: {"status": "success", "points_saved": 1, "chunks": 1, "dropped_chunks": 0},
    )

    deleted = {"called": False}

    def fake_delete(doc_id):
        deleted["called"] = True
        return True

    monkeypatch.setattr(dataHandler, "delete_extracted_pickle", fake_delete)

    result = dataHandler.process_document_pipeline("doc-2", b"bytes", "file.pdf")
    assert result["embedding"]["status"] == "completed"
    assert result["cleanup"]["pickle_deleted"] is True
    assert deleted["called"] is True
