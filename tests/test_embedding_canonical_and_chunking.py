import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import dataHandler, embedding_service
from src.api.config import Config
from src.api.statuses import STATUS_EXTRACTION_OR_CHUNKING_FAILED
from src.main import app
from src.embedding.pipeline import qdrant_ingestion
from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text


def test_fallback_chunking_produces_min_chunks(monkeypatch):
    long_text = "\n".join([f"Sentence {i} about experience and skills." for i in range(240)])
    min_required = int(getattr(Config.Retrieval, "MIN_REQUIRED_CHUNKS", 3))

    def fake_chunk_with_section_chunker(*_args, **_kwargs):
        meta = {
            "document_id": "doc-1",
            "section_title": "Resume",
            "section_path": "Resume",
            "chunk_index": 0,
            "chunk_type": "text",
            "chunk_kind": "section_text",
            "chunking_mode": "section_aware",
            "sentence_complete": True,
        }
        return [long_text], [meta], None

    monkeypatch.setattr(dataHandler, "_chunk_with_section_chunker", fake_chunk_with_section_chunker)
    monkeypatch.setattr(dataHandler, "encode_with_fallback", lambda texts, **_kw: [[0.0, 0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(dataHandler, "build_sparse_vectors", lambda texts: [{"indices": [], "values": []} for _ in texts])
    monkeypatch.setattr(dataHandler, "compute_section_summaries", lambda chunks, *_args, **_kwargs: [None for _ in chunks])
    monkeypatch.setattr(dataHandler, "save_embeddings_to_qdrant", lambda embeddings, *_args, **_kwargs: {"points_saved": len(embeddings["texts"]), "dropped_invalid": 0})
    monkeypatch.setattr(dataHandler, "get_metrics_store", lambda: SimpleNamespace(available=False))
    monkeypatch.setattr(dataHandler, "METRICS_V2_ENABLED", False)

    result = dataHandler.train_on_document(
        long_text,
        subscription_id="sub-1",
        profile_id="prof-1",
        doc_tag="doc-1",
        doc_name="resume.pdf",
    )

    assert result["chunks"] >= min_required


def test_embed_uses_embedding_text_only(monkeypatch):
    captured = {}

    def fake_embed(texts, model=None):
        captured["texts"] = list(texts)
        return [[0.0, 0.1, 0.2] for _ in texts]

    class FakeStore:
        def __init__(self):
            self.records = []

        def ensure_collection(self, *_args, **_kwargs):
            return None

        def upsert_records(self, _collection_name, records, batch_size=100):
            self.records.extend(records)
            return len(records)

    fake_store = FakeStore()

    monkeypatch.setattr(qdrant_ingestion, "_ollama_embed", fake_embed)
    monkeypatch.setattr(qdrant_ingestion, "QdrantVectorStore", lambda client=None: fake_store)

    content = "Python engineer with AWS and data pipelines." * 6
    expected_embedding_text = ensure_embedding_text(content)
    raw_payloads = [
        {
            "subscription_id": "sub-1",
            "profile_id": "prof-1",
            "document_id": "doc-1",
            "content": content,
            "text": "",
        }
    ]

    result = qdrant_ingestion.ingest_payloads(raw_payloads, client=None)

    assert result
    assert captured["texts"] == [expected_embedding_text]
    assert fake_store.records
    payload = fake_store.records[0].payload
    assert payload.get("embedding_text") == expected_embedding_text
    assert payload.get("embedding_text")  # not empty


def test_bm25_never_sees_empty_field(caplog):
    from src.api.dw_newron import HybridReranker, RetrievedChunk

    reranker = HybridReranker(cross_encoder=None)
    chunks = [
        RetrievedChunk(id="c1", text="", score=0.2, metadata={}, source=None, method="dense"),
        RetrievedChunk(id="c2", text="   ", score=0.1, metadata={}, source=None, method="dense"),
    ]
    diagnostics = {
        "retrieved_count": 2,
        "dropped_invalid_count": 2,
        "invalid_samples": [{"chunk_id": "c1", "length": 0, "sample": ""}],
    }

    with caplog.at_level(logging.WARNING):
        reranker.rerank(chunks=chunks, query="test", top_k=2, use_cross_encoder=False, diagnostics=diagnostics)

    assert "BM25 skipped" in caplog.text
    assert "retrieved=2" in caplog.text
    assert "dropped_invalid=2" in caplog.text


def test_insufficient_chunks_returns_diagnostic_not_500(monkeypatch):
    def fake_process_local_document(**kwargs):
        return {
            "document_id": kwargs["document_id"],
            "status": "FAILED",
            "chunks_count": 1,
            "points_upserted": 0,
            "error": "extraction_or_chunking_failed",
            "error_message": "EXTRACTION_OR_CHUNKING_FAILED",
            "failed_reason": "extraction_or_chunking_failed",
            "diagnostics": {
                "valid_chunks": 1,
                "min_required": 3,
                "chunking_mode": "sliding_window_fallback",
            },
        }

    monkeypatch.setattr(embedding_service, "blob_storage_configured", lambda: False)
    monkeypatch.setattr(embedding_service, "_fetch_document_ids_by_filters", lambda **_kwargs: [])
    monkeypatch.setattr(embedding_service, "_process_local_document", fake_process_local_document)

    client = TestClient(app)
    response = client.post("/api/documents/embed", json={"document_id": "doc-small"})

    assert response.status_code == 422
    payload = response.json()
    assert payload.get("status") == STATUS_EXTRACTION_OR_CHUNKING_FAILED
    assert payload.get("diagnostics", {}).get("min_required") == 3
