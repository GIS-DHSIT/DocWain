import pytest
from fastapi import HTTPException

from src.api import dataHandler
from src.api.config import Config
from src.api.documents_api import DocumentEmbedRequest, embed_document


class _DummyStore:
    def __init__(self, capture):
        self._capture = capture

    def ensure_collection(self, collection_name, vector_size):
        self._capture["collection"] = collection_name
        self._capture["vector_size"] = vector_size

    def upsert_records(self, collection_name, records, batch_size=100):
        self._capture["collection"] = collection_name
        self._capture["records"] = list(records)
        return len(self._capture["records"])


def test_embed_requires_subscription_and_profile_or_doc():
    with pytest.raises(HTTPException) as excinfo:
        embed_document(DocumentEmbedRequest())
    assert excinfo.value.status_code == 422

    with pytest.raises(HTTPException) as excinfo:
        embed_document(DocumentEmbedRequest(subscription_id="sub-1"))
    assert excinfo.value.status_code == 422


def test_embed_payload_contains_required_metadata(monkeypatch):
    capture = {}
    monkeypatch.setattr(dataHandler, "get_vector_store", lambda: _DummyStore(capture))
    monkeypatch.setattr(Config.Model, "EMBEDDING_DIM", None)

    embeddings = {
        "embeddings": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]],
        "texts": ["Line one.", "Line two."],
        "chunk_metadata": [
            {"document_id": "doc-1", "section_title": "SUMMARY"},
            {"document_id": "doc-1", "section_title": "SUMMARY"},
        ],
        "doc_metadata": {
            "document_type": "REPORT",
            "profile_name": "Test User",
            "products_name": "Widget",
            "description": "Sample",
            "languages": ["en"],
            "source_uri": "s3://bucket/doc-1",
        },
    }

    result = dataHandler.save_embeddings_to_qdrant(
        embeddings,
        subscription_id="sub-1",
        profile_id="profile-1",
        doctag="doc-1",
        source_filename="doc-1.pdf",
    )
    assert result["status"] == "success"

    records = capture.get("records") or []
    assert records
    payload = records[0].payload
    required = {
        "subscription_id",
        "profile_id",
        "document_name",
        "document_type",
        "chunk_id",
        "chunk_kind",
        "section",
        "page_start",
        "page_end",
        "source_uri",
        "text_hash",
    }
    missing = required.difference(payload.keys())
    assert not missing
