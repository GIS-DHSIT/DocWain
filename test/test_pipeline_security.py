import numpy as np
import pytest

from src.api import dataHandler
from src.api.vector_store import build_collection_name, compute_chunk_id
from src.api.dw_newron import QdrantRetriever


def test_no_profile_leakage():
    name_one = build_collection_name("tenantA", "profile1")
    name_two = build_collection_name("tenantA", "profile2")
    assert name_one == "tenantA"
    assert name_two == "tenantA"


def test_chunk_ids_deterministic():
    cid1 = compute_chunk_id("sub", "prof", "doc", "file", 0, "hello world")
    cid2 = compute_chunk_id("sub", "prof", "doc", "file", 0, "hello world")
    cid3 = compute_chunk_id("sub", "prof", "doc", "file", 1, "hello world")
    assert cid1 == cid2
    assert cid1 != cid3


def test_retrieval_requires_profile_filter():
    class DummyClient:
        pass

    class DummyModel:
        def encode(self, *args, **kwargs):
            return np.zeros((1,))

        def get_sentence_embedding_dimension(self):
            return 1

    retriever = QdrantRetriever(DummyClient(), DummyModel())
    with pytest.raises(ValueError):
        retriever.hybrid_retrieve(collection_name="col", query="test", profile_id=None)


def test_training_creates_expected_payload(monkeypatch):
    captured = {}

    class StubStore:
        def ensure_collection(self, collection_name, vector_size):
            captured["collection"] = collection_name
            captured["vector_size"] = vector_size

        def upsert_records(self, collection_name, records, batch_size=100):
            captured["records"] = records
            return len(records)

        def delete_document(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(dataHandler, "get_vector_store", lambda: StubStore())

    embeddings = {
        "embeddings": [[0.1, 0.2], [0.2, 0.3]],
        "texts": ["alpha text", "beta text"],
        "chunk_metadata": [{"document_id": "doc"}, {"document_id": "doc"}],
    }

    result = dataHandler.save_embeddings_to_qdrant(
        embeddings, subscription_id="sub", profile_id="prof", doctag="doc", source_filename="file.txt"
    )

    assert result["status"] == "success"
    assert captured["collection"] == build_collection_name("sub", "prof")
    assert captured["vector_size"] == 2
    records = captured["records"]
    assert all(r.payload["profile_id"] == "prof" for r in records)
    assert all(r.payload["document_id"] == "doc" for r in records)
    assert len({r.chunk_id for r in records}) == len(records)


def test_delete_embeddings_scoped(monkeypatch):
    deleted = {}

    class StubStore:
        def delete_document(self, subscription_id, profile_id, document_id):
            deleted["sub"] = subscription_id
            deleted["profile"] = profile_id
            deleted["doc"] = document_id
            deleted["collection"] = build_collection_name(subscription_id, profile_id)
            return {"status": "success", "collection": deleted["collection"], "document_id": document_id}

    monkeypatch.setattr(dataHandler, "get_vector_store", lambda: StubStore())
    result = dataHandler.delete_embeddings("subA", "profileZ", "doc42")
    assert result["status"] == "success"
    assert deleted["collection"] == build_collection_name("subA", "profileZ")
    assert deleted["doc"] == "doc42"
