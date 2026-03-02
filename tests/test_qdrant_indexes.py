from __future__ import annotations

from types import SimpleNamespace

from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS
from src.api.vector_store import QdrantVectorStore


class FakeQdrant:
    def __init__(self):
        self.payload_schema = {}

    def get_collection(self, collection_name):  # noqa: ANN001
        _ = collection_name
        return SimpleNamespace(payload_schema=self.payload_schema, config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4))))

    def create_payload_index(self, collection_name, field_name, field_schema):  # noqa: ANN001
        _ = (collection_name, field_schema)
        self.payload_schema[field_name] = {"data_type": "keyword"}


def test_qdrant_indexes_created():
    client = FakeQdrant()
    store = QdrantVectorStore(client=client)
    result = store.ensure_payload_indexes("sub-1", REQUIRED_PAYLOAD_INDEX_FIELDS, create_missing=True)
    assert result["missing"] == []
    for field in REQUIRED_PAYLOAD_INDEX_FIELDS:
        assert field in client.payload_schema
