from src.finetune import qdrant_discovery


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class _FakeCollectionsResponse:
    def __init__(self, names):
        self.collections = [type("Collection", (), {"name": name})() for name in names]


class _FakeQdrantClient:
    def __init__(self, batches):
        self.batches = batches

    def get_collections(self):
        return _FakeCollectionsResponse(list(self.batches.keys()))

    def scroll(self, collection_name, limit, with_vectors, with_payload, offset=None):
        data = self.batches.get(collection_name, [])
        start = offset or 0
        end = start + limit
        batch = [_FakePoint(payload) for payload in data[start:end]]
        next_offset = end if end < len(data) else None
        return batch, next_offset


def test_list_collections_returns_names():
    client = _FakeQdrantClient({"default": [], "team": []})
    assert qdrant_discovery.list_collections(client=client) == ["default", "team"]


def test_list_profile_ids_counts_profiles():
    client = _FakeQdrantClient(
        {
            "default": [
                {"profile_id": "alpha"},
                {"profileId": "beta"},
                {"profile_id": "alpha"},
            ]
        }
    )
    stats = qdrant_discovery.list_profile_ids("default", client=client, scan_limit=10, batch_size=2)
    assert stats["profile_ids"] == ["alpha", "beta"]
    assert stats["counts"]["alpha"] == 2
