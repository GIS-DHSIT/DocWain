from src.finetune.dataset_builder import (
    discover_collections_and_profiles,
    discover_profiles_for_collection,
)


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class _FakeCollectionsResponse:
    def __init__(self, names):
        self.collections = [type("Collection", (), {"name": name})() for name in names]


class _FakeQdrantClient:
    def __init__(self, batches):
        # batches: {collection_name: [payload_dict, ...]}
        self.batches = batches
        self.scroll_calls = []

    def get_collections(self):
        return _FakeCollectionsResponse(list(self.batches.keys()))

    def scroll(self, collection_name, limit, with_vectors, with_payload, offset=None, scroll_filter=None):
        # Simulate Qdrant scroll pagination using a simple offset index
        self.scroll_calls.append((collection_name, limit, offset))
        data = self.batches.get(collection_name, [])
        start = offset or 0
        end = start + limit
        batch = [_FakePoint(payload) for payload in data[start:end]]
        next_offset = end if end < len(data) else None
        return batch, next_offset


def test_discover_collections_and_profiles_reads_all_collections():
    client = _FakeQdrantClient(
        {
            "default": [{"profile_id": "alpha"}, {"profileId": "beta"}, {"profile_id": "alpha"}],
            "other": [{"profile_id": "gamma"}],
        }
    )

    discovered = discover_collections_and_profiles(client=client)

    assert discovered == {"default": ["alpha", "beta"], "other": ["gamma"]}


def test_discover_collections_and_profiles_honors_subscription_filter():
    client = _FakeQdrantClient(
        {
            "default": [{"profile_id": "alpha"}],
            "other": [{"profile_id": "gamma"}],
        }
    )

    discovered = discover_collections_and_profiles(subscription_ids=["other"], client=client)

    assert discovered == {"other": ["gamma"]}


def test_discover_profiles_for_collection_reads_payloads():
    client = _FakeQdrantClient(
        {
            "default": [{"profile_id": "alpha"}, {"profileId": "beta"}, {"profile_id": "alpha"}],
        }
    )

    profiles = discover_profiles_for_collection("default", client=client)

    assert profiles == ["alpha", "beta"]
