from types import SimpleNamespace

from azure.core.exceptions import ResourceExistsError

from src.api.blob_store import BlobStore


class FakeBlob:
    def __init__(self, name, metadata=None):
        self.name = name
        self.size = 10
        self.metadata = metadata or {}
        self.etag = "etag"
        self.last_modified = None
        self.content_settings = SimpleNamespace(content_type="application/octet-stream")


class FakeContainerClient:
    def __init__(self, blobs=None, blob_client=None):
        self._blobs = blobs or []
        self._blob_client = blob_client

    def list_blobs(self, name_starts_with=None, include=None):
        return list(self._blobs)

    def get_blob_client(self, name):
        return self._blob_client


class FakeServiceClient:
    def __init__(self, container_client):
        self._container_client = container_client

    def get_container_client(self, name):
        return self._container_client


class FakeBlobClient:
    def __init__(self, raise_on_lease=False):
        self.raise_on_lease = raise_on_lease

    def acquire_lease(self, lease_duration=60):
        if self.raise_on_lease:
            raise ResourceExistsError("lease already present")
        return SimpleNamespace(id="lease-id")


def test_list_pickle_blobs_filters_extensions():
    blobs = [
        FakeBlob("one.pkl"),
        FakeBlob("two.txt"),
        FakeBlob("three.pickle"),
    ]
    container = FakeContainerClient(blobs=blobs)
    store = BlobStore(container="unit-test", service_client=FakeServiceClient(container))

    names = [blob.name for blob in store.list_pickle_blobs()]
    assert "one.pkl" in names
    assert "three.pickle" in names
    assert "two.txt" not in names


def test_try_acquire_lease_returns_none_on_conflict():
    container = FakeContainerClient(blob_client=FakeBlobClient(raise_on_lease=True))
    store = BlobStore(container="unit-test", service_client=FakeServiceClient(container))

    lease_id = store.try_acquire_lease("doc-1.pkl", lease_duration=30)
    assert lease_id is None
