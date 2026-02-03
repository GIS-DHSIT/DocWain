import json
from types import SimpleNamespace

import pytest
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from src.api.blob_store import BlobStore
from src.storage import blob_persistence


class FakeLease:
    def __init__(self, lease_id: str):
        self.id = lease_id


class FakeBlobClient:
    def __init__(self, name: str):
        self.name = name
        self.uploaded = []
        self.exists = False
        self.lease_acquired = False
        self.props = SimpleNamespace(etag="etag", lease=SimpleNamespace(status="locked", state="leased"))

    def get_blob_properties(self):
        if not self.exists:
            raise ResourceNotFoundError("missing")
        return self.props

    def upload_blob(self, payload, overwrite=False, metadata=None, content_settings=None, lease=None):
        self.exists = True
        self.uploaded.append(
            {
                "payload": payload,
                "overwrite": overwrite,
                "metadata": metadata,
                "content_settings": content_settings,
                "lease": lease,
            }
        )
        if self.name.endswith(".pkl") and "latest.json" not in self.name and lease:
            exc = HttpResponseError(message="lease missing")
            exc.error_code = "LeaseIdMissing"
            raise exc

    def acquire_lease(self, lease_duration=30):
        self.lease_acquired = True
        return FakeLease("lease-id")

    def release_lease(self, lease=None):
        self.lease_acquired = False

    def download_blob(self, lease=None):
        return SimpleNamespace(readall=lambda: b"payload")


class FakeContainer:
    def __init__(self):
        self.clients = {}

    def get_blob_client(self, name):
        if name not in self.clients:
            self.clients[name] = FakeBlobClient(name)
        return self.clients[name]


def test_blob_lease_conflict_writes_versioned(monkeypatch):
    container = FakeContainer()
    monkeypatch.setattr(blob_persistence, "get_document_container_client", lambda: container)
    monkeypatch.setattr(blob_persistence.uuid, "uuid4", lambda: SimpleNamespace(hex="run123"))

    payload = b"data"
    result = blob_persistence.save_pickle_atomic(
        "doc123.pkl",
        payload,
        {"document_id": "doc123"},
        content_type="application/octet-stream",
        lease_seconds=15,
        retry_seconds=0,
    )

    assert result["versioned_blob_name"] == "doc123/run123.pkl"
    assert result["pointer_blob_name"] == "doc123/latest.json"
    pointer_client = container.get_blob_client("doc123/latest.json")
    pointer_payload = pointer_client.uploaded[-1]["payload"]
    payload_dict = json.loads(pointer_payload.decode("utf-8"))
    assert payload_dict["blob_name"] == "doc123/run123.pkl"


def test_blob_not_found_lease_creates_then_acquires(monkeypatch):
    class LeaseBlobClient(FakeBlobClient):
        def __init__(self, name):
            super().__init__(name)
            self.acquire_calls = 0

        def acquire_lease(self, lease_duration=30):
            self.acquire_calls += 1
            if self.acquire_calls == 1:
                raise ResourceNotFoundError("missing")
            return FakeLease("lease-ok")

    class LeaseContainer(FakeContainer):
        def get_blob_client(self, name):
            if name not in self.clients:
                self.clients[name] = LeaseBlobClient(name)
            return self.clients[name]

    container = LeaseContainer()
    store = BlobStore(service_client=SimpleNamespace(get_container_client=lambda _: container), container="docs")

    lease_id = store.try_acquire_lease("doc123.pkl", lease_duration=15)
    assert lease_id == "lease-ok"
    client = container.get_blob_client("doc123.pkl")
    assert client.exists
