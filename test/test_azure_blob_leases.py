from types import SimpleNamespace

from azure.core.exceptions import HttpResponseError

from src.storage.azure_blob_client import AzureBlob


class FakeResponse:
    def __init__(self, request_id="req-1"):
        self.headers = {"x-ms-request-id": request_id}


class FakeBlobClient:
    def __init__(self):
        self.upload_calls = []
        self.delete_calls = []
        self.raise_lease_missing = False
        self.raise_lease_lost = False

    def upload_blob(self, data, **kwargs):
        self.upload_calls.append(kwargs)
        if self.raise_lease_missing and not kwargs.get("lease"):
            exc = HttpResponseError("lease missing", response=FakeResponse("req-upload"))
            exc.error_code = "LeaseIdMissing"
            raise exc

    def delete_blob(self, **kwargs):
        self.delete_calls.append(kwargs)
        if self.raise_lease_lost and kwargs.get("lease"):
            exc = HttpResponseError("lease lost", response=FakeResponse("req-delete"))
            exc.error_code = "LeaseLost"
            raise exc


class FakeContainer:
    def __init__(self, blob_client):
        self.blob_client = blob_client

    def get_blob_client(self, _name):
        return self.blob_client


def _fake_lease(lease_id="lease-123"):
    lease = SimpleNamespace(id=lease_id)
    lease.release = lambda: None
    lease.renew = lambda: None
    return lease


def test_upload_with_lease_passes_lease_id(monkeypatch):
    blob_client = FakeBlobClient()
    azure_blob = AzureBlob()
    monkeypatch.setattr(azure_blob, "get_container_client", lambda _name: FakeContainer(blob_client))

    lease = _fake_lease("lease-abc")
    azure_blob.upload_bytes("container", "blob.pkl", b"data", lease=lease)

    assert blob_client.upload_calls
    assert blob_client.upload_calls[-1]["lease"] == "lease-abc"


def test_upload_retries_on_lease_id_missing(monkeypatch):
    blob_client = FakeBlobClient()
    blob_client.raise_lease_missing = True
    azure_blob = AzureBlob()
    monkeypatch.setattr(azure_blob, "get_container_client", lambda _name: FakeContainer(blob_client))
    monkeypatch.setattr(azure_blob, "acquire_lease", lambda *_args, **_kwargs: _fake_lease("lease-retry"))

    azure_blob.upload_bytes("container", "blob.pkl", b"data")

    assert len(blob_client.upload_calls) >= 2
    assert blob_client.upload_calls[-1]["lease"] == "lease-retry"


def test_delete_retries_on_lease_lost(monkeypatch):
    blob_client = FakeBlobClient()
    blob_client.raise_lease_lost = True
    azure_blob = AzureBlob()
    monkeypatch.setattr(azure_blob, "get_container_client", lambda _name: FakeContainer(blob_client))

    lease = _fake_lease("lease-del")
    deleted = azure_blob.delete_blob("container", "blob.pkl", lease=lease)

    assert deleted is True
    assert len(blob_client.delete_calls) >= 2
    assert blob_client.delete_calls[-1].get("lease") is None
