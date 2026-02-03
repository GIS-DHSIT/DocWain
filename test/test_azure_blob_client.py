import base64
from pathlib import Path

import pytest
from azure.core.exceptions import ResourceNotFoundError

from src.api.config import Config
from src.api import dw_chat
from src.api.blob_store import BlobStore
from src.storage import azure_blob_client


@pytest.fixture(autouse=True)
def reset_blob_clients(monkeypatch):
    monkeypatch.setattr(azure_blob_client, "_SERVICE_CLIENT", None)
    monkeypatch.setattr(azure_blob_client, "_CHAT_CONTAINER_CLIENT", None)
    monkeypatch.setattr(azure_blob_client, "_DOCUMENT_CONTAINER_CLIENT", None)


def test_decode_connection_string_if_base64():
    raw = (
        "DefaultEndpointsProtocol=https;AccountName=acct;"
        "AccountKey=YWJjZA==;EndpointSuffix=core.windows.net"
    )
    encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
    assert azure_blob_client.decode_connection_string_if_base64(encoded) == raw
    assert azure_blob_client.decode_connection_string_if_base64(raw) == raw


def test_get_blob_service_client_uses_azure_blob_connection_string(monkeypatch):
    calls = {"count": 0}

    class FakeClient:
        @classmethod
        def from_connection_string(cls, conn_str):
            calls["count"] += 1
            calls["conn_str"] = conn_str
            return "client"

    monkeypatch.setattr(azure_blob_client, "BlobServiceClient", FakeClient)
    monkeypatch.setattr(
        Config.AzureBlob,
        "CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=YWJjZA==;EndpointSuffix=core.windows.net",
    )

    first = azure_blob_client.get_blob_service_client()
    second = azure_blob_client.get_blob_service_client()

    assert first == "client"
    assert second == "client"
    assert calls["count"] == 1
    assert calls["conn_str"].startswith("DefaultEndpointsProtocol=")


def test_missing_connection_string_raises(monkeypatch):
    monkeypatch.setattr(Config.AzureBlob, "CONNECTION_STRING", "")
    with pytest.raises(azure_blob_client.CredentialError) as exc:
        azure_blob_client.get_blob_service_client()
    assert "AzureBlob.CONNECTION_STRING" in str(exc.value)


def test_container_clients_use_config_names(monkeypatch):
    calls = {"names": []}

    class FakeService:
        def get_container_client(self, name):
            calls["names"].append(name)
            return f"container:{name}"

    monkeypatch.setattr(azure_blob_client, "get_blob_service_client", lambda: FakeService())
    monkeypatch.setattr(Config.AzureBlob, "CONTAINER_NAME", "chat-container")
    monkeypatch.setattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "doc-container")

    chat_client = azure_blob_client.get_chat_container_client()
    doc_client = azure_blob_client.get_document_container_client()

    assert chat_client == "container:chat-container"
    assert doc_client == "container:doc-container"
    assert calls["names"] == ["chat-container", "doc-container"]


def test_chat_history_uses_chat_container(monkeypatch):
    class FakeBlobClient:
        def __init__(self):
            self.data = None

        def upload_blob(self, payload, overwrite=False, content_settings=None):
            self.data = payload

        def download_blob(self):
            class Downloader:
                @staticmethod
                def readall():
                    return b'{"sessions":[]}'

            return Downloader()

    class FakeContainer:
        def __init__(self):
            self.blob_names = []
            self.blob_client = FakeBlobClient()

        def get_blob_client(self, blob_name):
            self.blob_names.append(blob_name)
            return self.blob_client

    fake_container = FakeContainer()
    monkeypatch.setattr(azure_blob_client, "get_chat_container_client", lambda: fake_container)
    monkeypatch.setattr(dw_chat, "get_chat_container_client", lambda: fake_container)
    monkeypatch.setattr(azure_blob_client, "get_document_container_client", lambda: (_ for _ in ()).throw(RuntimeError()))

    dw_chat.save_chat_history("user-1", {"sessions": []})
    history = dw_chat.get_chat_history("user-1")

    assert history == {"sessions": []}
    assert fake_container.blob_names == ["chat_history/user-1.json", "chat_history/user-1.json"]


def test_blob_store_uses_document_container(monkeypatch):
    class FakeContainer:
        pass

    fake_container = FakeContainer()
    from src.api import blob_store as blob_store_module

    monkeypatch.setattr(blob_store_module, "get_document_container_client", lambda: fake_container)
    monkeypatch.setattr(azure_blob_client, "get_chat_container_client", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "document-content")

    store = BlobStore()
    assert store.container == "document-content"
    assert store.container_client is fake_container


def test_get_azure_docs_preserves_blob_name_with_spaces(monkeypatch):
    class FakeDownloader:
        def readall(self):
            return b"pdf-bytes"

    class FakeBlobClient:
        url = "https://acct.blob.core.windows.net/document-content/folder/ajay%20resume.pdf?sig=secret"

        def download_blob(self):
            return FakeDownloader()

    class FakeContainerClient:
        def __init__(self):
            self.requested_blob = None
            self.blob_client = FakeBlobClient()

        def get_blob_client(self, blob=None, **kwargs):
            self.requested_blob = blob
            return self.blob_client

    from src.api import dataHandler

    container = FakeContainerClient()
    monkeypatch.setattr(dataHandler, "get_document_container_client", lambda: container)
    monkeypatch.setattr(Config.Teams, "BLOB_CONTAINER", "")
    monkeypatch.setattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "document-content")

    payload = dataHandler.get_azure_docs("folder/ajay resume.pdf", document_id="doc-1")
    assert payload == b"pdf-bytes"
    assert container.requested_blob == "folder/ajay resume.pdf"

    payload = dataHandler.get_azure_docs("folder/ajay%20resume.pdf", document_id="doc-2")
    assert payload == b"pdf-bytes"
    assert container.requested_blob == "folder/ajay%20resume.pdf"


def test_get_azure_docs_falls_back_to_teams_container(monkeypatch):
    class MissingBlobClient:
        def download_blob(self):
            raise ResourceNotFoundError("missing")

    class DocContainer:
        def get_blob_client(self, blob=None, **kwargs):
            self.requested_blob = blob
            return MissingBlobClient()

    class TeamsBlobClient:
        def __init__(self):
            self.requested = None

        def download_blob(self):
            class Downloader:
                @staticmethod
                def readall():
                    return b"ok"

            return Downloader()

    class TeamsContainer:
        def __init__(self):
            self.requested_blob = None

        def get_blob_client(self, blob=None, **kwargs):
            self.requested_blob = blob
            return TeamsBlobClient()

    from src.api import dataHandler

    doc_container = DocContainer()
    teams_container = TeamsContainer()

    monkeypatch.setattr(dataHandler, "get_document_container_client", lambda: doc_container)
    monkeypatch.setattr(dataHandler, "get_container_client", lambda name: teams_container)
    monkeypatch.setattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "document-content")
    monkeypatch.setattr(Config.Teams, "BLOB_CONTAINER", "local-uploads")

    payload = dataHandler.get_azure_docs("folder/file.pdf", document_id="doc-1")
    assert payload == b"ok"
    assert teams_container.requested_blob == "local/folder/file.pdf"


def test_no_legacy_blob_config_symbols():
    root = Path(__file__).resolve().parents[1]
    needles = ["AZURE_BLOB_" + "CONNECTION_STRING"]
    paths = list((root / "src").rglob("*.py")) + list((root / "test").rglob("*.py"))
    for path in paths:
        content = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in content, f"Forbidden reference {needle} in {path}"
