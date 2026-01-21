import hashlib
import logging
import os
import pickle
from typing import Any, Dict

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from src.api.config import Config

logger = logging.getLogger(__name__)

_CONTAINER_NAME = "document-content"
_EXTRACTED_DOC_TYPE = "extracted_doc"


def _build_service_client() -> BlobServiceClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    credential = os.getenv("AZURE_STORAGE_CREDENTIAL") or os.getenv("AZURE_STORAGE_KEY")
    if account_url and credential:
        return BlobServiceClient(account_url=account_url, credential=credential)

    fallback_conn = getattr(Config.DocAzureBlob, "AZURE_BLOB_CONNECTION_STRING", None)
    if fallback_conn:
        return BlobServiceClient.from_connection_string(fallback_conn)

    raise ValueError(
        "Azure Blob connection settings missing: "
        "set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL/AZURE_STORAGE_CREDENTIAL"
    )


def get_blob_client():
    service_client = _build_service_client()
    container_client = service_client.get_container_client(_CONTAINER_NAME)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure blob container %s exists: %s", _CONTAINER_NAME, exc)
    return container_client


def save_extracted_pickle(document_id: str, extracted_obj: Any) -> Dict[str, Any]:
    payload = pickle.dumps(extracted_obj, protocol=pickle.HIGHEST_PROTOCOL)
    blob_name = f"{document_id}.pkl"
    container = get_blob_client()
    blob_client = container.get_blob_client(blob_name)
    metadata = {"document_id": str(document_id), "type": _EXTRACTED_DOC_TYPE, "version": "v1"}
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata=metadata,
        content_settings=ContentSettings(content_type="application/octet-stream"),
    )
    try:
        props = blob_client.get_blob_properties()
        etag = getattr(props, "etag", None)
    except Exception:  # noqa: BLE001
        etag = None
    return {
        "blob_name": blob_name,
        "etag": etag,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_extracted_pickle(document_id: str) -> Any:
    blob_name = f"{document_id}.pkl"
    container = get_blob_client()
    blob_client = container.get_blob_client(blob_name)
    try:
        props = blob_client.get_blob_properties()
    except ResourceNotFoundError as exc:
        raise ValueError(f"Extracted content not found in blob for document_id={document_id}") from exc
    metadata = getattr(props, "metadata", {}) or {}
    if metadata and metadata.get("type") and metadata.get("type") != _EXTRACTED_DOC_TYPE:
        raise ValueError(f"Blob {blob_name} is not a trusted extracted document payload")
    try:
        payload = blob_client.download_blob().readall()
    except ResourceNotFoundError as exc:
        raise ValueError(f"Extracted content not found in blob for document_id={document_id}") from exc
    return pickle.loads(payload)


def delete_extracted_pickle(document_id: str) -> bool:
    blob_name = f"{document_id}.pkl"
    container = get_blob_client()
    blob_client = container.get_blob_client(blob_name)
    try:
        blob_client.delete_blob()
        return True
    except ResourceNotFoundError:
        return False
