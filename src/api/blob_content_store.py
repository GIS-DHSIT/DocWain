import hashlib
from src.utils.logging_utils import get_logger
import pickle
from typing import Any, Dict

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import ContentSettings

from src.api.config import Config
from src.storage.azure_blob_client import get_document_container_client

logger = get_logger(__name__)

_EXTRACTED_DOC_TYPE = "extracted_doc"

def get_blob_client():
    container_client = get_document_container_client()
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not ensure document blob container %s exists: %s",
            getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", ""),
            exc,
        )
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
