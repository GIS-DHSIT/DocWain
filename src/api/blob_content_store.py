import hashlib
import logging
import pickle
from typing import Any, Dict

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import ContentSettings

from src.storage.azure_blob_client import get_azure_blob

logger = logging.getLogger(__name__)

_EXTRACTED_DOC_TYPE = "extracted_doc"


def get_blob_client():
    azure_blob = get_azure_blob()
    return azure_blob.get_container_client(azure_blob.document_container_name)


def save_extracted_pickle(document_id: str, extracted_obj: Any) -> Dict[str, Any]:
    payload = pickle.dumps(extracted_obj, protocol=pickle.HIGHEST_PROTOCOL)
    blob_name = f"{document_id}.pkl"
    metadata = {"document_id": str(document_id), "type": _EXTRACTED_DOC_TYPE, "version": "v1"}
    container = get_blob_client()
    blob_client = container.get_blob_client(blob_name)
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata=metadata,
        content_settings=ContentSettings(content_type="application/octet-stream"),
    )
    etag = None
    return {
        "blob_name": blob_name,
        "etag": etag,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_extracted_pickle(document_id: str) -> Any:
    blob_name = f"{document_id}.pkl"
    try:
        container = get_blob_client()
        payload = container.get_blob_client(blob_name).download_blob().readall()
    except ResourceNotFoundError as exc:
        raise ValueError(f"Extracted content not found in blob for document_id={document_id}") from exc
    return pickle.loads(payload)


def delete_extracted_pickle(document_id: str) -> bool:
    blob_name = f"{document_id}.pkl"
    try:
        container = get_blob_client()
        container.get_blob_client(blob_name).delete_blob()
        return True
    except ResourceNotFoundError:
        return False
