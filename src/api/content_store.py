import hashlib
from src.utils.logging_utils import get_logger
import pickle
from typing import Any, Dict

from src.api.blob_store import BlobStore, BlobConfigurationError, blob_storage_configured
from src.storage.blob_persistence import load_pickle as load_blob_pickle

logger = get_logger(__name__)


def save_extracted_pickle(document_id: str, extracted_obj: Any) -> Dict[str, Any]:
    """Save extracted document content to Azure Blob as pickle.

    Azure Blob is the ONLY storage for pickles — no local fallback.
    Pickles are intermediate data that persists until embedding completes.
    """
    if not blob_storage_configured():
        raise BlobConfigurationError(
            "Azure Blob storage is not configured. "
            "Pickle storage requires blob — set AZURE_BLOB_CONNECTION_STRING."
        )

    store = BlobStore()
    info = store.upload_pickle(document_id, extracted_obj)
    info.setdefault("path", info.get("blob_name"))
    return info


def load_extracted_pickle(document_id: str) -> Any:
    """Load extracted document content from Azure Blob pickle.

    Azure Blob is the ONLY source — no local fallback.
    """
    if not blob_storage_configured():
        raise ValueError(
            f"Azure Blob not configured — cannot load pickle for {document_id}"
        )

    store = BlobStore()
    blob_name = store.build_blob_name(document_id)
    payload = load_blob_pickle(blob_name)
    if payload is None:
        raise ValueError(f"Extracted content not found for document_id={document_id}")
    return pickle.loads(payload)


def delete_extracted_pickle(document_id: str) -> bool:
    """Delete pickle from Azure Blob after embedding completes."""
    if not blob_storage_configured():
        return False

    store = BlobStore()
    blob_name = store.build_blob_name(document_id)
    return store.delete_blob(blob_name)
