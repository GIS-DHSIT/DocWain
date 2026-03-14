import hashlib
from src.utils.logging_utils import get_logger
import os
import pickle
from pathlib import Path
from typing import Any, Dict

from src.api.blob_store import BlobStore, BlobConfigurationError, blob_storage_configured
from src.storage.blob_persistence import load_pickle as load_blob_pickle

logger = get_logger(__name__)

_DEFAULT_DIR = "document-content"
_BLOB_UNCONFIGURED_LOGGED = False

def _sanitize_document_id(document_id: str) -> str:
    text = str(document_id).strip()
    for sep in {os.sep, os.altsep}:
        if sep:
            text = text.replace(sep, "_")
    return text

def get_document_content_dir() -> Path:
    raw = os.getenv("DOCUMENT_CONTENT_DIR", _DEFAULT_DIR)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path

def ensure_document_content_dir() -> Path:
    path = get_document_content_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to ensure document content dir %s: %s", path, exc)
        raise
    return path

def build_pickle_path(document_id: str) -> Path:
    safe_id = _sanitize_document_id(document_id)
    return ensure_document_content_dir() / f"{safe_id}.pkl"

def save_extracted_pickle(document_id: str, extracted_obj: Any) -> Dict[str, Any]:
    global _BLOB_UNCONFIGURED_LOGGED
    if blob_storage_configured():
        try:
            store = BlobStore()
            info = store.upload_pickle(document_id, extracted_obj)
            info.setdefault("path", info.get("blob_name"))
            return info
        except BlobConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist extracted pickle to blob: %s", exc)
            logger.debug("Falling back to local pickle storage for %s", document_id)

    if not _BLOB_UNCONFIGURED_LOGGED:
        logger.debug("Blob storage not configured; falling back to local pickle storage.")
        _BLOB_UNCONFIGURED_LOGGED = True
    payload = pickle.dumps(extracted_obj, protocol=pickle.HIGHEST_PROTOCOL)
    path = build_pickle_path(document_id)
    tmp_path = path.with_suffix(".pkl.tmp")

    with open(tmp_path, "wb") as handle:
        handle.write(payload)
    tmp_path.replace(path)

    return {
        "path": str(path),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

def load_extracted_pickle(document_id: str) -> Any:
    if blob_storage_configured():
        store = BlobStore()
        blob_name = store.build_blob_name(document_id)
        payload = load_blob_pickle(blob_name)
        if payload is None:
            raise ValueError(f"Extracted content not found for document_id={document_id}")
        return pickle.loads(payload)

    path = build_pickle_path(document_id)
    if not path.exists():
        raise ValueError(f"Extracted content not found for document_id={document_id}")
    with open(path, "rb") as handle:
        return pickle.load(handle)

def delete_extracted_pickle(document_id: str) -> bool:
    if blob_storage_configured():
        store = BlobStore()
        blob_name = store.build_blob_name(document_id)
        return store.delete_blob(blob_name)

    path = build_pickle_path(document_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete pickle for %s: %s", document_id, exc)
        return False
