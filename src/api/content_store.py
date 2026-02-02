import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError, ResourceExistsError

from src.storage.azure_blob_client import (
    CredentialError,
    extract_azure_error_details,
    get_azure_blob,
    has_blob_credentials,
)

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "document-content"
_TRUSTED_TYPE = "extracted_doc"


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


def _blob_prefix() -> str:
    prefix = os.getenv("DOCWAIN_BLOB_PREFIX", "")
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _build_blob_name(document_id: str) -> str:
    return f"{_blob_prefix()}{document_id}.pkl"


def _legacy_blob_name(document_id: str) -> str:
    return f"{_blob_prefix()}{document_id}.pickle"


def _lease_conflict_code(error_code: Optional[str]) -> bool:
    if not error_code:
        return False
    return error_code in {
        "leaseidmissing",
        "leasealreadypresent",
        "leasealreadyacquired",
        "leaseidmismatchwithleaseoperation",
        "leaselost",
    }


def save_extracted_pickle(document_id: str, extracted_obj: Any) -> Dict[str, Any]:
    if has_blob_credentials():
        azure_blob = get_azure_blob()
        container = azure_blob.document_container_name
        try:
            payload = pickle.dumps(extracted_obj, protocol=pickle.HIGHEST_PROTOCOL)
            blob_name = _build_blob_name(document_id)
            metadata = {
                "docwain_artifact": "true",
                "document_id": str(document_id),
                "type": _TRUSTED_TYPE,
                "version": "v1",
            }
            try:
                if azure_blob.blob_exists(container, blob_name):
                    with azure_blob.lease_guard(container, blob_name, duration=60, renew_every=30) as lease:
                        azure_blob.upload_bytes(
                            container,
                            blob_name,
                            payload,
                            overwrite=True,
                            metadata=metadata,
                            lease=lease,
                        )
                else:
                    azure_blob.upload_bytes(
                        container,
                        blob_name,
                        payload,
                        overwrite=True,
                        metadata=metadata,
                    )
            except (ResourceExistsError, HttpResponseError) as exc:
                error_code, request_id = extract_azure_error_details(exc)
                if isinstance(exc, ResourceExistsError) and not error_code:
                    error_code = "leasealreadypresent"
                if _lease_conflict_code(error_code):
                    logger.warning(
                        "Blob write failed due to active lease without lease id error_code=%s request_id=%s",
                        error_code,
                        request_id,
                    )
                    logger.warning("Falling back to local pickle storage due to blob lease conflict")
                else:
                    logger.error(
                        "Failed to persist extracted pickle to blob error_code=%s request_id=%s",
                        error_code,
                        request_id,
                    )
                raise
            except Exception:
                logger.error("Failed to persist extracted pickle to blob")
                raise
            return {
                "blob_name": blob_name,
                "path": blob_name,
                "etag": None,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        except CredentialError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to local pickle storage for %s", document_id)

    if not has_blob_credentials():
        logger.warning("Blob storage not configured; falling back to local pickle storage.")
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
    if has_blob_credentials():
        azure_blob = get_azure_blob()
        container = azure_blob.document_container_name
        blob_name = _build_blob_name(document_id)
        try:
            payload = azure_blob.download_bytes(container, blob_name)
            return pickle.loads(payload)
        except ResourceNotFoundError:
            legacy_blob = _legacy_blob_name(document_id)
            try:
                payload = azure_blob.download_bytes(container, legacy_blob)
            except ResourceNotFoundError as exc:
                raise ValueError(f"Extracted content not found for document_id={document_id}") from exc
            try:
                azure_blob.upload_bytes(
                    container,
                    blob_name,
                    payload,
                    overwrite=True,
                )
            except Exception:
                logger.warning("Legacy pickle migration failed for %s", document_id)
            return pickle.loads(payload)

    path = build_pickle_path(document_id)
    if not path.exists():
        raise ValueError(f"Extracted content not found for document_id={document_id}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def delete_extracted_pickle(document_id: str) -> bool:
    if has_blob_credentials():
        azure_blob = get_azure_blob()
        container = azure_blob.document_container_name
        blob_name = _build_blob_name(document_id)
        try:
            if azure_blob.blob_exists(container, blob_name):
                with azure_blob.lease_guard(container, blob_name, duration=60, renew_every=30) as lease:
                    return azure_blob.delete_blob(container, blob_name, lease=lease)
            return azure_blob.delete_blob(container, blob_name)
        except ResourceNotFoundError:
            return False
        except Exception:
            return False

    path = build_pickle_path(document_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete pickle for %s: %s", document_id, exc)
        return False
