from __future__ import annotations

import hashlib
import json
from src.utils.logging_utils import get_logger
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.storage.blob import BlobLeaseClient, ContentSettings

from src.storage.azure_blob_client import get_document_container_client, normalize_blob_name

logger = get_logger(__name__)

_LATEST_POINTER_NAME = "latest.json"
_LATEST_METADATA_KEY = "docwain_latest_run_id"
_DEFAULT_LEASE_SECONDS = 30
_DEFAULT_RETRY_SECONDS = 5

@dataclass(frozen=True)
class BlobWriteResult:
    blob_name: str
    size: int
    sha256: str
    etag: Optional[str] = None
    leased: bool = False
    versioned_blob_name: Optional[str] = None
    pointer_blob_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "blob_name": self.blob_name,
            "size": self.size,
            "sha256": self.sha256,
            "etag": self.etag,
            "leased": self.leased,
        }
        if self.versioned_blob_name:
            payload["versioned_blob_name"] = self.versioned_blob_name
        if self.pointer_blob_name:
            payload["pointer_blob_name"] = self.pointer_blob_name
        payload.setdefault("path", payload["blob_name"])
        return payload

def _document_id_from_blob_name(blob_name: str) -> str:
    base = blob_name.split("/")[-1]
    if base.endswith(".pickle"):
        base = base[: -len(".pickle")]
    elif base.endswith(".pkl"):
        base = base[: -len(".pkl")]
    return base

def _pointer_blob_name(document_id: str) -> str:
    return f"{document_id}/{_LATEST_POINTER_NAME}"

def _versioned_blob_name(document_id: str, run_id: str) -> str:
    return f"{document_id}/{run_id}.pkl"

def _ensure_blob_exists(blob_client) -> None:
    try:
        blob_client.get_blob_properties()
        return
    except ResourceNotFoundError:
        pass
    try:
        blob_client.upload_blob(b"", overwrite=False)
    except ResourceExistsError:
        return

def _acquire_lease(blob_client, lease_duration: int) -> Optional[str]:
    try:
        lease = blob_client.acquire_lease(lease_duration=lease_duration)
        return lease.id
    except ResourceNotFoundError:
        _ensure_blob_exists(blob_client)
        lease = blob_client.acquire_lease(lease_duration=lease_duration)
        return lease.id
    except HttpResponseError as exc:
        if str(getattr(exc, "error_code", "")).lower() in {"leasealreadypresent", "leasealreadyacquired"}:
            return None
        raise

def _release_lease_with_retry(
    blob_client, lease_id: str, document_id: Optional[str], blob_name: str, *, max_attempts: int = 3
) -> None:
    """Release a blob lease with retry logic and short backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            BlobLeaseClient(blob_client, lease_id=lease_id).release()
            logger.info(
                "Released blob lease document_id=%s blob=%s (attempt %d)",
                document_id, blob_name, attempt,
            )
            return
        except Exception as exc:  # noqa: BLE001
            if attempt < max_attempts:
                backoff = 0.5 * attempt
                logger.debug(
                    "Blob lease release attempt %d/%d failed for %s: %s (retrying in %.1fs)",
                    attempt, max_attempts, blob_name, exc, backoff,
                )
                time.sleep(backoff)
            else:
                logger.warning(
                    "Failed to release blob lease after %d attempts document_id=%s blob=%s lease_id=%s: %s",
                    max_attempts, document_id, blob_name, lease_id, exc,
                )

def _log_lease_state(blob_name: str, document_id: Optional[str], props: Any, *, attempt: int, outcome: str) -> None:
    lease = getattr(props, "lease", None)
    lease_status = getattr(lease, "status", None)
    lease_state = getattr(lease, "state", None)
    logger.info(
        "Blob lease state document_id=%s blob=%s lease_status=%s lease_state=%s attempt=%s outcome=%s",
        document_id,
        blob_name,
        lease_status,
        lease_state,
        attempt,
        outcome,
    )

def _upload_pointer_blob(container_client, pointer_blob: str, payload: Dict[str, Any]) -> None:
    blob_client = container_client.get_blob_client(pointer_blob)
    blob_client.upload_blob(
        json.dumps(payload).encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

def _write_versioned_blob(
    *,
    container_client,
    document_id: str,
    payload: bytes,
    metadata: Dict[str, str],
    content_type: str,
    sha256: str,
) -> BlobWriteResult:
    run_id = uuid.uuid4().hex
    versioned_blob = _versioned_blob_name(document_id, run_id)
    blob_client = container_client.get_blob_client(versioned_blob)
    meta = dict(metadata)
    meta[_LATEST_METADATA_KEY] = run_id
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata=meta,
        content_settings=ContentSettings(content_type=content_type),
    )
    try:
        props = blob_client.get_blob_properties()
        etag = getattr(props, "etag", None)
    except Exception:  # noqa: BLE001
        etag = None
    pointer_blob = _pointer_blob_name(document_id)
    _upload_pointer_blob(
        container_client,
        pointer_blob,
        {
            "document_id": document_id,
            "run_id": run_id,
            "blob_name": versioned_blob,
            "updated_at": time.time(),
        },
    )
    logger.info(
        "Saved versioned blob for document_id=%s blob=%s pointer=%s",
        document_id,
        versioned_blob,
        pointer_blob,
    )
    return BlobWriteResult(
        blob_name=versioned_blob,
        versioned_blob_name=versioned_blob,
        pointer_blob_name=pointer_blob,
        size=len(payload),
        sha256=sha256,
        etag=etag,
        leased=True,
    )

def load_pickle(blob_name: str) -> Optional[bytes]:
    container_client = get_document_container_client()
    normalized = normalize_blob_name(blob_name)
    max_attempts = int(os.getenv("DOCWAIN_BLOB_DOWNLOAD_RETRIES", "3"))
    base_delay = float(os.getenv("DOCWAIN_BLOB_DOWNLOAD_RETRY_SECONDS", "0.5"))
    blob_client = container_client.get_blob_client(normalized)
    for attempt in range(max_attempts):
        try:
            return blob_client.download_blob().readall()
        except ResourceNotFoundError:
            break
        except (ServiceResponseError, ServiceRequestError, HttpResponseError) as exc:
            if attempt >= max_attempts - 1:
                raise
            delay = base_delay * (attempt + 1) + random.random() * 0.2
            logger.warning(
                "Blob download transient error blob=%s attempt=%s/%s err=%s; retrying in %.2fs",
                normalized,
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            blob_client = container_client.get_blob_client(normalized)

    document_id = _document_id_from_blob_name(normalized)
    pointer_blob = _pointer_blob_name(document_id)
    pointer_client = container_client.get_blob_client(pointer_blob)
    try:
        pointer_payload = pointer_client.download_blob().readall()
    except ResourceNotFoundError:
        return None
    except (ServiceResponseError, ServiceRequestError, HttpResponseError) as exc:
        logger.warning("Pointer blob download failed for %s: %s", pointer_blob, exc)
        return None
    try:
        pointer_info = json.loads(pointer_payload.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    versioned_blob = pointer_info.get("blob_name")
    if not versioned_blob:
        return None
    versioned_client = container_client.get_blob_client(versioned_blob)
    for attempt in range(max_attempts):
        try:
            return versioned_client.download_blob().readall()
        except ResourceNotFoundError:
            return None
        except (ServiceResponseError, ServiceRequestError, HttpResponseError) as exc:
            if attempt >= max_attempts - 1:
                raise
            delay = base_delay * (attempt + 1) + random.random() * 0.2
            logger.warning(
                "Versioned blob download transient error blob=%s attempt=%s/%s err=%s; retrying in %.2fs",
                versioned_blob,
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            versioned_client = container_client.get_blob_client(versioned_blob)

def save_pickle_atomic(
    blob_name: str,
    payload: bytes,
    metadata: Dict[str, str],
    *,
    content_type: str = "application/octet-stream",
    lease_seconds: Optional[int] = None,
    retry_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    container_client = get_document_container_client()
    normalized = normalize_blob_name(blob_name)
    document_id = metadata.get("document_id") or _document_id_from_blob_name(normalized)
    lease_seconds = lease_seconds or int(os.getenv("DOCWAIN_BLOB_LEASE_SECONDS", _DEFAULT_LEASE_SECONDS))
    retry_seconds = retry_seconds or int(os.getenv("DOCWAIN_BLOB_LEASE_RETRY_SECONDS", _DEFAULT_RETRY_SECONDS))

    blob_client = container_client.get_blob_client(normalized)
    sha256 = metadata.get("sha256") or hashlib.sha256(payload).hexdigest()
    meta = {str(k): str(v) for k, v in metadata.items() if v is not None}
    meta.setdefault("document_id", str(document_id))

    lease_id: Optional[str] = None
    try:
        _ensure_blob_exists(blob_client)
        lease_id = _acquire_lease(blob_client, lease_seconds)
        if lease_id:
            logger.info(
                "Acquired blob lease document_id=%s blob=%s lease_id=%s",
                document_id,
                normalized,
                lease_id,
            )
        elif lease_id is None:
            logger.info(
                "Blob lease unavailable document_id=%s blob=%s; using versioned write",
                document_id,
                normalized,
            )
            return _write_versioned_blob(
                container_client=container_client,
                document_id=document_id,
                payload=payload,
                metadata=meta,
                content_type=content_type,
                sha256=sha256,
            ).to_dict()

        blob_client.upload_blob(
            payload,
            overwrite=True,
            metadata=meta,
            content_settings=ContentSettings(content_type=content_type),
            lease=lease_id,
        )
    except HttpResponseError as exc:
        error_code = str(getattr(exc, "error_code", "")).lower()
        if error_code == "leaseidmissing":
            try:
                props = blob_client.get_blob_properties()
                _log_lease_state(normalized, document_id, props, attempt=1, outcome="lease_id_missing")
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Blob lease inspection failed document_id=%s blob=%s",
                    document_id,
                    normalized,
                )
            deadline = time.monotonic() + retry_seconds
            attempt = 1
            while time.monotonic() < deadline:
                attempt += 1
                sleep_for = min(0.5 * (2 ** (attempt - 1)), 2.0) + random.uniform(0, 0.25)
                time.sleep(sleep_for)
                try:
                    blob_client.upload_blob(
                        payload,
                        overwrite=True,
                        metadata=meta,
                        content_settings=ContentSettings(content_type=content_type),
                        lease=lease_id,
                    )
                    break
                except HttpResponseError as retry_exc:  # noqa: BLE001
                    if str(getattr(retry_exc, "error_code", "")).lower() != "leaseidmissing":
                        raise
            else:
                return _write_versioned_blob(
                    container_client=container_client,
                    document_id=document_id,
                    payload=payload,
                    metadata=meta,
                    content_type=content_type,
                    sha256=sha256,
                ).to_dict()
        else:
            raise
    finally:
        if lease_id:
            _release_lease_with_retry(blob_client, lease_id, document_id, normalized)

    try:
        props = blob_client.get_blob_properties()
        etag = getattr(props, "etag", None)
    except Exception:  # noqa: BLE001
        etag = None
    return BlobWriteResult(
        blob_name=normalized,
        size=len(payload),
        sha256=sha256,
        etag=etag,
        leased=bool(lease_id),
    ).to_dict()

__all__ = ["load_pickle", "save_pickle_atomic", "BlobWriteResult"]
