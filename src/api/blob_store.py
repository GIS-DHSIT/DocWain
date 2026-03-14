"""Azure Blob helpers for extracted document pickles.

Blob contract (expected pickle payload):
- Top-level object is either:
  - ExtractedDocument (see src/api/pipeline_models.py), or
  - dict[str, ExtractedDocument | dict | str], keyed by source filename.
- Structured dict payloads may include keys like "texts", "embeddings",
  "chunk_metadata", "full_text", "metadata", and related extraction fields.
- Blob metadata is expected to include:
  - docwain_artifact=true
  - document_id=<doc id>
  - type=extracted_doc
  - version=v1
  - optional subscription_id/profile_id/doc_type.
"""

from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from azure.core.exceptions import HttpResponseError, ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from src.api.config import Config
from src.storage.azure_blob_client import get_blob_service_client, get_document_container_client, has_blob_credentials
from src.storage.blob_persistence import save_pickle_atomic

logger = get_logger(__name__)

_DEFAULT_CONTAINER = "document-content"
_DEFAULT_PREFIX = ""
_TRUSTED_TYPE = "extracted_doc"

class BlobConfigurationError(ValueError):
    """Raised when blob auth config is missing or invalid."""

@dataclass
class BlobInfo:
    name: str
    size: Optional[int] = None
    metadata: Dict[str, str] = None
    etag: Optional[str] = None
    last_modified: Optional[Any] = None
    content_type: Optional[str] = None

def blob_storage_configured() -> bool:
    return has_blob_credentials()

def _normalize_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"

def _build_service_client() -> BlobServiceClient:
    try:
        return get_blob_service_client()
    except Exception as exc:  # noqa: BLE001
        raise BlobConfigurationError(str(exc)) from exc

class BlobStore:
    def __init__(
        self,
        *,
        container: Optional[str] = None,
        prefix: Optional[str] = None,
        service_client: Optional[BlobServiceClient] = None,
    ) -> None:
        default_container = getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", _DEFAULT_CONTAINER)
        self.container = container or default_container
        self.prefix = _normalize_prefix(prefix if prefix is not None else os.getenv("DOCWAIN_BLOB_PREFIX", _DEFAULT_PREFIX))
        if service_client is not None or container is not None:
            self._service_client = service_client or _build_service_client()
            self._container_client = self._service_client.get_container_client(self.container)
        else:
            self._service_client = None
            self._container_client = get_document_container_client()

    @property
    def container_client(self):
        return self._container_client

    def build_blob_name(self, document_id: str, extension: str = ".pkl", prefix_override: Optional[str] = None) -> str:
        prefix = self.prefix
        if prefix_override:
            prefix = f"{prefix}{_normalize_prefix(prefix_override)}"
        return f"{prefix}{document_id}{extension}"

    def get_blob_info(self, blob_name: str) -> Optional[BlobInfo]:
        blob_client = self._container_client.get_blob_client(blob_name)
        try:
            props = blob_client.get_blob_properties()
        except ResourceNotFoundError:
            return None
        metadata = getattr(props, "metadata", {}) or {}
        return BlobInfo(
            name=blob_name,
            size=getattr(props, "size", None),
            metadata=metadata,
            etag=getattr(props, "etag", None),
            last_modified=getattr(props, "last_modified", None),
            content_type=getattr(props, "content_settings", None).content_type
            if getattr(props, "content_settings", None)
            else None,
        )

    def list_blobs(
        self,
        *,
        name_prefix: Optional[str] = None,
        include_metadata: bool = True,
        limit: Optional[int] = None,
    ) -> List[BlobInfo]:
        prefix = f"{self.prefix}{_normalize_prefix(name_prefix)}" if name_prefix else self.prefix
        include = ["metadata"] if include_metadata else None
        results: List[BlobInfo] = []
        for blob in self._container_client.list_blobs(name_starts_with=prefix, include=include):
            results.append(
                BlobInfo(
                    name=blob.name,
                    size=getattr(blob, "size", None),
                    metadata=getattr(blob, "metadata", {}) or {},
                    etag=getattr(blob, "etag", None),
                    last_modified=getattr(blob, "last_modified", None),
                    content_type=getattr(blob, "content_settings", None).content_type
                    if getattr(blob, "content_settings", None)
                    else None,
                )
            )
            if limit and len(results) >= limit:
                break
        return results

    def list_pickle_blobs(
        self,
        *,
        name_prefix: Optional[str] = None,
        extensions: Iterable[str] = (".pkl", ".pickle"),
        limit: Optional[int] = None,
    ) -> List[BlobInfo]:
        normalized_ext = tuple(ext.lower() for ext in extensions)
        blobs = self.list_blobs(name_prefix=name_prefix, include_metadata=True, limit=limit)
        return [blob for blob in blobs if blob.name.lower().endswith(normalized_ext)]

    def upload_pickle(
        self,
        document_id: str,
        extracted_obj: Any,
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = pickle.dumps(extracted_obj, protocol=pickle.HIGHEST_PROTOCOL)
        blob_name = self.build_blob_name(document_id)
        meta = {
            "docwain_artifact": "true",
            "document_id": str(document_id),
            "type": _TRUSTED_TYPE,
            "version": "v1",
        }
        if metadata:
            meta.update({str(k): str(v) for k, v in metadata.items() if v is not None})
        sha256 = hashlib.sha256(payload).hexdigest()
        meta["sha256"] = sha256
        return save_pickle_atomic(
            blob_name,
            payload,
            meta,
            content_type="application/octet-stream",
        )

    def download_blob(self, blob_name: str, *, lease: Optional[str] = None) -> bytes:
        blob_client = self._container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob(lease=lease)
        return downloader.readall()

    def try_acquire_lease(self, blob_name: str, lease_duration: int = 60) -> Optional[str]:
        blob_client = self._container_client.get_blob_client(blob_name)
        try:
            lease = blob_client.acquire_lease(lease_duration=lease_duration)
            return lease.id
        except ResourceNotFoundError:
            try:
                blob_client.upload_blob(b"", overwrite=False)
            except ResourceExistsError:
                pass
            lease = blob_client.acquire_lease(lease_duration=lease_duration)
            return lease.id
        except ResourceExistsError:
            return None
        except HttpResponseError as exc:
            if str(getattr(exc, "error_code", "")).lower() in {"leasealreadypresent", "leasealreadyacquired"}:
                return None
            raise

    def release_lease(self, blob_name: str, lease_id: Optional[str]) -> bool:
        if not lease_id:
            return False
        blob_client = self._container_client.get_blob_client(blob_name)
        try:
            blob_client.release_lease(lease=lease_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def delete_blob(self, blob_name: str, *, lease: Optional[str] = None) -> bool:
        blob_client = self._container_client.get_blob_client(blob_name)
        try:
            blob_client.delete_blob(lease=lease)
            return True
        except ResourceNotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete blob %s: %s", blob_name, exc)
            return False

def is_trusted_blob(blob: BlobInfo, *, expected_prefix: str = "") -> bool:
    name = blob.name or ""
    if expected_prefix and not name.startswith(expected_prefix):
        logger.warning("Blob name %s does not match expected prefix %s", name, expected_prefix)
        return False
    metadata = blob.metadata or {}
    artifact_flag = metadata.get("docwain_artifact")
    if artifact_flag is None:
        logger.warning("Blob %s missing docwain_artifact tag", name)
    elif str(artifact_flag).lower() != "true":
        logger.warning("Blob %s has docwain_artifact=%s", name, artifact_flag)
        return False
    if metadata.get("type") and metadata.get("type") != _TRUSTED_TYPE:
        logger.warning("Blob %s has unexpected type=%s", name, metadata.get("type"))
        return False
    return True

def extract_document_id(blob_name: str, *, prefix: str = "") -> str:
    base = blob_name
    if prefix and base.startswith(prefix):
        base = base[len(prefix):]
    if base.endswith(".pickle"):
        base = base[: -len(".pickle")]
    elif base.endswith(".pkl"):
        base = base[: -len(".pkl")]
    return base.rsplit("/", 1)[-1]
