from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings

from src.api.config import Config

logger = logging.getLogger(__name__)

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
_TRUSTED_TYPE = "extracted_doc"

_SERVICE_CLIENT: Optional[BlobServiceClient] = None
_CHAT_CONTAINER_CLIENT: Optional[ContainerClient] = None
_DOCUMENT_CONTAINER_CLIENT: Optional[ContainerClient] = None
_CONTAINERS_VALIDATED = False


def _extract_error_code(exc: BaseException) -> str:
    return str(getattr(exc, "error_code", "") or "").lower()


def _coerce_lease_id(lease: Optional[Any]) -> Optional[str]:
    if lease is None:
        return None
    if isinstance(lease, str):
        return lease
    lease_id = getattr(lease, "id", None) or getattr(lease, "lease_id", None)
    return str(lease_id) if lease_id else None


def _log_azure_error(message: str, exc: BaseException) -> None:
    if isinstance(exc, HttpResponseError):
        logger.warning(
            "%s error_code=%s request_id=%s",
            message,
            _extract_error_code(exc),
            _extract_request_id(exc),
        )
    else:
        logger.warning("%s", message)


class BlobDownloadError(RuntimeError):
    error_code = "UnknownError"

    def __init__(
        self,
        message: str,
        *,
        document_id: Optional[str] = None,
        blob_name: Optional[str] = None,
        request_id: Optional[str] = None,
        original_exc: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.document_id = document_id
        self.blob_name = blob_name
        self.request_id = request_id
        self.original_exc = original_exc


class CredentialError(BlobDownloadError):
    error_code = "CredentialError"


class NotFoundError(BlobDownloadError):
    error_code = "NotFoundError"


class TransientError(BlobDownloadError):
    error_code = "TransientError"


class UnknownError(BlobDownloadError):
    error_code = "UnknownError"


@dataclass
class BlobInfo:
    name: str
    size: Optional[int] = None
    metadata: Dict[str, str] = None
    etag: Optional[str] = None
    last_modified: Optional[Any] = None
    content_type: Optional[str] = None


class AzureBlob:
    def __init__(self) -> None:
        self._container_clients: Dict[str, ContainerClient] = {}
        self._lock = threading.Lock()

    @property
    def chat_container_name(self) -> str:
        return getattr(Config.AzureBlob, "CONTAINER_NAME", "")

    @property
    def document_container_name(self) -> str:
        return getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "")

    def get_container_client(self, container_name: str) -> ContainerClient:
        if not container_name:
            raise CredentialError("AzureBlob container name is missing.")
        with self._lock:
            client = self._container_clients.get(container_name)
            if client is None:
                client = get_container_client(container_name)
                self._container_clients[container_name] = client
        return client

    def get_chat_container_client(self) -> ContainerClient:
        return self.get_container_client(self.chat_container_name)

    def get_document_container_client(self) -> ContainerClient:
        return self.get_container_client(self.document_container_name)

    def blob_exists(self, container: str, blob_name: str) -> bool:
        blob_client = self.get_container_client(container).get_blob_client(blob_name)
        try:
            if hasattr(blob_client, "exists"):
                return bool(blob_client.exists())
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception:  # noqa: BLE001
            return False

    def get_blob_info(self, container: str, blob_name: str) -> Optional[BlobInfo]:
        blob_client = self.get_container_client(container).get_blob_client(blob_name)
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
        container: str,
        *,
        prefix: str = "",
        include_metadata: bool = True,
        limit: Optional[int] = None,
    ) -> list[BlobInfo]:
        container_client = self.get_container_client(container)
        include = ["metadata"] if include_metadata else None
        results: list[BlobInfo] = []
        for blob in container_client.list_blobs(name_starts_with=prefix, include=include):
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

    def download_bytes(self, container: str, blob_name: str) -> bytes:
        blob_client = self.get_container_client(container).get_blob_client(blob_name)
        return blob_client.download_blob().readall()

    def upload_bytes(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        overwrite: bool = True,
        metadata: Optional[dict] = None,
        *,
        content_type: str = "application/octet-stream",
        lease: Optional[Any] = None,
    ) -> None:
        blob_client = self.get_container_client(container).get_blob_client(blob_name)
        kwargs: Dict[str, Any] = {
            "overwrite": overwrite,
            "metadata": metadata,
            "content_settings": ContentSettings(content_type=content_type),
        }
        lease_id = _coerce_lease_id(lease)
        if lease_id:
            kwargs["lease"] = lease_id
        try:
            blob_client.upload_blob(data, **kwargs)
            return
        except HttpResponseError as exc:
            if _extract_error_code(exc) != "leaseidmissing":
                raise
            _log_azure_error("Blob write failed due to active lease without lease id", exc)

        retry_lease = None
        try:
            retry_lease = self.acquire_lease(container, blob_name, lease_duration=60)
            retry_kwargs = dict(kwargs)
            retry_kwargs["lease"] = _coerce_lease_id(retry_lease)
            blob_client.upload_blob(data, **retry_kwargs)
        finally:
            if retry_lease:
                self.release_lease(retry_lease)

    def delete_blob(
        self,
        container: str,
        blob_name: str,
        *,
        include_snapshots: bool = False,
        lease: Optional[Any] = None,
    ) -> bool:
        blob_client = self.get_container_client(container).get_blob_client(blob_name)
        delete_snapshots = "include" if include_snapshots else None
        lease_id = _coerce_lease_id(lease)
        try:
            blob_client.delete_blob(lease=lease_id, delete_snapshots=delete_snapshots)
            return True
        except ResourceNotFoundError:
            return False
        except HttpResponseError as exc:
            if _extract_error_code(exc) != "leaselost":
                raise
            _log_azure_error("Blob delete lease lost; retrying", exc)

        try:
            blob_client.delete_blob(delete_snapshots=delete_snapshots)
            return True
        except HttpResponseError:
            pass

        retry_lease = None
        try:
            retry_lease = self.acquire_lease(container, blob_name, lease_duration=60)
            blob_client.delete_blob(
                lease=_coerce_lease_id(retry_lease),
                delete_snapshots=delete_snapshots,
            )
            return True
        finally:
            if retry_lease:
                self.release_lease(retry_lease)

    def acquire_lease(self, container: str, blob_name: str, lease_duration: int = 60):
        blob_client = self.get_container_client(container).get_blob_client(blob_name)
        return blob_client.acquire_lease(lease_duration=lease_duration)

    def renew_lease(self, lease) -> None:
        if lease is None:
            return
        lease.renew()

    def release_lease(self, lease) -> None:
        if lease is None:
            return
        try:
            lease.release()
        except ResourceNotFoundError:
            return
        except HttpResponseError as exc:
            if _extract_error_code(exc) in {"blobnotfound", "leaseidmissing", "leasenotpresent", "leasedalreadybroken"}:
                return
            raise

    @contextlib.contextmanager
    def lease_guard(
        self,
        container: str,
        blob_name: str,
        *,
        duration: int = 60,
        renew_every: int = 30,
    ):
        lease = None
        stop_event = threading.Event()
        renew_thread: Optional[threading.Thread] = None

        try:
            lease = self.acquire_lease(container, blob_name, lease_duration=duration)
            if duration != -1 and renew_every and renew_every > 0:
                def _renew_loop():
                    while not stop_event.wait(renew_every):
                        try:
                            self.renew_lease(lease)
                        except Exception as exc:  # noqa: BLE001
                            _log_azure_error("Lease renew failed", exc)
                            break

                renew_thread = threading.Thread(target=_renew_loop, daemon=True)
                renew_thread.start()
            yield lease
        finally:
            stop_event.set()
            if renew_thread:
                renew_thread.join(timeout=1)
            if lease:
                try:
                    self.release_lease(lease)
                except Exception as exc:  # noqa: BLE001
                    _log_azure_error("Lease release failed", exc)


_AZURE_BLOB: Optional[AzureBlob] = None


def get_azure_blob() -> AzureBlob:
    global _AZURE_BLOB
    if _AZURE_BLOB is None:
        _AZURE_BLOB = AzureBlob()
    return _AZURE_BLOB

def _pad_base64(value: str) -> str:
    remainder = len(value) % 4
    if remainder == 0:
        return value
    return value + ("=" * (4 - remainder))


def validate_base64_padding(value: str, *, label: str = "AZURE_STORAGE_ACCOUNT_KEY") -> None:
    if not value:
        raise CredentialError(f"{label} is missing.")
    if not _BASE64_RE.fullmatch(value):
        raise CredentialError(f"{label} contains invalid characters; expected base64.")

    if len(value) % 4 != 0:
        padded = _pad_base64(value)
        try:
            base64.b64decode(padded, validate=True)
        except binascii.Error as exc:
            raise CredentialError(f"{label} is not valid base64.") from exc
        raise CredentialError(
            f"{label} looks base64 but is missing '=' padding. "
            "Ensure the full account key is provided."
        )

    try:
        base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise CredentialError(f"{label} is not valid base64.") from exc


def decode_connection_string_if_base64(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    if "DefaultEndpointsProtocol=" in raw:
        return raw
    if ";" in raw or "AccountName=" in raw or "AccountKey=" in raw:
        return raw
    if not _BASE64_RE.fullmatch(raw):
        return raw

    padded = _pad_base64(raw)
    try:
        decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return raw
    if "DefaultEndpointsProtocol=" in decoded:
        return decoded.strip()
    return raw


def sanitize_blob_url(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def normalize_blob_name(file_path: str, *, container_name: Optional[str] = None) -> str:
    blob_name = (file_path or "").strip()
    if blob_name.startswith("az://"):
        parsed = urlparse(blob_name)
        netloc = parsed.netloc
        raw_path = parsed.path.lstrip("/")
        if container_name:
            if netloc == container_name and raw_path:
                return raw_path
            if raw_path.startswith(f"{container_name}/"):
                return raw_path[len(container_name) + 1 :]
        return raw_path or netloc
    return blob_name


def iter_blob_name_candidates(blob_name: str) -> Iterable[str]:
    seen = set()

    def _emit(value: str):
        if value and value not in seen:
            seen.add(value)
            return value
        return None

    first = _emit(blob_name)
    if first:
        yield first

    if "%" in blob_name:
        decoded = unquote(blob_name)
        next_value = _emit(decoded)
        if next_value:
            yield next_value

    if " " in blob_name:
        encoded = quote(blob_name, safe="/")
        next_value = _emit(encoded)
        if next_value:
            yield next_value


def _parse_connection_string(raw: str) -> dict:
    parts = {}
    for segment in raw.split(";"):
        if not segment:
            continue
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        parts[key] = value
    return parts


def has_blob_credentials() -> bool:
    return bool(getattr(Config.AzureBlob, "CONNECTION_STRING", "").strip())


def _connection_string() -> str:
    try:
        Config.AzureBlob.validate()
    except Exception as exc:  # noqa: BLE001
        raise CredentialError(str(exc)) from exc
    return getattr(Config.AzureBlob, "CONNECTION_STRING", "").strip()


def get_blob_service_client() -> BlobServiceClient:
    global _SERVICE_CLIENT
    if _SERVICE_CLIENT is not None:
        return _SERVICE_CLIENT

    raw_value = _connection_string()
    conn_str = decode_connection_string_if_base64(raw_value)
    if conn_str != raw_value:
        logger.info("Azure blob auth: decoded base64 AzureBlob.CONNECTION_STRING")
    logger.info("Azure blob auth: using AzureBlob.CONNECTION_STRING")

    account_key = _parse_connection_string(conn_str).get("AccountKey")
    if account_key:
        validate_base64_padding(account_key.strip(), label="AzureBlob.CONNECTION_STRING AccountKey")

    try:
        _SERVICE_CLIENT = BlobServiceClient.from_connection_string(conn_str)
    except Exception as exc:  # noqa: BLE001
        raise CredentialError("AzureBlob.CONNECTION_STRING is not a valid Azure storage connection string.") from exc
    return _SERVICE_CLIENT


def get_container_client(container_name: str) -> ContainerClient:
    service_client = get_blob_service_client()
    return service_client.get_container_client(container_name)


def get_chat_container_client() -> ContainerClient:
    global _CHAT_CONTAINER_CLIENT
    if _CHAT_CONTAINER_CLIENT is None:
        container_name = getattr(Config.AzureBlob, "CONTAINER_NAME", "")
        if not container_name:
            raise CredentialError("AzureBlob.CONTAINER_NAME is missing.")
        logger.info("Chat container: %s", container_name)
        _CHAT_CONTAINER_CLIENT = get_azure_blob().get_container_client(container_name)
    return _CHAT_CONTAINER_CLIENT


def get_document_container_client() -> ContainerClient:
    global _DOCUMENT_CONTAINER_CLIENT
    if _DOCUMENT_CONTAINER_CLIENT is None:
        container_name = getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "")
        if not container_name:
            raise CredentialError("AzureBlob.DOCUMENT_CONTAINER_NAME is missing.")
        logger.info("Document container: %s", container_name)
        _DOCUMENT_CONTAINER_CLIENT = get_azure_blob().get_container_client(container_name)
    return _DOCUMENT_CONTAINER_CLIENT


def validate_containers_once() -> None:
    """Validate configured containers exist, logging once without raising."""
    global _CONTAINERS_VALIDATED
    if _CONTAINERS_VALIDATED:
        return
    _CONTAINERS_VALIDATED = True

    for name in (
        getattr(Config.AzureBlob, "CONTAINER_NAME", ""),
        getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", ""),
    ):
        if not name:
            logger.warning("Configured container is empty; please verify Azure Blob setup.")
            continue
        try:
            container = get_container_client(name)
            exists = container.exists()
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, HttpResponseError):
                logger.warning(
                    "Container check failed for %s error_code=%s request_id=%s",
                    name,
                    _extract_error_code(exc),
                    _extract_request_id(exc),
                )
            else:
                logger.warning("Container check failed for %s", name)
            continue
        if not exists:
            logger.warning("Configured container %s not found. Please verify Azure Blob setup.", name)


def upload_chat_history(blob_name: str, payload: bytes) -> None:
    azure_blob = get_azure_blob()
    azure_blob.upload_bytes(
        azure_blob.chat_container_name,
        blob_name,
        payload,
        overwrite=True,
        content_type="application/json",
    )


def upload_pickle(
    blob_name: str,
    payload: bytes,
    *,
    metadata: Optional[dict] = None,
) -> None:
    azure_blob = get_azure_blob()
    azure_blob.upload_bytes(
        azure_blob.document_container_name,
        blob_name,
        payload,
        overwrite=True,
        metadata=metadata,
        content_type="application/octet-stream",
    )


def download_pickle(blob_name: str) -> bytes:
    azure_blob = get_azure_blob()
    return azure_blob.download_bytes(azure_blob.document_container_name, blob_name)


def _extract_request_id(exc: BaseException) -> Optional[str]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    return headers.get("x-ms-request-id") or headers.get("x-ms-client-request-id")


def classify_blob_error(
    exc: BaseException,
    *,
    document_id: Optional[str],
    blob_name: Optional[str],
) -> BlobDownloadError:
    request_id = _extract_request_id(exc)
    message = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, CredentialError):
        return exc
    if isinstance(exc, ResourceNotFoundError):
        return NotFoundError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
    if isinstance(exc, ClientAuthenticationError):
        return CredentialError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
    if isinstance(exc, ServiceRequestError):
        return TransientError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
    if isinstance(exc, HttpResponseError):
        status_code = getattr(exc, "status_code", None)
        if status_code in {401, 403}:
            return CredentialError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
        if status_code == 404:
            return NotFoundError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
        if status_code in {408, 429, 500, 502, 503, 504}:
            return TransientError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
        return UnknownError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)
    if isinstance(exc, binascii.Error) or "Incorrect padding" in str(exc):
        return CredentialError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)

    return UnknownError(message, document_id=document_id, blob_name=blob_name, request_id=request_id, original_exc=exc)


def extract_azure_error_details(exc: BaseException) -> tuple[Optional[str], Optional[str]]:
    if isinstance(exc, HttpResponseError):
        return _extract_error_code(exc), _extract_request_id(exc)
    return None, None


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
        base = base[len(prefix) :]
    if base.endswith(".pkl"):
        base = base[: -len(".pkl")]
    return base.rsplit("/", 1)[-1]
