from __future__ import annotations

import base64
import binascii
from src.utils.logging_utils import get_logger
import re
from typing import Iterable, Optional
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceNotFoundError, ServiceRequestError
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings

from src.api.config import Config

logger = get_logger(__name__)

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")

_SERVICE_CLIENT: Optional[BlobServiceClient] = None
_CHAT_CONTAINER_CLIENT: Optional[ContainerClient] = None
_DOCUMENT_CONTAINER_CLIENT: Optional[ContainerClient] = None
_CONTAINERS_VALIDATED = False
_STORAGE_CONFIG_VALIDATED = False

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
        _CHAT_CONTAINER_CLIENT = get_blob_service_client().get_container_client(container_name)
    return _CHAT_CONTAINER_CLIENT

def get_document_container_client() -> ContainerClient:
    global _DOCUMENT_CONTAINER_CLIENT
    if _DOCUMENT_CONTAINER_CLIENT is None:
        container_name = getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "")
        if not container_name:
            raise CredentialError("AzureBlob.DOCUMENT_CONTAINER_NAME is missing.")
        logger.info("Document container: %s", container_name)
        _DOCUMENT_CONTAINER_CLIENT = get_blob_service_client().get_container_client(container_name)
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
            logger.debug("Configured container is empty; please verify Azure Blob setup.")
            continue
        try:
            container = get_container_client(name)
            exists = container.exists()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Container check failed for %s: %s", name, exc)
            continue
        if not exists:
            logger.debug("Configured container %s not found. Please verify Azure Blob setup.", name)

def validate_storage_configured_once() -> bool:
    """Validate blob credentials once and log a single actionable message."""
    global _STORAGE_CONFIG_VALIDATED
    if _STORAGE_CONFIG_VALIDATED:
        return has_blob_credentials()
    _STORAGE_CONFIG_VALIDATED = True
    if not has_blob_credentials():
        logger.debug("Azure blob storage is not configured; set AzureBlob.CONNECTION_STRING to enable storage.")
        return False
    logger.info("Azure blob storage configured.")
    return True

def upload_chat_history(blob_name: str, payload: bytes) -> None:
    container_client = get_chat_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        payload,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

def upload_pickle(
    blob_name: str,
    payload: bytes,
    *,
    metadata: Optional[dict] = None,
) -> None:
    container_client = get_document_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata=metadata,
        content_settings=ContentSettings(content_type="application/octet-stream"),
    )

def download_pickle(blob_name: str) -> bytes:
    container_client = get_document_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()

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
