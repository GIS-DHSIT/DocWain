import asyncio
import hashlib
import logging
import mimetypes
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from src.api.config import Config
from src.api.dataHandler import fileProcessor, train_on_document
from src.teams.logic import TeamsChatContext
from src.teams.state import TeamsStateStore
from src.utils.logging_utils import get_logger

try:
    from azure.storage.blob import ContentSettings

    _BLOB_AVAILABLE = True
    _BLOB_WARNING_EMITTED = False
    _BLOB_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:
    ContentSettings = None  # type: ignore
    _BLOB_AVAILABLE = False
    _BLOB_WARNING_EMITTED = False
    _BLOB_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


class AttachmentIngestError(Exception):
    """Raised when a Teams attachment cannot be ingested."""


@dataclass
class AttachmentOutcome:
    filename: str
    documents_created: int
    doc_tag: str


@dataclass
class IngestionResult:
    filenames: List[str]
    documents_created: int
    doc_tags: List[str]


def _attachment_type(attachment: Dict[str, Any]) -> str:
    return (attachment.get("contentType") or "").lower()


def _blob_uploads_enabled(log: logging.LoggerAdapter) -> bool:
    if not _BLOB_AVAILABLE:
        global _BLOB_WARNING_EMITTED
        if not _BLOB_WARNING_EMITTED:
            log.warning(
                "azure-storage-blob is not installed; Teams attachments will be processed "
                "but not uploaded to blob storage. Install azure-storage-blob to enable uploads."
            )
            _BLOB_WARNING_EMITTED = True
        return False
    return True


def _blob_configured() -> bool:
    return bool(getattr(Config.AzureBlob, "CONNECTION_STRING", "")) and bool(getattr(Config.Teams, "BLOB_CONTAINER", ""))


def _build_blob_name(filename: str, subscription_id: str) -> str:
    prefix = getattr(Config.Teams, "BLOB_PATH_PREFIX", "") or getattr(Config.Teams, "UPLOAD_DIR", "")
    blob_name_parts = [prefix.strip("/"), subscription_id.strip("/"), filename]
    return "/".join(part for part in blob_name_parts if part)


def _upload_to_blob(file_bytes: bytes, filename: str, subscription_id: str, log: logging.LoggerAdapter) -> None:
    if not _blob_uploads_enabled(log):
        return
    if not _blob_configured():
        log.info("Teams blob storage is not configured; skipping upload.")
        return

    container_name = getattr(Config.Teams, "BLOB_CONTAINER", "")
    blob_name = _build_blob_name(filename, subscription_id)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        from src.storage.azure_blob_client import get_blob_service_client

        service = get_blob_service_client()
        container = service.get_container_client(container_name)
        try:
            container.create_container()
        except Exception:
            pass
        container.upload_blob(
            name=blob_name,
            data=file_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type) if ContentSettings else None,
        )
        log.info("Uploaded Teams attachment to blob storage at %s/%s", container_name, blob_name)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to upload Teams attachment to blob storage: %s", exc, exc_info=True)


async def _download_bytes(
    url: str,
    headers: Dict[str, str],
    timeout: float,
    retries: int,
    max_bytes: int,
) -> bytes:
    backoff = 0.5
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    content_length = resp.headers.get("Content-Length")
                    if content_length and int(content_length) > max_bytes:
                        raise ValueError("Attachment exceeds size limit")
                    data = bytearray()
                    async for chunk in resp.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            raise ValueError("Attachment exceeds size limit")
                    return bytes(data)
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            if attempt >= retries:
                raise exc
            await asyncio.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Download failed unexpectedly")


async def _resolve_auth_token(turn_context, provided_token: Optional[str]) -> str:
    if provided_token:
        return provided_token
    # Prefer adapter-issued connector token
    if turn_context:
        try:
            token = await turn_context.adapter.get_access_token()  # type: ignore[attr-defined]
            if token:
                return token
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unable to resolve connector token from adapter: %s", exc)
    fallback = getattr(Config.Teams, "BOT_ACCESS_TOKEN", "") or ""
    if fallback:
        return fallback
    raise AttachmentIngestError("Missing connector token for secure file download.")


def _resolve_doc_tag(attachment: Dict[str, Any], filename: str) -> str:
    content = attachment.get("content") or {}
    return (
        content.get("uniqueId")
        or content.get("id")
        or content.get("fileType")
        or hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    )


async def _process_file_download(
    attachment: Dict[str, Any],
    context: TeamsChatContext,
    log: logging.LoggerAdapter,
    auth_token: str,
    timeout: float,
    retries: int,
    max_bytes: int,
) -> Optional[AttachmentOutcome]:
    content = attachment.get("content") or {}
    download_url = content.get("downloadUrl") or content.get("download_url")
    filename = (
        content.get("fileName")
        or content.get("name")
        or attachment.get("name")
        or f"teams-upload-{uuid.uuid4()}"
    )
    if not download_url:
        raise AttachmentIngestError("Attachment missing a secure download URL.")

    headers = {"Authorization": f"Bearer {auth_token}"}
    file_bytes = await _download_bytes(download_url, headers=headers, timeout=timeout, retries=retries, max_bytes=max_bytes)
    extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
    if not extracted_docs:
        log.warning("No extractable content found for attachment %s", filename)
        return None

    doc_tag = _resolve_doc_tag(attachment, filename)
    documents_created = 0
    for doc_name, doc_content in extracted_docs.items():
        await asyncio.to_thread(
            train_on_document,
            doc_content,
            subscription_id=context.subscription_id,
            profile_id=context.profile_id,
            doc_tag=str(doc_tag),
            doc_name=doc_name,
        )
        documents_created += 1

    _upload_to_blob(file_bytes, filename, context.subscription_id, log)
    return AttachmentOutcome(filename=filename, documents_created=documents_created, doc_tag=str(doc_tag))


async def _process_inline_image(
    attachment: Dict[str, Any],
    context: TeamsChatContext,
    log: logging.LoggerAdapter,
    auth_token: str,
    timeout: float,
    retries: int,
    max_bytes: int,
) -> Optional[AttachmentOutcome]:
    content_url = attachment.get("contentUrl")
    content_type = _attachment_type(attachment)
    if not content_url:
        log.warning("Inline image missing contentUrl: %s", attachment)
        return None

    headers = {"Authorization": f"Bearer {auth_token}"}
    filename = attachment.get("name") or f"teams-image-{uuid.uuid4()}"
    file_bytes = await _download_bytes(content_url, headers=headers, timeout=timeout, retries=retries, max_bytes=max_bytes)
    extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
    if not extracted_docs:
        log.warning("No extractable content found for inline image %s", filename)
        return None

    documents_created = 0
    doc_tag = hashlib.sha256(content_url.encode("utf-8")).hexdigest()
    for doc_name, doc_content in extracted_docs.items():
        await asyncio.to_thread(
            train_on_document,
            doc_content,
            subscription_id=context.subscription_id,
            profile_id=context.profile_id,
            doc_tag=doc_tag,
            doc_name=doc_name,
        )
        documents_created += 1
    _upload_to_blob(file_bytes, filename, context.subscription_id, log)
    log.info("Processed inline image %s (%s)", filename, content_type)
    return AttachmentOutcome(filename=filename, documents_created=documents_created, doc_tag=doc_tag)


async def ingest_attachments(
    activity: Dict[str, Any],
    turn_context,
    context: TeamsChatContext,
    correlation_id: str,
    state_store: Optional[TeamsStateStore] = None,
    connector_token: Optional[str] = None,
) -> IngestionResult:
    attachments = activity.get("attachments") or []
    log = get_logger(__name__, correlation_id)
    if not attachments:
        raise AttachmentIngestError("No attachments found to ingest.")

    timeout = float(getattr(Config.Teams, "HTTP_TIMEOUT_SEC", 20))
    retries = int(getattr(Config.Teams, "HTTP_RETRIES", 2))
    max_bytes = int(getattr(Config.Teams, "MAX_ATTACHMENT_MB", 50)) * 1024 * 1024
    token = await _resolve_auth_token(turn_context, connector_token)

    outcomes: List[AttachmentOutcome] = []
    errors: List[str] = []

    for attachment in attachments:
        content_type = _attachment_type(attachment)
        try:
            if "file.download.info" in content_type or (attachment.get("content") or {}).get("downloadUrl"):
                outcome = await _process_file_download(
                    attachment,
                    context,
                    log,
                    auth_token=token,
                    timeout=timeout,
                    retries=retries,
                    max_bytes=max_bytes,
                )
            elif content_type.startswith("image/") or attachment.get("contentUrl"):
                outcome = await _process_inline_image(
                    attachment,
                    context,
                    log,
                    auth_token=token,
                    timeout=timeout,
                    retries=retries,
                    max_bytes=max_bytes,
                )
            else:
                errors.append(f"Unsupported attachment type: {content_type or 'unknown'}.")
                continue

            if outcome:
                outcomes.append(outcome)
                if state_store:
                    state_store.record_upload(
                        context.subscription_id,
                        context.profile_id,
                        outcome.filename,
                        outcome.doc_tag,
                        outcome.documents_created,
                    )
            else:
                errors.append("Attachment had no extractable content.")
        except AttachmentIngestError as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.error("Attachment processing failed: %s", exc, exc_info=True)
            errors.append("An attachment failed to process. Please try again.")

    if outcomes:
        return IngestionResult(
            filenames=[out.filename for out in outcomes],
            documents_created=sum(out.documents_created for out in outcomes),
            doc_tags=[out.doc_tag for out in outcomes],
        )

    message = " ".join(errors) if errors else "No attachments were ingested."
    raise AttachmentIngestError(message)
