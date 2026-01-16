import asyncio
import hashlib
import logging
import mimetypes
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.api.config import Config
from src.api.dataHandler import fileProcessor, train_on_document
from src.teams.logic import TeamsChatContext
from src.utils.logging_utils import get_logger

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings

    _BLOB_AVAILABLE = True
    _BLOB_WARNING_EMITTED = False
    _BLOB_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:
    BlobServiceClient = None  # type: ignore
    ContentSettings = None  # type: ignore
    _BLOB_AVAILABLE = False
    _BLOB_WARNING_EMITTED = False
    _BLOB_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


@dataclass
class AttachmentOutcome:
    filename: str
    documents_created: int


def _get_download_info(attachment: Dict[str, Any]) -> Tuple[Optional[str], str]:
    content = attachment.get("content") or {}
    download_url = content.get("downloadUrl") or content.get("download_url")
    filename = (
        content.get("fileName")
        or content.get("name")
        or attachment.get("name")
        or f"teams-upload-{uuid.uuid4()}"
    )
    return download_url, filename


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
    return bool(getattr(Config.Teams, "BLOB_CONNECTION_STRING", "")) and bool(
        getattr(Config.Teams, "BLOB_CONTAINER", "")
    )


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

    connection_string = getattr(Config.Teams, "BLOB_CONNECTION_STRING", "")
    container_name = getattr(Config.Teams, "BLOB_CONTAINER", "")
    blob_name = _build_blob_name(filename, subscription_id)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        service = BlobServiceClient.from_connection_string(connection_string)
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
    except Exception as exc:
        log.error("Failed to upload Teams attachment to blob storage: %s", exc, exc_info=True)


async def _download_bytes(
    url: str,
    headers: Optional[Dict[str, str]],
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


async def _process_file_download(
    attachment: Dict[str, Any],
    context: TeamsChatContext,
    log: logging.LoggerAdapter,
) -> Optional[AttachmentOutcome]:
    download_url, filename = _get_download_info(attachment)
    if not download_url:
        log.warning("Attachment missing download URL: %s", attachment)
        return None

    max_bytes = int(getattr(Config.Teams, "MAX_ATTACHMENT_MB", 50)) * 1024 * 1024
    timeout = float(getattr(Config.Teams, "HTTP_TIMEOUT_SEC", 20))
    retries = int(getattr(Config.Teams, "HTTP_RETRIES", 2))

    file_bytes = await _download_bytes(download_url, headers=None, timeout=timeout, retries=retries, max_bytes=max_bytes)
    extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
    if not extracted_docs:
        log.warning("No extractable content found for attachment %s", filename)
        return None

    doc_tag = (
        attachment.get("content", {}).get("uniqueId")
        or attachment.get("content", {}).get("id")
        or filename
    )
    documents_created = 0
    for doc_name, doc_content in extracted_docs.items():
        await asyncio.to_thread(
            train_on_document,
            doc_content,
            subscription_id=context.subscription_id,
            profile_tag=context.profile_id,
            doc_tag=str(doc_tag),
            doc_name=doc_name,
        )
        documents_created += 1

    _upload_to_blob(file_bytes, filename, context.subscription_id, log)
    return AttachmentOutcome(filename=filename, documents_created=documents_created)


async def _process_inline_image(
    attachment: Dict[str, Any],
    context: TeamsChatContext,
    log: logging.LoggerAdapter,
) -> Optional[AttachmentOutcome]:
    content_url = attachment.get("contentUrl")
    content_type = _attachment_type(attachment)
    if not content_url:
        log.warning("Inline image missing contentUrl: %s", attachment)
        return None

    headers = {}
    if "/v3/attachments/" in content_url and getattr(Config.Teams, "BOT_ACCESS_TOKEN", ""):
        headers["Authorization"] = f"Bearer {Config.Teams.BOT_ACCESS_TOKEN}"
    elif "/v3/attachments/" in content_url:
        log.warning("Bot Framework token missing for attachment download")
        return None

    max_bytes = int(getattr(Config.Teams, "MAX_ATTACHMENT_MB", 50)) * 1024 * 1024
    timeout = float(getattr(Config.Teams, "HTTP_TIMEOUT_SEC", 20))
    retries = int(getattr(Config.Teams, "HTTP_RETRIES", 2))

    filename = attachment.get("name") or f"teams-image-{uuid.uuid4()}"
    file_bytes = await _download_bytes(content_url, headers=headers or None, timeout=timeout, retries=retries, max_bytes=max_bytes)
    extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
    if not extracted_docs:
        log.warning("No extractable content found for inline image %s", filename)
        return None

    documents_created = 0
    for doc_name, doc_content in extracted_docs.items():
        await asyncio.to_thread(
            train_on_document,
            doc_content,
            subscription_id=context.subscription_id,
            profile_tag=context.profile_id,
            doc_tag=hashlib.sha256(content_url.encode("utf-8")).hexdigest(),
            doc_name=doc_name,
        )
        documents_created += 1
    _upload_to_blob(file_bytes, filename, context.subscription_id, log)
    log.info("Processed inline image %s (%s)", filename, content_type)
    return AttachmentOutcome(filename=filename, documents_created=documents_created)


async def handle_attachments(activity: Dict[str, Any], context: TeamsChatContext, correlation_id: str) -> Dict[str, Any]:
    attachments = activity.get("attachments") or []
    log = get_logger(__name__, correlation_id)
    if not attachments:
        return {"type": "message", "text": "I did not find any attachments to process."}

    outcomes: List[AttachmentOutcome] = []
    errors: List[str] = []

    for attachment in attachments:
        content_type = _attachment_type(attachment)
        try:
            if "file.download.info" in content_type:
                outcome = await _process_file_download(attachment, context, log)
                if outcome:
                    outcomes.append(outcome)
                else:
                    errors.append("File attachment could not be processed.")
            elif content_type.startswith("image/"):
                outcome = await _process_inline_image(attachment, context, log)
                if outcome:
                    outcomes.append(outcome)
                else:
                    errors.append(
                        f"Received image attachment ({content_type}) but could not download it. "
                        "Please upload the image as a file attachment."
                    )
            else:
                errors.append(f"Unsupported attachment type: {content_type or 'unknown'}.")
        except Exception as exc:
            log.error("Attachment processing failed: %s", exc, exc_info=True)
            errors.append("An attachment failed to process. Please try again.")

    if outcomes:
        filenames = ", ".join(sorted({out.filename for out in outcomes}))
        doc_count = sum(out.documents_created for out in outcomes)
        return {
            "type": "message",
            "text": (
                f"Successfully processed file(s): {filenames}. "
                f"Extracted {doc_count} document(s). "
                "Ask a question now and I will use the new content!"
            ),
        }

    if errors:
        return {"type": "message", "text": " ".join(errors)}

    return {"type": "message", "text": "Unable to process the attachment. Please try again."}
