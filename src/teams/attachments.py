import asyncio
import hashlib
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
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

logger = get_logger(__name__)

class AttachmentIngestError(Exception):
    """Raised when a Teams attachment cannot be ingested."""

@dataclass
class ScreeningSummary:
    risk_level: str = "LOW"
    overall_score: float = 0.0
    top_findings: List[str] = field(default_factory=list)
    tools_run: int = 0

@dataclass
class DocumentIntelligence:
    doc_type: str = "general"
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    intent_tags: List[str] = field(default_factory=list)
    chunks_created: int = 0

@dataclass
class AttachmentOutcome:
    filename: str
    documents_created: int
    doc_tag: str
    extracted_text: str = ""
    screening: Optional[ScreeningSummary] = None
    intelligence: Optional[DocumentIntelligence] = None

@dataclass
class IngestionResult:
    filenames: List[str]
    documents_created: int
    doc_tags: List[str]
    screening_results: List[ScreeningSummary] = field(default_factory=list)
    intelligence_results: List[DocumentIntelligence] = field(default_factory=list)

def _attachment_type(attachment: Dict[str, Any]) -> str:
    return (attachment.get("contentType") or "").lower()

def _attachment_content(attachment: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract the content dict from an attachment.

    Teams may set content to a string (e.g. HTML body for text/html messages)
    or None. Always returns a dict.
    """
    content = attachment.get("content")
    return content if isinstance(content, dict) else {}

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
    # Try multiple paths to obtain a connector token from the Bot Framework adapter
    if turn_context:
        # Path 1: adapter.get_access_token() (some SDK versions)
        try:
            token = await turn_context.adapter.get_access_token()  # type: ignore[attr-defined]
            if token:
                return token
        except Exception:  # noqa: BLE001
            pass
        # Path 2: credentials on the adapter (BotFrameworkAdapter stores MicrosoftAppCredentials)
        try:
            creds = getattr(turn_context.adapter, "_credentials", None) or getattr(turn_context.adapter, "credentials", None)
            if creds and hasattr(creds, "get_token"):
                token_obj = creds.get_token()
                if token_obj:
                    token_str = getattr(token_obj, "token", None) or str(token_obj)
                    if token_str:
                        return token_str
        except Exception:  # noqa: BLE001
            pass
        # Path 3: connector_client on the turn context
        try:
            connector_client = turn_context.turn_state.get("ConnectorClient")
            if connector_client and hasattr(connector_client, "config"):
                creds = getattr(connector_client.config, "credentials", None)
                if creds and hasattr(creds, "get_token"):
                    token_obj = creds.get_token()
                    if token_obj:
                        return getattr(token_obj, "token", None) or str(token_obj)
        except Exception:  # noqa: BLE001
            pass
    fallback = getattr(Config.Teams, "BOT_ACCESS_TOKEN", "") or ""
    if fallback:
        return fallback
    logger.warning("No connector token available for Teams attachment download; will try without auth.")
    return ""

def _build_download_headers(auth_token: str) -> Dict[str, str]:
    """Build headers for attachment download. Skip Authorization if token is empty."""
    if auth_token:
        return {"Authorization": f"Bearer {auth_token}"}
    return {}

def _resolve_doc_tag(attachment: Dict[str, Any], filename: str) -> str:
    content = _attachment_content(attachment)
    return (
        content.get("uniqueId")
        or content.get("id")
        or content.get("fileType")
        or hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    )

def _resolve_download_url(attachment: Dict[str, Any]) -> Optional[str]:
    """Resolve download URL from all possible sources in the attachment."""
    content = _attachment_content(attachment)
    return (
        content.get("downloadUrl")
        or content.get("download_url")
        or attachment.get("contentUrl")
    )

def _resolve_filename(attachment: Dict[str, Any]) -> str:
    """Resolve filename from all possible sources in the attachment."""
    content = _attachment_content(attachment)
    return (
        content.get("fileName")
        or content.get("name")
        or attachment.get("name")
        or f"teams-upload-{uuid.uuid4()}"
    )

def _run_security_screening(text: str, filename: str, doc_tag: str, log: logging.LoggerAdapter) -> Optional[ScreeningSummary]:
    """Run security-only screening (PII, secrets, private data) on extracted text.

    Uses SecurityScreeningService which focuses on data protection checks only —
    no legality, compliance, or quality checks that are irrelevant for Teams uploads.
    """
    try:
        from src.screening.security_service import SecurityScreeningService
        service = SecurityScreeningService()
        result = service.screen_text(
            text=text,
            doc_id=doc_tag,
            metadata={"filename": filename, "source": "teams_upload"},
        )
        risk_level = result.get("overall_risk_level", "MINIMAL")
        overall_score = float(result.get("overall_risk_score", 0))
        findings = result.get("security_findings", [])

        # Build human-readable findings summary
        top_findings: List[str] = []
        pii_count = sum(1 for f in findings if f.get("finding_type") == "PII")
        secret_count = sum(1 for f in findings if f.get("finding_type") == "SECRET")
        private_count = sum(1 for f in findings if f.get("finding_type") == "PRIVATE_DATA")
        if pii_count:
            top_findings.append(f"{pii_count} PII item(s) detected (personal identifiers)")
        if secret_count:
            top_findings.append(f"{secret_count} secret(s)/credential(s) detected")
        if private_count:
            top_findings.append(f"{private_count} private business data item(s) detected")
        if not top_findings:
            top_findings.append("No sensitive data detected")

        log.info(
            "Security screening complete for %s: risk=%s score=%.0f pii=%d secrets=%d private=%d",
            filename, risk_level, overall_score, pii_count, secret_count, private_count,
        )
        return ScreeningSummary(
            risk_level=risk_level,
            overall_score=overall_score,
            top_findings=top_findings,
            tools_run=3,  # PII + Secrets + Private Data
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Security screening failed for %s: %s", filename, exc)
        return None

def _run_document_intelligence(extracted, filename: str, doc_tag: str, context: "TeamsChatContext", log: logging.LoggerAdapter) -> Optional[DocumentIntelligence]:
    """Run Document Intelligence pipeline to enrich document metadata.

    Uses the 3-stage pipeline: identify → content_map → understand.
    Results are stored in MongoDB for RAG enrichment.
    Returns DocumentIntelligence for use by the Teams card builder.
    """
    try:
        from src.doc_understanding.identify import identify_document
        from src.doc_understanding.content_map import build_content_map
        from src.doc_understanding.understand import understand_document
        from src.llm.gateway import get_llm_gateway
        cloud_llm = get_llm_gateway()

        # Stage 1: Identify document type (Teams uses cloud models)
        id_result = identify_document(extracted=extracted, filename=filename, llm_client=cloud_llm)
        doc_type = getattr(id_result, "document_type", None) or (
            id_result.get("document_type", "other") if isinstance(id_result, dict) else "other"
        )
        log.info("DI identify: %s → type=%s", filename, doc_type)

        # Stage 2: Build content map
        content_map = build_content_map(extracted)

        # Stage 3: Understand document (Teams uses cloud models)
        understanding = understand_document(
            extracted=extracted,
            doc_type=doc_type,
            llm_client=cloud_llm,
        )
        summary = ""
        key_entities: List[str] = []
        key_facts: List[str] = []
        intent_tags: List[str] = []
        if isinstance(understanding, dict):
            summary = understanding.get("document_summary", "")
            key_entities = understanding.get("key_entities", [])
            key_facts = understanding.get("key_facts", [])
            intent_tags = understanding.get("intent_tags", [])
        log.info(
            "DI understand: %s → entities=%d facts=%d tags=%s",
            filename, len(key_entities), len(key_facts), intent_tags,
        )

        # Persist to MongoDB if available
        try:
            from src.api.dataHandler import db
            from src.api.config import Config
            docs_collection = db[Config.MongoDB.DOCUMENTS]
            update_fields = {
                "document_type": doc_type,
                "document_domain": doc_type,
            }
            if isinstance(understanding, dict):
                update_fields["document_summary"] = summary
                update_fields["key_entities"] = key_entities
                update_fields["key_facts"] = key_facts
                update_fields["intent_tags"] = intent_tags
            docs_collection.update_one(
                {"document_id": doc_tag, "subscription_id": context.subscription_id},
                {"$set": update_fields},
                upsert=True,
            )
        except Exception as db_exc:  # noqa: BLE001
            log.debug("DI MongoDB persist skipped: %s", db_exc)

        return DocumentIntelligence(
            doc_type=doc_type,
            summary=summary,
            key_entities=key_entities,
            key_facts=key_facts,
            intent_tags=intent_tags,
        )

    except Exception as exc:  # noqa: BLE001
        log.warning("Document intelligence pipeline error for %s: %s", filename, exc)
        return None

async def _process_file_attachment(
    attachment: Dict[str, Any],
    context: TeamsChatContext,
    log: logging.LoggerAdapter,
    auth_token: str,
    timeout: float,
    retries: int,
    max_bytes: int,
) -> Optional[AttachmentOutcome]:
    """Process any file attachment — handles both content.downloadUrl and contentUrl sources."""
    download_url = _resolve_download_url(attachment)
    filename = _resolve_filename(attachment)

    if not download_url:
        raise AttachmentIngestError(f"Attachment '{filename}' has no download URL.")

    log.info("Downloading attachment: %s (url_source=%s)", filename,
             "content.downloadUrl" if _attachment_content(attachment).get("downloadUrl") else "contentUrl")

    headers = _build_download_headers(auth_token)
    try:
        file_bytes = await _download_bytes(download_url, headers=headers, timeout=timeout, retries=retries, max_bytes=max_bytes)
    except Exception as exc:
        log.error("Download failed for %s: %s (url=%s...)", filename, exc, download_url[:80])
        raise AttachmentIngestError(f"Failed to download '{filename}': {exc}")

    log.info("Downloaded %s: %d bytes", filename, len(file_bytes))

    try:
        extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
    except Exception as exc:
        log.error("Text extraction failed for %s: %s", filename, exc, exc_info=True)
        raise AttachmentIngestError(f"Failed to extract text from '{filename}': {exc}")

    if not extracted_docs:
        log.warning("No extractable content found for %s", filename)
        return None

    doc_tag = _resolve_doc_tag(attachment, filename)
    documents_created = 0
    all_text_parts: List[str] = []
    first_extracted = None
    for doc_name, doc_content in extracted_docs.items():
        if first_extracted is None:
            first_extracted = doc_content
        await asyncio.to_thread(
            train_on_document,
            doc_content,
            subscription_id=context.subscription_id,
            profile_id=context.profile_id,
            doc_tag=str(doc_tag),
            doc_name=doc_name,
        )
        documents_created += 1
        if isinstance(doc_content, str):
            all_text_parts.append(doc_content)
        elif hasattr(doc_content, "full_text") and doc_content.full_text:
            all_text_parts.append(str(doc_content.full_text))
        elif hasattr(doc_content, "text") and doc_content.text:
            all_text_parts.append(str(doc_content.text))

    log.info("Ingested %s: %d document(s), tag=%s", filename, documents_created, doc_tag)

    extracted_text = "\n\n".join(all_text_parts)

    # Run DI, blob upload, and security screening concurrently
    async def _di_task():
        if first_extracted is None:
            return None
        try:
            result = await asyncio.to_thread(
                _run_document_intelligence, first_extracted, filename, str(doc_tag), context, log,
            )
            if result:
                result.chunks_created = documents_created
            return result
        except Exception as exc:  # noqa: BLE001
            log.warning("Document intelligence failed for %s (non-blocking): %s", filename, exc)
            return None

    async def _screening_task():
        if not extracted_text.strip():
            return None
        return await asyncio.to_thread(
            _run_security_screening, extracted_text[:50000], filename, str(doc_tag), log,
        )

    async def _blob_task():
        await asyncio.to_thread(_upload_to_blob, file_bytes, filename, context.subscription_id, log)

    di_result, screening, _ = await asyncio.gather(
        _di_task(), _screening_task(), _blob_task(),
    )

    return AttachmentOutcome(
        filename=filename,
        documents_created=documents_created,
        doc_tag=str(doc_tag),
        extracted_text=extracted_text[:500],
        screening=screening,
        intelligence=di_result,
    )

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
    token: Optional[str] = None

    outcomes: List[AttachmentOutcome] = []
    errors: List[str] = []

    # Resolve auth token once (shared across all attachments)
    token = connector_token

    # Separate attachments into processable vs skipped
    processable: List[Dict[str, Any]] = []
    for idx, attachment in enumerate(attachments):
        content_type = _attachment_type(attachment)
        log.info(
            "Processing attachment %d/%d: contentType=%s name=%s has_contentUrl=%s has_downloadUrl=%s",
            idx + 1, len(attachments), content_type,
            attachment.get("name"),
            bool(attachment.get("contentUrl")),
            bool(_attachment_content(attachment).get("downloadUrl")),
        )
        download_url = _resolve_download_url(attachment)
        if not download_url:
            if content_type and content_type not in ("text/html", "unknown"):
                errors.append(f"Attachment '{attachment.get('name', 'unknown')}' has no download URL.")
            else:
                errors.append(f"Unsupported attachment type: {content_type or 'unknown'}.")
            continue
        processable.append(attachment)

    if processable:
        if token is None:
            token = await _resolve_auth_token(turn_context, connector_token)

        async def _ingest_one(attachment: Dict[str, Any]) -> Optional[AttachmentOutcome]:
            try:
                return await _process_file_attachment(
                    attachment, context, log,
                    auth_token=token or "",
                    timeout=timeout, retries=retries, max_bytes=max_bytes,
                )
            except AttachmentIngestError as exc:
                log.warning("Attachment ingest error: %s", exc)
                errors.append(str(exc))
                return None
            except Exception as exc:  # noqa: BLE001
                log.error("Attachment processing failed: %s", exc, exc_info=True)
                errors.append(f"Failed to process '{attachment.get('name', 'unknown')}'. Please try again.")
                return None

        # Process all attachments concurrently
        results = await asyncio.gather(*[_ingest_one(a) for a in processable])

        for outcome in results:
            if outcome:
                outcomes.append(outcome)
                if state_store:
                    state_store.record_upload(
                        context.subscription_id,
                        context.profile_id,
                        outcome.filename,
                        outcome.doc_tag,
                        outcome.documents_created,
                        document_type=outcome.intelligence.doc_type if outcome.intelligence else None,
                    )
            else:
                if not errors:  # Only add generic error if no specific error was recorded
                    errors.append("Attachment had no extractable content.")

    if outcomes:
        return IngestionResult(
            filenames=[out.filename for out in outcomes],
            documents_created=sum(out.documents_created for out in outcomes),
            doc_tags=[out.doc_tag for out in outcomes],
            screening_results=[out.screening for out in outcomes if out.screening],
            intelligence_results=[out.intelligence for out in outcomes if out.intelligence],
        )

    message = " ".join(errors) if errors else "No attachments were ingested."
    raise AttachmentIngestError(message)
