import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.api.content_store import save_extracted_pickle
from src.api.blob_store import blob_storage_configured
from src.api.structured_extraction import get_extraction_engine

# Import intelligence layer
try:
    from src.intelligence.integration import (
        DocumentIntelligenceProcessor,
        process_document_intelligence,
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False
    DocumentIntelligenceProcessor = None
    process_document_intelligence = None
try:
    from src.api.dataHandler import (
        extract_document_info,
        decrypt_data,
        fileProcessor,
        get_azure_docs,
        get_s3_client,
        get_subscription_pii_setting,
        mask_document_content,
        read_s3_file,
        resolve_subscription_id,
        update_extraction_metadata,
        update_layout_graph_metadata,
        update_pii_stats,
    )
except Exception as _datahandler_exc:  # noqa: BLE001
    def _datahandler_unavailable(*_args, **_kwargs):
        raise RuntimeError("dataHandler unavailable") from _datahandler_exc

    extract_document_info = _datahandler_unavailable
    decrypt_data = _datahandler_unavailable
    fileProcessor = _datahandler_unavailable
    get_azure_docs = _datahandler_unavailable
    get_s3_client = _datahandler_unavailable
    get_subscription_pii_setting = _datahandler_unavailable
    mask_document_content = _datahandler_unavailable
    read_s3_file = _datahandler_unavailable
    resolve_subscription_id = _datahandler_unavailable
    update_extraction_metadata = _datahandler_unavailable
    update_layout_graph_metadata = _datahandler_unavailable
    update_pii_stats = _datahandler_unavailable
from src.api.layout_graph_store import save_layout_graph, save_layout_graph_local
from src.api.document_status import init_document_record, set_error, update_document_fields, update_stage
from src.api.pipeline_models import ExtractedDocument
from src.api.statuses import (
    STATUS_DELETED,
    STATUS_EXTRACTION_COMPLETED,
    STATUS_EXTRACTION_FAILED,
    STATUS_SCREENING_COMPLETED,
    STATUS_UNDER_REVIEW,
)
from src.embedding.pipeline.payload_normalizer import normalize_chunk_metadata
from src.storage.azure_blob_client import BlobDownloadError, CredentialError, normalize_blob_name
from src.utils.idempotency import acquire_lock, release_lock
from src.embedding.layout_graph import build_layout_graph

logger = logging.getLogger(__name__)


def _persist_layout_graph(
    *,
    document_id: str,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    extracted_docs: Any,
) -> Optional[Dict[str, Any]]:
    try:
        file_name = "document"
        content = extracted_docs
        if isinstance(extracted_docs, dict) and extracted_docs:
            file_name, content = next(iter(extracted_docs.items()))
        layout_graph = build_layout_graph(content, document_id=document_id, file_name=file_name)
        if blob_storage_configured():
            info = save_layout_graph(
                document_id=document_id,
                layout_graph=layout_graph,
                subscription_id=subscription_id,
                profile_id=profile_id,
            )
        else:
            info = save_layout_graph_local(document_id=document_id, layout_graph=layout_graph)
        update_layout_graph_metadata(
            document_id,
            layout_latest_path=info.get("latest_path"),
            layout_versioned_path=info.get("versioned_path"),
            layout_hash=info.get("sha256"),
        )
        return info
    except Exception as exc:  # noqa: BLE001
        logger.warning("LayoutGraph persistence skipped for %s: %s", document_id, exc)
        return None


def _build_extraction_summary(extracted_obj: Any) -> Dict[str, Any]:
    total_chars = 0
    total_pages = 0
    total_chunks = 0

    def handle_extracted(value: Any) -> None:
        nonlocal total_chars, total_pages, total_chunks
        if isinstance(value, ExtractedDocument):
            text_val = value.full_text or ""
            if not text_val:
                text_val = "\n".join([sec.text for sec in value.sections if sec.text])
            total_chars += len(text_val)
            total_chunks += len(value.sections)
            for sec in value.sections:
                total_pages = max(total_pages, sec.end_page or sec.start_page or 0)
        elif isinstance(value, str):
            total_chars += len(value)
            total_chunks += 1
        elif isinstance(value, dict):
            inner_text = value.get("text") or value.get("content")
            if isinstance(inner_text, str):
                total_chars += len(inner_text)
                total_chunks += 1
            elif isinstance(inner_text, list):
                total_chunks += len(inner_text)
        elif isinstance(value, list):
            total_chunks += len(value)

    if isinstance(extracted_obj, dict):
        for entry in extracted_obj.values():
            handle_extracted(entry)
    else:
        handle_extracted(extracted_obj)

    return {
        "pages": total_pages or None,
        "chunks": total_chunks or None,
        "chars": total_chars,
        "language": None,
    }


def _process_document_intelligence(
    document_id: str,
    extracted_docs: Dict[str, Any],
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process document through intelligence layer for entity extraction and Q&A generation.

    Args:
        document_id: Document ID.
        extracted_docs: Extracted document content.
        filename: Original filename.
        subscription_id: Optional subscription ID.
        profile_id: Optional profile ID.

    Returns:
        Intelligence result dictionary or None on failure.
    """
    if not INTELLIGENCE_AVAILABLE:
        logger.debug("Intelligence layer not available, skipping")
        return None

    try:
        # Get text content from extracted docs
        raw_text = ""
        file_size = 0

        if isinstance(extracted_docs, dict):
            for fname, content in extracted_docs.items():
                if isinstance(content, dict):
                    text = content.get("full_text") or content.get("text") or ""
                    if not text and content.get("sections"):
                        text = "\n".join(
                            sec.get("text", "") for sec in content.get("sections", [])
                            if sec.get("text")
                        )
                    raw_text += text
                    file_size = content.get("file_size", 0) or len(raw_text)
                elif isinstance(content, str):
                    raw_text += content
                    file_size = len(raw_text)

        if not raw_text:
            logger.debug("No text content for intelligence processing")
            return None

        # Get Redis client if available
        redis_client = None
        try:
            from src.api.dataHandler import get_redis_client
            redis_client = get_redis_client()
        except Exception:
            pass

        # Process through intelligence layer
        result = process_document_intelligence(
            document_id=document_id,
            content=raw_text,
            filename=filename,
            subscription_id=subscription_id,
            profile_id=profile_id,
            file_size=file_size,
            redis_client=redis_client,
        )

        logger.info(
            "Intelligence processing complete for %s: domain=%s, entities=%d, qa_pairs=%d",
            document_id,
            result.domain,
            len(result.entities.get_all_searchable_terms()) if result.entities else 0,
            len(result.qa_pairs),
        )

        return result.to_dict()

    except Exception as exc:
        logger.warning("Intelligence processing failed for %s: %s", document_id, exc)
        return None


def _normalize_extracted_metadata(extracted_docs: Any, *, document_id: str) -> Any:
    if isinstance(extracted_docs, dict):
        normalized: Dict[str, Any] = {}
        for name, content in extracted_docs.items():
            if isinstance(content, dict):
                content = dict(content)
                if isinstance(content.get("chunk_metadata"), list):
                    content["chunk_metadata"] = normalize_chunk_metadata(
                        content.get("chunk_metadata") or [],
                        document_id=str(document_id),
                    )
                if content.get("document") is not None:
                    content["document"] = _normalize_extracted_metadata(
                        content.get("document"),
                        document_id=document_id,
                    )
            normalized[name] = content
        return normalized
    return extracted_docs


def _extract_classification_from_structured(structured_docs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract document type/classification from structured extraction results."""
    for fname, value in (structured_docs or {}).items():
        if isinstance(value, dict):
            return {
                "document_type": value.get("document_type", "GENERIC"),
                "domain": (value.get("document_classification") or {}).get("domain", "generic"),
                "confidence": (value.get("document_classification") or {}).get("confidence", 0.0),
                "filename": fname,
            }
        if hasattr(value, "document_type"):
            doc_cls = getattr(value, "document_classification", None)
            domain = doc_cls.get("domain", "generic") if isinstance(doc_cls, dict) else "generic"
            confidence = doc_cls.get("confidence", 0.0) if isinstance(doc_cls, dict) else 0.0
            return {
                "document_type": getattr(value, "document_type", "GENERIC"),
                "domain": domain,
                "confidence": confidence,
                "filename": fname,
            }
    return {"document_type": "GENERIC", "domain": "generic", "confidence": 0.0}


# ── Ingestion-time entity metadata extraction ──────────────────────────

import re as _re

_EMAIL_RE = _re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = _re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
)
_LINKEDIN_RE = _re.compile(r"linkedin\.com/in/[\w\-]+", _re.IGNORECASE)


def _extract_entity_metadata(raw_docs: Any, filename: str = "") -> Dict[str, Any]:
    """Extract entity name, email, and phone from raw text at ingestion time.

    Uses the first 500 chars of raw text plus filename-based heuristics.
    Returns a dict suitable for storing as ``entity_metadata`` in the pickle.
    """
    text = ""
    if isinstance(raw_docs, dict):
        for _fname, content in raw_docs.items():
            if isinstance(content, dict):
                text = str(content.get("full_text") or content.get("text") or "")[:500]
            elif isinstance(content, str):
                text = content[:500]
            if text:
                break
    elif isinstance(raw_docs, str):
        text = raw_docs[:500]

    result: Dict[str, Any] = {}

    # Email
    email_match = _EMAIL_RE.search(text)
    if email_match:
        result["entity_email"] = email_match.group()

    # Phone
    phone_match = _PHONE_RE.search(text)
    if phone_match:
        candidate = phone_match.group().strip()
        # Only accept if it has enough digits (avoids matching year ranges)
        digits = _re.sub(r"\D", "", candidate)
        if len(digits) >= 7:
            result["entity_phone"] = candidate

    # LinkedIn
    linkedin_match = _LINKEDIN_RE.search(text)
    if linkedin_match:
        result["entity_linkedin"] = linkedin_match.group()

    # Name — try filename first (most reliable), then first line of text
    name = ""
    if filename:
        try:
            from src.rag_v3.extract import _name_from_filename
            name = _name_from_filename(filename) or ""
        except ImportError:
            pass
    if not name and text:
        # Use first non-empty line that looks like a name (2-4 words, no special chars)
        for line in text.split("\n")[:5]:
            line = line.strip()
            if not line or len(line) > 60:
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                # Reject lines that look like headers/metadata
                lower = line.lower()
                if not any(kw in lower for kw in ("resume", "curriculum", "invoice", "page", "date")):
                    name = line
                    break
    if name:
        result["entity_name"] = name

    return result


def _validate_extraction_fields(
    doc_classification: Dict[str, Any],
    raw_docs: Any,
    filename: str,
) -> None:
    """Log warnings for missing required fields per document type.

    No hard failures — just structured logging for observability.
    """
    doc_type = (doc_classification.get("document_type") or "").lower()
    domain = (doc_classification.get("domain") or "").lower()

    # Build a quick text sample for checking
    text = ""
    if isinstance(raw_docs, dict):
        for _fname, content in raw_docs.items():
            if isinstance(content, dict):
                text = str(content.get("full_text") or "")[:2000]
            elif isinstance(content, str):
                text = content[:2000]
            if text:
                break
    elif isinstance(raw_docs, str):
        text = raw_docs[:2000]

    text_lower = text.lower()

    if domain in ("resume", "hr") or "resume" in doc_type or "cv" in doc_type:
        # Check for name (at least one capitalized word pair in first 300 chars)
        has_name = bool(_re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", text[:300]))
        has_skills = any(kw in text_lower for kw in ("skill", "python", "java", "sql", "experience"))
        has_experience = any(kw in text_lower for kw in ("experience", "worked", "years", "company"))
        if not has_name:
            logger.warning("extraction_validation: resume missing entity name — doc=%s file=%s", filename, doc_type)
        if not has_skills:
            logger.warning("extraction_validation: resume missing skills — doc=%s file=%s", filename, doc_type)
        if not has_experience:
            logger.warning("extraction_validation: resume missing experience — doc=%s file=%s", filename, doc_type)

    elif domain == "invoice" or "invoice" in doc_type:
        has_number = any(kw in text_lower for kw in ("invoice number", "invoice no", "inv #", "inv no"))
        has_date = any(kw in text_lower for kw in ("invoice date", "date:", "due date"))
        has_items = any(kw in text_lower for kw in ("line item", "description", "qty", "quantity", "total"))
        if not has_number and not has_date:
            logger.warning("extraction_validation: invoice missing number/date — doc=%s file=%s", filename, doc_type)
        if not has_items:
            logger.warning("extraction_validation: invoice missing line items/total — doc=%s file=%s", filename, doc_type)


def _sanitize_raw_text_fields(docs: Any) -> Any:
    """Ensure no stringified repr or dict garbage in text fields."""
    if not isinstance(docs, dict):
        return docs
    from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage

    for fname, content in docs.items():
        if not isinstance(content, dict):
            continue
        # Sanitize full_text
        full_text = content.get("full_text")
        if isinstance(full_text, str) and _is_metadata_garbage(full_text):
            section_texts = []
            for sec in content.get("sections") or []:
                if isinstance(sec, dict):
                    t = sec.get("text") or sec.get("content") or ""
                    if isinstance(t, str) and t.strip() and not _is_metadata_garbage(t):
                        section_texts.append(t.strip())
            if section_texts:
                content["full_text"] = "\n\n".join(section_texts)
                logger.info("Recovered full_text from sections for %s", fname)
            else:
                logger.warning("full_text is garbage for %s and no clean sections found", fname)

        # Sanitize texts list
        texts = content.get("texts")
        if isinstance(texts, list):
            clean_texts = []
            for t in texts:
                if isinstance(t, str) and t.strip() and not _is_metadata_garbage(t):
                    clean_texts.append(t)
                elif isinstance(t, dict):
                    extracted = t.get("text") or t.get("content") or ""
                    if isinstance(extracted, str) and extracted.strip():
                        clean_texts.append(extracted)
            content["texts"] = clean_texts
    return docs


def _set_document_status(
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    fields: Dict[str, Any] = {"status": status, "updated_at": time.time()}
    if error_msg:
        fields["training_error"] = error_msg
        fields["training_failed_at"] = time.time()
    else:
        fields["training_error"] = None
    if extra_fields:
        fields.update(extra_fields)
    update_document_fields(document_id, fields)
    logger.info("Document %s status updated to %s", document_id, status)


def _run_auto_screening(document_id: str, doc_type: Optional[str] = None) -> None:
    """Run screening automatically after extraction completes.

    When auto_attach_on_ingest is enabled, runs the screening engine and
    transitions status to SCREENING_COMPLETED (or TRAINING_BLOCKED_SECURITY).
    When disabled, bypasses screening and directly sets SCREENING_COMPLETED.
    On any failure, sets SCREENING_COMPLETED so embedding is not blocked.
    """
    try:
        from src.screening.config import ScreeningConfig
        cfg = ScreeningConfig.load()
    except Exception:  # noqa: BLE001
        logger.warning("Screening config unavailable for %s; bypassing screening", document_id)
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
        return

    if not cfg.auto_attach_on_ingest:
        logger.info("Auto-screening disabled; bypassing screening for %s", document_id)
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
        return

    try:
        from src.screening.engine import ScreeningEngine
        from src.api.screening_service import apply_security_result

        engine = ScreeningEngine(config=cfg)
        report = engine.run_all(document_id, doc_type=doc_type)
        report_dict = report.to_dict()
        apply_security_result(document_id, report_dict)
        logger.info(
            "Auto-screening completed for %s: risk=%s score=%.1f",
            document_id,
            report.risk_level,
            report.overall_score_0_100,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Auto-screening failed for %s: %s; setting SCREENING_COMPLETED to unblock pipeline",
            document_id,
            exc,
        )
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)


_DEBOUNCE_TTL_SECONDS = 5


def _debounce_extraction(subscription_id: str, doc_id: str) -> bool:
    """Return True if this extraction request is a duplicate within the debounce window."""
    try:
        from src.api.config import Config
        redis_client = getattr(Config.Redis, "get_client", lambda: None)()
        if redis_client is None:
            from src.api.rag_state import get_app_state
            state = get_app_state()
            redis_client = getattr(state, "redis_client", None) if state else None
        if redis_client is None:
            return False
        debounce_key = f"extract:debounce:{subscription_id}:{doc_id}"
        if redis_client.get(debounce_key):
            return True
        redis_client.setex(debounce_key, _DEBOUNCE_TTL_SECONDS, "1")
        return False
    except Exception:
        return False


def _extract_from_connector(doc_id: str, doc_data: Dict[str, Any], conn_data: Dict[str, Any]) -> Dict[str, Any]:
    profile_id = str(doc_data.get("profile")) if doc_data.get("profile") else None
    subscription_candidate = (
        doc_data.get("subscriptionId")
        or doc_data.get("subscription_id")
        or doc_data.get("subscription")
        or (conn_data.get("subscriptionId") if isinstance(conn_data, dict) else None)
        or (conn_data.get("subscription") if isinstance(conn_data, dict) else None)
    )

    try:
        subscription_id = resolve_subscription_id(doc_id, subscription_candidate)
    except Exception as exc:  # noqa: BLE001
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, str(exc))
        return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    if _debounce_extraction(subscription_id, doc_id):
        logger.info("Extraction debounced for %s (subscription %s); duplicate within %ds", doc_id, subscription_id, _DEBOUNCE_TTL_SECONDS)
        return {"document_id": doc_id, "status": "CONFLICT", "reason": "debounced_duplicate"}

    lock = acquire_lock(stage="extraction", document_id=doc_id, subscription_id=subscription_id)
    if not lock.acquired:
        logger.info("Extraction already in progress for %s; skipping duplicate trigger.", doc_id)
        return {"document_id": doc_id, "status": "CONFLICT", "reason": "duplicate_extraction_in_progress"}

    pii_masking_enabled = get_subscription_pii_setting(subscription_id)
    logger.info("Document %s (subscription %s): PII masking=%s", doc_id, subscription_id, pii_masking_enabled)

    try:
        all_extracted_docs: Dict[str, Any] = {}
        try:
            if doc_data.get("type") == "S3":
                bk_name = conn_data["s3_details"]["bucketName"]
                region = conn_data["s3_details"]["region"]
                ak = decrypt_data(conn_data["s3_details"]["accessKey"]).split("\x0c")[0].strip()
                sk = decrypt_data(conn_data["s3_details"]["secretKey"]).split("\x08")[0].strip()
                s3 = get_s3_client(ak, sk, region)
                if not s3:
                    raise ValueError("Failed to create S3 client")

                objs = s3.list_objects_v2(Bucket=bk_name)
                file_keys = [obj["Key"] for obj in objs.get("Contents", []) if obj["Key"] == doc_data.get("name")]
                if not file_keys:
                    raise ValueError("File not found in S3")

                doc_content = read_s3_file(s3, bk_name, file_keys[0])
                if doc_content is None:
                    raise ValueError("Failed to read S3 file")

                extracted_doc = fileProcessor(doc_content, file_keys[0])
                if not extracted_doc:
                    raise ValueError("Content extraction failed")

                all_extracted_docs.update(extracted_doc)
            elif doc_data.get("type") == "LOCAL":
                doc_name = doc_data.get("name", "")
                if not doc_name:
                    raise ValueError("No filename specified")

                all_connector_files = conn_data.get("locations", [])
                matching_files: List[str] = []
                for file_path in all_connector_files:
                    file_name_only = file_path.split("/")[-1] if "/" in file_path else file_path
                    if file_name_only == doc_name:
                        matching_files.append(file_path)
                        break

                if not matching_files:
                    raise ValueError(f"Exact file match not found: {doc_name}")
                if len(matching_files) > 1:
                    raise ValueError("Multiple file matches")

                file_path = matching_files[0]
                file_key = normalize_blob_name(file_path, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME)
                doc_content = get_azure_docs(file_key, document_id=doc_id)
                if doc_content is None:
                    raise ValueError(f"Failed to read file {file_key}")

                extracted_doc = fileProcessor(doc_content, file_path)
                if not extracted_doc:
                    raise ValueError("Content extraction failed")

                all_extracted_docs.update(extracted_doc)
            else:
                raise ValueError(f"Unsupported connector type: {doc_data.get('type')}")
        except CredentialError as exc:
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"CredentialError: {exc}")
            raise
        except BlobDownloadError as exc:
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"{exc.__class__.__name__}: {exc}")
            return {
                "document_id": doc_id,
                "status": STATUS_EXTRACTION_FAILED,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, str(exc))
            return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

        if not all_extracted_docs:
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, "No content extracted")
            return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": "No content extracted"}

        masked_docs = all_extracted_docs
        pii_count = 0
        pii_items: List[Any] = []
        if pii_masking_enabled:
            masked_docs, pii_count, _high_conf, pii_items = mask_document_content(all_extracted_docs)
            update_pii_stats(doc_id, pii_count, False, pii_items)
        else:
            update_pii_stats(doc_id, 0, False, [])

        if pii_masking_enabled:
            for fname, content in masked_docs.items():
                if isinstance(content, dict) and "texts" in content:
                    texts = content.get("texts") or []
                    if texts:
                        from src.api.dataHandler import encode_with_fallback

                        content["embeddings"] = encode_with_fallback(
                            texts,
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                        )
                    masked_docs[fname] = content

        masked_docs = _normalize_extracted_metadata(masked_docs, document_id=doc_id)
        masked_docs = _sanitize_raw_text_fields(masked_docs)

        _persist_layout_graph(
            document_id=doc_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            extracted_docs=masked_docs,
        )

        # Structured extraction: build structured JSON from masked/raw text and persist alongside raw extraction
        try:
            engine = get_extraction_engine()
            # For each file in masked_docs, create structured doc; if multiple files, pick the first as primary
            # Compose a combined structured representation keyed by filename
            structured_docs = {}
            for fname, content in masked_docs.items():
                try:
                    # Determine raw text to feed into structured extractor
                    raw_text = ""
                    if isinstance(content, dict):
                        raw_text = content.get("full_text") or content.get("text") or "\n".join(
                            [sec.get("text") for sec in (content.get("sections") or []) if sec.get("text")]
                        )
                        page_count = content.get("pages") or (len(content.get("sections") or []) or 1)
                    else:
                        raw_text = str(content)
                        page_count = 1

                    structured = engine.extract_document(
                        document_id=doc_id,
                        text=raw_text,
                        filename=fname,
                        metadata={"source": "extraction", "orig_filename": fname},
                        page_count=page_count,
                    )
                    structured_docs[fname] = structured
                except Exception as sexc:  # noqa: BLE001
                    logger.warning("Structured extraction failed for %s: %s", fname, sexc)
                    # fallback: store raw content under structured as minimal
                    structured_docs[fname] = {
                        "document_id": doc_id,
                        "original_filename": fname,
                        "document_type": "GENERIC",
                        "sections": [{"section_id": "section_0", "section_type": "content", "content": raw_text}],
                        "extraction_quality_score": 0.0,
                    }

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to run structured extraction engine for %s: %s", doc_id, exc)
            structured_docs = {}

        # Intelligence layer processing: entity extraction, Q&A generation
        intelligence_result = _process_document_intelligence(
            document_id=doc_id,
            extracted_docs=masked_docs,
            filename=doc_data.get("name", "document"),
            subscription_id=subscription_id,
            profile_id=profile_id,
        )

        try:
            # Persist raw, structured extraction and intelligence in the pickle for source-of-truth
            doc_classification = _extract_classification_from_structured(structured_docs)
            entity_meta = _extract_entity_metadata(masked_docs, doc_data.get("name", ""))
            _validate_extraction_fields(doc_classification, masked_docs, doc_data.get("name", ""))
            payload_to_save = {
                "raw": masked_docs,
                "structured": structured_docs,
                "intelligence": intelligence_result,
                "document_classification": doc_classification,
                "entity_metadata": entity_meta,
            }
            save_info = save_extracted_pickle(doc_id, payload_to_save)
            update_extraction_metadata(doc_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
        except Exception as exc:  # noqa: BLE001
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"Failed to persist extracted content: {exc}")
            return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

        extra_fields = {
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "extracted_pickle_path": save_info.get("path"),
            "extracted_hash": save_info.get("sha256"),
        }
        _set_document_status(doc_id, STATUS_EXTRACTION_COMPLETED, extra_fields=extra_fields)

        # Auto-screen after extraction so embedding can proceed
        _run_auto_screening(doc_id, doc_type=doc_classification.get("document_type"))

        summary = _build_extraction_summary(masked_docs)
        return {
            "document_id": doc_id,
            "status": STATUS_EXTRACTION_COMPLETED,
            "pickle_path": save_info.get("path"),
            "summary": summary,
            "doc_name": doc_data.get("name", "Unknown"),
        }
    finally:
        if lock.acquired:
            release_lock(lock)


def extract_documents() -> Dict[str, Any]:
    try:
        doc_coll = extract_document_info()
        if not doc_coll:
            return {"status": "no_documents", "message": "No documents found for extraction"}

        allowed_statuses = {STATUS_UNDER_REVIEW, STATUS_EXTRACTION_FAILED}
        eligible_docs = {
            doc_id: doc_info
            for doc_id, doc_info in doc_coll.items()
            if doc_info.get("dataDict", {}).get("status") in allowed_statuses
        }

        if not eligible_docs:
            return {"status": "no_documents", "message": "No documents eligible for extraction"}

        results = {"successful": [], "failed": [], "total": len(eligible_docs)}
        for doc_id, doc_info in eligible_docs.items():
            if doc_info.get("dataDict", {}).get("status") == STATUS_DELETED:
                continue
            try:
                res = _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))
            except CredentialError as exc:
                logger.error("Credential error during extraction; failing batch: %s", exc)
                return {"status": "error", "message": f"CredentialError: {exc}", "results": None}
            if res.get("status") == STATUS_EXTRACTION_COMPLETED:
                results["successful"].append(res)
            else:
                results["failed"].append(res)

        return {"status": "completed", "results": results}
    except Exception as exc:  # noqa: BLE001
        logger.error("Extraction process failed: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "results": None}


def extract_single_document(doc_id: str) -> Dict[str, Any]:
    doc_coll = extract_document_info()
    if not doc_coll or doc_id not in doc_coll:
        return {"status": "not_found", "message": f"Document {doc_id} not found"}
    doc_info = doc_coll[doc_id]
    return _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))


def extract_uploaded_document(
    *,
    document_id: str,
    file_bytes: bytes,
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    profile_name: Optional[str] = None,
    doc_type: Optional[str] = None,
    content_type: Optional[str] = None,
    content_size: Optional[int] = None,
) -> Dict[str, Any]:
    lock = acquire_lock(stage="extraction", document_id=document_id, subscription_id=subscription_id)
    if not lock.acquired:
        logger.info("Extraction already in progress for %s; skipping duplicate upload.", document_id)
        return {"status": "skipped", "reason": "duplicate_extraction_in_progress", "document_id": document_id}
    init_document_record(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        doc_type=doc_type,
        filename=filename,
        content_type=content_type,
        size=content_size,
    )
    if profile_name:
        update_document_fields(document_id, {"profile_name": profile_name, "metadata.profile_name": profile_name})

    update_stage(document_id, "extraction", {"status": "IN_PROGRESS", "started_at": time.time(), "error": None})

    try:
        extracted = fileProcessor(file_bytes, filename)
        if not extracted:
            raise ValueError("No content extracted from file")
    except Exception as exc:  # noqa: BLE001
        set_error(document_id, "extraction", exc)
        _set_document_status(document_id, STATUS_EXTRACTION_FAILED, str(exc))
        raise

    extracted = _normalize_extracted_metadata(extracted, document_id=document_id)
    extracted = _sanitize_raw_text_fields(extracted)

    _persist_layout_graph(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        extracted_docs=extracted,
    )

    # Structured extraction for uploaded document
    try:
        engine = get_extraction_engine()
        structured_docs = {}
        for fname, content in extracted.items() if isinstance(extracted, dict) else [(filename, extracted)]:
            try:
                raw_text = ""
                if isinstance(content, dict):
                    raw_text = content.get("full_text") or content.get("text") or "\n".join(
                        [sec.get("text") for sec in (content.get("sections") or []) if sec.get("text")]
                    )
                    page_count = content.get("pages") or (len(content.get("sections") or []) or 1)
                else:
                    raw_text = str(content)
                    page_count = 1

                structured = engine.extract_document(
                    document_id=document_id,
                    text=raw_text,
                    filename=fname,
                    metadata={"source": "upload", "orig_filename": fname},
                    page_count=page_count,
                )
                structured_docs[fname] = structured
            except Exception as sexc:  # noqa: BLE001
                logger.warning("Structured extraction failed for upload %s: %s", fname, sexc)
                structured_docs[fname] = {"document_id": document_id, "original_filename": fname, "document_type": "GENERIC", "sections": [{"section_id": "section_0", "section_type": "content", "content": raw_text}], "extraction_quality_score": 0.0}
    except Exception:
        structured_docs = {}

    # Intelligence layer processing: entity extraction, Q&A generation
    intelligence_result = _process_document_intelligence(
        document_id=document_id,
        extracted_docs=extracted if isinstance(extracted, dict) else {filename: extracted},
        filename=filename,
        subscription_id=subscription_id,
        profile_id=profile_id,
    )

    try:
        # Persist uploaded file extraction (raw, structured, intelligence, classification) in the pickle
        doc_classification = _extract_classification_from_structured(structured_docs)
        entity_meta = _extract_entity_metadata(extracted, filename)
        _validate_extraction_fields(doc_classification, extracted, filename)
        payload_to_save = {
            "raw": extracted,
            "structured": structured_docs,
            "intelligence": intelligence_result,
            "document_classification": doc_classification,
            "entity_metadata": entity_meta,
        }
        save_info = save_extracted_pickle(document_id, payload_to_save)
        update_extraction_metadata(document_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
    except Exception as exc:  # noqa: BLE001
        _set_document_status(document_id, STATUS_EXTRACTION_FAILED, f"Failed to persist extracted content: {exc}")
        return {"document_id": document_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    extra_fields = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "extracted_pickle_path": save_info.get("path"),
        "extracted_hash": save_info.get("sha256"),
    }
    _set_document_status(document_id, STATUS_EXTRACTION_COMPLETED, extra_fields=extra_fields)

    # Auto-screen after extraction so embedding can proceed
    _run_auto_screening(document_id, doc_type=doc_classification.get("document_type"))

    summary = _build_extraction_summary(extracted)
    return {
        "document_id": document_id,
        "status": STATUS_EXTRACTION_COMPLETED,
        "pickle_path": save_info.get("path"),
        "summary": summary,
        "doc_name": filename,
        "intelligence": intelligence_result is not None,
    }

