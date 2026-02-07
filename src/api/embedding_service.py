import logging
import os
import pickle
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from bson import ObjectId
except Exception:  # noqa: BLE001
    ObjectId = None  # type: ignore[assignment]

from src.api.blob_store import (
    BlobConfigurationError,
    BlobInfo,
    BlobStore,
    blob_storage_configured,
    extract_document_id,
    is_trusted_blob,
)
from src.api.config import Config
from src.api.content_store import build_pickle_path, delete_extracted_pickle, load_extracted_pickle, save_extracted_pickle
from src.api.dataHandler import (
    ChunkingDiagnosticError,
    db,
    decrypt_data,
    fileProcessor,
    get_azure_docs,
    get_qdrant_client,
    get_s3_client,
    get_subscription_pii_setting,
    get_model,
    read_s3_file,
    resolve_profile_id,
    resolve_subscription_id,
    train_on_document,
    update_extraction_metadata,
    update_pii_stats,
)
from src.api.document_status import get_document_record, update_document_fields, update_stage
from src.api.pipeline_models import ExtractedDocument
from src.api.pii_masking import mask_document_content
from src.api.statuses import (
    STATUS_EMBEDDING_COMPLETED,
    STATUS_EXTRACTION_OR_CHUNKING_FAILED,
    STATUS_SCREENING_COMPLETED,
    STATUS_TRAINING_FAILED,
    STATUS_TRAINING_STARTED,
    STATUS_TRAINING_COMPLETED,
    STATUS_TRAINING_PARTIALLY_COMPLETED,
)
from src.api.vector_store import build_collection_name, build_qdrant_filter
from src.embedding.chunking.section_chunker import SectionChunker, normalize_text
from src.embedding.model_loader import embed_request_context
from src.embedding.pipeline.payload_normalizer import normalize_chunk_metadata
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from src.storage.azure_blob_client import normalize_blob_name
from src.storage.blob_persistence import load_pickle as load_blob_pickle
from src.utils.idempotency import acquire_lock, release_lock

logger = logging.getLogger(__name__)

COMPLETED_STATUSES = {
    STATUS_EMBEDDING_COMPLETED,
    STATUS_TRAINING_STARTED,
    STATUS_TRAINING_COMPLETED,
    STATUS_TRAINING_PARTIALLY_COMPLETED,
}


def _telemetry():
    return telemetry_store() if METRICS_V2_ENABLED else None


def _metrics_store():
    return get_metrics_store()


def _truncate_error_message(message: Optional[str], limit: int = 500) -> str:
    if not message:
        return ""
    text = str(message)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _build_error_payload(
    *,
    stage: str,
    message: str,
    exc: Optional[Exception] = None,
    details: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    code: Optional[Any] = None,
) -> Dict[str, Any]:
    error: Dict[str, Any] = {
        "stage": stage,
        "message": _truncate_error_message(message),
        "code": code or getattr(exc, "code", None) or getattr(exc, "status_code", None),
        "details": details or getattr(exc, "details", None),
        "at": time.time(),
        "run_id": run_id,
    }
    return {k: v for k, v in error.items() if v is not None}


def _safe_update_stage(
    document_id: str,
    stage: str,
    patch: Dict[str, Any],
    *,
    cause: Optional[Exception] = None,
) -> None:
    try:
        update_stage(document_id, stage, patch)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update stage %s for %s: %s", stage, document_id, exc, exc_info=True)
        if cause:
            logger.error("Original error for %s stage %s: %s", document_id, stage, cause, exc_info=True)


def _safe_set_document_status(
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
    *,
    error_summary: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    try:
        _set_document_status(
            document_id,
            status,
            error_msg,
            error_summary=error_summary,
            extra_fields=extra_fields,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update document status for %s: %s", document_id, exc, exc_info=True)
        if cause:
            logger.error("Original error for %s status update: %s", document_id, cause, exc_info=True)


def _build_failed_result(
    *,
    document_id: Optional[str],
    blob_name: Optional[str],
    error_message: str,
    failed_reason: str = "training_failed",
) -> Dict[str, Any]:
    return {
        "blob_name": blob_name,
        "document_id": document_id,
        "status": "FAILED",
        "chunks_count": 0,
        "points_upserted": 0,
        "error": failed_reason,
        "error_message": error_message,
        "failed_reason": failed_reason,
    }


def _set_document_status(
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
    error_summary: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    now = time.time()
    fields: Dict[str, Any] = {"status": status, "updated_at": now}
    if status == STATUS_TRAINING_STARTED:
        fields["training_started_at"] = now
    if error_msg:
        error_msg = _truncate_error_message(error_msg)
        fields["training_error"] = error_msg
        fields["training_failed_at"] = now
    else:
        fields["training_error"] = None
    if error_summary is not None:
        fields["error_summary"] = error_summary
    elif not error_msg:
        fields["error_summary"] = None
    if extra_fields:
        fields.update(extra_fields)
    update_document_fields(document_id, fields)
    logger.info("Document %s status updated to %s", document_id, status)


def _training_success_fields() -> Dict[str, Any]:
    now = time.time()
    return {
        "embedding_status": STATUS_EMBEDDING_COMPLETED,
        "embedding_completed_at": now,
        "trained_at": now,
    }


def _get_max_workers(total: int) -> int:
    max_workers_env = os.getenv("EMBEDDING_MAX_WORKERS")
    try:
        max_workers = int(max_workers_env) if max_workers_env else None
    except ValueError:
        max_workers = None
    if max_workers is None:
        max_workers = min(total, max(os.cpu_count() or 2, 2))
    return max_workers


def _get_max_blobs(requested: Optional[int] = None) -> int:
    if requested is not None and requested > 0:
        return requested
    env_val = os.getenv("DOCWAIN_EMBED_MAX_BLOBS")
    try:
        parsed = int(env_val) if env_val else None
    except ValueError:
        parsed = None
    return parsed or 25


def _lease_seconds() -> int:
    env_val = os.getenv("DOCWAIN_BLOB_LEASE_SECONDS")
    try:
        parsed = int(env_val) if env_val else None
    except ValueError:
        parsed = None
    return parsed or 60


def _normalize_requested_ids(document_id: Optional[str], document_ids: Optional[List[str]]) -> List[str]:
    requested: List[str] = []
    if document_id:
        requested.append(str(document_id))
    if document_ids:
        requested.extend([str(doc_id).strip() for doc_id in document_ids if str(doc_id).strip()])
    return list(dict.fromkeys(requested))


def _extract_doc_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("_id", "document_id", "documentId", "doc_id", "id"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def _fetch_document_ids_by_filters(
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> List[str]:
    if not subscription_id and not profile_id:
        return []

    from src.api.document_status import get_documents_collection

    collection = get_documents_collection()
    if collection is None:
        raise ValueError("Document store is not accessible")

    filters: List[Dict[str, Any]] = [{"status": STATUS_SCREENING_COMPLETED}]
    if subscription_id:
        filters.append(
            {
                "$or": [
                    {"subscriptionId": subscription_id},
                    {"subscription_id": subscription_id},
                    {"subscription": subscription_id},
                ]
            }
        )
    if profile_id:
        filters.append(
            {
                "$or": [
                    {"profileId": profile_id},
                    {"profile_id": profile_id},
                    {"profile": profile_id},
                ]
            }
        )

    query = {"$and": filters} if len(filters) > 1 else filters[0]
    cursor = collection.find(query, projection={"_id": 1, "document_id": 1, "documentId": 1, "doc_id": 1})
    doc_ids: List[str] = []
    for record in cursor:
        doc_id = _extract_doc_id(record)
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def _fetch_document_ids_for_integrity(
    subscription_id: Optional[str],
    profile_id: Optional[str],
    limit: int,
) -> List[str]:
    from src.api.document_status import get_documents_collection

    collection = get_documents_collection()
    if collection is None:
        raise ValueError("Document store is not accessible")

    eligible_statuses = [
        STATUS_SCREENING_COMPLETED,
        STATUS_EMBEDDING_COMPLETED,
        STATUS_TRAINING_STARTED,
        STATUS_TRAINING_COMPLETED,
        STATUS_TRAINING_PARTIALLY_COMPLETED,
        STATUS_TRAINING_FAILED,
    ]
    filters: List[Dict[str, Any]] = [{"status": {"$in": eligible_statuses}}]
    if subscription_id:
        filters.append(
            {
                "$or": [
                    {"subscriptionId": subscription_id},
                    {"subscription_id": subscription_id},
                    {"subscription": subscription_id},
                ]
            }
        )
    if profile_id:
        filters.append(
            {
                "$or": [
                    {"profileId": profile_id},
                    {"profile_id": profile_id},
                    {"profile": profile_id},
                ]
            }
        )

    query = {"$and": filters} if len(filters) > 1 else filters[0]
    cursor = collection.find(
        query,
        projection={"_id": 1, "document_id": 1, "documentId": 1, "doc_id": 1},
    ).limit(max(1, int(limit)))

    doc_ids: List[str] = []
    for record in cursor:
        doc_id = _extract_doc_id(record)
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def _load_extracted_for_doc(document_id: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    details: Dict[str, Any] = {"source": "missing"}
    if blob_storage_configured():
        store = _build_blob_store()
        for ext in (".pkl", ".pickle"):
            blob_name = store.build_blob_name(document_id, extension=ext)
            payload = load_blob_pickle(blob_name)
            if payload:
                details.update(
                    {
                        "source": "blob",
                        "blob_name": blob_name,
                        "bytes": len(payload or b""),
                    }
                )
                return pickle.loads(payload), details
        return None, details

    path = build_pickle_path(document_id)
    if not path.exists():
        return None, details
    payload = path.read_bytes()
    details.update({"source": "local", "path": str(path), "bytes": len(payload or b"")})
    return pickle.loads(payload), details


def _min_chars_threshold() -> int:
    raw = os.getenv("EMBEDDING_MIN_CHARS", "50")
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg


def _coverage_threshold() -> float:
    try:
        return float(getattr(Config.Retrieval, "CHUNK_COVERAGE_THRESHOLD", 0.98))
    except Exception:  # noqa: BLE001
        return 0.98


def _basename(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value)
    return text.split("/")[-1] if "/" in text else text


def _sum_strings(values: Iterable[Optional[str]]) -> int:
    total = 0
    for value in values:
        if isinstance(value, str):
            total += len(value.strip())
    return total


def _raw_coverage_for_content(content: Any) -> Optional[float]:
    threshold = _coverage_threshold()
    try:
        if isinstance(content, ExtractedDocument) and content.full_text:
            full_len = len(content.full_text or "")
            if full_len <= 0:
                return None
            candidate_len = _sum_strings([cand.text for cand in content.chunk_candidates or []])
            if candidate_len <= 0:
                candidate_len = _sum_strings([sec.text for sec in content.sections or []])
            if candidate_len <= 0:
                return None
            ratio = candidate_len / max(1, full_len)
            # Clamp for sanity in case of overlap inflation.
            return min(max(ratio, 0.0), max(1.0, threshold))

        if isinstance(content, dict):
            full_text = content.get("full_text")
            texts = content.get("texts")
            if isinstance(full_text, str) and isinstance(texts, list):
                full_len = len(full_text.strip())
                joined_len = _sum_strings([t for t in texts if isinstance(t, str)])
                if full_len > 0 and joined_len > 0:
                    ratio = joined_len / max(1, full_len)
                    return min(max(ratio, 0.0), max(1.0, threshold))
    except Exception:  # noqa: BLE001
        return None
    return None


def _content_char_count(content: Any) -> int:
    if isinstance(content, ExtractedDocument):
        if content.full_text:
            return len(content.full_text.strip())
        section_len = _sum_strings([sec.text for sec in content.sections or []])
        if section_len > 0:
            return section_len
        return _sum_strings([cand.text for cand in content.chunk_candidates or []])

    if isinstance(content, dict):
        if isinstance(content.get("full_text"), str):
            return len((content.get("full_text") or "").strip())
        texts = content.get("texts")
        if isinstance(texts, list):
            return _sum_strings([t for t in texts if isinstance(t, str)])
        return _sum_strings([content.get("text"), content.get("content")])

    if isinstance(content, str):
        return len(content.strip())

    return 0


def _payload_has_required_schema(content: Any) -> bool:
    """Validate that the extracted payload contains at least one usable text field."""
    if isinstance(content, ExtractedDocument):
        return bool(
            (content.full_text or "").strip()
            or any((sec.text or "").strip() for sec in content.sections or [])
            or any((cand.text or "").strip() for cand in content.chunk_candidates or [])
        )
    if isinstance(content, dict):
        schema_keys = {"text", "pages", "sections", "content", "full_text", "texts"}
        if schema_keys.intersection(content.keys()):
            return True
        # Nested "document" payloads are common in pickles.
        doc_value = content.get("document")
        if doc_value is not None:
            return _payload_has_required_schema(doc_value)
        return False
    if isinstance(content, str):
        return bool(content.strip())
    return False


def _normalize_content_in_place(content: Any) -> Any:
    """Normalize extracted text fields without losing layout cues."""
    if isinstance(content, ExtractedDocument):
        content.full_text = normalize_text(content.full_text or "")
        for sec in content.sections or []:
            sec.text = normalize_text(sec.text or "")
        for cand in content.chunk_candidates or []:
            cand.text = normalize_text(cand.text or "")
        return content
    if isinstance(content, dict):
        for key in ("full_text", "text", "content"):
            if isinstance(content.get(key), str):
                content[key] = normalize_text(content.get(key) or "")
        if isinstance(content.get("texts"), list):
            content["texts"] = [normalize_text(str(t)) for t in content.get("texts") or []]
        if isinstance(content.get("sections"), list):
            normalized_sections = []
            for sec in content.get("sections") or []:
                if isinstance(sec, dict):
                    sec = dict(sec)
                    if isinstance(sec.get("text"), str):
                        sec["text"] = normalize_text(sec.get("text") or "")
                normalized_sections.append(sec)
            content["sections"] = normalized_sections
        if isinstance(content.get("pages"), list):
            normalized_pages = []
            for page in content.get("pages") or []:
                if isinstance(page, dict):
                    page = dict(page)
                    for key in ("text", "content"):
                        if isinstance(page.get(key), str):
                            page[key] = normalize_text(page.get(key) or "")
                normalized_pages.append(page)
            content["pages"] = normalized_pages
        if content.get("document") is not None:
            content["document"] = _normalize_content_in_place(content.get("document"))
        return content
    if isinstance(content, str):
        return normalize_text(content)
    return content


def _normalize_metadata_in_place(content: Any, *, document_id: Optional[str]) -> Any:
    if isinstance(content, dict):
        if isinstance(content.get("chunk_metadata"), list):
            content["chunk_metadata"] = normalize_chunk_metadata(
                content.get("chunk_metadata") or [],
                document_id=str(document_id) if document_id else None,
            )
        if content.get("document") is not None:
            content["document"] = _normalize_metadata_in_place(
                content.get("document"),
                document_id=document_id,
            )
        return content
    return content


def _assess_extracted_docs(extracted_docs: Dict[str, Any]) -> Dict[str, Any]:
    total_chars = 0
    coverage_values: List[float] = []
    for _name, content in extracted_docs.items():
        total_chars += _content_char_count(content)
        coverage = _raw_coverage_for_content(content)
        if isinstance(coverage, (int, float)):
            coverage_values.append(float(coverage))

    min_chars = _min_chars_threshold()
    has_data = total_chars >= min_chars
    incomplete = any(cov < _coverage_threshold() for cov in coverage_values) if coverage_values else False

    return {
        "total_chars": total_chars,
        "coverage_values": coverage_values,
        "has_data": has_data,
        "incomplete": incomplete,
        "min_chars": min_chars,
    }


def _connector_lookup(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    connector_id = record.get("connector") or record.get("connector_id") or record.get("connectorId")
    if not connector_id:
        return None
    connectors = db[Config.MongoDB.CONNECTOR]
    try:
        if ObjectId and ObjectId.is_valid(str(connector_id)):
            conn = connectors.find_one({"_id": ObjectId(str(connector_id))})
            if conn:
                return conn
        return connectors.find_one({"_id": str(connector_id)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Connector lookup failed for %s: %s", connector_id, exc)
        return None


def _download_from_connector(
    *,
    document_id: str,
    record: Dict[str, Any],
    connector: Dict[str, Any],
) -> Tuple[Optional[bytes], Optional[str]]:
    doc_name = _basename(record.get("name") or record.get("source_file"))
    conn_type = str(record.get("type") or connector.get("type") or "").upper()

    if conn_type == "S3":
        try:
            s3_details = connector.get("s3_details") or {}
            bucket_name = s3_details.get("bucketName")
            region = s3_details.get("region")
            ak_raw = s3_details.get("accessKey")
            sk_raw = s3_details.get("secretKey")
            if not (bucket_name and region and ak_raw and sk_raw and doc_name):
                return None, None
            ak = decrypt_data(ak_raw).split("\x0c")[0].strip()
            sk = decrypt_data(sk_raw).split("\x08")[0].strip()
            s3 = get_s3_client(ak, sk, region)
            if not s3:
                return None, None
            content = read_s3_file(s3, bucket_name, doc_name)
            return content, doc_name
        except Exception as exc:  # noqa: BLE001
            logger.warning("S3 fallback download failed for %s: %s", document_id, exc)
            return None, None

    if conn_type == "LOCAL":
        locations = connector.get("locations") or []
        if not doc_name or not locations:
            return None, None
        for file_path in locations:
            if _basename(file_path) != doc_name:
                continue
            try:
                blob_name = normalize_blob_name(file_path, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME)
                content = get_azure_docs(blob_name, document_id=document_id)
                return content, file_path
            except Exception as exc:  # noqa: BLE001
                logger.warning("Azure fallback download failed for %s (%s): %s", document_id, file_path, exc)
                return None, None
    return None, None


def _download_from_source_file(record: Dict[str, Any], document_id: str) -> Tuple[Optional[bytes], Optional[str]]:
    source_file = record.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        return None, None
    try:
        blob_name = normalize_blob_name(source_file, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME)
        content = get_azure_docs(blob_name, document_id=document_id)
        return content, source_file
    except Exception as exc:  # noqa: BLE001
        logger.warning("Source-file fallback download failed for %s (%s): %s", document_id, source_file, exc)
        return None, None


def _reextract_from_source(
    *,
    document_id: str,
    record: Dict[str, Any],
    subscription_id: str,
) -> Optional[Dict[str, Any]]:
    connector = _connector_lookup(record) or {}
    content_bytes, source_id = _download_from_connector(document_id=document_id, record=record, connector=connector)
    if content_bytes is None:
        content_bytes, source_id = _download_from_source_file(record, document_id)
    if content_bytes is None or not source_id:
        logger.warning("Fallback extraction skipped for %s: source content unavailable", document_id)
        return None

    extracted = fileProcessor(content_bytes, source_id)
    if not extracted:
        logger.warning("Fallback extraction produced no content for %s", document_id)
        return None

    pii_enabled = False
    try:
        pii_enabled = get_subscription_pii_setting(subscription_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PII setting lookup failed for %s: %s", document_id, exc)

    masked_docs = extracted
    pii_count = 0
    pii_items: List[Any] = []
    if pii_enabled:
        masked_docs, pii_count, _high_conf, pii_items = mask_document_content(extracted)
        update_pii_stats(document_id, pii_count, False, pii_items)
    else:
        update_pii_stats(document_id, 0, False, [])

    try:
        save_info = save_extracted_pickle(document_id, masked_docs)
        update_extraction_metadata(document_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fallback pickle persist failed for %s: %s", document_id, exc)

    logger.info("Fallback extraction completed for %s using source %s", document_id, source_id)
    return masked_docs


def _prepare_extracted_docs(
    *,
    document_id: str,
    extracted: Any,
    record: Dict[str, Any],
    subscription_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], List[float], Optional[str]]:
    metrics_store = _metrics_store()
    extracted_docs = _normalize_extracted_docs(extracted)
    for doc_name, content in list(extracted_docs.items()):
        extracted_docs[doc_name] = _normalize_content_in_place(content)
        extracted_docs[doc_name] = _normalize_metadata_in_place(
            extracted_docs[doc_name],
            document_id=document_id,
        )

    invalid_docs = [doc_name for doc_name, content in extracted_docs.items() if not _payload_has_required_schema(content)]
    if invalid_docs:
        if metrics_store.available:
            metrics_store.record(counters={"empty_docs_count": len(invalid_docs)}, document_id=document_id, agent="embedding")
        logger.warning("Extracted payload missing required schema keys for %s: %s", document_id, invalid_docs)
        return None, None, [], "empty_extraction"

    assessment = _assess_extracted_docs(extracted_docs)
    logger.info(
        "Extraction assessment for %s: total_chars=%s coverage=%s",
        document_id,
        assessment.get("total_chars"),
        assessment.get("coverage_values"),
    )

    if not assessment.get("has_data"):
        if metrics_store.available:
            metrics_store.record(counters={"empty_docs_count": 1}, document_id=document_id, agent="embedding")
        return None, None, [], "empty_extraction"

    if assessment.get("incomplete"):
        logger.warning(
            "Extracted pickle appears incomplete for %s; attempting source-file fallback",
            document_id,
        )
        fallback_docs = _reextract_from_source(
            document_id=document_id,
            record=record,
            subscription_id=subscription_id,
        )
        if fallback_docs:
            extracted_docs = _normalize_extracted_docs(fallback_docs)
            for doc_name, content in list(extracted_docs.items()):
                extracted_docs[doc_name] = _normalize_content_in_place(content)
            assessment = _assess_extracted_docs(extracted_docs)
            logger.info(
                "Fallback assessment for %s: total_chars=%s coverage=%s",
                document_id,
                assessment.get("total_chars"),
                assessment.get("coverage_values"),
            )
            if not assessment.get("has_data"):
                if metrics_store.available:
                    metrics_store.record(counters={"empty_docs_count": 1}, document_id=document_id, agent="embedding")
                return None, None, [], "empty_extraction"

    try:
        expected_chunks, coverage_values = _screen_payload(extracted_docs, document_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chunk screening failed for %s: %s", document_id, exc)
        return None, None, [], "chunking_failed"

    return extracted_docs, expected_chunks, coverage_values, None


def _select_blob_candidates(
    store: BlobStore,
    document_ids: List[str],
    subscription_id: Optional[str],
    profile_id: Optional[str],
    max_blobs: int,
) -> List[BlobInfo]:
    candidates: List[BlobInfo] = []
    if document_ids:
        for doc_id in document_ids:
            for ext in (".pkl", ".pickle"):
                blob_name = store.build_blob_name(doc_id, extension=ext)
                info = store.get_blob_info(blob_name)
                if info:
                    candidates.append(info)
        if not candidates:
            candidates = store.list_pickle_blobs(limit=max_blobs)
    else:
        candidates = store.list_pickle_blobs(limit=max_blobs)

    filtered: List[BlobInfo] = []
    doc_id_set = set(document_ids)
    for blob in candidates:
        if not is_trusted_blob(blob, expected_prefix=store.prefix):
            continue
        doc_id = (blob.metadata or {}).get("document_id") or extract_document_id(blob.name, prefix=store.prefix)
        if doc_id_set and doc_id not in doc_id_set:
            continue

        if subscription_id:
            meta_sub = (blob.metadata or {}).get("subscription_id") or (blob.metadata or {}).get("subscriptionId")
            if meta_sub and str(meta_sub) != str(subscription_id):
                continue
        if profile_id:
            meta_profile = (blob.metadata or {}).get("profile_id") or (blob.metadata or {}).get("profileId")
            if meta_profile and str(meta_profile) != str(profile_id):
                continue
        filtered.append(blob)
        if len(filtered) >= max_blobs:
            break

    return filtered


def _split_text_preserve(input_text: str) -> List[str]:
    if not input_text:
        return []
    chunk_size = max(1, int(getattr(Config.Retrieval, "CHUNK_SIZE", 800)))
    overlap = max(0, int(getattr(Config.Retrieval, "CHUNK_OVERLAP", 200)))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    step = max(1, chunk_size - overlap)
    chunks_out = []
    text_len = len(input_text)
    for start in range(0, text_len, step):
        end = min(text_len, start + chunk_size)
        chunks_out.append(input_text[start:end])
        if end >= text_len:
            break
    return chunks_out


def _merge_candidates(candidates: List[Any], min_len: int):
    merged = []
    buffer_text = ""
    buffer_meta = None
    for cand in candidates:
        if not getattr(cand, "text", ""):
            continue
        if buffer_meta is None:
            buffer_text = cand.text
            buffer_meta = cand
            continue

        if len(buffer_text) < min_len and cand.section_id == buffer_meta.section_id:
            buffer_text = f"{buffer_text}\n{cand.text}"
        else:
            merged.append((buffer_text, buffer_meta))
            buffer_text = cand.text
            buffer_meta = cand

    if buffer_meta and buffer_text:
        merged.append((buffer_text, buffer_meta))
    return merged


def _build_chunks_for_extracted_doc(
    extracted: ExtractedDocument,
    *,
    document_id: str,
    doc_name: str,
) -> Tuple[List[str], List[Dict[str, Any]], Optional[float], int]:
    doc_type = extracted.doc_type
    ocr_confidences = (extracted.metrics or {}).get("ocr_confidences", []) if extracted.metrics else []
    doc_ocr_confidence = None
    if ocr_confidences:
        try:
            doc_ocr_confidence = float(sum(ocr_confidences) / len(ocr_confidences))
        except Exception:  # noqa: BLE001
            doc_ocr_confidence = None

    try:
        from src.embedding.layout_semantics import build_semantic_payloads
        from src.api.layout_graph_store import load_layout_graph

        layout_graph = load_layout_graph(document_id)
        semantic = build_semantic_payloads(
            layout_graph=layout_graph,
            extracted=extracted,
            document_id=str(document_id),
            source_name=doc_name,
        )
        chunks = [normalize_text(chunk) for chunk in semantic.chunks if normalize_text(chunk)]
        chunk_metadata = normalize_chunk_metadata(
            list(semantic.chunk_metadata),
            document_id=str(document_id),
            default_doc_type=doc_type,
            default_chunking_mode="layout_semantics",
        )
        dropped = max(0, len(semantic.chunks) - len(chunks))
        full_text = normalize_text(extracted.full_text or "")
        coverage_ratio = None
        if full_text:
            coverage_ratio = len("".join(chunks)) / max(1, len(full_text))
        return chunks, chunk_metadata, coverage_ratio, dropped
    except Exception as exc:  # noqa: BLE001
        logger.debug("LayoutGraph chunk preview failed for %s: %s", doc_name, exc)

    chunker = SectionChunker()
    try:
        section_chunks = chunker.chunk_document(extracted, doc_internal_id=document_id, source_filename=doc_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Section chunking failed for {doc_name}: {exc}") from exc

    chunks: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []
    dropped = 0
    for section_chunk in section_chunks:
        chunk_text = normalize_text(section_chunk.text)
        if not chunk_text:
            dropped += 1
            continue
        section_title = (section_chunk.section_title or "Untitled Section").strip() or "Untitled Section"
        section_path = (section_chunk.section_path or section_title).strip() or section_title
        page_start = section_chunk.page_start
        page_end = section_chunk.page_end
        chunks.append(chunk_text)
        chunk_metadata.append(
            {
                "document_id": document_id,
                "section_title": section_title,
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "page_number": page_start,
                "chunk_index": len(chunks) - 1,
                "chunk_type": "text",
                "doc_type": doc_type,
                "ocr_confidence": doc_ocr_confidence,
                "sentence_complete": bool(section_chunk.sentence_complete),
            }
        )

    if not chunks:
        raise ValueError(f"No chunk candidates extracted for {doc_name}")

    chunk_metadata = normalize_chunk_metadata(
        chunk_metadata,
        document_id=str(document_id),
        default_doc_type=doc_type,
        default_chunking_mode="section_aware",
    )

    full_text = normalize_text(extracted.full_text or "")
    coverage_ratio = None
    if full_text:
        coverage_ratio = len("".join(chunks)) / max(1, len(full_text))
        coverage_threshold = float(getattr(Config.Retrieval, "CHUNK_COVERAGE_THRESHOLD", 0.98))
        if coverage_ratio < coverage_threshold:
            logger.warning(
                "Chunk coverage %.3f below threshold %.3f for %s; falling back to full_text chunking",
                coverage_ratio,
                coverage_threshold,
                doc_name,
            )
            section_chunks = chunker.chunk_document(full_text, doc_internal_id=document_id, source_filename=doc_name)
            chunks = []
            chunk_metadata = []
            dropped = 0
            for section_chunk in section_chunks:
                chunk_text = normalize_text(section_chunk.text)
                if not chunk_text:
                    dropped += 1
                    continue
                section_title = (section_chunk.section_title or "Untitled Section").strip() or "Untitled Section"
                section_path = (section_chunk.section_path or section_title).strip() or section_title
                page_start = section_chunk.page_start
                page_end = section_chunk.page_end
                chunks.append(chunk_text)
                chunk_metadata.append(
                    {
                        "document_id": document_id,
                        "section_title": section_title,
                        "section_path": section_path,
                        "page_start": page_start,
                        "page_end": page_end,
                        "page_number": page_start,
                        "chunk_index": len(chunks) - 1,
                        "chunk_type": "text",
                        "doc_type": doc_type,
                        "ocr_confidence": doc_ocr_confidence,
                        "sentence_complete": bool(section_chunk.sentence_complete),
                    }
                )
            coverage_ratio = len("".join(chunks)) / max(1, len(full_text))
            chunk_metadata = normalize_chunk_metadata(
                chunk_metadata,
                document_id=str(document_id),
                default_doc_type=doc_type,
                default_chunking_mode="section_aware",
            )

    return chunks, chunk_metadata, coverage_ratio, dropped


def _estimate_chunks_for_content(content: Any, *, document_id: str, doc_name: str) -> Tuple[int, Optional[float]]:
    if isinstance(content, ExtractedDocument):
        chunks, _, coverage_ratio, _ = _build_chunks_for_extracted_doc(
            content, document_id=document_id, doc_name=doc_name
        )
        return len(chunks), coverage_ratio
    if isinstance(content, dict):
        if isinstance(content.get("texts"), list):
            return len([text for text in content.get("texts") if text]), None
        if isinstance(content.get("embeddings"), (list, tuple)):
            return len(content.get("embeddings") or []), None
        if isinstance(content.get("chunk_metadata"), list):
            return len(content.get("chunk_metadata") or []), None
        text_value = content.get("full_text") or content.get("text") or content.get("content")
        if isinstance(text_value, str) and text_value.strip():
            try:
                chunker = SectionChunker()
                section_chunks = chunker.chunk_document(text_value, doc_internal_id=document_id, source_filename=doc_name)
                return len(section_chunks), None
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section chunk estimation failed for %s: %s", doc_name, exc)
                return 0, None
        if isinstance(content.get("pages"), list):
            pages_text: List[str] = []
            for page in content.get("pages") or []:
                if isinstance(page, dict):
                    pages_text.append(str(page.get("text") or page.get("content") or ""))
                else:
                    pages_text.append(str(page))
            joined = "\n\n".join(pages_text).strip()
            if joined:
                try:
                    chunker = SectionChunker()
                    section_chunks = chunker.chunk_document(joined, doc_internal_id=document_id, source_filename=doc_name)
                    return len(section_chunks), None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Section chunk estimation failed for %s pages: %s", doc_name, exc)
                    return 0, None
    if isinstance(content, str):
        try:
            chunker = SectionChunker()
            section_chunks = chunker.chunk_document(content, doc_internal_id=document_id, source_filename=doc_name)
            return len(section_chunks), None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Section chunk estimation failed for %s: %s", doc_name, exc)
            return 0, None
    return 0, None


def _normalize_extracted_docs(extracted: Any) -> Dict[str, Any]:
    if isinstance(extracted, ExtractedDocument):
        return {"document": extracted}
    if isinstance(extracted, dict):
        if {"texts", "embeddings", "chunk_metadata", "full_text"}.intersection(extracted.keys()):
            return {"document": extracted}
        return extracted
    if isinstance(extracted, str):
        return {"document": extracted}
    raise ValueError(f"Unsupported extracted payload type: {type(extracted)}")


def _screen_payload(extracted_docs: Dict[str, Any], document_id: str) -> Tuple[int, List[float]]:
    total_chunks = 0
    coverage_values: List[float] = []
    min_len = int(getattr(Config.Retrieval, "MIN_CHUNK_SIZE", 200))

    for doc_name, content in extracted_docs.items():
        chunks_count, coverage = _estimate_chunks_for_content(content, document_id=document_id, doc_name=doc_name)
        if chunks_count <= 0:
            raise ValueError(f"No chunks available for {doc_name}")
        total_chunks += chunks_count
        if coverage is not None:
            coverage_values.append(float(coverage))

        if isinstance(content, dict):
            if "texts" in content and not isinstance(content.get("chunk_metadata"), list):
                logger.warning("chunk_metadata missing for structured payload %s", doc_name)
            text_values = [text for text in (content.get("texts") or []) if isinstance(text, str)]
            if text_values and max(len(text) for text in text_values) < min_len:
                logger.warning("Structured payload chunks below minimum length for %s", doc_name)

        if isinstance(content, (str, ExtractedDocument)):
            longest = 0
            if isinstance(content, str):
                try:
                    chunker = SectionChunker()
                    section_chunks = chunker.chunk_document(content, doc_internal_id=document_id, source_filename=doc_name)
                    longest = max((len(normalize_text(ch.text)) for ch in section_chunks), default=0)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Section chunk length check failed for %s: %s", doc_name, exc)
                    longest = 0
            else:
                chunks, _, _, _ = _build_chunks_for_extracted_doc(content, document_id=document_id, doc_name=doc_name)
                longest = max((len(chunk) for chunk in chunks), default=0)
            if longest < min_len:
                logger.warning("Chunk length below minimum for %s (max=%s)", doc_name, longest)

    return total_chunks, coverage_values


def _count_qdrant_points(subscription_id: str, profile_id: str, document_id: str, *, exact: bool = False) -> int:
    client = get_qdrant_client()
    collection_name = build_collection_name(subscription_id)
    count_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_id=str(document_id),
    )
    result = client.count(collection_name=collection_name, count_filter=count_filter, exact=bool(exact))
    return int(getattr(result, "count", 0) or 0)


def _verify_post_upsert_count(
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str,
    expected_chunks: int,
    attempts: int = 3,
    delay_seconds: float = 1.0,
) -> Tuple[Optional[int], bool]:
    """Best-effort verification that Qdrant reflects the upserted chunks.

    We retry a few times to avoid deleting the pickle while Qdrant is still
    catching up, and only allow cleanup once the observed count meets or
    exceeds the expected chunk count.
    """
    expected = max(0, int(expected_chunks))
    tries = max(1, int(attempts))
    last_count: Optional[int] = None

    for attempt in range(1, tries + 1):
        try:
            last_count = _count_qdrant_points(subscription_id, profile_id, document_id, exact=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Post-upsert Qdrant count failed for %s on attempt %s/%s: %s",
                document_id,
                attempt,
                tries,
                exc,
            )
            last_count = None
        else:
            if expected == 0 or last_count >= expected:
                return last_count, True
            logger.warning(
                "Post-upsert Qdrant points for %s below expected on attempt %s/%s: %s < %s",
                document_id,
                attempt,
                tries,
                last_count,
                expected,
            )

        if attempt < tries and delay_seconds > 0:
            time.sleep(delay_seconds)

    return last_count, expected == 0


def _build_blob_store() -> BlobStore:
    return BlobStore()


def _process_blob(
    *,
    store: BlobStore,
    blob: BlobInfo,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
    embed_request_id: Optional[str],
) -> Dict[str, Any]:
    telemetry = _telemetry()
    result = {
        "blob_name": blob.name,
        "document_id": (blob.metadata or {}).get("document_id")
        or extract_document_id(blob.name, prefix=store.prefix),
        "status": "FAILED",
        "chunks_count": 0,
        "points_upserted": 0,
        "error": None,
        "failed_reason": None,
    }

    request_ctx = embed_request_context(embed_request_id)
    request_ctx.__enter__()
    logger.info(
        "embed_request_id=%s doc=%s embedding start blob=%s",
        embed_request_id,
        result.get("document_id"),
        blob.name,
    )

    lease_id = None
    lock = None
    try:
        lease_id = store.try_acquire_lease(blob.name, lease_duration=_lease_seconds())
        if not lease_id:
            if telemetry:
                telemetry.increment("embed_pickles_lease_conflict_total")
            result["status"] = "SKIPPED"
            result["error"] = "lease_conflict"
            result["failed_reason"] = "lease_conflict"
            return result
        if telemetry:
            telemetry.increment("embed_pickles_leased_total")

        try:
            payload = store.download_blob(blob.name, lease=lease_id)
        except Exception as exc:  # noqa: BLE001
            if telemetry:
                telemetry.increment("embed_pickles_download_fail_total")
            error_message = _truncate_error_message(str(exc) or repr(exc))
            result["error"] = "blob_read_failed"
            result["failed_reason"] = "blob_read_failed"
            doc_id_hint = result.get("document_id")
            if doc_id_hint:
                _safe_set_document_status(
                    doc_id_hint,
                    STATUS_TRAINING_FAILED,
                    error_message,
                    error_summary="blob_read_failed",
                    cause=exc,
                )
            return result
        logger.info("Downloaded pickle blob %s (bytes=%s)", blob.name, len(payload or b""))

        try:
            extracted = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001
            if telemetry:
                telemetry.increment("embed_pickles_download_fail_total")
            result["error"] = "pickle_deserialize_failed"
            result["failed_reason"] = "blob_read_failed"
            doc_id_hint = result.get("document_id")
            if doc_id_hint:
                _safe_set_document_status(
                    doc_id_hint,
                    STATUS_TRAINING_FAILED,
                    f"pickle_deserialize_failed: {exc}",
                    error_summary="blob_read_failed",
                    cause=exc,
                )
            return result

        doc_id = (blob.metadata or {}).get("document_id") or extract_document_id(blob.name, prefix=store.prefix)
        if not doc_id:
            result["error"] = "document_id_missing"
            result["failed_reason"] = "document_id_missing"
            return result
        result["document_id"] = doc_id

        record = get_document_record(doc_id) or {}
        _set_document_status(doc_id, STATUS_TRAINING_STARTED)

        try:
            subscription_id = resolve_subscription_id(
                doc_id,
                subscription_id
                or (blob.metadata or {}).get("subscription_id")
                or (blob.metadata or {}).get("subscriptionId")
                or record.get("subscription_id")
                or record.get("subscriptionId")
                or record.get("subscription"),
            )
            profile_id = resolve_profile_id(
                doc_id,
                profile_id
                or (blob.metadata or {}).get("profile_id")
                or (blob.metadata or {}).get("profileId")
                or record.get("profile_id")
                or record.get("profileId")
                or record.get("profile"),
            )
        except Exception as exc:  # noqa: BLE001
            error_message = _truncate_error_message(str(exc) or repr(exc))
            _safe_set_document_status(
                doc_id,
                STATUS_TRAINING_FAILED,
                error_message,
                error_summary="resolve_ids_failed",
                cause=exc,
            )
            result["error"] = "resolve_ids_failed"
            result["failed_reason"] = "resolve_ids_failed"
            return result

        lock = acquire_lock(stage="embedding", document_id=doc_id, subscription_id=subscription_id)
        if not lock.acquired:
            logger.info("Embedding already in progress for %s; skipping duplicate.", doc_id)
            result["status"] = "SKIPPED"
            result["error"] = "duplicate_embedding_in_progress"
            result["failed_reason"] = "duplicate_embedding_in_progress"
            return result

        update_stage(
            doc_id,
            "embedding",
            {"status": "IN_PROGRESS", "started_at": time.time(), "error": None, "reason": None},
        )

        extracted_docs, expected_chunks, coverage_values, prep_error = _prepare_extracted_docs(
            document_id=doc_id,
            extracted=extracted,
            record=record,
            subscription_id=subscription_id,
        )
        if prep_error == "empty_extraction":
            reextracted = _reextract_from_source(
                document_id=doc_id,
                record=record,
                subscription_id=subscription_id,
            )
            if reextracted:
                extracted_docs, expected_chunks, coverage_values, prep_error = _prepare_extracted_docs(
                    document_id=doc_id,
                    extracted=reextracted,
                    record=record,
                    subscription_id=subscription_id,
                )
                if not prep_error:
                    extracted = reextracted

        if prep_error or extracted_docs is None or expected_chunks is None:
            if telemetry:
                telemetry.increment("embed_screening_fail_total")
            reason = prep_error or "empty_extraction"
            message = {
                "empty_extraction": "empty extraction",
                "chunking_failed": "chunking failed",
            }.get(reason, "embedding preparation failed")
            error_payload = _build_error_payload(
                stage="embedding",
                message=message,
                run_id=embed_request_id,
                code=reason,
            )
            _safe_update_stage(
                doc_id,
                "embedding",
                {
                    "status": "FAILED",
                    "completed_at": time.time(),
                    "reason": reason,
                    "error": error_payload,
                },
            )
            _safe_set_document_status(
                doc_id,
                STATUS_TRAINING_FAILED,
                message,
                error_summary=reason,
            )
            result["error"] = reason
            result["error_message"] = message
            result["failed_reason"] = reason
            return result

        result["chunks_count"] = expected_chunks
        logger.info("Embedding pre-check for %s: expected_chunks=%s", doc_id, expected_chunks)

        qdrant_count = 0
        if subscription_id and profile_id:
            try:
                qdrant_count = _count_qdrant_points(subscription_id, profile_id, doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant count failed for %s: %s", doc_id, exc)
        logger.info("Existing Qdrant points for %s: %s", doc_id, qdrant_count)

        if qdrant_count and expected_chunks and qdrant_count >= expected_chunks:
            if telemetry:
                telemetry.increment("embed_skipped_already_embedded_total")
            update_stage(
                doc_id,
                "embedding",
                {
                    "status": "COMPLETED",
                    "completed_at": time.time(),
                    "error": None,
                    "already_embedded": True,
                    "qdrant": {"collection": build_collection_name(subscription_id), "upserted": qdrant_count},
                },
            )
            _set_document_status(doc_id, STATUS_TRAINING_COMPLETED, extra_fields=_training_success_fields())
            deleted = False
            try:
                deleted = store.delete_blob(blob.name, lease=lease_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_request_id=%s doc=%s blob delete failed: %s",
                    embed_request_id,
                    doc_id,
                    exc,
                )
            if telemetry:
                telemetry.increment("embed_pickles_deleted_total" if deleted else "embed_pickles_delete_fail_total")
            _safe_update_stage(
                doc_id,
                "cleanup",
                {
                    "pickle_deleted": deleted,
                    "deleted_at": time.time() if deleted else None,
                    "cleanup_pending": not deleted,
                    "error": None
                    if deleted
                    else _build_error_payload(
                        stage="cleanup",
                        message="blob_delete_failed",
                        run_id=embed_request_id,
                        details={"message": "blob_delete_failed"},
                    ),
                },
            )
            result["status"] = "SKIPPED"
            result["error"] = None
            result["points_upserted"] = qdrant_count
            result["failed_reason"] = None
            return result

        if qdrant_count and expected_chunks and qdrant_count < expected_chunks:
            logger.warning(
                "Partial embeddings detected for %s: %s/%s; upserting all chunks",
                doc_id,
                qdrant_count,
                expected_chunks,
            )
        elif qdrant_count and not expected_chunks:
            logger.warning("Embeddings found for %s but expected count unknown; leaving blob", doc_id)

        total_chunks = 0
        total_upserted = 0
        total_dropped = 0
        coverage_values = coverage_values or []

        try:
            for file_name, content in extracted_docs.items():
                try:
                    embed_result = train_on_document(content, subscription_id, profile_id, doc_id, file_name)
                except Exception as exc:  # noqa: BLE001
                    if _is_meta_tensor_error(exc):
                        logger.warning(
                            "embed_request_id=%s doc=%s meta tensor error; retrying on cpu",
                            embed_request_id,
                            doc_id,
                        )
                        try:
                            get_model(reload=True, device="cpu")
                            embed_result = train_on_document(
                                content,
                                subscription_id,
                                profile_id,
                                doc_id,
                                file_name,
                                device="cpu",
                            )
                            logger.info(
                                "embed_request_id=%s doc=%s cpu fallback succeeded",
                                embed_request_id,
                                doc_id,
                            )
                        except Exception as retry_exc:  # noqa: BLE001
                            raise retry_exc from exc
                    else:
                        raise
                total_chunks += int(embed_result.get("chunks", 0))
                total_upserted += int(embed_result.get("points_saved", 0))
                total_dropped += int(embed_result.get("dropped_chunks", 0))
                ratio = embed_result.get("coverage_ratio")
                if isinstance(ratio, (int, float)):
                    coverage_values.append(float(ratio))
        except ChunkingDiagnosticError as exc:
            error_message = _truncate_error_message(str(exc) or repr(exc))
            diagnostics = exc.diagnostics or {}
            logger.error(
                "embed_request_id=%s doc=%s chunking diagnostics failed: %s",
                embed_request_id,
                doc_id,
                exc,
                exc_info=True,
            )
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_message,
                exc=exc,
                details=diagnostics,
                run_id=embed_request_id,
                code="extraction_or_chunking_failed",
            )
            _safe_update_stage(
                doc_id,
                "embedding",
                {
                    "status": "FAILED",
                    "completed_at": time.time(),
                    "reason": "extraction_or_chunking_failed",
                    "error": error_payload,
                    "diagnostics": diagnostics,
                },
                cause=exc,
            )
            _safe_set_document_status(
                doc_id,
                STATUS_EXTRACTION_OR_CHUNKING_FAILED,
                error_message,
                error_summary="extraction_or_chunking_failed",
                cause=exc,
            )
            result["error"] = "extraction_or_chunking_failed"
            result["error_message"] = error_message
            result["failed_reason"] = "extraction_or_chunking_failed"
            result["diagnostics"] = diagnostics
            return result
        except Exception as exc:  # noqa: BLE001
            error_message = _truncate_error_message(str(exc) or repr(exc))
            if telemetry:
                telemetry.increment("embed_qdrant_upsert_fail_total")
            logger.error(
                "embed_request_id=%s doc=%s embedding failed: %s",
                embed_request_id,
                doc_id,
                exc,
                exc_info=True,
            )
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_message,
                exc=exc,
                run_id=embed_request_id,
                code="training_failed",
            )
            _safe_update_stage(
                doc_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": error_payload},
                cause=exc,
            )
            _safe_set_document_status(
                doc_id,
                STATUS_TRAINING_FAILED,
                error_message,
                error_summary="training_failed",
                cause=exc,
            )
            result["error"] = "training_failed"
            result["error_message"] = error_message
            result["failed_reason"] = "training_failed"
            return result

        if total_chunks != total_upserted:
            error_msg = f"Embedding upsert mismatch: expected {total_chunks}, saved {total_upserted}"
            if telemetry:
                telemetry.increment("embed_qdrant_upsert_fail_total")
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_msg,
                run_id=embed_request_id,
                code="qdrant_upsert_failed",
            )
            _safe_update_stage(
                doc_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": error_payload},
            )
            _safe_set_document_status(
                doc_id,
                STATUS_TRAINING_FAILED,
                error_msg,
                error_summary="qdrant_upsert_failed",
            )
            result["error"] = "qdrant_upsert_failed"
            result["error_message"] = error_msg
            result["failed_reason"] = "qdrant_upsert_failed"
            result["points_upserted"] = total_upserted
            return result

        collection_name = build_collection_name(subscription_id)
        logger.info(
            "Embedding results for %s: chunks=%s upserted=%s dropped=%s collection=%s",
            doc_id,
            total_chunks,
            total_upserted,
            total_dropped,
            collection_name,
        )
        coverage_ratio = min(coverage_values) if coverage_values else None
        update_stage(
            doc_id,
            "embedding",
            {
                "status": "COMPLETED",
                "completed_at": time.time(),
                "error": None,
                "chunking": {
                    "chunks": total_chunks,
                    "coverage_ratio": coverage_ratio,
                    "dropped_empty": total_dropped,
                },
                "qdrant": {"collection": collection_name, "expected": total_chunks, "upserted": total_upserted},
            },
        )

        _set_document_status(doc_id, STATUS_TRAINING_COMPLETED, extra_fields=_training_success_fields())

        post_count: Optional[int] = None
        cleanup_allowed = True
        cleanup_error: Optional[Dict[str, Any]] = None
        if subscription_id and profile_id:
            try:
                post_count, cleanup_allowed = _verify_post_upsert_count(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    document_id=doc_id,
                    expected_chunks=total_chunks,
                )
                if post_count is not None:
                    logger.info("Post-upsert Qdrant points for %s: %s", doc_id, post_count)
                if not cleanup_allowed:
                    cleanup_error = (
                        {"message": "post_upsert_count_unavailable"}
                        if post_count is None
                        else {
                            "message": "post_upsert_count_mismatch",
                            "post_count": post_count,
                            "expected": total_chunks,
                        }
                    )
                    logger.warning(
                        "Skipping pickle cleanup for %s because embedding is not yet verified (post_count=%s, expected=%s)",
                        doc_id,
                        post_count,
                        total_chunks,
                    )
            except Exception as exc:  # noqa: BLE001
                cleanup_allowed = False
                cleanup_error = {"message": "post_upsert_count_failed"}
                logger.warning(
                    "embed_request_id=%s doc=%s post-upsert count failed: %s",
                    embed_request_id,
                    doc_id,
                    exc,
                )

        deleted = False
        if cleanup_allowed:
            try:
                deleted = store.delete_blob(blob.name, lease=lease_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_request_id=%s doc=%s blob delete failed: %s",
                    embed_request_id,
                    doc_id,
                    exc,
                )
                deleted = False
            if telemetry:
                telemetry.increment("embed_pickles_deleted_total" if deleted else "embed_pickles_delete_fail_total")
            if not deleted:
                cleanup_error = {"message": "blob_delete_failed"}
        cleanup_payload = None
        if not deleted:
            cleanup_details = cleanup_error or {"message": "cleanup_deferred"}
            cleanup_payload = _build_error_payload(
                stage="cleanup",
                message=cleanup_details.get("message", "cleanup_deferred"),
                run_id=embed_request_id,
                details=cleanup_details,
            )
        _safe_update_stage(
            doc_id,
            "cleanup",
            {
                "pickle_deleted": deleted,
                "deleted_at": time.time() if deleted else None,
                "cleanup_pending": not deleted,
                "error": cleanup_payload,
            },
        )

        result["status"] = "COMPLETED"
        result["error"] = None
        result["chunks_count"] = total_chunks
        result["points_upserted"] = total_upserted
        result["failed_reason"] = None
        return result
    except Exception as exc:  # noqa: BLE001
        error_message = _truncate_error_message(str(exc) or repr(exc))
        logger.error(
            "embed_request_id=%s doc=%s embedding failed: %s",
            embed_request_id,
            result.get("document_id"),
            exc,
            exc_info=True,
        )
        result["error"] = "training_failed"
        result["error_message"] = error_message
        result["failed_reason"] = "training_failed"
        doc_id_hint = result.get("document_id")
        if doc_id_hint:
            _safe_set_document_status(
                doc_id_hint,
                STATUS_TRAINING_FAILED,
                error_message,
                error_summary="training_failed",
                cause=exc,
            )
        return result
    finally:
        if lease_id:
            try:
                store.release_lease(blob.name, lease_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_request_id=%s doc=%s lease release failed: %s",
                    embed_request_id,
                    result.get("document_id"),
                    exc,
                )
        if lock and lock.acquired:
            try:
                release_lock(lock)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_request_id=%s doc=%s lock release failed: %s",
                    embed_request_id,
                    result.get("document_id"),
                    exc,
                )
        logger.info(
            "embed_request_id=%s doc=%s embedding end status=%s",
            embed_request_id,
            result.get("document_id"),
            result.get("status"),
        )
        request_ctx.__exit__(*sys.exc_info())


def _process_local_document(
    *,
    document_id: str,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
    embed_request_id: Optional[str],
) -> Dict[str, Any]:
    result = {
        "blob_name": f"{document_id}.pkl",
        "document_id": document_id,
        "status": "FAILED",
        "chunks_count": 0,
        "points_upserted": 0,
        "error": None,
        "failed_reason": None,
    }

    request_ctx = embed_request_context(embed_request_id)
    request_ctx.__enter__()
    logger.info(
        "embed_request_id=%s doc=%s embedding start local",
        embed_request_id,
        document_id,
    )

    def _early_return(res: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "embed_request_id=%s doc=%s embedding end status=%s",
            embed_request_id,
            document_id,
            res.get("status"),
        )
        request_ctx.__exit__(None, None, None)
        return res

    try:
        record = get_document_record(document_id) or {}
    except Exception as exc:  # noqa: BLE001
        error_message = _truncate_error_message(str(exc) or repr(exc))
        result["error"] = "document_record_failed"
        result["error_message"] = error_message
        result["failed_reason"] = "document_record_failed"
        return _early_return(result)
    current_status = record.get("status")
    if current_status in COMPLETED_STATUSES:
        result["status"] = "SKIPPED"
        result["failed_reason"] = None
        return _early_return(result)
    if current_status != STATUS_SCREENING_COMPLETED:
        result["status"] = "SKIPPED"
        result["error"] = "screening_not_completed"
        result["failed_reason"] = "screening_not_completed"
        return _early_return(result)

    try:
        subscription_id = resolve_subscription_id(
            document_id,
            subscription_id
            or record.get("subscription_id")
            or record.get("subscriptionId")
            or record.get("subscription"),
        )
        profile_id = resolve_profile_id(
            document_id,
            profile_id or record.get("profile_id") or record.get("profileId") or record.get("profile"),
        )
    except Exception as exc:  # noqa: BLE001
        error_message = _truncate_error_message(str(exc) or repr(exc))
        _safe_set_document_status(
            document_id,
            STATUS_TRAINING_FAILED,
            error_message,
            error_summary="resolve_ids_failed",
            cause=exc,
        )
        result["error"] = "resolve_ids_failed"
        result["error_message"] = error_message
        result["failed_reason"] = "resolve_ids_failed"
        return _early_return(result)

    lock = acquire_lock(stage="embedding", document_id=document_id, subscription_id=subscription_id)
    if not lock.acquired:
        logger.info("Embedding already in progress for %s; skipping duplicate.", document_id)
        result["status"] = "SKIPPED"
        result["error"] = "duplicate_embedding_in_progress"
        result["failed_reason"] = "duplicate_embedding_in_progress"
        return _early_return(result)

    _set_document_status(document_id, STATUS_TRAINING_STARTED)

    update_stage(
        document_id,
        "embedding",
        {"status": "IN_PROGRESS", "started_at": time.time(), "error": None, "reason": None},
    )

    try:
        try:
            extracted = load_extracted_pickle(document_id)
        except Exception as exc:  # noqa: BLE001
            error_message = _truncate_error_message(str(exc) or repr(exc))
            logger.error(
                "embed_request_id=%s doc=%s load pickle failed: %s",
                embed_request_id,
                document_id,
                exc,
                exc_info=True,
            )
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_message,
                exc=exc,
                run_id=embed_request_id,
                code="blob_read_failed",
            )
            _safe_update_stage(
                document_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": error_payload},
                cause=exc,
            )
            _safe_set_document_status(
                document_id,
                STATUS_TRAINING_FAILED,
                error_message,
                error_summary="blob_read_failed",
                cause=exc,
            )
            result["error"] = "blob_read_failed"
            result["error_message"] = error_message
            result["failed_reason"] = "blob_read_failed"
            return result

        extracted_docs, expected_chunks, coverage_values, prep_error = _prepare_extracted_docs(
            document_id=document_id,
            extracted=extracted,
            record=record,
            subscription_id=subscription_id,
        )
        if prep_error or extracted_docs is None or expected_chunks is None:
            reason = prep_error or "empty_extraction"
            message = {
                "empty_extraction": "empty extraction",
                "chunking_failed": "chunking failed",
            }.get(reason, "embedding preparation failed")
            error_payload = _build_error_payload(
                stage="embedding",
                message=message,
                run_id=embed_request_id,
                code=reason,
            )
            _safe_update_stage(
                document_id,
                "embedding",
                {
                    "status": "FAILED",
                    "completed_at": time.time(),
                    "reason": reason,
                    "error": error_payload,
                },
            )
            _safe_set_document_status(
                document_id,
                STATUS_TRAINING_FAILED,
                message,
                error_summary=reason,
            )
            result["error"] = reason
            result["error_message"] = message
            result["failed_reason"] = reason
            return result

        result["chunks_count"] = expected_chunks
        logger.info("Embedding pre-check for %s: expected_chunks=%s", document_id, expected_chunks)

        total_chunks = 0
        total_upserted = 0
        total_dropped = 0
        coverage_values = coverage_values or []
        try:
            for file_name, content in extracted_docs.items():
                try:
                    embed_result = train_on_document(content, subscription_id, profile_id, document_id, file_name)
                except Exception as exc:  # noqa: BLE001
                    if _is_meta_tensor_error(exc):
                        logger.warning(
                            "embed_request_id=%s doc=%s meta tensor error; retrying on cpu",
                            embed_request_id,
                            document_id,
                        )
                        try:
                            get_model(reload=True, device="cpu")
                            embed_result = train_on_document(
                                content,
                                subscription_id,
                                profile_id,
                                document_id,
                                file_name,
                                device="cpu",
                            )
                            logger.info(
                                "embed_request_id=%s doc=%s cpu fallback succeeded",
                                embed_request_id,
                                document_id,
                            )
                        except Exception as retry_exc:  # noqa: BLE001
                            raise retry_exc from exc
                    else:
                        raise
                total_chunks += int(embed_result.get("chunks", 0))
                total_upserted += int(embed_result.get("points_saved", 0))
                total_dropped += int(embed_result.get("dropped_chunks", 0))
                ratio = embed_result.get("coverage_ratio")
                if isinstance(ratio, (int, float)):
                    coverage_values.append(float(ratio))
        except ChunkingDiagnosticError as exc:
            error_message = _truncate_error_message(str(exc) or repr(exc))
            diagnostics = exc.diagnostics or {}
            logger.error(
                "embed_request_id=%s doc=%s chunking diagnostics failed: %s",
                embed_request_id,
                document_id,
                exc,
                exc_info=True,
            )
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_message,
                exc=exc,
                details=diagnostics,
                run_id=embed_request_id,
                code="extraction_or_chunking_failed",
            )
            _safe_update_stage(
                document_id,
                "embedding",
                {
                    "status": "FAILED",
                    "completed_at": time.time(),
                    "reason": "extraction_or_chunking_failed",
                    "error": error_payload,
                    "diagnostics": diagnostics,
                },
                cause=exc,
            )
            _safe_set_document_status(
                document_id,
                STATUS_EXTRACTION_OR_CHUNKING_FAILED,
                error_message,
                error_summary="extraction_or_chunking_failed",
                cause=exc,
            )
            result["error"] = "extraction_or_chunking_failed"
            result["error_message"] = error_message
            result["failed_reason"] = "extraction_or_chunking_failed"
            result["diagnostics"] = diagnostics
            return result
        except Exception as exc:  # noqa: BLE001
            error_message = _truncate_error_message(str(exc) or repr(exc))
            logger.error(
                "embed_request_id=%s doc=%s embedding failed: %s",
                embed_request_id,
                document_id,
                exc,
                exc_info=True,
            )
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_message,
                exc=exc,
                run_id=embed_request_id,
                code="training_failed",
            )
            _safe_update_stage(
                document_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": error_payload},
                cause=exc,
            )
            _safe_set_document_status(
                document_id,
                STATUS_TRAINING_FAILED,
                error_message,
                error_summary="training_failed",
                cause=exc,
            )
            result["error"] = "training_failed"
            result["error_message"] = error_message
            result["failed_reason"] = "training_failed"
            return result

        if total_chunks != total_upserted:
            error_msg = f"Embedding upsert mismatch: expected {total_chunks}, saved {total_upserted}"
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_msg,
                run_id=embed_request_id,
                code="qdrant_upsert_failed",
            )
            _safe_update_stage(
                document_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": error_payload},
            )
            _safe_set_document_status(
                document_id,
                STATUS_TRAINING_FAILED,
                error_msg,
                error_summary="qdrant_upsert_failed",
            )
            result["error"] = "qdrant_upsert_failed"
            result["error_message"] = error_msg
            result["failed_reason"] = "qdrant_upsert_failed"
            result["points_upserted"] = total_upserted
            return result

        collection_name = build_collection_name(subscription_id)
        logger.info(
            "Embedding results for %s: chunks=%s upserted=%s dropped=%s collection=%s",
            document_id,
            total_chunks,
            total_upserted,
            total_dropped,
            collection_name,
        )
        coverage_ratio = min(coverage_values) if coverage_values else None
        update_stage(
            document_id,
            "embedding",
            {
                "status": "COMPLETED",
                "completed_at": time.time(),
                "error": None,
                "chunking": {
                    "chunks": total_chunks,
                    "coverage_ratio": coverage_ratio,
                    "dropped_empty": total_dropped,
                },
                "qdrant": {"collection": collection_name, "expected": total_chunks, "upserted": total_upserted},
            },
        )

        _set_document_status(document_id, STATUS_TRAINING_COMPLETED, extra_fields=_training_success_fields())

        post_count: Optional[int] = None
        cleanup_allowed = True
        cleanup_error: Optional[Dict[str, Any]] = None
        if subscription_id and profile_id:
            try:
                post_count, cleanup_allowed = _verify_post_upsert_count(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    document_id=document_id,
                    expected_chunks=total_chunks,
                )
                if post_count is not None:
                    logger.info("Post-upsert Qdrant points for %s: %s", document_id, post_count)
                if not cleanup_allowed:
                    cleanup_error = (
                        {"message": "post_upsert_count_unavailable"}
                        if post_count is None
                        else {
                            "message": "post_upsert_count_mismatch",
                            "post_count": post_count,
                            "expected": total_chunks,
                        }
                    )
                    logger.warning(
                        "Skipping pickle cleanup for %s because embedding is not yet verified (post_count=%s, expected=%s)",
                        document_id,
                        post_count,
                        total_chunks,
                    )
            except Exception as exc:  # noqa: BLE001
                cleanup_allowed = False
                cleanup_error = {"message": "post_upsert_count_failed"}
                logger.warning(
                    "embed_request_id=%s doc=%s post-upsert count failed: %s",
                    embed_request_id,
                    document_id,
                    exc,
                )

        deleted = False
        if cleanup_allowed:
            try:
                deleted = delete_extracted_pickle(document_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_request_id=%s doc=%s local pickle delete failed: %s",
                    embed_request_id,
                    document_id,
                    exc,
                )
                deleted = False
            if not deleted:
                cleanup_error = {"message": "local_pickle_delete_failed"}
        cleanup_payload = None
        if not deleted:
            cleanup_details = cleanup_error or {"message": "cleanup_deferred"}
            cleanup_payload = _build_error_payload(
                stage="cleanup",
                message=cleanup_details.get("message", "cleanup_deferred"),
                run_id=embed_request_id,
                details=cleanup_details,
            )
        _safe_update_stage(
            document_id,
            "cleanup",
            {
                "pickle_deleted": deleted,
                "deleted_at": time.time() if deleted else None,
                "cleanup_pending": not deleted,
                "error": cleanup_payload,
            },
        )

        result["status"] = "COMPLETED"
        result["error"] = None
        result["chunks_count"] = total_chunks
        result["points_upserted"] = total_upserted
        result["failed_reason"] = None
        return result
    except Exception as exc:  # noqa: BLE001
        error_message = _truncate_error_message(str(exc) or repr(exc))
        logger.error(
            "embed_request_id=%s doc=%s embedding failed: %s",
            embed_request_id,
            document_id,
            exc,
            exc_info=True,
        )
        result["error"] = "training_failed"
        result["error_message"] = error_message
        result["failed_reason"] = "training_failed"
        _safe_set_document_status(
            document_id,
            STATUS_TRAINING_FAILED,
            error_message,
            error_summary="training_failed",
            cause=exc,
        )
        return result
    finally:
        if lock and lock.acquired:
            try:
                release_lock(lock)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_request_id=%s doc=%s lock release failed: %s",
                    embed_request_id,
                    document_id,
                    exc,
                )
        logger.info(
            "embed_request_id=%s doc=%s embedding end status=%s",
            embed_request_id,
            document_id,
            result.get("status"),
        )
        request_ctx.__exit__(*sys.exc_info())


def _build_embed_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    succeeded = [entry for entry in results if entry.get("status") == "COMPLETED"]
    failed = [entry for entry in results if entry.get("status") == "FAILED"]
    processed = len(succeeded) + len(failed)
    total_chunks = sum(int(entry.get("chunks_count") or 0) for entry in succeeded)
    total_points = sum(int(entry.get("points_upserted") or 0) for entry in succeeded)

    reason_counts: Dict[str, int] = {}
    for entry in failed:
        reason = str(entry.get("failed_reason") or entry.get("error") or "unknown_failure")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if not results:
        overall_status = "EMPTY"
    elif succeeded and not failed:
        overall_status = "COMPLETED"
    elif failed and not succeeded:
        overall_status = "FAILED"
    else:
        overall_status = "PARTIAL"

    return {
        "overall_status": overall_status,
        "documents_processed": processed,
        "documents_succeeded": len(succeeded),
        "documents_failed": len(failed),
        "total_chunks": total_chunks,
        "total_points_upserted": total_points,
        "failure_reasons": reason_counts,
        "documents": results,
    }


def _embed_from_local_pickles(
    *,
    document_id: Optional[str],
    document_ids: Optional[List[str]],
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
    max_blobs: Optional[int],
    embed_request_id: Optional[str],
) -> Dict[str, Any]:
    logger.warning("Blob storage not configured; using local pickles (deprecated).")
    requested_ids = _normalize_requested_ids(document_id, document_ids)
    filter_ids = _fetch_document_ids_by_filters(subscription_id=subscription_id, profile_id=profile_id)
    all_ids = requested_ids + filter_ids
    seen = set()
    ordered_ids = []
    for doc_id in all_ids:
        if not doc_id or not str(doc_id).strip():
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered_ids.append(doc_id)

    if not ordered_ids:
        return _build_embed_summary([])

    limit = _get_max_blobs(max_blobs)
    ordered_ids = ordered_ids[:limit]

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = _get_max_workers(len(ordered_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for index, doc_id in enumerate(ordered_ids):
            future = executor.submit(
                _process_local_document,
                document_id=doc_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                doc_type=doc_type,
                embed_request_id=embed_request_id,
            )
            future_map[future] = {"index": index, "document_id": doc_id}
        for future in as_completed(future_map):
            info = future_map[future]
            index = info["index"]
            doc_id = info["document_id"]
            try:
                results_by_index[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                error_message = _truncate_error_message(str(exc) or repr(exc))
                logger.error(
                    "embed_request_id=%s doc=%s embedding failed: %s",
                    embed_request_id,
                    doc_id,
                    exc,
                    exc_info=True,
                )
                if doc_id:
                    _safe_set_document_status(
                        doc_id,
                        STATUS_TRAINING_FAILED,
                        error_message,
                        error_summary="training_failed",
                        cause=exc,
                    )
                results_by_index[index] = _build_failed_result(
                    document_id=doc_id,
                    blob_name=f"{doc_id}.pkl",
                    error_message=error_message,
                )

    results = [results_by_index[index] for index in sorted(results_by_index)]
    return _build_embed_summary(results)


def embed_documents(
    *,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    max_blobs: Optional[int] = None,
) -> Dict[str, Any]:
    embed_request_id = str(uuid.uuid4())
    logger.info("embed_request_id=%s embedding request received", embed_request_id)
    if not blob_storage_configured():
        return _embed_from_local_pickles(
            document_id=document_id,
            document_ids=document_ids,
            subscription_id=subscription_id,
            profile_id=profile_id,
            doc_type=doc_type,
            max_blobs=max_blobs,
            embed_request_id=embed_request_id,
        )

    requested_ids = _normalize_requested_ids(document_id, document_ids)
    limit = _get_max_blobs(max_blobs)

    try:
        store = _build_blob_store()
    except BlobConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BlobConfigurationError(str(exc)) from exc

    blob_candidates = _select_blob_candidates(
        store,
        requested_ids,
        subscription_id,
        profile_id,
        limit,
    )

    telemetry = _telemetry()
    if telemetry:
        telemetry.increment("embed_pickles_listed_total", amount=len(blob_candidates))

    if not blob_candidates:
        return _build_embed_summary([])

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = _get_max_workers(len(blob_candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for index, blob in enumerate(blob_candidates):
            future = executor.submit(
                _process_blob,
                store=store,
                blob=blob,
                subscription_id=subscription_id,
                profile_id=profile_id,
                doc_type=doc_type,
                embed_request_id=embed_request_id,
            )
            doc_id_hint = (blob.metadata or {}).get("document_id") or extract_document_id(blob.name, prefix=store.prefix)
            future_map[future] = {"index": index, "blob": blob, "document_id": doc_id_hint}
        for future in as_completed(future_map):
            info = future_map[future]
            index = info["index"]
            doc_id = info["document_id"]
            blob = info["blob"]
            try:
                results_by_index[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                error_message = _truncate_error_message(str(exc) or repr(exc))
                logger.error(
                    "embed_request_id=%s doc=%s embedding failed: %s",
                    embed_request_id,
                    doc_id,
                    exc,
                    exc_info=True,
                )
                if doc_id:
                    _safe_set_document_status(
                        doc_id,
                        STATUS_TRAINING_FAILED,
                        error_message,
                        error_summary="training_failed",
                        cause=exc,
                    )
                results_by_index[index] = _build_failed_result(
                    document_id=doc_id,
                    blob_name=getattr(blob, "name", None),
                    error_message=error_message,
                )

    results = [results_by_index[index] for index in sorted(results_by_index)]
    for _entry in results:
        if telemetry:
            telemetry.increment("embed_pickles_processed_total")

    return _build_embed_summary(results)


def embedding_integrity_report(
    *,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    requested_ids = _normalize_requested_ids(document_id, document_ids)
    max_docs = int(limit or _get_max_blobs(None))
    max_docs = max(1, min(max_docs, 200))

    if not requested_ids:
        requested_ids = _fetch_document_ids_for_integrity(subscription_id, profile_id, max_docs)

    requested_ids = requested_ids[:max_docs]
    if not requested_ids:
        return {"documents": [], "summary": {"total": 0, "matched": 0, "mismatched": 0, "errors": 0}}

    results: List[Dict[str, Any]] = []
    for doc_id in requested_ids:
        entry: Dict[str, Any] = {"document_id": doc_id}
        try:
            record = get_document_record(doc_id) or {}
            entry["status"] = record.get("status")

            extracted, extracted_details = _load_extracted_for_doc(doc_id)
            entry["extracted"] = extracted_details

            expected_chunks: Optional[int] = None
            coverage_ratio: Optional[float] = None
            if extracted is not None:
                extracted_docs = _normalize_extracted_docs(extracted)
                expected_chunks, coverage_values = _screen_payload(extracted_docs, doc_id)
                if coverage_values:
                    coverage_ratio = min(coverage_values)
            else:
                chunking = ((record.get("embedding") or {}).get("chunking") or {}) if isinstance(record, dict) else {}
                if chunking:
                    expected_chunks = int(chunking.get("chunks") or 0) or None
                    cov_val = chunking.get("coverage_ratio")
                    coverage_ratio = float(cov_val) if isinstance(cov_val, (int, float)) else None
                    entry["extracted"]["source"] = "stage"

            entry["expected_chunks"] = expected_chunks
            if coverage_ratio is not None:
                entry["coverage_ratio"] = coverage_ratio

            try:
                resolved_subscription_id = resolve_subscription_id(
                    doc_id,
                    subscription_id
                    or record.get("subscription_id")
                    or record.get("subscriptionId")
                    or record.get("subscription"),
                )
                resolved_profile_id = resolve_profile_id(
                    doc_id,
                    profile_id
                    or record.get("profile_id")
                    or record.get("profileId")
                    or record.get("profile"),
                )
                entry["subscription_id"] = resolved_subscription_id
                entry["profile_id"] = resolved_profile_id
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"resolve_ids_failed: {exc}"
                results.append(entry)
                continue

            try:
                qdrant_count = _count_qdrant_points(
                    entry["subscription_id"],
                    entry["profile_id"],
                    doc_id,
                )
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"qdrant_count_failed: {exc}"
                results.append(entry)
                continue

            entry["qdrant_count"] = qdrant_count
            if expected_chunks is not None:
                delta = qdrant_count - expected_chunks
                entry["delta"] = delta
                entry["matches"] = delta == 0
            else:
                entry["matches"] = None

        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        results.append(entry)

    summary = {"total": len(results), "matched": 0, "mismatched": 0, "errors": 0}
    for entry in results:
        if entry.get("error"):
            summary["errors"] += 1
            continue
        matches = entry.get("matches")
        if matches is True:
            summary["matched"] += 1
        elif matches is False:
            summary["mismatched"] += 1

    return {"documents": results, "summary": summary}
