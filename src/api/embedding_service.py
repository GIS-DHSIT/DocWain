from src.utils.logging_utils import get_logger
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
from src.api.content_store import load_extracted_pickle, save_extracted_pickle
try:
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
except Exception as _datahandler_exc:  # noqa: BLE001
    class ChunkingDiagnosticError(RuntimeError):
        pass

    db = None

    def _datahandler_unavailable(*_args, **_kwargs):
        raise RuntimeError("dataHandler unavailable") from _datahandler_exc

    decrypt_data = _datahandler_unavailable
    fileProcessor = _datahandler_unavailable
    get_azure_docs = _datahandler_unavailable
    get_qdrant_client = _datahandler_unavailable
    get_s3_client = _datahandler_unavailable
    get_subscription_pii_setting = _datahandler_unavailable
    get_model = _datahandler_unavailable
    read_s3_file = _datahandler_unavailable
    resolve_profile_id = _datahandler_unavailable
    resolve_subscription_id = _datahandler_unavailable
    train_on_document = _datahandler_unavailable
    update_extraction_metadata = _datahandler_unavailable
    update_pii_stats = _datahandler_unavailable
from src.api.document_status import emit_progress, get_document_record, update_document_fields, update_stage
from src.api.pipeline_models import ExtractedDocument
from dataclasses import is_dataclass, asdict
from src.api.structured_extraction import StructuredDocument
from src.api.pii_masking import mask_document_content
from src.api.statuses import (
    STATUS_EMBEDDING_COMPLETED,
    STATUS_EXTRACTION_COMPLETED,
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

try:
    from src.docwain_intel.integration import run_intel_pipeline_hook, INTEL_PIPELINE_ENABLED
    from src.docwain_intel.extraction import build_document_json_from_extracted
except ImportError:
    INTEL_PIPELINE_ENABLED = False
    run_intel_pipeline_hook = None  # type: ignore[assignment]
    build_document_json_from_extracted = None  # type: ignore[assignment]

logger = get_logger(__name__)

COMPLETED_STATUSES = {
    STATUS_EMBEDDING_COMPLETED,
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

def _ingest_chunks_to_knowledge_graph(
    document_id: str,
    subscription_id: str,
    profile_id: str,
    doc_name: str,
    extracted_docs: Dict[str, Any],
) -> None:
    """Non-blocking chunk-level KG ingestion after Qdrant upsert.

    Builds a graph payload from the embedding texts+metadata and enqueues
    for async processing.  KG failure must never block embedding.
    """
    try:
        from src.kg.ingest import build_graph_payload, get_graph_ingest_queue

        texts: List[str] = []
        chunk_metadata: List[Dict[str, Any]] = []
        for fname, content in (extracted_docs or {}).items():
            if isinstance(content, dict):
                raw_texts = content.get("texts") or []
                raw_meta = content.get("chunk_metadata") or []
                if isinstance(raw_texts, list):
                    texts.extend(raw_texts)
                if isinstance(raw_meta, list):
                    chunk_metadata.extend(raw_meta)

        if not texts:
            return

        graph_payload = build_graph_payload(
            embeddings_payload={"texts": texts, "chunk_metadata": chunk_metadata},
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            document_id=str(document_id),
            doc_name=doc_name,
        )
        if graph_payload:
            queue = get_graph_ingest_queue()
            queue.enqueue(graph_payload)
            logger.info("KG chunk-level ingestion enqueued for %s (%d texts)", document_id, len(texts))
    except Exception as exc:  # noqa: BLE001
        logger.debug("KG chunk ingestion skipped for %s: %s", document_id, exc)

def _get_max_workers(total: int) -> int:
    max_workers_env = os.getenv("EMBEDDING_MAX_WORKERS")
    try:
        max_workers = int(max_workers_env) if max_workers_env else None
    except ValueError:
        max_workers = None
    if max_workers is None:
        # Default to 2: prevents GPU OOM from concurrent docs and
        # avoids CPU thrashing when GPU is unavailable.
        max_workers = min(total, 2)
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

def _extract_full_text_from_pickle(extracted: Any) -> str:
    """Extract full document text from a deserialized pickle."""
    if extracted is None:
        return ""
    # Handle dict with 'raw' key containing ExtractedDocument objects
    if isinstance(extracted, dict):
        for key in ("raw", "structured", "document"):
            content = extracted.get(key)
            if isinstance(content, dict):
                for _fname, _doc in content.items():
                    if hasattr(_doc, "full_text") and _doc.full_text:
                        return _doc.full_text
                    if isinstance(_doc, dict) and _doc.get("full_text"):
                        return _doc["full_text"]
            elif hasattr(content, "full_text") and content.full_text:
                return content.full_text
        # Try direct full_text
        if extracted.get("full_text"):
            return extracted["full_text"]
    elif hasattr(extracted, "full_text") and extracted.full_text:
        return extracted.full_text
    return ""


def _extract_full_text_from_qdrant(
    doc_id: str, subscription_id: str, profile_id: str
) -> str:
    """Reconstruct document text from existing Qdrant chunks when pickle is unavailable."""
    try:
        from src.api.dataHandler import get_vector_store
        from src.api.vector_store import build_collection_name
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        _vs = get_vector_store()
        collection = build_collection_name(subscription_id)
        points, _ = _vs.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
            ]),
            limit=100,
            with_payload=["canonical_text", "resolution", "chunk_index"],
            with_vectors=False,
        )
        chunks = []
        for p in points:
            payload = p.payload or {}
            if payload.get("resolution") in ("doc_index", "doc_intelligence"):
                continue
            chunks.append({
                "text": payload.get("canonical_text", ""),
                "index": payload.get("chunk_index", 0),
            })
        chunks.sort(key=lambda c: c["index"])
        full_text = "\n".join(c["text"] for c in chunks if c["text"])
        if full_text:
            logger.info("[DOC_INTELLIGENCE] Reconstructed %d chars from %d Qdrant chunks for %s",
                       len(full_text), len(chunks), doc_id)
        return full_text
    except Exception as exc:
        logger.debug("[DOC_INTELLIGENCE] Qdrant text reconstruction failed: %s", exc)
        return ""


def _check_doc_intelligence_exists(doc_id: str, subscription_id: str, profile_id: str) -> bool:
    """Check if doc_index point already exists in Qdrant for this document."""
    try:
        from src.api.vector_store import build_collection_name
        from src.api.dataHandler import get_vector_store
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        _vs = get_vector_store()
        _count = _vs.client.count(
            collection_name=build_collection_name(subscription_id),
            count_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="resolution", match=MatchValue(value="doc_index")),
            ]),
        ).count
        return _count > 0
    except Exception:
        return False


def _upsert_doc_intelligence(
    doc_id: str,
    subscription_id: str,
    profile_id: str,
    source_filename: str,
    full_text: str,
    collection_name: str,
) -> None:
    """Extract document intelligence and upsert doc_index + doc_intelligence points."""
    import uuid

    from src.extraction.document_intelligence import (
        extract_document_intelligence,
        build_doc_index_text,
        build_doc_intelligence_text,
    )
    from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
    from src.api.dataHandler import get_vector_store, get_model

    intelligence = extract_document_intelligence(full_text, source_filename)
    doc_index_text = build_doc_index_text(source_filename, intelligence)
    doc_intel_text = build_doc_intelligence_text(source_filename, intelligence)

    if not doc_index_text.strip():
        logger.warning("[DOC_INTELLIGENCE] Empty doc_index text for %s", doc_id)
        return

    model = get_model()
    _vs = get_vector_store()

    # Build and upsert doc_index point
    _base = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": doc_id,
        "source_name": source_filename,
        "resolution": "doc_index",
        "chunk_kind": "doc_index",
        "chunk_id": f"doc_index_{doc_id}",
        "chunk_index": 0,
        "section_title": "Document Index",
        "canonical_text": doc_index_text,
        "embedding_text": doc_index_text,
    }
    idx_payload = build_qdrant_payload(_base)
    idx_payload["doc_intelligence"] = intelligence

    _base_intel = {**_base,
        "canonical_text": doc_intel_text,
        "embedding_text": doc_intel_text,
        "resolution": "doc_intelligence",
        "chunk_kind": "doc_intelligence",
        "chunk_id": f"doc_intelligence_{doc_id}",
        "section_title": "Document Intelligence",
    }
    intel_payload = build_qdrant_payload(_base_intel)
    intel_payload["doc_intelligence"] = intelligence

    # Encode vectors
    idx_vector = model.encode([doc_index_text])[0].tolist()
    intel_vector = model.encode([doc_intel_text])[0].tolist()

    from qdrant_client.models import PointStruct
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_index_{doc_id}")),
            vector={"content_vector": idx_vector},
            payload=idx_payload,
        ),
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_intelligence_{doc_id}")),
            vector={"content_vector": intel_vector},
            payload=intel_payload,
        ),
    ]
    _vs.client.upsert(collection_name=collection_name, points=points)
    logger.info("[DOC_INTELLIGENCE] Upserted doc_index + doc_intelligence for %s (%s)", doc_id, source_filename)


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

    filters: List[Dict[str, Any]] = [{"status": {"$in": [STATUS_SCREENING_COMPLETED]}}]
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
        STATUS_EXTRACTION_COMPLETED,
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
    if not blob_storage_configured():
        return None, details
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

def _min_chars_threshold() -> int:
    raw = os.getenv("EMBEDDING_MIN_CHARS", "50")
    try:
        return max(1, int(raw))
    except ValueError:
        return 50

def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg

def _is_cuda_oom(exc: Exception) -> bool:
    """Detect CUDA out-of-memory errors including CUBLAS allocation failures."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or ("cuda error" in msg and "alloc" in msg)
    )

def _clear_gpu_cache() -> None:
    """Best-effort GPU cache clear after CUDA OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

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

def _safe_text_item(item: Any) -> str:
    """Extract text from a list item without stringifying dicts/objects."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "full_text", "raw_text", "canonical_text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""
    if hasattr(item, "text"):
        val = getattr(item, "text", "")
        if isinstance(val, str) and val.strip():
            return val
    return ""

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
            content["texts"] = [normalize_text(_safe_text_item(t)) for t in content.get("texts") or []]
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
        logger.warning("Extracted payload missing required schema keys for %s: %s; attempting re-extraction", document_id, invalid_docs)
        # Attempt re-extraction before giving up
        fallback_docs = _reextract_from_source(
            document_id=document_id,
            record=record,
            subscription_id=subscription_id,
        )
        if fallback_docs:
            extracted_docs = _normalize_extracted_docs(fallback_docs)
            for doc_name, content in list(extracted_docs.items()):
                extracted_docs[doc_name] = _normalize_content_in_place(content)
                extracted_docs[doc_name] = _normalize_metadata_in_place(
                    extracted_docs[doc_name], document_id=document_id,
                )
            invalid_docs = [doc_name for doc_name, content in extracted_docs.items() if not _payload_has_required_schema(content)]
        if invalid_docs:
            if metrics_store.available:
                metrics_store.record(counters={"empty_docs_count": len(invalid_docs)}, document_id=document_id, agent="embedding")
            logger.warning("Re-extraction also failed for %s: %s", document_id, invalid_docs)
            return None, None, [], "empty_extraction"

    assessment = _assess_extracted_docs(extracted_docs)
    logger.info(
        "Extraction assessment for %s: total_chars=%s coverage=%s",
        document_id,
        assessment.get("total_chars"),
        assessment.get("coverage_values"),
    )

    if not assessment.get("has_data"):
        # Try re-extraction as last resort before returning empty
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
        if chunks:
            return chunks, chunk_metadata, coverage_ratio, dropped
        # Layout graph produced 0 chunks — fall through to SectionChunker
        logger.info("LayoutGraph produced 0 chunks for %s; falling back to SectionChunker", doc_name)
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
    """
    Estimate the number of chunks that will be produced from content.

    Improved to properly handle cases where texts array is empty but full_text exists.
    """
    if isinstance(content, ExtractedDocument):
        chunks, _, coverage_ratio, _ = _build_chunks_for_extracted_doc(
            content, document_id=document_id, doc_name=doc_name
        )
        return len(chunks), coverage_ratio

    if isinstance(content, dict):
        # Check for pre-computed texts array with actual content
        texts_list = content.get("texts")
        if isinstance(texts_list, list):
            non_empty_texts = [text for text in texts_list if isinstance(text, str) and text.strip()]
            if non_empty_texts:
                return len(non_empty_texts), None
            # texts array exists but is empty - fall through to try full_text

        # Check for pre-computed embeddings
        if isinstance(content.get("embeddings"), (list, tuple)):
            embeddings = content.get("embeddings") or []
            if embeddings:
                return len(embeddings), None

        # Check for pre-computed chunk_metadata
        chunk_meta = content.get("chunk_metadata")
        if isinstance(chunk_meta, list) and chunk_meta:
            return len(chunk_meta), None

        # Try to chunk from full_text or equivalent fields
        text_value = content.get("full_text") or content.get("text") or content.get("content")
        if isinstance(text_value, str) and text_value.strip():
            try:
                chunker = SectionChunker()
                section_chunks = chunker.chunk_document(
                    text_value,
                    doc_internal_id=document_id,
                    source_filename=doc_name
                )
                if section_chunks:
                    return len(section_chunks), None
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section chunk estimation failed for %s: %s", doc_name, exc)
                # Don't return 0 yet - try other sources

        # Try to extract from sections
        sections = content.get("sections")
        if isinstance(sections, list) and sections:
            section_texts = []
            for sec in sections:
                if isinstance(sec, dict):
                    sec_text = (sec.get("text") or sec.get("content") or "").strip()
                    if sec_text:
                        section_texts.append(sec_text)
            if section_texts:
                try:
                    chunker = SectionChunker()
                    joined = "\n\n".join(section_texts)
                    section_chunks = chunker.chunk_document(
                        joined,
                        doc_internal_id=document_id,
                        source_filename=doc_name
                    )
                    if section_chunks:
                        return len(section_chunks), None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Section chunk estimation from sections failed for %s: %s", doc_name, exc)

        # Try to extract from pages
        pages = content.get("pages")
        if isinstance(pages, list) and pages:
            pages_text: List[str] = []
            for page in pages:
                if isinstance(page, dict):
                    page_text = (page.get("text") or page.get("content") or "").strip()
                    if page_text:
                        pages_text.append(page_text)
                elif isinstance(page, str) and page.strip():
                    pages_text.append(page.strip())
            if pages_text:
                joined = "\n\n".join(pages_text)
                try:
                    chunker = SectionChunker()
                    section_chunks = chunker.chunk_document(
                        joined,
                        doc_internal_id=document_id,
                        source_filename=doc_name
                    )
                    if section_chunks:
                        return len(section_chunks), None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Section chunk estimation failed for %s pages: %s", doc_name, exc)

        # Last resort: if we have any text, use simple character-based estimation
        text_value = content.get("full_text") or content.get("text") or content.get("content")
        if isinstance(text_value, str) and len(text_value.strip()) > 100:
            # Rough estimate: ~500 chars per chunk on average
            estimated_chunks = max(1, len(text_value.strip()) // 500)
            logger.info("Using character-based chunk estimate for %s: %d chunks", doc_name, estimated_chunks)
            return estimated_chunks, None

    if isinstance(content, str) and content.strip():
        try:
            chunker = SectionChunker()
            section_chunks = chunker.chunk_document(
                content,
                doc_internal_id=document_id,
                source_filename=doc_name
            )
            if section_chunks:
                return len(section_chunks), None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Section chunk estimation failed for %s: %s", doc_name, exc)
            # Fall back to character-based estimation
            if len(content.strip()) > 100:
                estimated_chunks = max(1, len(content.strip()) // 500)
                return estimated_chunks, None

    return 0, None

def _normalize_extracted_docs(extracted: Any) -> Dict[str, Any]:
    """
    Normalize extracted document payload to a consistent format for embedding.

    Handles multiple input formats:
    - New format: {"raw": {...}, "structured": {...}, "intelligence": {...}}
    - Legacy format: ExtractedDocument dataclass
    - Simple dict with texts/embeddings/chunk_metadata
    - Raw string

    Key improvement: Falls back to raw extraction when structured extraction
    produces empty/insufficient content.
    """
    # New format: payload may be {"raw": {...}, "structured": {...}}
    if isinstance(extracted, dict) and ("raw" in extracted or "structured" in extracted):
        structured = extracted.get("structured") or {}
        raw = extracted.get("raw") or {}

        structured_norm = None
        raw_norm = None

        # Try to normalize structured
        if structured:
            structured_norm = _normalize_structured_payload(structured)
            if not _has_useful_content(structured_norm):
                structured_norm = None

        # Also try raw extraction — it often has better section granularity
        if raw:
            raw_norm = _normalize_raw_payload(raw)
            if not raw_norm:
                raw_norm = None

        # Pick the better normalization: prefer the one with more texts (sections)
        if structured_norm and raw_norm:
            s_texts = sum(len(d.get("texts", [])) for d in structured_norm.values() if isinstance(d, dict))
            r_texts = sum(len(d.get("texts", [])) for d in raw_norm.values() if isinstance(d, dict))
            if r_texts > s_texts:
                logger.info(
                    "Raw extraction has better section granularity (%d vs %d); using raw",
                    r_texts, s_texts,
                )
                return raw_norm
            return structured_norm
        if structured_norm:
            return structured_norm
        if raw_norm:
            return raw_norm

        # Last resort: try to extract any usable content
        logger.warning("Both structured and raw extraction failed; attempting recovery")
        return _recover_from_payload(extracted)

    if isinstance(extracted, ExtractedDocument):
        return {"document": extracted}
    if isinstance(extracted, dict):
        if {"texts", "embeddings", "chunk_metadata", "full_text"}.intersection(extracted.keys()):
            return {"document": extracted}
        return extracted
    if isinstance(extracted, str):
        return {"document": extracted}
    raise ValueError(f"Unsupported extracted payload type: {type(extracted)}")

def _salvage_repr_text(text: str) -> str:
    """Salvage real content from ExtractedDocument/Section repr strings."""
    if not text or not isinstance(text, str):
        return text or ""
    from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
    if not _is_metadata_garbage(text):
        return text
    from src.embedding.pipeline.embed_pipeline import _salvage_chunk_text
    salvaged = _salvage_chunk_text(text)
    if salvaged and len(salvaged.strip()) >= 20:
        return salvaged
    return text

def _normalize_structured_payload(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Convert structured extraction to embedding-compatible format."""
    normalized: Dict[str, Any] = {}

    for name, value in structured.items():
        try:
            if is_dataclass(value):
                sd = asdict(value)
            elif isinstance(value, dict):
                sd = value
            else:
                # Check if it's an object with raw_text/full_text attrs before str()
                raw_text_attr = getattr(value, "raw_text", None) or getattr(value, "full_text", None)
                if isinstance(raw_text_attr, str) and raw_text_attr.strip():
                    text_content = _salvage_repr_text(raw_text_attr.strip())
                else:
                    text_content = _salvage_repr_text(str(value).strip())
                from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
                if _is_metadata_garbage(text_content):
                    logger.debug("Skipping garbage text from structured value for %s", name)
                    continue
                if text_content:
                    normalized[name] = {
                        "full_text": text_content,
                        "sections": [{"text": text_content, "start_page": 1, "end_page": 1}],
                        "texts": [text_content],
                    }
                continue

            # Extract full text - prefer raw_text, fall back to joining sections
            section_contents = []
            for sec in sd.get("sections", []):
                content = _salvage_repr_text((sec.get("content") or sec.get("text") or "").strip())
                if content:
                    section_contents.append(content)

            full_text = _salvage_repr_text((sd.get("raw_text") or sd.get("full_text") or "").strip())
            if not full_text and section_contents:
                full_text = "\n\n".join(section_contents)

            # If no full_text at all, skip this document
            if not full_text:
                logger.warning("Structured document %s has no extractable text", name)
                continue

            # Build sections list
            sections = []
            texts = []
            chunk_meta = []

            for idx, sec in enumerate(sd.get("sections", [])):
                raw_content = _salvage_repr_text((sec.get("content") or sec.get("text") or "").strip())
                # Prepend section title to content for better embedding context
                sec_title = (sec.get("title") or sec.get("heading") or sec.get("section_type") or "").strip()
                if sec_title and raw_content and sec_title.lower() not in raw_content[:80].lower():
                    text_val = f"{sec_title}\n{raw_content}"
                else:
                    text_val = raw_content
                sections.append({
                    "text": text_val,
                    "start_page": sec.get("start_page") or 1,
                    "end_page": sec.get("end_page") or 1,
                })
                if text_val:
                    texts.append(text_val)
                    chunk_meta.append({
                        "document_id": sd.get("document_id"),
                        "section_title": sec.get("section_type") or sec.get("title") or "Section",
                        "section_path": sec.get("section_type") or sec.get("title") or "Section",
                        "page_start": sec.get("start_page") or 1,
                        "page_end": sec.get("end_page") or 1,
                        "page_number": sec.get("start_page") or 1,
                        "chunk_index": idx,
                        "chunk_type": "text",
                        "doc_type": sd.get("document_type"),
                        "sentence_complete": text_val.endswith(('.', '?', '!')),
                    })

            # Merge short adjacent sections to prevent content loss during chunking.
            # The integrity enforcer groups by section_id, so very short sections
            # (e.g. "Diagnosis: 43 chars") stay isolated and get dropped by the
            # validity filter (min_chars=80). Merging them here preserves all content.
            _MIN_SECTION_LEN = 200
            if len(texts) > 1:
                merged_texts = []
                merged_sections = []
                merged_meta = []
                i = 0
                while i < len(texts):
                    current_text = texts[i]
                    current_section = dict(sections[i])
                    current_meta = dict(chunk_meta[i])
                    # Keep merging forward while current chunk is short
                    while len(current_text) < _MIN_SECTION_LEN and i + 1 < len(texts):
                        i += 1
                        current_text = f"{current_text}\n\n{texts[i]}"
                        current_section["end_page"] = sections[i].get("end_page") or current_section.get("end_page", 1)
                        combined_title = current_meta.get("section_title", "")
                        next_title = chunk_meta[i].get("section_title", "")
                        if next_title and next_title not in combined_title:
                            current_meta["section_title"] = f"{combined_title} + {next_title}"
                            current_meta["section_path"] = current_meta["section_title"]
                    current_section["text"] = current_text
                    current_meta["chunk_index"] = len(merged_texts)
                    merged_texts.append(current_text)
                    merged_sections.append(current_section)
                    merged_meta.append(current_meta)
                    i += 1
                # Merge last chunk backward if still too short
                if len(merged_texts) > 1 and len(merged_texts[-1]) < _MIN_SECTION_LEN:
                    merged_texts[-2] = f"{merged_texts[-2]}\n\n{merged_texts[-1]}"
                    merged_sections[-2]["text"] = merged_texts[-2]
                    merged_sections[-2]["end_page"] = merged_sections[-1].get("end_page", 1)
                    merged_texts.pop()
                    merged_sections.pop()
                    merged_meta.pop()
                texts = merged_texts
                sections = merged_sections
                chunk_meta = merged_meta

            # If no section texts but we have full_text, use full_text as single chunk
            if not texts and full_text:
                texts = [full_text]
                sections = [{"text": full_text, "start_page": 1, "end_page": 1}]
                chunk_meta = [{
                    "document_id": sd.get("document_id"),
                    "section_title": "Content",
                    "section_path": "Content",
                    "page_start": 1,
                    "page_end": sd.get("total_pages") or 1,
                    "page_number": 1,
                    "chunk_index": 0,
                    "chunk_type": "text",
                    "doc_type": sd.get("document_type"),
                    "sentence_complete": True,
                }]

            normalized[name] = {
                "full_text": full_text,
                "sections": sections,
                "texts": texts,
                "chunk_metadata": chunk_meta,
                "doc_type": sd.get("document_type"),
                "doc_domain": (sd.get("document_classification") or {}).get("domain", "generic"),
            }

        except Exception as exc:
            logger.warning("Failed to normalize structured document %s: %s", name, exc)
            # Try raw_text/full_text attrs — never fall back to str() on ExtractedDocument
            raw_text_attr = getattr(value, "raw_text", None) or getattr(value, "full_text", None)
            if isinstance(raw_text_attr, str) and raw_text_attr.strip():
                text_content = raw_text_attr.strip()
            elif isinstance(value, ExtractedDocument):
                # Use dedicated extraction to avoid garbage repr
                from src.api.extraction_service import _extract_text_from_extracted_document
                text_content = _extract_text_from_extracted_document(value)
            elif isinstance(value, str):
                text_content = value.strip()
            else:
                text_content = ""
            from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
            if text_content and not _is_metadata_garbage(text_content):
                normalized[name] = {
                    "full_text": text_content,
                    "sections": [{"text": text_content, "start_page": 1, "end_page": 1}],
                    "texts": [text_content],
                }

    return normalized

def _normalize_raw_payload(raw: Any) -> Dict[str, Any]:
    """Convert raw extraction to embedding-compatible format."""
    if not raw:
        return {}

    if isinstance(raw, str):
        text = raw.strip()
        if text:
            return {"document": {"full_text": text, "texts": [text]}}
        return {}

    if isinstance(raw, dict):
        # Check if it's already in the right format
        if {"texts", "embeddings", "chunk_metadata", "full_text"}.intersection(raw.keys()):
            return {"document": raw}

        # Process each document in raw
        normalized: Dict[str, Any] = {}
        for name, content in raw.items():
            if isinstance(content, ExtractedDocument):
                # Extract sections with titles from the ExtractedDocument
                sections_list = getattr(content, "sections", None) or []
                full_text = getattr(content, "full_text", None) or ""
                # Also try chunk_candidates and tables when sections/full_text are empty
                if not full_text and not sections_list:
                    candidate_texts = [cand.text.strip() for cand in (getattr(content, "chunk_candidates", None) or []) if (getattr(cand, "text", "") or "").strip()]
                    table_texts = [t.text.strip() for t in (getattr(content, "tables", None) or []) if (getattr(t, "text", "") or "").strip()]
                    all_fallback = candidate_texts + table_texts
                    if all_fallback:
                        full_text = "\n\n".join(all_fallback)
                if sections_list and len(sections_list) >= 1:
                    texts = []
                    chunk_meta = []
                    sections_data = []
                    for idx, sec in enumerate(sections_list):
                        sec_title = getattr(sec, "title", "") or ""
                        sec_text = getattr(sec, "text", "") or getattr(sec, "content", "") or ""
                        sec_text = sec_text.strip()
                        if not sec_text:
                            continue
                        # Prepend section title for embedding context
                        if sec_title and sec_title.lower() not in sec_text[:80].lower():
                            text_val = f"{sec_title}\n{sec_text}"
                        else:
                            text_val = sec_text
                        texts.append(text_val)
                        sections_data.append({
                            "text": text_val,
                            "start_page": getattr(sec, "start_page", 1) or 1,
                            "end_page": getattr(sec, "end_page", 1) or 1,
                        })
                        chunk_meta.append({
                            "document_id": None,
                            "section_title": sec_title or "Section",
                            "section_path": sec_title or "Section",
                            "page_start": getattr(sec, "start_page", 1) or 1,
                            "page_end": getattr(sec, "end_page", 1) or 1,
                            "page_number": getattr(sec, "start_page", 1) or 1,
                            "chunk_index": idx,
                            "chunk_type": "text",
                            "doc_type": getattr(content, "doc_type", None),
                            "sentence_complete": text_val.endswith((".", "?", "!")),
                        })
                    # Merge short adjacent sections so they survive validity filters
                    _MIN_SECTION_LEN = 200  # noqa: N806
                    if len(texts) > 1:
                        merged_texts = []
                        merged_sections = []
                        merged_meta = []
                        i = 0
                        while i < len(texts):
                            cur_text = texts[i]
                            cur_sec = dict(sections_data[i])
                            cur_meta = dict(chunk_meta[i])
                            # Forward-merge while current chunk is too short
                            while len(cur_text) < _MIN_SECTION_LEN and i + 1 < len(texts):
                                i += 1
                                cur_text = f"{cur_text}\n\n{texts[i]}"
                                cur_sec["end_page"] = sections_data[i].get("end_page") or cur_sec.get("end_page", 1)
                                next_title = chunk_meta[i].get("section_title", "")
                                if next_title and next_title not in cur_meta.get("section_title", ""):
                                    cur_meta["section_title"] = f"{cur_meta.get('section_title', '')} + {next_title}"
                                    cur_meta["section_path"] = cur_meta["section_title"]
                            cur_sec["text"] = cur_text
                            cur_meta["chunk_index"] = len(merged_texts)
                            merged_texts.append(cur_text)
                            merged_sections.append(cur_sec)
                            merged_meta.append(cur_meta)
                            i += 1
                        # Backward-merge last chunk if still too short
                        if len(merged_texts) > 1 and len(merged_texts[-1]) < _MIN_SECTION_LEN:
                            merged_texts[-2] = f"{merged_texts[-2]}\n\n{merged_texts[-1]}"
                            merged_sections[-2]["text"] = merged_texts[-2]
                            merged_sections[-2]["end_page"] = merged_sections[-1].get("end_page", 1)
                            merged_texts.pop()
                            merged_sections.pop()
                            merged_meta.pop()
                        texts = merged_texts
                        sections_data = merged_sections
                        chunk_meta = merged_meta

                    if texts:
                        normalized[name] = {
                            "full_text": full_text or "\n\n".join(texts),
                            "texts": texts,
                            "sections": sections_data,
                            "chunk_metadata": chunk_meta,
                            "doc_type": getattr(content, "doc_type", None),
                        }
                        continue
                # Fallback: run full-section chunker on full_text to get proper sections
                if full_text and full_text.strip():
                    try:
                        from src.embedding.chunking.section_chunker import SectionChunker
                        _chunker = SectionChunker()
                        # Pass full_text as string — chunker infers sections from text
                        _chunks = _chunker.chunk_document(full_text, doc_internal_id="", source_filename=name)
                        if _chunks and len(_chunks) >= 1:
                            texts = []
                            chunk_meta = []
                            sections_data = []
                            for idx, ch in enumerate(_chunks):
                                ch_text = ch.text.strip()
                                if not ch_text or len(ch_text) < 50:
                                    continue
                                # Prepend section title
                                if ch.section_title and ch.section_title.lower() not in ch_text[:80].lower():
                                    ch_text = f"{ch.section_title}\n{ch_text}"
                                texts.append(ch_text)
                                sections_data.append({
                                    "text": ch_text,
                                    "start_page": ch.page_start or 1,
                                    "end_page": ch.page_end or 1,
                                })
                                chunk_meta.append({
                                    "document_id": None,
                                    "section_title": ch.section_title or "Section",
                                    "section_path": ch.section_path or ch.section_title or "Section",
                                    "page_start": ch.page_start or 1,
                                    "page_end": ch.page_end or 1,
                                    "page_number": ch.page_start or 1,
                                    "chunk_index": idx,
                                    "chunk_type": "section_full",
                                    "doc_type": getattr(content, "doc_type", None),
                                    "sentence_complete": ch_text.endswith((".", "?", "!")),
                                })
                            if texts:
                                logger.info(
                                    "[SECTION_CHUNKER] Produced %d full-section chunks for %s (avg %d chars)",
                                    len(texts), name, sum(len(t) for t in texts) // max(len(texts), 1),
                                )
                                normalized[name] = {
                                    "full_text": full_text,
                                    "texts": texts,
                                    "sections": sections_data,
                                    "chunk_metadata": chunk_meta,
                                    "doc_type": getattr(content, "doc_type", None),
                                }
                                continue
                    except Exception as _sc_exc:
                        logger.debug("[SECTION_CHUNKER] Fallback chunking failed: %s", _sc_exc)

                    # Final fallback: full_text as single chunk
                    normalized[name] = {
                        "full_text": full_text,
                        "texts": [full_text],
                        "sections": [{"text": full_text, "start_page": 1, "end_page": 1}],
                        "doc_type": getattr(content, "doc_type", None),
                    }
                else:
                    # Last resort: store ExtractedDocument as-is for legacy handling
                    normalized[name] = content
            elif isinstance(content, dict):
                # Extract text from various possible fields
                full_text = (
                    content.get("full_text") or
                    content.get("text") or
                    content.get("content") or
                    ""
                ).strip()

                # Try to get text from sections
                if not full_text and content.get("sections"):
                    section_texts = []
                    for sec in content.get("sections", []):
                        if isinstance(sec, dict):
                            sec_text = (sec.get("text") or sec.get("content") or "").strip()
                            if sec_text:
                                section_texts.append(sec_text)
                    full_text = "\n\n".join(section_texts)

                # Try to get text from pages
                if not full_text and content.get("pages"):
                    page_texts = []
                    for page in content.get("pages", []):
                        if isinstance(page, dict):
                            page_text = (page.get("text") or page.get("content") or "").strip()
                            if page_text:
                                page_texts.append(page_text)
                        elif isinstance(page, str):
                            page_texts.append(page.strip())
                    full_text = "\n\n".join(page_texts)

                if full_text:
                    # Preserve existing texts array if present, otherwise create from full_text
                    texts = content.get("texts")
                    if not texts or not any(t.strip() for t in texts if isinstance(t, str)):
                        texts = [full_text]

                    # Include translated English text alongside original for non-English docs
                    translated_text = content.get("translated_text")
                    if translated_text and content.get("detected_language", "en") != "en":
                        # Append translated version of each section for bilingual embedding
                        sections = content.get("sections", [{"text": full_text, "start_page": 1, "end_page": 1}])
                        translated_sections = []
                        translated_texts = []
                        for sec in sections:
                            if isinstance(sec, dict):
                                sec_translated = sec.get("translated_text")
                                if sec_translated:
                                    translated_sections.append({
                                        **sec,
                                        "text": sec_translated,
                                        "chunk_type": "translated",
                                    })
                                    translated_texts.append(sec_translated)
                        if not translated_texts and translated_text:
                            translated_sections = [{"text": translated_text, "start_page": 1, "end_page": 1, "chunk_type": "translated"}]
                            translated_texts = [translated_text]
                        all_texts = texts + translated_texts
                        all_sections = list(sections) + translated_sections
                    else:
                        all_texts = texts
                        all_sections = content.get("sections", [{"text": full_text, "start_page": 1, "end_page": 1}])

                    normalized[name] = {
                        "full_text": full_text,
                        "texts": all_texts,
                        "sections": all_sections,
                        "chunk_metadata": content.get("chunk_metadata", []),
                        "doc_type": content.get("doc_type"),
                        "detected_language": content.get("detected_language"),
                    }
            elif isinstance(content, str) and content.strip():
                normalized[name] = {
                    "full_text": content.strip(),
                    "texts": [content.strip()],
                }

        return normalized if normalized else {}

    return {}

def _has_useful_content(normalized: Dict[str, Any]) -> bool:
    """Check if normalized payload has useful content for embedding."""
    if not normalized:
        return False

    for name, content in normalized.items():
        if isinstance(content, ExtractedDocument):
            if content.full_text and content.full_text.strip():
                return True
            if content.sections and any(s.text.strip() for s in content.sections):
                return True
        elif isinstance(content, dict):
            # Check full_text
            full_text = content.get("full_text", "")
            if isinstance(full_text, str) and len(full_text.strip()) > 50:
                return True
            # Check texts array
            texts = content.get("texts", [])
            if texts and any(t.strip() for t in texts if isinstance(t, str)):
                return True
        elif isinstance(content, str) and len(content.strip()) > 50:
            return True

    return False

def _recover_from_payload(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to recover any usable text from the payload."""
    recovered: Dict[str, Any] = {}

    # Try to find text anywhere in the payload
    def extract_text_recursive(obj: Any, path: str = "") -> List[str]:
        texts = []
        if isinstance(obj, str) and len(obj.strip()) > 20:
            texts.append(obj.strip())
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("full_text", "text", "content", "raw_text"):
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())
                else:
                    texts.extend(extract_text_recursive(value, f"{path}.{key}"))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                texts.extend(extract_text_recursive(item, f"{path}[{i}]"))
        return texts

    all_texts = extract_text_recursive(extracted)

    if all_texts:
        # Use the longest text as the main content
        all_texts.sort(key=len, reverse=True)
        main_text = all_texts[0]
        recovered["recovered_document"] = {
            "full_text": main_text,
            "texts": [main_text],
            "sections": [{"text": main_text, "start_page": 1, "end_page": 1}],
        }
        logger.info("Recovered %d characters from payload", len(main_text))

    return recovered

def _diagnose_content(content: Any, doc_name: str) -> str:
    """Generate diagnostic info for failed chunk estimation."""
    if content is None:
        return "content is None"
    if isinstance(content, str):
        return f"string content length={len(content)}, stripped={len(content.strip())}"
    if isinstance(content, ExtractedDocument):
        return f"ExtractedDocument full_text={len(content.full_text or '')}, sections={len(content.sections or [])}"
    if isinstance(content, dict):
        fields = []
        for key in ["full_text", "text", "content", "texts", "sections", "pages"]:
            val = content.get(key)
            if val is not None:
                if isinstance(val, str):
                    fields.append(f"{key}:str({len(val)})")
                elif isinstance(val, list):
                    non_empty = sum(1 for x in val if x)
                    fields.append(f"{key}:list({len(val)}, non_empty={non_empty})")
                else:
                    fields.append(f"{key}:{type(val).__name__}")
        return f"dict with {', '.join(fields) if fields else 'no text fields'}"
    return f"unknown type: {type(content).__name__}"

def _screen_payload(extracted_docs: Dict[str, Any], document_id: str) -> Tuple[int, List[float]]:
    total_chunks = 0
    coverage_values: List[float] = []
    min_len = int(getattr(Config.Retrieval, "MIN_CHUNK_SIZE", 200))

    for doc_name, content in extracted_docs.items():
        chunks_count, coverage = _estimate_chunks_for_content(content, document_id=document_id, doc_name=doc_name)
        if chunks_count <= 0:
            # Enhanced diagnostic logging
            content_info = _diagnose_content(content, doc_name)
            logger.error(
                "No chunks available for %s (document_id=%s). Diagnosis: %s",
                doc_name, document_id, content_info
            )
            raise ValueError(f"No chunks available for {doc_name}: {content_info}")
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
        # Retry lease acquisition up to 3 times with exponential backoff
        lease_id = None
        for _lease_attempt in range(3):
            lease_id = store.try_acquire_lease(blob.name, lease_duration=_lease_seconds())
            if lease_id:
                break
            import time as _t
            _backoff = 2 ** _lease_attempt  # 1s, 2s, 4s
            logger.info("Lease conflict for %s, retrying in %ds (attempt %d/3)", blob.name, _backoff, _lease_attempt + 1)
            _t.sleep(_backoff)
        if not lease_id:
            if telemetry:
                telemetry.increment("embed_pickles_lease_conflict_total")
            # Reset stuck documents so they can be retried
            _lease_doc_id = result.get("document_id")
            if _lease_doc_id:
                try:
                    _lease_record = get_document_record(_lease_doc_id) or {}
                    if _lease_record.get("status") == STATUS_TRAINING_STARTED:
                        _safe_set_document_status(_lease_doc_id, STATUS_TRAINING_FAILED,
                                                  "lease_conflict_after_retries",
                                                  error_summary="lease_conflict")
                except Exception:  # noqa: BLE001
                    pass
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
            logger.warning("Pickle deserialization failed for blob %s: %s; attempting re-extraction", blob.name, exc)
            # Try to resolve doc_id and attempt re-extraction from source
            doc_id_hint = result.get("document_id") or (blob.metadata or {}).get("document_id") or extract_document_id(blob.name, prefix=store.prefix)
            if doc_id_hint:
                record = get_document_record(doc_id_hint) or {}
                fallback_docs = _reextract_from_source(
                    document_id=doc_id_hint,
                    record=record,
                    subscription_id=record.get("subscription_id") or subscription_id or "",
                )
                if fallback_docs:
                    logger.info("Re-extraction succeeded for %s after pickle failure", doc_id_hint)
                    extracted = fallback_docs
                else:
                    if telemetry:
                        telemetry.increment("embed_pickles_download_fail_total")
                    result["error"] = "pickle_deserialize_failed"
                    result["failed_reason"] = "blob_read_failed"
                    _safe_set_document_status(
                        doc_id_hint,
                        STATUS_TRAINING_FAILED,
                        f"pickle_deserialize_failed: {exc}",
                        error_summary="blob_read_failed",
                        cause=exc,
                    )
                    return result
            else:
                if telemetry:
                    telemetry.increment("embed_pickles_download_fail_total")
                result["error"] = "pickle_deserialize_failed"
                result["failed_reason"] = "blob_read_failed"
                return result

        doc_id = (blob.metadata or {}).get("document_id") or extract_document_id(blob.name, prefix=store.prefix)
        if not doc_id:
            result["error"] = "document_id_missing"
            result["failed_reason"] = "document_id_missing"
            return result
        result["document_id"] = doc_id

        record = get_document_record(doc_id) or {}
        current_status = record.get("status")
        if current_status in COMPLETED_STATUSES:
            # Still generate doc_index/doc_intelligence if missing
            try:
                _has_di = _check_doc_intelligence_exists(doc_id, subscription_id, profile_id)
                if not _has_di:
                    _di_full_text = _extract_full_text_from_pickle(extracted)
                    # Fallback: reconstruct text from Qdrant chunks if pickle is purged
                    if not _di_full_text:
                        _di_full_text = _extract_full_text_from_qdrant(
                            doc_id, subscription_id or "", profile_id or ""
                        )
                    if _di_full_text:
                        _source_fn = record.get("name", blob.name or "")
                        _upsert_doc_intelligence(
                            doc_id, subscription_id, profile_id,
                            _source_fn, _di_full_text,
                            collection_name=build_collection_name(subscription_id or ""),
                        )
                    else:
                        logger.warning("[DOC_INTELLIGENCE] No text source for %s (pickle purged, no Qdrant chunks)", doc_id)
            except Exception as _di_exc:
                logger.warning("[DOC_INTELLIGENCE] Skipped-path extraction failed: %s", _di_exc, exc_info=True)
            result["status"] = "SKIPPED"
            result["failed_reason"] = None
            return result
        # HITL gate: only screened or retryable docs can be embedded
        _EMBEDDING_ELIGIBLE_STATUSES = {STATUS_SCREENING_COMPLETED, STATUS_TRAINING_FAILED, STATUS_TRAINING_STARTED}
        if current_status not in _EMBEDDING_ELIGIBLE_STATUSES:
            result["status"] = "SKIPPED"
            if current_status == STATUS_EXTRACTION_COMPLETED:
                result["error"] = "screening_not_completed"
                result["failed_reason"] = f"screening_not_completed (current: {current_status})"
                logger.info(
                    "HITL gate: embedding rejected for %s (status=%s). Run screening first (POST /api/gateway/screen).",
                    doc_id, current_status,
                )
            else:
                result["error"] = "not_eligible"
                result["failed_reason"] = f"not_eligible (current: {current_status})"
                logger.info(
                    "HITL gate: embedding rejected for %s (status=%s). Run extraction then screening first.",
                    doc_id, current_status,
                )
            return result
        # Zombie guard: if stuck in TRAINING_STARTED for >30 min, auto-fail before retrying
        if current_status == STATUS_TRAINING_STARTED:
            started_at = record.get("training_started_at", 0)
            if started_at and (time.time() - started_at) > 1800:
                logger.warning("Document %s stuck in TRAINING_STARTED for >30min — recovering", doc_id)
                _safe_set_document_status(doc_id, STATUS_TRAINING_FAILED,
                    "zombie_timeout: stuck in TRAINING_STARTED",
                    error_summary="zombie_timeout")
                emit_progress(doc_id, "failed", 0.0, "Auto-recovered: training timeout exceeded")
        _set_document_status(doc_id, STATUS_TRAINING_STARTED)
        emit_progress(doc_id, "extraction", 0.10, "Starting document processing")
        from src.api.document_status import emit_status_log
        emit_status_log(doc_id, "embedding", "pipeline_start", "Embedding pipeline started",
                        extra={"blob_name": blob.name, "subscription_id": subscription_id, "profile_id": profile_id})

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
        emit_progress(doc_id, "chunking", 0.20, "Preparing document chunks")
        emit_status_log(doc_id, "embedding", "lock_acquired", "Embedding lock acquired, preparing chunks")

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

        # Inject document understanding metadata from pickle into each doc's metadata
        _understanding = None
        if isinstance(extracted, dict):
            _understanding = extracted.get("understanding") or {}
        if _understanding and extracted_docs:
            for _doc_key, _doc_content in extracted_docs.items():
                if isinstance(_doc_content, dict):
                    if "doc_metadata" not in _doc_content:
                        _doc_content["doc_metadata"] = {}
                    if isinstance(_doc_content.get("doc_metadata"), dict):
                        if _understanding.get("document_summary"):
                            _doc_content["doc_metadata"]["document_summary"] = str(
                                _understanding["document_summary"]
                            )[:500]
                        if _understanding.get("key_entities"):
                            _doc_content["doc_metadata"]["key_entities"] = _understanding["key_entities"]
                        if _understanding.get("key_facts"):
                            _doc_content["doc_metadata"]["key_facts"] = _understanding["key_facts"]
                        if _understanding.get("intent_tags"):
                            _doc_content["doc_metadata"]["intent_tags"] = _understanding["intent_tags"]

            # Fallback: persist understanding from pickle to MongoDB if extraction missed it
            try:
                from src.api.extraction_service import _update_understanding_fields
                _update_understanding_fields(doc_id, _understanding)
            except Exception as exc:
                logger.warning("Embedding fallback: failed to persist understanding from pickle for %s: %s",
                               doc_id, exc)

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
            emit_progress(doc_id, "failed", 0.25, message)
            emit_status_log(doc_id, "embedding", "preparation_failed", f"Embedding preparation failed: {message}",
                            extra={"reason": reason})
            result["error"] = reason
            result["error_message"] = message
            result["failed_reason"] = reason
            return result

        emit_progress(doc_id, "chunking", 0.25,
                      f"Prepared {expected_chunks} chunks for embedding",
                      extra={"chunks_total": expected_chunks})
        emit_status_log(doc_id, "embedding", "chunks_prepared",
                        f"Prepared {expected_chunks} chunks for embedding",
                        extra={"chunks_total": expected_chunks})

        result["chunks_count"] = expected_chunks
        logger.info("Embedding pre-check for %s: expected_chunks=%s", doc_id, expected_chunks)

        qdrant_count = 0
        if subscription_id and profile_id:
            try:
                qdrant_count = _count_qdrant_points(subscription_id, profile_id, doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant count failed for %s: %s", doc_id, exc)
        logger.info("Existing Qdrant points for %s: %s", doc_id, qdrant_count)

        emit_status_log(doc_id, "embedding", "dedup_check",
                        f"Qdrant dedup check: {qdrant_count} existing points, {expected_chunks} expected",
                        extra={"existing_points": qdrant_count, "expected_chunks": expected_chunks})

        if qdrant_count and expected_chunks and qdrant_count >= expected_chunks:
            # Generate doc_intelligence if missing even for already-embedded docs
            try:
                _has_di = _check_doc_intelligence_exists(doc_id, subscription_id, profile_id)
                if not _has_di:
                    _di_text = _extract_full_text_from_pickle(extracted)
                    if not _di_text:
                        _di_text = _extract_full_text_from_qdrant(
                            doc_id, subscription_id or "", profile_id or ""
                        )
                    if _di_text:
                        _upsert_doc_intelligence(
                            doc_id, subscription_id, profile_id,
                            source_filename or blob.name or "",
                            _di_text,
                            collection_name=build_collection_name(subscription_id or ""),
                        )
            except Exception as _di_exc:
                logger.warning("[DOC_INTELLIGENCE] Already-embedded path extraction failed: %s", _di_exc, exc_info=True)

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
            # Profile intelligence: auto-generate insights (background, non-blocking)
            try:
                from src.intelligence.profile_intelligence import generate_profile_intelligence
                import threading as _threading
                _threading.Thread(
                    target=generate_profile_intelligence,
                    args=(doc_id, profile_id, subscription_id),
                    daemon=True,
                    name=f"profile-intel-{doc_id[:12]}",
                ).start()
            except Exception:
                logger.debug("Profile intelligence trigger skipped", exc_info=True)
            deleted = False
            logger.info("Preserving pickle for %s after embedding (cleanup disabled)", doc_id)
            if telemetry:
                telemetry.increment("embed_pickles_retained_total")
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
            # Clean up partial embeddings before re-upserting to prevent duplicates
            try:
                from src.api.dataHandler import get_vector_store
                _cleanup_store = get_vector_store()
                _cleanup_store.delete_document(subscription_id, profile_id, doc_id)
                logger.info(
                    "Cleaned %d partial embeddings for %s before re-embedding",
                    qdrant_count, doc_id,
                )
            except Exception as _cleanup_exc:
                logger.warning("Partial embedding cleanup failed for %s: %s", doc_id, _cleanup_exc)
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
            _file_idx = 0
            _file_total = max(len(extracted_docs), 1)
            for file_name, content in extracted_docs.items():
                # --- Intel pipeline hook (non-blocking, feature-flagged) ---
                if INTEL_PIPELINE_ENABLED and run_intel_pipeline_hook is not None:
                    try:
                        _intel_t0 = time.time()
                        _intel_doc_json = build_document_json_from_extracted(content, document_id=doc_id)
                        _intel_result = run_intel_pipeline_hook(
                            extracted_doc=_intel_doc_json,
                            document_id=doc_id,
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                        )
                        logger.info(
                            "intel_pipeline doc=%s file=%s elapsed=%.3fs result=%s",
                            doc_id, file_name, time.time() - _intel_t0,
                            _intel_result.stage_reached if _intel_result else "none",
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "intel_pipeline hook failed for doc=%s file=%s; continuing with embedding",
                            doc_id, file_name, exc_info=True,
                        )
                # --- End intel pipeline hook ---
                try:
                    embed_result = train_on_document(content, subscription_id, profile_id, doc_id, file_name)
                except Exception as exc:  # noqa: BLE001
                    if _is_meta_tensor_error(exc) or _is_cuda_oom(exc):
                        device_type = "meta tensor" if _is_meta_tensor_error(exc) else "CUDA OOM"
                        logger.warning(
                            "embed_request_id=%s doc=%s %s error; retrying on cpu",
                            embed_request_id,
                            doc_id,
                            device_type,
                        )
                        if _is_cuda_oom(exc):
                            _clear_gpu_cache()
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
                _file_idx += 1
                _file_progress = 0.30 + (0.50 * _file_idx / _file_total)
                emit_progress(doc_id, "encoding", _file_progress,
                              f"Encoded file {_file_idx}/{_file_total} ({total_upserted} chunks stored)",
                              extra={"files_done": _file_idx, "files_total": _file_total, "chunks_stored": total_upserted})
                emit_status_log(doc_id, "embedding", "file_encoded",
                                f"Encoded file {_file_idx}/{_file_total}: {int(embed_result.get('points_saved', 0))} chunks stored, {int(embed_result.get('dropped_chunks', 0))} dropped",
                                extra={"file_index": _file_idx, "files_total": _file_total,
                                       "chunks_stored": int(embed_result.get("points_saved", 0)),
                                       "chunks_dropped": int(embed_result.get("dropped_chunks", 0)),
                                       "coverage_ratio": embed_result.get("coverage_ratio")})
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

        effective_expected = total_chunks - total_dropped
        logger.info(
            "embed_request_id=%s doc=%s mismatch_check: total_chunks=%s total_dropped=%s total_upserted=%s effective_expected=%s",
            embed_request_id, doc_id, total_chunks, total_dropped, total_upserted, effective_expected,
        )
        if effective_expected > 0 and effective_expected != total_upserted:
            shortfall_ratio = total_upserted / effective_expected if effective_expected > 0 else 0
            if shortfall_ratio >= 0.9 and total_upserted > 0:
                logger.warning(
                    "embed_request_id=%s doc=%s minor mismatch (%.0f%%): expected %s, saved %s — treating as success",
                    embed_request_id, doc_id, shortfall_ratio * 100, effective_expected, total_upserted,
                )
            else:
                error_msg = f"Embedding upsert mismatch: expected {effective_expected} (prepared {total_chunks}, dropped {total_dropped}), saved {total_upserted}"
                logger.error(
                    "embed_request_id=%s doc=%s MISMATCH: %s",
                    embed_request_id, doc_id, error_msg,
                )
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
        emit_progress(doc_id, "upserting", 0.85,
                      f"Stored {total_upserted} embeddings in Qdrant",
                      extra={"chunks_stored": total_upserted, "chunks_total": total_chunks})
        emit_status_log(doc_id, "embedding", "embeddings_stored",
                        f"Stored {total_upserted} embeddings in Qdrant (collection: {collection_name})",
                        extra={"chunks_stored": total_upserted, "chunks_total": total_chunks,
                               "chunks_dropped": total_dropped, "collection": collection_name,
                               "coverage_ratio": coverage_ratio})

        # ── Multi-resolution: create doc-level + section-level vectors ──
        try:
            from src.embedding.multi_resolution import build_multi_resolution_extras
            from src.doc_understanding.schema_detector import detect_and_extract_schema
            from src.intelligence.answerability_index import build_answerability_index
            from src.doc_understanding.form_extractor import extract_all_form_fields, form_fields_to_chunk_text

            _mr_understanding = _understanding or {}
            _mr_doc_type = record.get("document_type") or record.get("doc_type") or "other"
            _mr_doc_domain = record.get("doc_domain") or "generic"

            # Schema detection (Phase 3)
            _schema_result = {}
            _answerability = []
            try:
                _mr_sections = []
                _mr_entities = _mr_understanding.get("key_entities") or []
                _mr_full_text = ""
                if extracted_docs:
                    _ex_data = list(extracted_docs.values())[0] if extracted_docs else {}
                    if isinstance(_ex_data, dict):
                        _mr_full_text = _ex_data.get("full_text") or ""
                        for sec in (_ex_data.get("sections") or []):
                            if isinstance(sec, dict):
                                _mr_sections.append(sec)
                _mr_section_roles = {}
                _struct = _mr_understanding.get("structure_inference") or {}
                for s_info in (_struct.get("sections") or []):
                    if isinstance(s_info, dict):
                        _mr_section_roles[s_info.get("section_title", "")] = s_info.get("inferred_section_role", "")

                _schema_result = detect_and_extract_schema(
                    doc_type=_mr_doc_type,
                    sections=_mr_sections,
                    entities=_mr_entities,
                    full_text=_mr_full_text,
                    section_roles=_mr_section_roles,
                )
                logger.info(
                    "Schema detection for %s: type=%s completeness=%.2f found=%s missing=%s",
                    doc_id, _mr_doc_type,
                    _schema_result.get("completeness_score", 0),
                    _schema_result.get("found_sections", []),
                    _schema_result.get("missing_sections", []),
                )

                # Answerability index (Phase 4)
                _answerability_result = build_answerability_index(
                    doc_type=_mr_doc_type,
                    schema_result=_schema_result,
                    entities=_mr_entities,
                    full_text=_mr_full_text,
                    section_summaries=_mr_understanding.get("section_summaries"),
                )
                _answerability = _answerability_result.get("answerable_query_types") or []
                logger.info(
                    "Answerability index for %s: %d query types (confidence=%.2f)",
                    doc_id, len(_answerability), _answerability_result.get("confidence", 0),
                )

                # Form field extraction (Phase 6)
                _form_fields = []
                try:
                    _form_fields = extract_all_form_fields(_mr_sections)
                    if _form_fields:
                        logger.info("Form field extraction for %s: %d fields", doc_id, len(_form_fields))
                except Exception as ff_exc:
                    logger.debug("Form field extraction failed for %s: %s", doc_id, ff_exc)

                # Persist schema + answerability + form fields to MongoDB
                _schema_mongo_fields = {
                    "schema_extraction": _schema_result,
                    "answerability": _answerability,
                }
                if _form_fields:
                    _schema_mongo_fields["form_fields"] = [
                        {"label": f.label, "value": f.value, "confidence": f.confidence,
                         "page": f.page, "section_title": f.section_title}
                        for f in _form_fields
                    ]
                try:
                    from src.api.document_status import update_document_fields
                    update_document_fields(doc_id, _schema_mongo_fields)
                except Exception as exc:
                    logger.debug("Failed to persist schema fields to MongoDB for %s", doc_id, exc_info=True)
            except Exception as schema_exc:
                logger.debug("Schema/answerability detection failed for %s: %s", doc_id, schema_exc)

            # Build multi-resolution vectors
            _mr_extras = build_multi_resolution_extras(
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_id,
                doc_name=file_name or doc_id,
                doc_domain=_mr_doc_domain,
                doc_type=_mr_doc_type,
                understanding=_mr_understanding,
                rescued_fragments={},
                answerability=_answerability,
                schema_completeness=_schema_result.get("completeness_score") if _schema_result else None,
            )

            # Encode and upsert multi-resolution extras
            if _mr_extras:
                try:
                    from src.embedding.model_loader import get_embedding_model as get_model
                    _mr_model_result = get_model()
                    # get_embedding_model returns (model, dim) tuple
                    _mr_model = _mr_model_result[0] if isinstance(_mr_model_result, tuple) else _mr_model_result
                    _mr_texts = [e["text"] for e in _mr_extras]
                    _mr_vectors = _mr_model.encode(_mr_texts, show_progress_bar=False)
                    from src.api.vector_store import build_collection_name as _bcn
                    from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
                    _mr_collection = _bcn(subscription_id)
                    from qdrant_client.models import PointStruct
                    import uuid as _uuid
                    _mr_points = []
                    for i, extra in enumerate(_mr_extras):
                        _payload = build_qdrant_payload(extra["metadata"])
                        _mr_points.append(PointStruct(
                            id=str(_uuid.uuid4()),
                            vector=[float(x) for x in _mr_vectors[i]],
                            payload=_payload,
                        ))
                    from src.api.dataHandler import get_vector_store
                    _vs = get_vector_store()
                    _vs.client.upsert(collection_name=_mr_collection, points=_mr_points)
                    total_upserted += len(_mr_points)
                    logger.info(
                        "Multi-resolution: upserted %d extra vectors (doc+section) for %s",
                        len(_mr_points), doc_id,
                    )
                except Exception as mr_upsert_exc:
                    logger.warning("Multi-resolution upsert failed for %s: %s", doc_id, mr_upsert_exc)
        except Exception as mr_exc:
            logger.debug("Multi-resolution/schema processing skipped for %s: %s", doc_id, mr_exc)

        # Post-upsert verification BEFORE setting TRAINING_COMPLETED
        post_count: Optional[int] = None
        cleanup_allowed = True
        cleanup_error: Optional[Dict[str, Any]] = None
        if subscription_id and profile_id and total_upserted > 0:
            emit_progress(doc_id, "verifying", 0.90, "Verifying storage integrity")
            emit_status_log(doc_id, "embedding", "verification_start", "Post-upsert storage integrity verification started")
            try:
                post_count, cleanup_allowed = _verify_post_upsert_count(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    document_id=doc_id,
                    expected_chunks=total_upserted,
                )
                if post_count is not None:
                    logger.info("Post-upsert Qdrant points for %s: %s", doc_id, post_count)
                if post_count is not None and post_count < total_upserted * 0.5:
                    error_msg = f"Post-upsert verification failed: Qdrant has {post_count} points, expected {total_upserted}"
                    logger.error("embed_request_id=%s doc=%s %s", embed_request_id, doc_id, error_msg)
                    _safe_set_document_status(doc_id, STATUS_TRAINING_FAILED, error_msg, error_summary="qdrant_verification_failed")
                    emit_progress(doc_id, "failed", 0.90, error_msg)
                    result["error"] = "qdrant_verification_failed"
                    result["error_message"] = error_msg
                    result["failed_reason"] = "qdrant_verification_failed"
                    return result
                if not cleanup_allowed:
                    cleanup_error = (
                        {"message": "post_upsert_count_unavailable"}
                        if post_count is None
                        else {
                            "message": "post_upsert_count_mismatch",
                            "post_count": post_count,
                            "expected": total_upserted,
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

        # --- Document Intelligence Points ---
        try:
            from src.extraction.document_intelligence import (
                extract_document_intelligence,
                build_doc_index_text,
                build_doc_intelligence_text,
            )

            # Reconstruct full document text from extracted_docs
            _di_full_text = ""
            for _di_fname, _di_content in (extracted_docs or {}).items():
                if isinstance(_di_content, ExtractedDocument) and _di_content.full_text:
                    _di_full_text += _di_content.full_text + "\n"
                elif isinstance(_di_content, dict) and isinstance(_di_content.get("full_text"), str):
                    _di_full_text += _di_content["full_text"] + "\n"
                elif isinstance(_di_content, str):
                    _di_full_text += _di_content + "\n"

            _di_source_filename = file_name or doc_id

            if _di_full_text.strip():
                _intelligence = extract_document_intelligence(_di_full_text, _di_source_filename)
                _doc_index_text = build_doc_index_text(_di_source_filename, _intelligence)
                _doc_intel_text = build_doc_intelligence_text(_di_source_filename, _intelligence)

                from src.embedding.pipeline.schema_normalizer import build_qdrant_payload as _di_build_payload
                import uuid as _di_uuid

                _di_base_payload = {
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "document_id": doc_id,
                    "source_name": _di_source_filename,
                    "canonical_text": _doc_index_text,
                    "embedding_text": _doc_index_text,
                    "resolution": "doc_index",
                    "chunk_kind": "doc_index",
                    "chunk_id": f"doc_index_{doc_id}",
                    "chunk_index": 0,
                    "section_title": "Document Index",
                }

                _index_payload = _di_build_payload(_di_base_payload)
                _index_payload["doc_intelligence"] = _intelligence

                # Encode embedding vectors
                from src.embedding.model_loader import get_embedding_model as _di_get_model
                _di_model_result = _di_get_model()
                _di_model = _di_model_result[0] if isinstance(_di_model_result, tuple) else _di_model_result

                _index_vector = _di_model.encode([_doc_index_text], show_progress_bar=False)[0].tolist()

                from qdrant_client.models import PointStruct as _di_PointStruct

                _index_point = _di_PointStruct(
                    id=str(_di_uuid.uuid5(_di_uuid.NAMESPACE_DNS, f"doc_index_{doc_id}")),
                    vector={"content_vector": _index_vector},
                    payload=_index_payload,
                )

                # Build doc_intelligence point
                _intel_payload = _di_build_payload({
                    **_di_base_payload,
                    "canonical_text": _doc_intel_text,
                    "embedding_text": _doc_intel_text,
                    "resolution": "doc_intelligence",
                    "chunk_kind": "doc_intelligence",
                    "chunk_id": f"doc_intelligence_{doc_id}",
                    "section_title": "Document Intelligence",
                })
                _intel_payload["doc_intelligence"] = _intelligence

                _intel_vector = _di_model.encode([_doc_intel_text], show_progress_bar=False)[0].tolist()

                _intel_point = _di_PointStruct(
                    id=str(_di_uuid.uuid5(_di_uuid.NAMESPACE_DNS, f"doc_intelligence_{doc_id}")),
                    vector={"content_vector": _intel_vector},
                    payload=_intel_payload,
                )

                from src.api.dataHandler import get_vector_store as _di_get_vs
                _di_vs = _di_get_vs()
                _di_vs.client.upsert(
                    collection_name=collection_name,
                    points=[_index_point, _intel_point],
                )

                logger.info(
                    "[DOC_INTELLIGENCE] Upserted doc_index + doc_intelligence for %s (%s)",
                    doc_id, _di_source_filename,
                )
        except Exception as _di_exc:
            logger.warning("[DOC_INTELLIGENCE] Failed for %s: %s", doc_id, _di_exc)

        _set_document_status(doc_id, STATUS_TRAINING_COMPLETED, extra_fields=_training_success_fields())
        # Profile intelligence: auto-generate insights (background, non-blocking)
        try:
            from src.intelligence.profile_intelligence import generate_profile_intelligence
            import threading as _threading
            _threading.Thread(
                target=generate_profile_intelligence,
                args=(doc_id, profile_id, subscription_id),
                daemon=True,
                name=f"profile-intel-{doc_id[:12]}",
            ).start()
        except Exception:
            logger.debug("Profile intelligence trigger skipped", exc_info=True)
        emit_progress(doc_id, "completed", 1.0,
                      f"Training completed — {total_upserted} chunks stored",
                      extra={"chunks_stored": total_upserted, "collection": collection_name})
        emit_status_log(doc_id, "embedding", "training_completed",
                        f"Training completed — {total_upserted} chunks stored in {collection_name}",
                        extra={"chunks_stored": total_upserted, "collection": collection_name,
                               "post_upsert_count": post_count})

        # KG chunk-level ingestion (async, non-blocking)
        _ingest_chunks_to_knowledge_graph(
            document_id=doc_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            doc_name=file_name or doc_id,
            extracted_docs=extracted_docs,
        )

        # ── Cross-document intelligence (Phase 5, non-blocking daemon thread) ──
        try:
            import threading as _cd_threading
            from src.intelligence.cross_doc import run_cross_document_intelligence

            _cd_doc_vector = None
            # Use the first multi-resolution doc vector if available
            if _mr_extras:
                for _mre in _mr_extras:
                    if _mre.get("metadata", {}).get("resolution") == "doc":
                        # Re-encode the doc text to get the vector
                        try:
                            from src.embedding.model_loader import get_embedding_model as _cd_get_model
                            _cd_model_result = _cd_get_model()
                            _cd_model = _cd_model_result[0] if isinstance(_cd_model_result, tuple) else _cd_model_result
                            _cd_doc_vector = _cd_model.encode([_mre["text"]], show_progress_bar=False)[0].tolist()
                        except Exception as exc:
                            logger.debug("Failed to encode document vector for cross-doc intelligence", exc_info=True)
                        break

            _cd_entities = (_understanding or {}).get("key_entities") or []

            def _run_cross_doc():
                try:
                    run_cross_document_intelligence(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        document_id=doc_id,
                        doc_name=file_name or doc_id,
                        doc_summary=str((_understanding or {}).get("document_summary") or ""),
                        doc_entities=_cd_entities,
                        doc_vector=_cd_doc_vector,
                    )
                except Exception as _cd_exc:
                    logger.debug("Cross-doc intelligence failed for %s: %s", doc_id, _cd_exc)

            _cd_thread = _cd_threading.Thread(target=_run_cross_doc, daemon=True)
            _cd_thread.start()
        except Exception as exc:
            logger.debug("Cross-doc intelligence setup failed (purely additive, never blocks pipeline)", exc_info=True)

        # Purge pickle after successful embedding — content now lives in vector DB.
        deleted = False
        if cleanup_allowed:
            try:
                from src.api.content_store import delete_extracted_pickle
                deleted = delete_extracted_pickle(doc_id)
                if deleted:
                    logger.info("Pickle purged for %s after successful embedding", doc_id)
                else:
                    logger.info("Pickle not found for deletion for %s (may already be purged)", doc_id)
            except Exception as del_exc:  # noqa: BLE001
                logger.warning("Pickle purge failed for %s: %s", doc_id, del_exc)
                cleanup_error = {"message": f"purge_failed: {del_exc}"}
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
            emit_progress(doc_id_hint, "failed", 0.0, error_message)
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
        # Generate doc_intelligence if missing (local path)
        try:
            _has_di = _check_doc_intelligence_exists(document_id, subscription_id or "", profile_id or "")
            if not _has_di:
                _di_text = _extract_full_text_from_qdrant(document_id, subscription_id or "", profile_id or "")
                if _di_text:
                    _source_fn = record.get("name", "")
                    _upsert_doc_intelligence(
                        document_id, subscription_id or "", profile_id or "",
                        _source_fn, _di_text,
                        collection_name=build_collection_name(subscription_id or ""),
                    )
        except Exception as _di_exc:
            logger.warning("[DOC_INTELLIGENCE] Local-path extraction failed: %s", _di_exc, exc_info=True)
        result["status"] = "SKIPPED"
        result["failed_reason"] = None
        return _early_return(result)
    # HITL gate: only screened or retryable docs can be embedded
    _EMBEDDING_ELIGIBLE_STATUSES = {STATUS_SCREENING_COMPLETED, STATUS_TRAINING_FAILED, STATUS_TRAINING_STARTED}
    if current_status not in _EMBEDDING_ELIGIBLE_STATUSES:
        result["status"] = "SKIPPED"
        if current_status == STATUS_EXTRACTION_COMPLETED:
            result["error"] = "screening_not_completed"
            result["failed_reason"] = f"screening_not_completed (current: {current_status})"
            logger.info(
                "HITL gate: embedding rejected for %s (status=%s). Run screening first (POST /api/gateway/screen).",
                document_id, current_status,
            )
        else:
            result["error"] = "not_eligible"
            result["failed_reason"] = f"not_eligible (current: {current_status})"
            logger.info(
                "HITL gate: embedding rejected for %s (status=%s). Run extraction then screening first.",
                document_id, current_status,
            )
        return _early_return(result)
    # Zombie guard: if stuck in TRAINING_STARTED for >30 min, auto-fail before retrying
    if current_status == STATUS_TRAINING_STARTED:
        started_at = record.get("training_started_at", 0)
        if started_at and (time.time() - started_at) > 1800:
            logger.warning("Document %s stuck in TRAINING_STARTED for >30min — recovering", document_id)
            _safe_set_document_status(document_id, STATUS_TRAINING_FAILED,
                "zombie_timeout: stuck in TRAINING_STARTED",
                error_summary="zombie_timeout")
            emit_progress(document_id, "failed", 0.0, "Auto-recovered: training timeout exceeded")

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
    emit_progress(document_id, "extraction", 0.10, "Starting document processing")
    telemetry = _telemetry()

    update_stage(
        document_id,
        "embedding",
        {"status": "IN_PROGRESS", "started_at": time.time(), "error": None, "reason": None},
    )
    emit_progress(document_id, "chunking", 0.20, "Preparing document chunks")

    try:
        try:
            extracted = load_extracted_pickle(document_id)
            if isinstance(extracted, dict):
                pickle_keys = list(extracted.keys())
                has_screening = "screening" in extracted
                logger.info(
                    "Loaded pickle for %s: keys=%s, has_screening=%s",
                    document_id, pickle_keys, has_screening,
                )
        except Exception as exc:  # noqa: BLE001
            error_message = _truncate_error_message(str(exc) or repr(exc))
            logger.error(
                "embed_request_id=%s doc=%s load pickle failed: %s",
                embed_request_id, document_id, exc, exc_info=True,
            )
            error_payload = _build_error_payload(
                stage="embedding",
                message=error_message,
                exc=exc,
                run_id=embed_request_id,
                code="pickle_not_found",
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
                f"Extraction pickle not found — run extraction before embedding. {error_message}",
                error_summary="pickle_not_found",
                cause=exc,
            )
            result["error"] = "pickle_not_found"
            result["error_message"] = error_message
            result["failed_reason"] = "pickle_not_found"
            return result

        extracted_docs, expected_chunks, coverage_values, prep_error = _prepare_extracted_docs(
            document_id=document_id,
            extracted=extracted,
            record=record,
            subscription_id=subscription_id,
        )

        # Inject document understanding metadata from pickle into each doc's metadata
        _understanding_local = None
        if isinstance(extracted, dict):
            _understanding_local = extracted.get("understanding") or {}
        if _understanding_local and extracted_docs:
            for _doc_key, _doc_content in extracted_docs.items():
                if isinstance(_doc_content, dict):
                    if "doc_metadata" not in _doc_content:
                        _doc_content["doc_metadata"] = {}
                    if isinstance(_doc_content.get("doc_metadata"), dict):
                        if _understanding_local.get("document_summary"):
                            _doc_content["doc_metadata"]["document_summary"] = str(
                                _understanding_local["document_summary"]
                            )[:500]
                        if _understanding_local.get("key_entities"):
                            _doc_content["doc_metadata"]["key_entities"] = _understanding_local["key_entities"]
                        if _understanding_local.get("key_facts"):
                            _doc_content["doc_metadata"]["key_facts"] = _understanding_local["key_facts"]
                        if _understanding_local.get("intent_tags"):
                            _doc_content["doc_metadata"]["intent_tags"] = _understanding_local["intent_tags"]

            # Fallback: if extraction_service failed to write understanding to MongoDB,
            # persist from pickle now so _fetch_document_metadata() and Qdrant payload get it.
            try:
                from src.api.extraction_service import _update_understanding_fields
                _update_understanding_fields(document_id, _understanding_local)
            except Exception as exc:
                logger.warning("Embedding fallback: failed to persist understanding from pickle for %s: %s",
                               document_id, exc)

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
            emit_progress(document_id, "failed", 0.25, message)
            result["error"] = reason
            result["error_message"] = message
            result["failed_reason"] = reason
            return result

        emit_progress(document_id, "chunking", 0.25,
                      f"Prepared {expected_chunks} chunks for embedding",
                      extra={"chunks_total": expected_chunks})

        result["chunks_count"] = expected_chunks
        logger.info("Embedding pre-check for %s: expected_chunks=%s", document_id, expected_chunks)

        total_chunks = 0
        total_upserted = 0
        total_dropped = 0
        coverage_values = coverage_values or []
        try:
            _file_idx_local = 0
            _file_total_local = max(len(extracted_docs), 1)
            for file_name, content in extracted_docs.items():
                # --- Intel pipeline hook (non-blocking, feature-flagged) ---
                if INTEL_PIPELINE_ENABLED and run_intel_pipeline_hook is not None:
                    try:
                        _intel_t0 = time.time()
                        _intel_doc_json = build_document_json_from_extracted(content, document_id=document_id)
                        _intel_result = run_intel_pipeline_hook(
                            extracted_doc=_intel_doc_json,
                            document_id=document_id,
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                        )
                        logger.info(
                            "intel_pipeline doc=%s file=%s elapsed=%.3fs result=%s",
                            document_id, file_name, time.time() - _intel_t0,
                            _intel_result.stage_reached if _intel_result else "none",
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "intel_pipeline hook failed for doc=%s file=%s; continuing with embedding",
                            document_id, file_name, exc_info=True,
                        )
                # --- End intel pipeline hook ---
                try:
                    embed_result = train_on_document(content, subscription_id, profile_id, document_id, file_name)
                except Exception as exc:  # noqa: BLE001
                    if _is_meta_tensor_error(exc) or _is_cuda_oom(exc):
                        device_type = "meta tensor" if _is_meta_tensor_error(exc) else "CUDA OOM"
                        logger.warning(
                            "embed_request_id=%s doc=%s %s error; retrying on cpu",
                            embed_request_id,
                            document_id,
                            device_type,
                        )
                        if _is_cuda_oom(exc):
                            _clear_gpu_cache()
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
                _file_idx_local += 1
                _file_progress_local = 0.30 + (0.50 * _file_idx_local / _file_total_local)
                emit_progress(document_id, "encoding", _file_progress_local,
                              f"Encoded file {_file_idx_local}/{_file_total_local} ({total_upserted} chunks stored)",
                              extra={"files_done": _file_idx_local, "files_total": _file_total_local, "chunks_stored": total_upserted})
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

        effective_expected = total_chunks - total_dropped
        logger.info(
            "embed_request_id=%s doc=%s mismatch_check: total_chunks=%s total_dropped=%s total_upserted=%s effective_expected=%s",
            embed_request_id, document_id, total_chunks, total_dropped, total_upserted, effective_expected,
        )
        if effective_expected > 0 and effective_expected != total_upserted:
            shortfall_ratio = total_upserted / effective_expected if effective_expected > 0 else 0
            if shortfall_ratio >= 0.9 and total_upserted > 0:
                logger.warning(
                    "embed_request_id=%s doc=%s minor mismatch (%.0f%%): expected %s, saved %s — treating as success",
                    embed_request_id, document_id, shortfall_ratio * 100, effective_expected, total_upserted,
                )
            else:
                error_msg = f"Embedding upsert mismatch: expected {effective_expected} (prepared {total_chunks}, dropped {total_dropped}), saved {total_upserted}"
                logger.error(
                    "embed_request_id=%s doc=%s MISMATCH: %s",
                    embed_request_id, document_id, error_msg,
                )
                if telemetry:
                    telemetry.increment("embed_qdrant_upsert_fail_total")
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
        emit_progress(document_id, "upserting", 0.85,
                      f"Stored {total_upserted} embeddings in Qdrant",
                      extra={"chunks_stored": total_upserted, "chunks_total": total_chunks})

        # Post-upsert verification BEFORE setting TRAINING_COMPLETED
        post_count: Optional[int] = None
        cleanup_allowed = True
        cleanup_error: Optional[Dict[str, Any]] = None
        if subscription_id and profile_id and total_upserted > 0:
            emit_progress(document_id, "verifying", 0.90, "Verifying storage integrity")
            try:
                post_count, cleanup_allowed = _verify_post_upsert_count(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    document_id=document_id,
                    expected_chunks=total_upserted,
                )
                if post_count is not None:
                    logger.info("Post-upsert Qdrant points for %s: %s", document_id, post_count)
                if post_count is not None and post_count < total_upserted * 0.5:
                    error_msg = f"Post-upsert verification failed: Qdrant has {post_count} points, expected {total_upserted}"
                    logger.error("embed_request_id=%s doc=%s %s", embed_request_id, document_id, error_msg)
                    _safe_set_document_status(document_id, STATUS_TRAINING_FAILED, error_msg, error_summary="qdrant_verification_failed")
                    emit_progress(document_id, "failed", 0.90, error_msg)
                    result["error"] = "qdrant_verification_failed"
                    result["error_message"] = error_msg
                    result["failed_reason"] = "qdrant_verification_failed"
                    return result
                if not cleanup_allowed:
                    cleanup_error = (
                        {"message": "post_upsert_count_unavailable"}
                        if post_count is None
                        else {
                            "message": "post_upsert_count_mismatch",
                            "post_count": post_count,
                            "expected": total_upserted,
                        }
                    )
                    logger.warning(
                        "Skipping pickle cleanup for %s because embedding is not yet verified (post_count=%s, expected=%s)",
                        document_id,
                        post_count,
                        total_upserted,
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

        # --- Document Intelligence Points ---
        try:
            from src.extraction.document_intelligence import (
                extract_document_intelligence,
                build_doc_index_text,
                build_doc_intelligence_text,
            )

            # Reconstruct full document text from extracted_docs
            _di_full_text = ""
            for _di_fname, _di_content in (extracted_docs or {}).items():
                if isinstance(_di_content, ExtractedDocument) and _di_content.full_text:
                    _di_full_text += _di_content.full_text + "\n"
                elif isinstance(_di_content, dict) and isinstance(_di_content.get("full_text"), str):
                    _di_full_text += _di_content["full_text"] + "\n"
                elif isinstance(_di_content, str):
                    _di_full_text += _di_content + "\n"

            _di_source_filename = file_name or document_id

            if _di_full_text.strip():
                _intelligence = extract_document_intelligence(_di_full_text, _di_source_filename)
                _doc_index_text = build_doc_index_text(_di_source_filename, _intelligence)
                _doc_intel_text = build_doc_intelligence_text(_di_source_filename, _intelligence)

                from src.embedding.pipeline.schema_normalizer import build_qdrant_payload as _di_build_payload
                import uuid as _di_uuid

                _di_base_payload = {
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "document_id": document_id,
                    "source_name": _di_source_filename,
                    "canonical_text": _doc_index_text,
                    "embedding_text": _doc_index_text,
                    "resolution": "doc_index",
                    "chunk_kind": "doc_index",
                    "chunk_id": f"doc_index_{document_id}",
                    "chunk_index": 0,
                    "section_title": "Document Index",
                }

                _index_payload = _di_build_payload(_di_base_payload)
                _index_payload["doc_intelligence"] = _intelligence

                # Encode embedding vectors
                from src.embedding.model_loader import get_embedding_model as _di_get_model
                _di_model_result = _di_get_model()
                _di_model = _di_model_result[0] if isinstance(_di_model_result, tuple) else _di_model_result

                _index_vector = _di_model.encode([_doc_index_text], show_progress_bar=False)[0].tolist()

                from qdrant_client.models import PointStruct as _di_PointStruct

                _index_point = _di_PointStruct(
                    id=str(_di_uuid.uuid5(_di_uuid.NAMESPACE_DNS, f"doc_index_{document_id}")),
                    vector={"content_vector": _index_vector},
                    payload=_index_payload,
                )

                # Build doc_intelligence point
                _intel_payload = _di_build_payload({
                    **_di_base_payload,
                    "canonical_text": _doc_intel_text,
                    "embedding_text": _doc_intel_text,
                    "resolution": "doc_intelligence",
                    "chunk_kind": "doc_intelligence",
                    "chunk_id": f"doc_intelligence_{document_id}",
                    "section_title": "Document Intelligence",
                })
                _intel_payload["doc_intelligence"] = _intelligence

                _intel_vector = _di_model.encode([_doc_intel_text], show_progress_bar=False)[0].tolist()

                _intel_point = _di_PointStruct(
                    id=str(_di_uuid.uuid5(_di_uuid.NAMESPACE_DNS, f"doc_intelligence_{document_id}")),
                    vector={"content_vector": _intel_vector},
                    payload=_intel_payload,
                )

                from src.api.dataHandler import get_vector_store as _di_get_vs
                _di_vs = _di_get_vs()
                _di_vs.client.upsert(
                    collection_name=collection_name,
                    points=[_index_point, _intel_point],
                )

                logger.info(
                    "[DOC_INTELLIGENCE] Upserted doc_index + doc_intelligence for %s (%s)",
                    document_id, _di_source_filename,
                )
        except Exception as _di_exc:
            logger.warning("[DOC_INTELLIGENCE] Failed for %s: %s", document_id, _di_exc)

        _set_document_status(document_id, STATUS_TRAINING_COMPLETED, extra_fields=_training_success_fields())
        # Profile intelligence: auto-generate insights (background, non-blocking)
        try:
            from src.intelligence.profile_intelligence import generate_profile_intelligence
            import threading as _threading
            _threading.Thread(
                target=generate_profile_intelligence,
                args=(document_id, profile_id, subscription_id),
                daemon=True,
                name=f"profile-intel-{document_id[:12]}",
            ).start()
        except Exception:
            logger.debug("Profile intelligence trigger skipped", exc_info=True)
        emit_progress(document_id, "completed", 1.0,
                      f"Training completed — {total_upserted} chunks stored",
                      extra={"chunks_stored": total_upserted, "collection": collection_name})

        # KG chunk-level ingestion (async, non-blocking)
        _ingest_chunks_to_knowledge_graph(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            doc_name=file_name or document_id,
            extracted_docs=extracted_docs,
        )

        # ── Cross-document intelligence (Phase 5, non-blocking daemon thread) ──
        try:
            import threading as _cd_threading_local
            from src.intelligence.cross_doc import run_cross_document_intelligence

            _cd_entities_local = (_understanding_local or {}).get("key_entities") or []

            def _run_cross_doc_local():
                try:
                    run_cross_document_intelligence(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        document_id=document_id,
                        doc_name=file_name or document_id,
                        doc_summary=str((_understanding_local or {}).get("document_summary") or ""),
                        doc_entities=_cd_entities_local,
                        doc_vector=None,  # no doc vector in local path
                    )
                except Exception as _cd_exc_local:
                    logger.debug("Cross-doc intelligence failed for %s: %s", document_id, _cd_exc_local)

            _cd_thread_local = _cd_threading_local.Thread(target=_run_cross_doc_local, daemon=True)
            _cd_thread_local.start()
        except Exception as exc:
            logger.debug("Cross-doc intelligence setup failed (purely additive, never blocks pipeline)", exc_info=True)

        # Purge pickle after successful embedding — content now lives in vector DB.
        deleted = False
        if cleanup_allowed:
            try:
                from src.api.content_store import delete_extracted_pickle
                deleted = delete_extracted_pickle(document_id)
                if deleted:
                    logger.info("Pickle purged for %s after successful embedding", document_id)
                else:
                    logger.info("Pickle not found for deletion for %s (may already be purged)", document_id)
            except Exception as del_exc:  # noqa: BLE001
                logger.warning("Pickle purge failed for %s: %s", document_id, del_exc)
                cleanup_error = {"message": f"purge_failed: {del_exc}"}
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
        emit_progress(document_id, "failed", 0.0, error_message)
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
    logger.debug("Blob storage not configured; using local pickles (deprecated).")
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
    total_local = len(ordered_ids)
    completed_local = 0

    logger.info(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║  BATCH EMBEDDING (local): %d documents queued (sub=%s)  ║",
        total_local, subscription_id or "global",
    )
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝"
    )
    _emit_embedding_batch_progress(subscription_id, 0, total_local, stage="starting")

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
                completed_local += 1
                logger.info("[EMBEDDING %d/%d] ✓ Completed: doc=%s", completed_local, total_local, doc_id)
            except Exception as exc:  # noqa: BLE001
                error_message = _truncate_error_message(str(exc) or repr(exc))
                logger.error(
                    "[EMBEDDING %d/%d] ✗ Failed: doc=%s error=%s",
                    completed_local + 1, total_local, doc_id, error_message,
                )
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
            _emit_embedding_batch_progress(
                subscription_id, completed_local, total_local,
                current_doc=doc_id or "", stage=f"processed {completed_local}/{total_local}",
            )

    results = [results_by_index[index] for index in sorted(results_by_index)]
    summary = _build_embed_summary(results)
    logger.info(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║  BATCH EMBEDDING (local) DONE: %d/%d succeeded (sub=%s)  ║",
        completed_local, total_local, subscription_id or "global",
    )
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝"
    )
    _emit_embedding_batch_progress(subscription_id, completed_local, total_local, stage="completed")
    return summary

def _emit_embedding_batch_progress(subscription_id: str, completed: int, total: int,
                                    current_doc: str = "", stage: str = "") -> None:
    """Publish batch embedding progress to Redis for frontend polling."""
    try:
        import json as _json
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return
        progress = round(completed / total, 3) if total > 0 else 0.0
        event = {
            "subscription_id": subscription_id or "global",
            "completed": completed,
            "total": total,
            "progress": progress,
            "current_document": current_doc,
            "stage": stage,
            "timestamp": time.time(),
        }
        payload = _json.dumps(event)
        client.setex(f"dw:embedding:batch_progress:{subscription_id or 'global'}", 3600, payload)
        client.publish("dw:embedding:progress", payload)
    except Exception:
        pass


def get_batch_embedding_progress(subscription_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest batch embedding progress for a subscription (for polling)."""
    try:
        import json as _json
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return None
        raw = client.get(f"dw:embedding:batch_progress:{subscription_id or 'global'}")
        return _json.loads(raw) if raw else None
    except Exception:
        return None


def embed_documents(
    *,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    max_blobs: Optional[int] = None,
) -> Dict[str, Any]:
    from src.utils.logging_utils import set_pipeline_profile, clear_pipeline_profile, clear_live_logs
    if profile_id:
        set_pipeline_profile(profile_id)
        clear_live_logs(profile_id)
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
        # Blob storage is configured but no pickles found in blob.
        # Extraction may have fallen back to local storage — try local pickles.
        logger.info(
            "embed_request_id=%s no blob candidates found; falling back to local pickles",
            embed_request_id,
        )
        return _embed_from_local_pickles(
            document_id=document_id,
            document_ids=document_ids,
            subscription_id=subscription_id,
            profile_id=profile_id,
            doc_type=doc_type,
            max_blobs=max_blobs,
            embed_request_id=embed_request_id,
        )

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = _get_max_workers(len(blob_candidates))
    total_blobs = len(blob_candidates)
    completed_blobs = 0

    logger.info(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║  BATCH EMBEDDING START: %d documents queued (sub=%s)  ║",
        total_blobs, subscription_id or "global",
    )
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝"
    )
    _emit_embedding_batch_progress(subscription_id, 0, total_blobs, stage="starting")

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
                completed_blobs += 1
                logger.info(
                    "[EMBEDDING %d/%d] ✓ Completed: doc=%s",
                    completed_blobs, total_blobs, doc_id,
                )
            except Exception as exc:  # noqa: BLE001
                error_message = _truncate_error_message(str(exc) or repr(exc))
                logger.error(
                    "[EMBEDDING %d/%d] ✗ Failed: doc=%s error=%s",
                    completed_blobs + 1, total_blobs, doc_id, error_message,
                )
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
            _emit_embedding_batch_progress(
                subscription_id, completed_blobs, total_blobs,
                current_doc=doc_id or "",
                stage=f"processed {completed_blobs}/{total_blobs}",
            )

    results = [results_by_index[index] for index in sorted(results_by_index)]
    for _entry in results:
        if telemetry:
            telemetry.increment("embed_pickles_processed_total")

    summary = _build_embed_summary(results)
    logger.info(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║  BATCH EMBEDDING DONE: %d documents, %d succeeded (sub=%s)  ║",
        total_blobs, completed_blobs, subscription_id or "global",
    )
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝"
    )
    _emit_embedding_batch_progress(subscription_id, completed_blobs, total_blobs, stage="completed")
    clear_pipeline_profile()
    return summary

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
