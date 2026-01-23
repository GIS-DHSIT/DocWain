import hashlib
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.api.blob_store import (
    BlobConfigurationError,
    BlobInfo,
    BlobStore,
    blob_storage_configured,
    extract_document_id,
    is_trusted_blob,
)
from src.api.config import Config
from src.api.content_store import delete_extracted_pickle, load_extracted_pickle
from src.api.dataHandler import get_qdrant_client, resolve_profile_id, resolve_subscription_id, train_on_document
from src.api.document_status import get_document_record, update_document_fields, update_stage
from src.api.pipeline_models import ExtractedDocument
from src.api.statuses import STATUS_EMBEDDING_COMPLETED, STATUS_EMBEDDING_FAILED, STATUS_SCREENING_COMPLETED
from src.api.vector_store import build_collection_name
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from src.api.enhanced_retrieval import chunk_text_for_embedding

logger = logging.getLogger(__name__)


def _telemetry():
    return telemetry_store() if METRICS_V2_ENABLED else None


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

    candidates = extracted.chunk_candidates or []
    chunks: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []

    if candidates:
        min_len = int(getattr(Config.Retrieval, "MIN_CHUNK_SIZE", 200))
        merged_candidates = _merge_candidates(candidates, min_len)
        for chunk_text, cand_meta in merged_candidates:
            chunks.append(chunk_text)
            chunk_metadata.append(
                {
                    "document_id": document_id,
                    "section_title": cand_meta.section_title or "Document",
                    "section_id": cand_meta.section_id
                    or hashlib.sha1(
                        f"{document_id}|{cand_meta.section_title or 'Document'}".encode("utf-8")
                    ).hexdigest()[:12],
                    "chunk_type": cand_meta.chunk_type,
                    "page_number": cand_meta.page,
                    "doc_type": doc_type,
                    "ocr_confidence": doc_ocr_confidence,
                }
            )
    else:
        for section in extracted.sections or []:
            section_text = section.text or ""
            if not section_text.strip():
                continue
            for chunk_text in _split_text_preserve(section_text):
                chunks.append(chunk_text)
                chunk_metadata.append(
                    {
                        "document_id": document_id,
                        "section_title": section.title or "Section",
                        "section_id": section.section_id
                        or hashlib.sha1(
                            f"{document_id}|{section.title or 'Section'}".encode("utf-8")
                        ).hexdigest()[:12],
                        "chunk_type": "section",
                        "page_number": section.start_page,
                        "doc_type": doc_type,
                        "ocr_confidence": doc_ocr_confidence,
                    }
                )

        if not chunks and extracted.full_text:
            for chunk_text in _split_text_preserve(extracted.full_text):
                chunks.append(chunk_text)
                chunk_metadata.append(
                    {
                        "document_id": document_id,
                        "section_title": "Document",
                        "section_id": hashlib.sha1(f"{document_id}|Document".encode("utf-8")).hexdigest()[:12],
                        "chunk_type": "text",
                        "page_number": None,
                        "doc_type": doc_type,
                        "ocr_confidence": doc_ocr_confidence,
                    }
                )

    if not chunks:
        raise ValueError(f"No chunk candidates extracted for {doc_name}")

    full_text = extracted.full_text or ""
    coverage_ratio = None
    if full_text:
        coverage_ratio = len("".join(chunks)) / max(1, len(full_text))
        coverage_threshold = float(getattr(Config.Retrieval, "CHUNK_COVERAGE_THRESHOLD", 0.98))
        if coverage_ratio < coverage_threshold:
            logger.error(
                "Chunk coverage %.3f below threshold %.3f for %s; falling back to full_text chunking",
                coverage_ratio,
                coverage_threshold,
                doc_name,
            )
            chunks = _split_text_preserve(full_text)
            chunk_metadata = [
                {
                    "document_id": document_id,
                    "section_title": "Document",
                    "section_id": hashlib.sha1(f"{document_id}|Document".encode("utf-8")).hexdigest()[:12],
                    "chunk_type": "text",
                    "page_number": None,
                    "doc_type": doc_type,
                    "ocr_confidence": doc_ocr_confidence,
                }
                for _ in chunks
            ]
            coverage_ratio = len("".join(chunks)) / max(1, len(full_text))

    filtered_chunks: List[str] = []
    filtered_meta: List[Dict[str, Any]] = []
    dropped = 0
    for chunk_text, meta in zip(chunks, chunk_metadata):
        if not (chunk_text or "").strip():
            dropped += 1
            continue
        filtered_chunks.append(chunk_text)
        filtered_meta.append(meta)

    return filtered_chunks, filtered_meta, coverage_ratio, dropped


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
        if isinstance(content.get("full_text"), str):
            return len(_split_text_preserve(content.get("full_text") or "")), None
    if isinstance(content, str):
        chunks_with_meta = chunk_text_for_embedding(content, doc_name, document_id=document_id)
        return len(chunks_with_meta), None
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
                longest = max((len(chunk) for chunk, _ in chunk_text_for_embedding(content, doc_name, document_id)), default=0)
            else:
                chunks, _, _, _ = _build_chunks_for_extracted_doc(content, document_id=document_id, doc_name=doc_name)
                longest = max((len(chunk) for chunk in chunks), default=0)
            if longest < min_len:
                logger.warning("Chunk length below minimum for %s (max=%s)", doc_name, longest)

    return total_chunks, coverage_values


def _count_qdrant_points(subscription_id: str, profile_id: str, document_id: str) -> int:
    client = get_qdrant_client()
    collection_name = build_collection_name(subscription_id)
    count_filter = Filter(
        must=[
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        ]
    )
    result = client.count(collection_name=collection_name, count_filter=count_filter, exact=False)
    return int(getattr(result, "count", 0) or 0)


def _build_blob_store() -> BlobStore:
    return BlobStore()


def _process_blob(
    *,
    store: BlobStore,
    blob: BlobInfo,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
) -> Dict[str, Any]:
    telemetry = _telemetry()
    result = {
        "blob_name": blob.name,
        "document_id": (blob.metadata or {}).get("document_id")
        or extract_document_id(blob.name, prefix=store.prefix),
        "status": "FAILED",
        "chunks_count": 0,
        "error": None,
    }

    lease_id = None
    try:
        lease_id = store.try_acquire_lease(blob.name, lease_duration=_lease_seconds())
        if not lease_id:
            if telemetry:
                telemetry.increment("embed_pickles_lease_conflict_total")
            result["status"] = "SKIPPED"
            result["error"] = "lease_conflict"
            return result
        if telemetry:
            telemetry.increment("embed_pickles_leased_total")

        try:
            payload = store.download_blob(blob.name, lease=lease_id)
        except Exception as exc:  # noqa: BLE001
            if telemetry:
                telemetry.increment("embed_pickles_download_fail_total")
            result["error"] = str(exc)
            return result

        try:
            extracted = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001
            if telemetry:
                telemetry.increment("embed_pickles_download_fail_total")
            result["error"] = f"pickle_deserialize_failed: {exc}"
            return result

        doc_id = (blob.metadata or {}).get("document_id") or extract_document_id(blob.name, prefix=store.prefix)
        if not doc_id:
            result["error"] = "document_id_missing"
            return result
        result["document_id"] = doc_id

        record = get_document_record(doc_id) or {}
        current_status = record.get("status")

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
            _set_document_status(doc_id, current_status or STATUS_SCREENING_COMPLETED, str(exc))
            result["error"] = str(exc)
            return result

        update_stage(
            doc_id,
            "embedding",
            {"status": "IN_PROGRESS", "started_at": time.time(), "error": None, "reason": None},
        )

        try:
            extracted_docs = _normalize_extracted_docs(extracted)
            expected_chunks, coverage_values = _screen_payload(extracted_docs, doc_id)
        except Exception as exc:  # noqa: BLE001
            if telemetry:
                telemetry.increment("embed_screening_fail_total")
            update_stage(
                doc_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": {"message": str(exc)}},
            )
            _set_document_status(doc_id, STATUS_EMBEDDING_FAILED, str(exc))
            result["error"] = str(exc)
            return result

        result["chunks_count"] = expected_chunks

        qdrant_count = 0
        if subscription_id and profile_id:
            try:
                qdrant_count = _count_qdrant_points(subscription_id, profile_id, doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant count failed for %s: %s", doc_id, exc)

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
            _set_document_status(doc_id, STATUS_EMBEDDING_COMPLETED)
            deleted = store.delete_blob(blob.name, lease=lease_id)
            if telemetry:
                telemetry.increment("embed_pickles_deleted_total" if deleted else "embed_pickles_delete_fail_total")
            update_stage(
                doc_id,
                "cleanup",
                {
                    "pickle_deleted": deleted,
                    "deleted_at": time.time() if deleted else None,
                    "cleanup_pending": not deleted,
                    "error": None if deleted else {"message": "blob_delete_failed"},
                },
            )
            result["status"] = "SKIPPED"
            result["error"] = None
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
                embed_result = train_on_document(content, subscription_id, profile_id, doc_id, file_name)
                total_chunks += int(embed_result.get("chunks", 0))
                total_upserted += int(embed_result.get("points_saved", 0))
                total_dropped += int(embed_result.get("dropped_chunks", 0))
                ratio = embed_result.get("coverage_ratio")
                if isinstance(ratio, (int, float)):
                    coverage_values.append(float(ratio))
        except Exception as exc:  # noqa: BLE001
            if telemetry:
                telemetry.increment("embed_qdrant_upsert_fail_total")
            update_stage(
                doc_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": {"message": str(exc)}},
            )
            _set_document_status(doc_id, STATUS_EMBEDDING_FAILED, str(exc))
            result["error"] = str(exc)
            return result

        if total_chunks != total_upserted:
            error_msg = f"Embedding upsert mismatch: expected {total_chunks}, saved {total_upserted}"
            if telemetry:
                telemetry.increment("embed_qdrant_upsert_fail_total")
            update_stage(
                doc_id,
                "embedding",
                {"status": "FAILED", "completed_at": time.time(), "error": {"message": error_msg}},
            )
            _set_document_status(doc_id, STATUS_EMBEDDING_FAILED, error_msg)
            result["error"] = error_msg
            return result

        collection_name = build_collection_name(subscription_id)
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

        _set_document_status(doc_id, STATUS_EMBEDDING_COMPLETED)
        deleted = store.delete_blob(blob.name, lease=lease_id)
        if telemetry:
            telemetry.increment("embed_pickles_deleted_total" if deleted else "embed_pickles_delete_fail_total")
        update_stage(
            doc_id,
            "cleanup",
            {
                "pickle_deleted": deleted,
                "deleted_at": time.time() if deleted else None,
                "cleanup_pending": not deleted,
                "error": None if deleted else {"message": "blob_delete_failed"},
            },
        )

        result["status"] = "COMPLETED"
        result["error"] = None
        result["chunks_count"] = total_chunks
        return result
    finally:
        if lease_id:
            store.release_lease(blob.name, lease_id)


def _process_local_document(
    *,
    document_id: str,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
) -> Dict[str, Any]:
    result = {
        "blob_name": f"{document_id}.pkl",
        "document_id": document_id,
        "status": "FAILED",
        "chunks_count": 0,
        "error": None,
    }

    record = get_document_record(document_id) or {}
    current_status = record.get("status")
    if current_status == STATUS_EMBEDDING_COMPLETED:
        result["status"] = "SKIPPED"
        return result
    if current_status != STATUS_SCREENING_COMPLETED:
        result["status"] = "SKIPPED"
        result["error"] = "screening_not_completed"
        return result

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
        _set_document_status(document_id, current_status or STATUS_SCREENING_COMPLETED, str(exc))
        result["error"] = str(exc)
        return result

    update_stage(
        document_id,
        "embedding",
        {"status": "IN_PROGRESS", "started_at": time.time(), "error": None, "reason": None},
    )

    try:
        extracted = load_extracted_pickle(document_id)
    except Exception as exc:  # noqa: BLE001
        update_stage(
            document_id,
            "embedding",
            {"status": "FAILED", "completed_at": time.time(), "error": {"message": str(exc)}},
        )
        _set_document_status(document_id, STATUS_EMBEDDING_FAILED, str(exc))
        result["error"] = str(exc)
        return result

    try:
        extracted_docs = _normalize_extracted_docs(extracted)
        expected_chunks, coverage_values = _screen_payload(extracted_docs, document_id)
    except Exception as exc:  # noqa: BLE001
        update_stage(
            document_id,
            "embedding",
            {"status": "FAILED", "completed_at": time.time(), "error": {"message": str(exc)}},
        )
        _set_document_status(document_id, STATUS_EMBEDDING_FAILED, str(exc))
        result["error"] = str(exc)
        return result

    result["chunks_count"] = expected_chunks

    total_chunks = 0
    total_upserted = 0
    total_dropped = 0
    coverage_values = coverage_values or []
    try:
        for file_name, content in extracted_docs.items():
            embed_result = train_on_document(content, subscription_id, profile_id, document_id, file_name)
            total_chunks += int(embed_result.get("chunks", 0))
            total_upserted += int(embed_result.get("points_saved", 0))
            total_dropped += int(embed_result.get("dropped_chunks", 0))
            ratio = embed_result.get("coverage_ratio")
            if isinstance(ratio, (int, float)):
                coverage_values.append(float(ratio))
    except Exception as exc:  # noqa: BLE001
        update_stage(
            document_id,
            "embedding",
            {"status": "FAILED", "completed_at": time.time(), "error": {"message": str(exc)}},
        )
        _set_document_status(document_id, STATUS_EMBEDDING_FAILED, str(exc))
        result["error"] = str(exc)
        return result

    if total_chunks != total_upserted:
        error_msg = f"Embedding upsert mismatch: expected {total_chunks}, saved {total_upserted}"
        update_stage(
            document_id,
            "embedding",
            {"status": "FAILED", "completed_at": time.time(), "error": {"message": error_msg}},
        )
        _set_document_status(document_id, STATUS_EMBEDDING_FAILED, error_msg)
        result["error"] = error_msg
        return result

    collection_name = build_collection_name(subscription_id)
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

    _set_document_status(document_id, STATUS_EMBEDDING_COMPLETED)
    deleted = delete_extracted_pickle(document_id)
    update_stage(
        document_id,
        "cleanup",
        {
            "pickle_deleted": deleted,
            "deleted_at": time.time() if deleted else None,
            "cleanup_pending": not deleted,
            "error": None if deleted else {"message": "local_pickle_delete_failed"},
        },
    )

    result["status"] = "COMPLETED"
    result["error"] = None
    result["chunks_count"] = total_chunks
    return result


def _embed_from_local_pickles(
    *,
    document_id: Optional[str],
    document_ids: Optional[List[str]],
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
    max_blobs: Optional[int],
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
        return {
            "documents": [],
            "summary": {"processed": 0, "skipped": 0, "failed": 0},
            "message": "no matching pickles found",
        }

    limit = _get_max_blobs(max_blobs)
    ordered_ids = ordered_ids[:limit]

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = _get_max_workers(len(ordered_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_local_document,
                document_id=doc_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                doc_type=doc_type,
            ): index
            for index, doc_id in enumerate(ordered_ids)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results_by_index[index] = future.result()

    results = [results_by_index[index] for index in sorted(results_by_index)]
    summary = {"processed": 0, "skipped": 0, "failed": 0}
    for entry in results:
        status = entry.get("status")
        if status == "COMPLETED":
            summary["processed"] += 1
        elif status == "SKIPPED":
            summary["skipped"] += 1
        else:
            summary["failed"] += 1

    return {"documents": results, "summary": summary}


def embed_documents(
    *,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    max_blobs: Optional[int] = None,
) -> Dict[str, Any]:
    if not blob_storage_configured():
        return _embed_from_local_pickles(
            document_id=document_id,
            document_ids=document_ids,
            subscription_id=subscription_id,
            profile_id=profile_id,
            doc_type=doc_type,
            max_blobs=max_blobs,
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
        return {
            "documents": [],
            "summary": {"processed": 0, "skipped": 0, "failed": 0},
            "message": "no matching pickles found",
        }

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = _get_max_workers(len(blob_candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_blob,
                store=store,
                blob=blob,
                subscription_id=subscription_id,
                profile_id=profile_id,
                doc_type=doc_type,
            ): index
            for index, blob in enumerate(blob_candidates)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results_by_index[index] = future.result()

    results = [results_by_index[index] for index in sorted(results_by_index)]

    summary = {"processed": 0, "skipped": 0, "failed": 0}
    for entry in results:
        status = entry.get("status")
        if status == "COMPLETED":
            summary["processed"] += 1
        elif status == "SKIPPED":
            summary["skipped"] += 1
        else:
            summary["failed"] += 1
        if telemetry:
            telemetry.increment("embed_pickles_processed_total")

    return {"documents": results, "summary": summary}
