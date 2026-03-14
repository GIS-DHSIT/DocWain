from __future__ import annotations

from src.utils.logging_utils import get_logger
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId
from qdrant_client import QdrantClient

from src.api.content_store import load_extracted_pickle
from src.api.config import Config
from src.api.pipeline_models import ExtractedDocument
from src.api.dataHandler import db, get_qdrant_client
from src.api.vector_store import build_collection_name

logger = get_logger(__name__)

def _get_document_collection():
    try:
        return db[Config.MongoDB.DOCUMENTS]
    except Exception:  # noqa: BLE001
        return None

def _get_document_record(doc_id: str) -> Dict[str, Any]:
    collection = _get_document_collection()
    if collection is None:
        raise ValueError("Document store is not accessible")

    queries: List[Dict[str, Any]] = []
    if ObjectId.is_valid(doc_id):
        queries.append({"_id": ObjectId(doc_id)})

    alt_filters = [
        {"_id": doc_id},
        {"document_id": doc_id},
        {"doc_id": doc_id},
        {"documentId": doc_id},
        {"docId": doc_id},
        {"id": doc_id},
    ]
    queries.append({"$or": alt_filters})

    for query in queries:
        try:
            record = collection.find_one(query)
            if record:
                return record
        except Exception as exc:  # noqa: BLE001
            logger.debug("Document lookup failed for %s with %s: %s", doc_id, query, exc)
            continue

    raise ValueError(f"Document not found for doc_id={doc_id}")

def _extract_text_from_record(record: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    # Primary text fields
    text_fields = ["text", "full_text", "raw_text", "content", "document_text", "extracted_text", "body", "summary"]
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())

    # Secondary content fields
    secondary_fields = ["description", "keywords", "abstract", "overview", "extracted_content", "rawContent"]
    for field in secondary_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())

    # Extracted document data
    extracted = record.get("extractedDoc")
    if isinstance(extracted, dict):
        for _, value in extracted.items():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                inner_text = value.get("text") or value.get("content")
                if isinstance(inner_text, str):
                    texts.append(inner_text)
                elif isinstance(inner_text, list):
                    texts.append(" ".join(str(item) for item in inner_text if item))
            elif isinstance(value, list):
                texts.append(" ".join(str(item) for item in value if item))

    # Metadata fields that may contain useful info
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ["summary", "description", "content", "text"]:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

    return texts

def _extract_text_from_extracted(extracted: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(extracted, ExtractedDocument):
        if extracted.full_text:
            texts.append(extracted.full_text)
        else:
            texts.extend([sec.text for sec in extracted.sections if sec.text])
        return texts

    if isinstance(extracted, dict):
        for value in extracted.values():
            if isinstance(value, ExtractedDocument):
                if value.full_text:
                    texts.append(value.full_text)
                else:
                    texts.extend([sec.text for sec in value.sections if sec.text])
            elif isinstance(value, str):
                if value.strip():
                    texts.append(value.strip())
            elif isinstance(value, dict):
                # Recurse into nested dicts — pickle structure can be
                # {"raw": {"filename.pdf": ExtractedDocument(...)}, ...}
                for inner_value in value.values():
                    if isinstance(inner_value, ExtractedDocument):
                        if inner_value.full_text:
                            texts.append(inner_value.full_text)
                        else:
                            texts.extend([sec.text for sec in inner_value.sections if sec.text])
                    elif isinstance(inner_value, str) and inner_value.strip():
                        texts.append(inner_value.strip())
                if not texts:
                    # Fallback: check for "text"/"content" keys directly
                    inner_text = value.get("text") or value.get("content")
                    if isinstance(inner_text, str) and inner_text.strip():
                        texts.append(inner_text.strip())
                    elif isinstance(inner_text, list):
                        texts.append(" ".join(str(item) for item in inner_text if item))
            elif isinstance(value, list):
                texts.append(" ".join(str(item) for item in value if item))
    return texts

def get_extracted_document(document_id: str) -> ExtractedDocument:
    extracted = load_extracted_pickle(document_id)
    if isinstance(extracted, ExtractedDocument):
        return extracted
    if isinstance(extracted, dict):
        for value in extracted.values():
            if isinstance(value, ExtractedDocument):
                return value
    raise ValueError(f"Extracted content is not an ExtractedDocument for document_id={document_id}")

def _fetch_chunks_from_qdrant(
    doc_id: str,
    subscription_id: str,
    profile_id: Optional[str],
    client: Optional[QdrantClient] = None,
    limit: int = 256,
) -> List[Dict[str, Any]]:
    if not subscription_id or str(subscription_id).strip().lower() == "default":
        logger.warning("Qdrant lookup unavailable for doc_id=%s: subscription_id missing", doc_id)
        return []
    if not profile_id:
        logger.warning("Qdrant lookup unavailable for doc_id=%s: profile_id missing", doc_id)
        return []
    client = client or get_qdrant_client()
    collection_name = build_collection_name(subscription_id)
    from src.api.vector_store import build_qdrant_filter

    # Check if collection exists before attempting to scroll
    try:
        client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001
        error_str = str(exc).lower()
        if "not found" in error_str or "doesn't exist" in error_str or "does not exist" in error_str:
            logger.debug(
                "Qdrant collection '%s' not found for doc_id=%s (subscription=%s); skipping Qdrant lookup",
                collection_name,
                doc_id,
                subscription_id
            )
        else:
            logger.warning(
                "Qdrant collection check failed for doc_id=%s (collection=%s): %s",
                doc_id,
                collection_name,
                exc
            )
        return []

    scroll_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_id=str(doc_id),
    )

    offset = None
    payloads: List[Dict[str, Any]] = []
    try:
        while len(payloads) < limit:
            batch, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=min(64, limit - len(payloads)),
                with_vectors=False,
                with_payload=True,
                offset=offset,
            )
            if not batch:
                break
            payloads.extend([pt.payload or {} for pt in batch])
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        error_str = str(exc).lower()
        if "not found" in error_str or "doesn't exist" in error_str:
            logger.debug(
                "Qdrant collection '%s' not found during scroll for doc_id=%s; no chunks available",
                collection_name,
                doc_id
            )
        else:
            logger.warning("Qdrant scroll failed for doc_id=%s: %s", doc_id, exc)
    return payloads

def _combine_payload_texts(payloads: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for payload in payloads:
        for key in ("canonical_text", "embedding_text", "text", "chunk", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
                break
    return texts

def get_document_subscription_id(doc_id: str) -> Optional[str]:
    record = _get_document_record(doc_id)
    candidates = [
        record.get("subscriptionId"),
        record.get("subscription_id"),
        record.get("subscription"),
        record.get("tenantId"),
        record.get("collection_name"),
    ]
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    candidates.extend(
        [
            metadata.get("subscriptionId"),
            metadata.get("subscription_id"),
            metadata.get("subscription"),
            metadata.get("tenantId"),
            metadata.get("collection_name"),
        ]
    )
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None

def get_document_metadata(doc_id: str) -> Dict[str, Any]:
    return _get_document_record(doc_id)

def get_document_doc_type(doc_id: str) -> Optional[str]:
    record = _get_document_record(doc_id)
    for key in ("doc_type", "doctype", "document_type", "type"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None

def extract_text_from_payload(extracted: Any) -> Optional[str]:
    texts = _extract_text_from_extracted(extracted)
    if texts:
        return "\n\n".join(texts)
    return None

def get_document_text(doc_id: str, extracted: Any = None, allow_fallback: bool = True) -> str:
    if extracted is not None:
        text = extract_text_from_payload(extracted)
        if text:
            return text
        if not allow_fallback:
            raise ValueError(f"Extracted payload had no text for doc_id={doc_id}")

    try:
        extracted = load_extracted_pickle(doc_id)
        texts = _extract_text_from_extracted(extracted)
        if texts:
            return "\n\n".join(texts)
        logger.debug("Extracted pickle for doc_id=%s had no text; falling back", doc_id)
    except Exception as exc:  # noqa: BLE001
        if allow_fallback:
            logger.warning("Blob pickle unavailable for doc_id=%s: %s", doc_id, exc)
        else:
            raise

    record = _get_document_record(doc_id)
    texts = _extract_text_from_record(record)

    subscription_id = get_document_subscription_id(doc_id)
    profile_id = record.get("profileId") or record.get("profile_id") or record.get("profile")

    if not texts:
        if subscription_id:
            try:
                payloads = _fetch_chunks_from_qdrant(
                    doc_id=doc_id,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id) if profile_id else None,
                )
                texts = _combine_payload_texts(payloads)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Qdrant retrieval failed for doc_id=%s (sub=%s, prof=%s): %s",
                    doc_id,
                    subscription_id,
                    profile_id,
                    exc
                )
                texts = []
        else:
            logger.warning(
                "No subscription_id available for doc_id=%s; qdrant_unavailable=true",
                doc_id,
            )

    # If text still missing, construct minimal text from metadata
    if not texts:
        logger.warning(
            "No text extracted for doc_id=%s from any source; constructing from metadata",
            doc_id
        )
        # Build comprehensive text from available metadata for screening
        metadata_text = []

        # Document identification
        if record.get("name"):
            metadata_text.append(f"Document Name: {record.get('name')}")
        if record.get("doc_type") or record.get("document_type"):
            doc_type = record.get('doc_type') or record.get('document_type')
            metadata_text.append(f"Document Type: {doc_type}")

        # Document status and metadata
        if record.get("description"):
            metadata_text.append(f"Description: {record.get('description')}")
        if record.get("status"):
            metadata_text.append(f"Status: {record.get('status')}")
        if record.get("createdAt") or record.get("created_at"):
            metadata_text.append(f"Created: {record.get('createdAt') or record.get('created_at')}")

        # Profile and subscription info
        if profile_id:
            metadata_text.append(f"Profile: {profile_id}")
        if subscription_id:
            metadata_text.append(f"Subscription: {subscription_id}")

        # Keywords and summary
        if record.get("keywords"):
            keywords = record.get("keywords")
            if isinstance(keywords, list):
                keywords = ", ".join(str(k) for k in keywords)
            metadata_text.append(f"Keywords: {keywords}")
        if record.get("summary"):
            metadata_text.append(f"Summary: {record.get('summary')}")
        if record.get("abstract"):
            metadata_text.append(f"Abstract: {record.get('abstract')}")

        # Additional metadata
        if record.get("source") or record.get("source_name"):
            metadata_text.append(f"Source: {record.get('source') or record.get('source_name')}")
        if record.get("tags"):
            tags = record.get("tags")
            if isinstance(tags, list):
                tags = ", ".join(str(t) for t in tags)
            metadata_text.append(f"Tags: {tags}")

        if metadata_text:
            logger.info(
                "Constructing content from %d metadata fields for doc_id=%s",
                len(metadata_text),
                doc_id
            )
            return "\n".join(metadata_text)

        # Last resort: return document ID as placeholder to allow screening to proceed
        logger.warning(
            "No metadata available for doc_id=%s; returning minimal placeholder",
            doc_id
        )
        return f"[Screening Placeholder] Document ID: {doc_id} - Original content unavailable"

    return "\n\n".join(texts)

def get_document_bytes(doc_id: str) -> Optional[bytes]:
    record = _get_document_record(doc_id)
    raw_bytes = record.get("raw_bytes") or record.get("file_bytes")
    if isinstance(raw_bytes, (bytes, bytearray)):
        return bytes(raw_bytes)

    path_value = record.get("file_path") or record.get("local_path") or record.get("path")
    if path_value:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = Path(Config.Path.DOCUMENTS_DIR) / candidate
        try:
            if candidate.exists():
                return candidate.read_bytes()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed reading bytes from %s: %s", candidate, exc)
    return None
