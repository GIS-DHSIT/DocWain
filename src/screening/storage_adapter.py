from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId
from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.dataHandler import db, get_qdrant_client
from src.api.vector_store import build_collection_name

logger = logging.getLogger(__name__)


def _get_document_collection():
    try:
        return db[Config.MongoDB.DOCUMENTS]
    except Exception:  # noqa: BLE001
        return None


def _lookup_document(doc_id: str) -> Optional[Dict[str, Any]]:
    collection = _get_document_collection()
    if collection is None:
        return None

    queries: List[Dict[str, Any]] = []
    if ObjectId.is_valid(doc_id):
        queries.append({"_id": ObjectId(doc_id)})
    queries.extend([{"_id": doc_id}, {"document_id": doc_id}, {"documentId": doc_id}])

    for query in queries:
        try:
            record = collection.find_one(query)
            if record:
                return record
        except Exception as exc:  # noqa: BLE001
            logger.debug("Document lookup failed for %s with %s: %s", doc_id, query, exc)
            continue
    return None


def _extract_text_from_record(record: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    text_fields = ["text", "full_text", "raw_text", "content", "document_text", "extracted_text", "body", "summary"]
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())

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
    return texts


def _fetch_chunks_from_qdrant(
    doc_id: str,
    subscription_id: str,
    profile_id: Optional[str],
    client: Optional[QdrantClient] = None,
    limit: int = 256,
) -> List[Dict[str, Any]]:
    client = client or get_qdrant_client()
    collection_name = build_collection_name(subscription_id)
    scroll_filter: Dict[str, Any] = {"must": [{"key": "document_id", "match": {"value": str(doc_id)}}]}
    if profile_id:
        scroll_filter["must"].append({"key": "profile_id", "match": {"value": str(profile_id)}})

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
        logger.warning("Qdrant scroll failed for doc_id=%s: %s", doc_id, exc)
    return payloads


def _combine_payload_texts(payloads: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for payload in payloads:
        for key in ("text", "chunk", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
                break
    return texts


def get_document_metadata(doc_id: str) -> Dict[str, Any]:
    record = _lookup_document(doc_id)
    return record or {}


def get_document_doc_type(doc_id: str) -> Optional[str]:
    record = _lookup_document(doc_id)
    if not record:
        return None
    for key in ("doc_type", "doctype", "document_type", "type"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def get_document_text(doc_id: str) -> str:
    record = _lookup_document(doc_id) or {}
    texts = _extract_text_from_record(record)

    subscription_id = (
        record.get("subscriptionId")
        or record.get("subscription_id")
        or record.get("subscription")
        or "default"
    )
    profile_id = record.get("profileId") or record.get("profile_id") or record.get("profile")

    if not texts:
        payloads = _fetch_chunks_from_qdrant(
            doc_id=doc_id,
            subscription_id=str(subscription_id),
            profile_id=str(profile_id) if profile_id else None,
        )
        texts = _combine_payload_texts(payloads)

    if not texts:
        raise ValueError(f"Document text not found for doc_id={doc_id}")
    return "\n\n".join(texts)


def get_document_bytes(doc_id: str) -> Optional[bytes]:
    record = _lookup_document(doc_id) or {}
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
