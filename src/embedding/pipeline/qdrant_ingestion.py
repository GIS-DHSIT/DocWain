from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ollama
from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.pipeline_models import ChunkRecord
from src.api.vector_store import QdrantVectorStore, build_collection_name, compute_chunk_id

logger = logging.getLogger(__name__)

_DOC_TYPE_MAP = {
    "report_table_heavy": "resume",
}


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_present(*values: Any) -> Optional[str]:
    for value in values:
        text = _stringify(value)
        if text:
            return text
    return None


def _merge_texts(raw_data: Dict[str, Any]) -> str:
    candidates = [
        raw_data.get("text"),
        (raw_data.get("text_data") or {}).get("clean"),
        raw_data.get("text_clean"),
    ]
    merged: List[str] = []
    seen = set()
    for value in candidates:
        text = _stringify(value)
        if not text:
            continue
        if text in seen:
            continue
        merged.append(text)
        seen.add(text)
    return "\n\n".join(merged).strip()


def _map_doc_type(raw_doc_type: Optional[str]) -> Optional[str]:
    if not raw_doc_type:
        return None
    return _DOC_TYPE_MAP.get(raw_doc_type, raw_doc_type)


def extract_skills_stub(content: str) -> List[str]:
    """Placeholder for future skill extraction; returns empty list for now."""
    _ = content
    return []


def transform_payload(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize messy payloads into a retrieval-ready schema."""
    subscription_id = _first_present(
        raw_data.get("subscription_id"),
        raw_data.get("subscriptionId"),
        raw_data.get("subscription"),
    )
    profile_id = _first_present(
        raw_data.get("profile_id"),
        raw_data.get("profileId"),
        raw_data.get("profile"),
    )
    document_id = _first_present(
        raw_data.get("document_id"),
        raw_data.get("documentId"),
        raw_data.get("doc_id"),
        raw_data.get("docId"),
    )

    content = _merge_texts(raw_data)

    source_file = _first_present(
        (raw_data.get("source") or {}).get("name"),
        raw_data.get("source_file"),
        raw_data.get("source_name"),
        raw_data.get("filename"),
    )
    ingestion_source = _first_present(
        (raw_data.get("document") or {}).get("ingestion_source"),
        raw_data.get("ingestion_source"),
        raw_data.get("source_type"),
    )
    raw_doc_type = _first_present(
        (raw_data.get("document") or {}).get("type"),
        raw_data.get("document_type"),
        raw_data.get("doc_type"),
    )
    doc_type = _map_doc_type(raw_doc_type)

    chunk_index = raw_data.get("chunk_index")
    if chunk_index is None:
        chunk_index = (raw_data.get("chunk") or {}).get("index")

    prev_chunk_id = _first_present(
        raw_data.get("prev_chunk_id"),
        (raw_data.get("chunk") or {}).get("links", {}).get("prev"),
    )
    next_chunk_id = _first_present(
        raw_data.get("next_chunk_id"),
        (raw_data.get("chunk") or {}).get("links", {}).get("next"),
    )

    skills = extract_skills_stub(content)

    payload: Dict[str, Any] = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": document_id,
        "doc_id": document_id,
        "content": content,
        "doc_type": doc_type,
        "metadata": {
            "source_file": source_file,
            "ingestion_source": ingestion_source,
            "doc_type": doc_type,
            "skills": skills,
        },
        "navigation": {
            "chunk_index": chunk_index,
            "prev_chunk_id": prev_chunk_id,
            "next_chunk_id": next_chunk_id,
        },
    }

    return {k: v for k, v in payload.items() if v is not None}


def _ollama_embed(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    model_name = model or getattr(Config.Model, "OLLAMA_EMBEDDING_MODEL", "bge-m3")
    vectors: List[List[float]] = []
    for text in texts:
        if not text:
            vectors.append([])
            continue
        response = ollama.embeddings(model=model_name, prompt=text)
        embedding = response.get("embedding")
        if embedding is None and isinstance(response.get("data"), list):
            embedding = response["data"][0].get("embedding") if response["data"] else None
        if not embedding:
            raise ValueError("Ollama embeddings response missing embedding vector")
        vectors.append([float(x) for x in embedding])
    return vectors


def ingest_payloads(
    raw_payloads: Iterable[Dict[str, Any]],
    *,
    client: Optional[QdrantClient] = None,
    batch_size: int = 100,
) -> Dict[str, int]:
    """Transform, embed, and upsert payloads into subscription-scoped Qdrant collections."""
    vector_store = QdrantVectorStore(client=client)
    transformed: List[Dict[str, Any]] = []
    for raw in raw_payloads:
        payload = transform_payload(raw)
        if not payload.get("content"):
            logger.warning("Skipping payload without content (document_id=%s)", payload.get("document_id"))
            continue
        if not payload.get("subscription_id"):
            raise ValueError("subscription_id is required for Qdrant ingestion")
        transformed.append(payload)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for payload in transformed:
        grouped.setdefault(str(payload["subscription_id"]), []).append(payload)

    results: Dict[str, int] = {}
    for subscription_id, payloads in grouped.items():
        collection_name = build_collection_name(subscription_id)
        contents = [p["content"] for p in payloads]
        vectors = _ollama_embed(contents)
        vector_size = len(next((v for v in vectors if v), []))
        if not vector_size:
            raise ValueError("Failed to compute embedding dimension from Ollama response")
        vector_store.ensure_collection(collection_name, vector_size)

        records: List[ChunkRecord] = []
        for payload, vector in zip(payloads, vectors):
            if not vector:
                continue
            nav = payload.get("navigation") or {}
            chunk_index = nav.get("chunk_index") or 0
            try:
                chunk_index_val = int(chunk_index)
            except (TypeError, ValueError):
                chunk_index_val = 0
            source_file = (payload.get("metadata") or {}).get("source_file") or "unknown"
            chunk_id = compute_chunk_id(
                payload.get("subscription_id"),
                payload.get("profile_id"),
                payload.get("document_id"),
                source_file,
                chunk_index_val,
                payload.get("content") or "",
            )
            payload["chunk_id"] = payload.get("chunk_id") or chunk_id
            records.append(
                ChunkRecord(
                    chunk_id=str(chunk_id),
                    dense_vector=vector,
                    sparse_vector=None,
                    payload=payload,
                )
            )

        if not records:
            results[collection_name] = 0
            continue

        saved = vector_store.upsert_records(collection_name, records, batch_size=batch_size)
        results[collection_name] = saved

    return results


__all__ = ["transform_payload", "ingest_payloads"]
