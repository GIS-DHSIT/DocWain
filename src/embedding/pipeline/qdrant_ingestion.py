from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Iterable, List, Optional

import ollama
from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.pipeline_models import ChunkRecord
from src.api.vector_store import QdrantVectorStore, build_collection_name, compute_chunk_id
from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text
from src.utils.payload_utils import get_canonical_text, token_count

logger = get_logger(__name__)

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
        raw_data.get("canonical_text"),
        raw_data.get("text"),
        raw_data.get("content"),
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

    content = raw_data.get("content") or _merge_texts(raw_data)
    canonical_text = get_canonical_text({"text": raw_data.get("text"), "content": content})
    canonical_len = len(canonical_text) if canonical_text else 0
    canonical_tokens = token_count(canonical_text) if canonical_text else 0
    embedding_text = raw_data.get("embedding_text") or ensure_embedding_text(content or "", raw_data.get("doc_domain"), raw_data.get("section_kind"))

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

    chunk_id = raw_data.get("chunk_id") or (raw_data.get("chunk") or {}).get("id")
    if not chunk_id and subscription_id and profile_id and document_id:
        chunk_id = compute_chunk_id(
            str(subscription_id),
            str(profile_id),
            str(document_id),
            source_file or "unknown",
            int(chunk_index or 0),
            content or "",
        )

    payload: Dict[str, Any] = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": document_id,
        "doc_id": document_id,
        "source_name": source_file or "unknown",
        "doc_domain": raw_data.get("doc_domain") or (raw_data.get("document") or {}).get("domain") or "unknown",
        "section_kind": raw_data.get("section_kind") or "unknown",
        "section_id": raw_data.get("section_id") or "unknown",
        "chunk_id": chunk_id or "unknown",
        "page": raw_data.get("page") or raw_data.get("page_start") or 0,
        "content": content,
        "canonical_text": canonical_text,
        "embedding_text": embedding_text,
        "canonical_text_len": canonical_len,
        "canonical_token_count": canonical_tokens,
        "chunking_mode": raw_data.get("chunking_mode") or "external",
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
        if not payload.get("canonical_text") and not payload.get("content"):
            logger.debug("Skipping payload without content (document_id=%s)", payload.get("document_id"))
            continue
        if not payload.get("subscription_id"):
            raise ValueError("subscription_id is required for Qdrant ingestion")
        transformed.append(payload)

    # --- Document profiling: learn domain from content at ingestion ---
    _profiler_enabled = getattr(Config, "DocumentProfiler", None) and getattr(Config.DocumentProfiler, "ENABLED", False)
    if _profiler_enabled:
        try:
            from src.intelligence.document_profiler import DocumentProfiler
            from src.llm.gateway import get_llm_gateway

            _profiler = DocumentProfiler(get_llm_gateway())
            # Group chunks by document_id, profile first 5 chunks per doc
            _doc_chunks: Dict[str, List[str]] = {}
            _doc_filenames: Dict[str, str] = {}
            for p in transformed:
                doc_id = p.get("document_id", "unknown")
                text = p.get("canonical_text") or p.get("content") or ""
                if text and doc_id not in _doc_filenames:
                    _doc_chunks.setdefault(doc_id, [])
                    _doc_filenames[doc_id] = p.get("source_name") or (p.get("metadata") or {}).get("source_file") or "document"
                if text and len(_doc_chunks.get(doc_id, [])) < 5:
                    _doc_chunks.setdefault(doc_id, []).append(text)

            _profiles: Dict[str, Dict[str, Any]] = {}
            for doc_id, chunks in _doc_chunks.items():
                try:
                    profile = _profiler.profile(chunks, _doc_filenames.get(doc_id, "document"))
                    _profiles[doc_id] = {
                        "domain": profile.domain,
                        "document_type": profile.document_type,
                        "key_terminology": profile.key_terminology,
                        "field_types": profile.field_types,
                        "structure_pattern": profile.structure_pattern,
                        "language_register": profile.language_register,
                    }
                    logger.info("Document profiled: %s → %s (%s)", doc_id, profile.domain, profile.document_type)
                except Exception as exc:
                    logger.debug("Document profiling failed for %s: %s", doc_id, exc)

            # Inject profiles into payloads
            for p in transformed:
                doc_id = p.get("document_id", "unknown")
                if doc_id in _profiles:
                    p["domain_profile"] = _profiles[doc_id]
                    # Also set doc_domain for backward compat
                    p["doc_domain"] = _profiles[doc_id].get("domain", "generic")

        except Exception as exc:
            logger.warning("Document profiling skipped: %s", exc)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for payload in transformed:
        grouped.setdefault(str(payload["subscription_id"]), []).append(payload)

    results: Dict[str, int] = {}
    for subscription_id, payloads in grouped.items():
        collection_name = build_collection_name(subscription_id)
        contents = [p.get("embedding_text") or p.get("canonical_text") or p.get("content") or "" for p in payloads]
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
                payload.get("canonical_text") or payload.get("content") or "",
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
