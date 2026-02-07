from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.embedding.pipeline.chunk_integrity import (
    ChunkIntegrityConfig,
    clean_text_for_embedding,
    enforce_chunk_integrity,
    is_chunk_complete,
)
from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text
from src.embedding.pipeline.dedupe_gate import DedupeConfig, apply_dedupe_gate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkPrepStats:
    dropped_quality: int = 0
    dropped_dedupe: int = 0
    merged_dedupe: int = 0
    merged_small: int = 0


def compute_stable_chunk_id(
    subscription_id: str,
    profile_id: str,
    document_id: str,
    section_id: str,
    chunk_index: int,
    chunk_hash: str,
) -> str:
    base = "|".join(
        [
            str(subscription_id),
            str(profile_id),
            str(document_id),
            str(section_id),
            str(chunk_index),
            str(chunk_hash),
        ]
    )
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"chunk_{digest}"


def normalize_chunk_chain(
    chunk_metadata: List[Dict[str, Any]],
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str,
    chunks: List[str],
) -> List[Dict[str, Any]]:
    if len(chunk_metadata) != len(chunks):
        raise ValueError("chunk_metadata and chunks length mismatch")

    normalized: List[Dict[str, Any]] = []
    for idx, meta in enumerate(chunk_metadata):
        m = dict(meta) if meta else {}
        chunk_hash = m.get("chunk_hash") or hashlib.sha256(chunks[idx].encode("utf-8")).hexdigest()
        section_id = m.get("section_id") or "section"
        m["chunk_index"] = idx
        m["chunk_count"] = len(chunks)
        m["chunk_hash"] = chunk_hash
        m["chunk_id"] = compute_stable_chunk_id(
            subscription_id,
            profile_id,
            document_id,
            section_id,
            idx,
            chunk_hash,
        )
        normalized.append(m)

    for idx, meta in enumerate(normalized):
        meta["prev_chunk_id"] = normalized[idx - 1]["chunk_id"] if idx > 0 else None
        meta["next_chunk_id"] = normalized[idx + 1]["chunk_id"] if idx < len(normalized) - 1 else None
        meta["document_id"] = document_id

    return normalized


def prepare_embedding_chunks(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str,
    doc_name: str,
    quality_filter: Optional[
        Callable[[List[str], List[Dict[str, Any]]], Tuple[List[str], List[Dict[str, Any]], int]]
    ] = None,
    integrity_config: Optional[ChunkIntegrityConfig] = None,
    dedupe_config: Optional[DedupeConfig] = None,
) -> Tuple[List[str], List[Dict[str, Any]], ChunkPrepStats]:
    if len(chunks) != len(metadata):
        raise ValueError("chunk and metadata length mismatch")

    prepared_chunks: List[str] = []
    prepared_meta: List[Dict[str, Any]] = []
    for chunk_text, meta in zip(chunks, metadata):
        m = dict(meta)
        m["document_id"] = document_id
        section_title = (m.get("section_title") or "Untitled Section").strip() or "Untitled Section"
        section_path = (m.get("section_path") or section_title).strip() or section_title
        m["section_title"] = section_title
        m["section_path"] = section_path
        m["section_id"] = m.get("section_id") or hashlib.sha1(
            f"{document_id}|{section_path}".encode("utf-8")
        ).hexdigest()[:12]
        m["chunk_type"] = m.get("chunk_type", "text")
        m["chunk_kind"] = m.get("chunk_kind", "section_text")
        raw_text = chunk_text
        clean_text = clean_text_for_embedding(raw_text)
        embedding_text = ensure_embedding_text(raw_text, m.get("doc_domain"), m.get("section_kind"))
        m["content"] = raw_text
        m["embedding_text"] = embedding_text
        m["text_clean"] = embedding_text
        if raw_text != embedding_text:
            m["text_raw"] = raw_text
        prepared_chunks.append(embedding_text or clean_text)
        prepared_meta.append(m)

    prepared_chunks, prepared_meta, integrity_stats = enforce_chunk_integrity(
        prepared_chunks, prepared_meta, config=integrity_config
    )
    prepared_chunks, prepared_meta, dedupe_stats = apply_dedupe_gate(
        prepared_chunks, prepared_meta, config=dedupe_config
    )

    dropped_quality = 0
    if quality_filter:
        prepared_chunks, prepared_meta, dropped_quality = quality_filter(prepared_chunks, prepared_meta)

    prepared_meta = normalize_chunk_chain(
        prepared_meta,
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
        chunks=prepared_chunks,
    )

    for idx, meta in enumerate(prepared_meta):
        meta["chunk_char_len"] = meta.get("chunk_char_len") or len(prepared_chunks[idx])
        chunk_type = meta.get("chunk_type") or "text"
        meta["sentence_complete"] = bool(
            meta.get("sentence_complete", is_chunk_complete(prepared_chunks[idx], str(chunk_type)))
        )

    stats = ChunkPrepStats(
        dropped_quality=dropped_quality,
        dropped_dedupe=dedupe_stats.get("dropped", 0),
        merged_dedupe=dedupe_stats.get("merged", 0),
        merged_small=integrity_stats.get("merged_small", 0),
    )
    logger.info(
        "Chunk prep for %s: prepared=%s dropped_quality=%s dropped_dedupe=%s merged_dedupe=%s merged_small=%s",
        doc_name,
        len(prepared_chunks),
        stats.dropped_quality,
        stats.dropped_dedupe,
        stats.merged_dedupe,
        stats.merged_small,
    )
    return prepared_chunks, prepared_meta, stats


__all__ = [
    "ChunkPrepStats",
    "compute_stable_chunk_id",
    "normalize_chunk_chain",
    "prepare_embedding_chunks",
]
