from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
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

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Text salvage for ExtractedDocument repr strings
# ---------------------------------------------------------------------------
_EXTRACTED_DOC_REPR_RE = re.compile(
    r"^Extracted\s*Document\s*\(\s*full_text=['\"]", re.IGNORECASE,
)
_EXTRACTED_DOC_FULLTEXT_RE = re.compile(
    r"full_text=['\"](.+?)(?:['\"],\s*\w+=|['\"]\s*\)$)", re.DOTALL,
)
_EXTRACTED_DOC_PREFIX_RE = re.compile(
    r"^Extracted\s+Document\s+(?:full_text|[\w.]+)\s*", re.IGNORECASE,
)

def _salvage_chunk_text(text: str) -> str:
    """Try to extract real document content from garbage/metadata-contaminated text.

    Handles two formats:
    1. Python repr: ``ExtractedDocument(full_text='...', sections=[...])``
    2. Space-delimited: ``Extracted Document full_text actual content, section_id ...``
    """
    if not text:
        return ""

    # Strategy 1: Extract full_text from ExtractedDocument(...) Python repr
    if _EXTRACTED_DOC_REPR_RE.match(text):
        m = _EXTRACTED_DOC_FULLTEXT_RE.search(text)
        if m:
            extracted = m.group(1).replace("\\n", "\n").strip()
            if extracted and len(extracted) > 20:
                return extracted
        # Fallback: strip the wrapper, find end of full_text value
        stripped = re.sub(r"^Extracted\s*Document\s*\(\s*full_text=['\"]", "", text)
        for end_marker in ("', sections=", '", sections=', "', tables=", '", tables=',
                           "', metadata=", '", metadata=', "', doc_type=", '", doc_type=',
                           "')", '")'):
            idx = stripped.find(end_marker)
            if idx != -1:
                stripped = stripped[:idx]
                break
        stripped = stripped.replace("\\n", "\n").strip()
        if stripped and len(stripped) > 20:
            return stripped

    # Strategy 2: Strip "Extracted Document full_text" prefix (space-delimited format)
    if text.startswith("Extracted Document"):
        stripped = _EXTRACTED_DOC_PREFIX_RE.sub("", text).strip()
        if stripped and len(stripped) > 20:
            return stripped

    return ""

@dataclass(frozen=True)
class ChunkPrepStats:
    dropped_quality: int = 0
    dropped_dedupe: int = 0
    merged_dedupe: int = 0
    merged_small: int = 0
    rescued_fragments: int = 0

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
) -> Tuple[List[str], List[Dict[str, Any]], ChunkPrepStats, Dict[str, List[str]]]:
    if len(chunks) != len(metadata):
        raise ValueError("chunk and metadata length mismatch")

    prepared_chunks: List[str] = []
    prepared_meta: List[Dict[str, Any]] = []
    # Multi-resolution: collect short fragments that would be dropped, keyed by section_id
    rescued_fragments: Dict[str, List[str]] = {}
    rescued_count = 0
    from src.embedding.pipeline.schema_normalizer import _is_encoding_garbage, _is_metadata_garbage
    for chunk_text, meta in zip(chunks, metadata):
        m = dict(meta)
        m["document_id"] = document_id
        section_title = (m.get("section_title") or "Untitled Section").strip() or "Untitled Section"
        raw_path = m.get("section_path") or section_title
        if isinstance(raw_path, list):
            section_path = " > ".join(str(p).strip() for p in raw_path if str(p).strip()) or section_title
        else:
            section_path = str(raw_path).strip() or section_title
        m["section_title"] = section_title
        m["section_path"] = section_path
        m["section_id"] = m.get("section_id") or hashlib.sha1(
            f"{document_id}|{section_path}".encode("utf-8")
        ).hexdigest()[:12]
        m["chunk_type"] = m.get("chunk_type", "text")
        m["chunk_kind"] = m.get("chunk_kind", "section_text")
        raw_text = chunk_text
        clean_text = clean_text_for_embedding(raw_text)
        # Guard against encoding garbage (UTF-16 null interleaving, etc.)
        if _is_encoding_garbage(raw_text):
            logger.warning(
                "Dropping encoding-garbage chunk %d for %s (null/control char ratio too high)",
                len(prepared_chunks), doc_name,
            )
            continue
        # Guard against metadata garbage in raw text — try salvage before fallback
        if _is_metadata_garbage(raw_text):
            salvaged = _salvage_chunk_text(raw_text)
            if salvaged and len(salvaged.strip()) >= 20:
                raw_text = salvaged
                clean_text = clean_text_for_embedding(raw_text)
            else:
                raw_text = clean_text
        embedding_text = ensure_embedding_text(raw_text, m.get("doc_domain"), m.get("section_kind"))
        # Pre-embedding validation gate: never embed garbage or trivially short text
        final_text = embedding_text or clean_text
        if _is_metadata_garbage(final_text):
            # Last-chance salvage before dropping
            salvaged = _salvage_chunk_text(final_text)
            if salvaged and len(salvaged.strip()) >= 20 and not _is_metadata_garbage(salvaged):
                raw_text = salvaged
                embedding_text = ensure_embedding_text(salvaged, m.get("doc_domain"), m.get("section_kind"))
                final_text = embedding_text or salvaged
        if _is_metadata_garbage(final_text) or len(final_text.strip()) < 20:
            # Multi-resolution rescue: collect short fragments instead of losing them
            stripped = final_text.strip()
            if stripped and len(stripped) >= 3 and not _is_metadata_garbage(stripped):
                sec_id = m["section_id"]
                rescued_fragments.setdefault(sec_id, []).append(stripped)
                rescued_count += 1
                logger.debug(
                    "Rescued short fragment from chunk %d for %s into section %s (len=%d)",
                    len(prepared_chunks), doc_name, sec_id, len(stripped),
                )
            else:
                logger.warning(
                    "Dropping garbage/short chunk %d for %s (len=%d)",
                    len(prepared_chunks), doc_name, len(final_text.strip()),
                )
            continue
        m["content"] = raw_text
        m["embedding_text"] = embedding_text
        m["text_clean"] = embedding_text
        m["resolution"] = "chunk"
        if raw_text != embedding_text:
            m["text_raw"] = raw_text
        prepared_chunks.append(final_text)
        prepared_meta.append(m)
    if rescued_count:
        logger.info(
            "Multi-resolution: rescued %d short fragments across %d sections for %s",
            rescued_count, len(rescued_fragments), doc_name,
        )

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
        rescued_fragments=rescued_count,
    )
    logger.info(
        "Chunk prep for %s: prepared=%s dropped_quality=%s dropped_dedupe=%s merged_dedupe=%s merged_small=%s rescued=%s",
        doc_name,
        len(prepared_chunks),
        stats.dropped_quality,
        stats.dropped_dedupe,
        stats.merged_dedupe,
        stats.merged_small,
        stats.rescued_fragments,
    )
    return prepared_chunks, prepared_meta, stats, rescued_fragments

__all__ = [
    "ChunkPrepStats",
    "compute_stable_chunk_id",
    "normalize_chunk_chain",
    "prepare_embedding_chunks",
]
