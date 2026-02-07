from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, Optional

from src.embedding.pipeline import schema_normalizer as _schema_normalizer
from src.embedding.pipeline.chunk_integrity import clean_text_for_embedding
from src.metadata.normalizer import normalize_chunk_kind
from src.utils.payload_utils import get_canonical_text, token_count

_EVIDENCE_RE = re.compile(r"Section:\s*(.*?)(?:,|$)\s*Page:\s*(.*)$", re.IGNORECASE)

_CANONICAL_REQUIRED_FIELDS = (
    "subscription_id",
    "profile_id",
    "document_id",
    "source_name",
    "doc_domain",
    "section_kind",
    "section_id",
    "chunk_id",
    "page",
    "canonical_text",
)

_CANONICAL_DEFAULTS = {
    "source_name": "unknown",
    "doc_domain": "unknown",
    "section_kind": "unknown",
    "section_id": "unknown",
    "chunk_id": "unknown",
    "page": 0,
    "canonical_text": "",
}


def _drop_empty(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        cleaned = [v for v in (_drop_empty(v) for v in value) if v is not None]
        return cleaned or None
    if isinstance(value, dict):
        cleaned = {k: v for k, v in ((k, _drop_empty(v)) for k, v in value.items()) if v is not None}
        return cleaned or None
    return value


def _ensure_canonical_payload_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    missing_required = [
        field
        for field in ("subscription_id", "profile_id", "document_id")
        if not _stringify(payload.get(field))
    ]
    if missing_required:
        raise ValueError(f"Missing required payload fields: {', '.join(missing_required)}")

    for field in _CANONICAL_REQUIRED_FIELDS:
        if field in payload and payload.get(field) not in (None, ""):
            continue
        fallback = _CANONICAL_DEFAULTS.get(field)
        if fallback is None:
            payload[field] = payload.get(field)
        else:
            payload[field] = fallback
    return payload


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _intify(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_evidence_pointer(pointer: Optional[str]) -> Dict[str, Any]:
    if not pointer:
        return {}
    match = _EVIDENCE_RE.search(pointer)
    if not match:
        return {}
    section_title = match.group(1).strip() or None
    page_range = match.group(2).strip() if match.group(2) else ""
    page_start = None
    page_end = None
    if page_range and page_range.upper() != "N/A":
        if "-" in page_range:
            start, end = page_range.split("-", 1)
            page_start = _intify(start.strip())
            page_end = _intify(end.strip())
        else:
            page_start = _intify(page_range)
            page_end = page_start
    result: Dict[str, Any] = {}
    if section_title:
        result["section_title"] = section_title
    if page_start is not None:
        result["page_start"] = page_start
    if page_end is not None:
        result["page_end"] = page_end
    return result


def _section_path_list(section_path: Optional[str]) -> Optional[list[str]]:
    if not section_path:
        return None
    parts = [part.strip() for part in section_path.split(">")]
    parts = [part for part in parts if part]
    return parts or None


def normalize_content(text: str) -> str:
    return _schema_normalizer.normalize_content(text)


def build_qdrant_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    return _schema_normalizer.build_qdrant_payload(raw)


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean_text = clean_text_for_embedding(payload.get("text") or "")
    raw_text = payload.get("text_raw") or payload.get("text") or ""
    chunk_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest() if clean_text else None

    section_title = _stringify(payload.get("section_title") or payload.get("section"))
    section_path = _stringify(payload.get("section_path") or section_title)
    section_id = _stringify(payload.get("section_id"))
    section_kind = _stringify(payload.get("section_kind"))

    page_start = _intify(payload.get("page_start") or payload.get("page"))
    page_end = _intify(payload.get("page_end") or payload.get("page"))

    evidence = _parse_evidence_pointer(_stringify(payload.get("evidence_pointer")))
    if evidence.get("section_title") and not section_title:
        section_title = evidence.get("section_title")
    if evidence.get("page_start") is not None:
        page_start = evidence.get("page_start")
    if evidence.get("page_end") is not None:
        page_end = evidence.get("page_end")

    chunk_index = _intify(payload.get("chunk_index"))
    chunk_count = _intify(payload.get("chunk_count"))

    chunk_type = _stringify(payload.get("chunk_type")) or "text"
    chunk_role = normalize_chunk_kind({"chunk_kind": payload.get("chunk_kind"), "chunk_type": chunk_type}, strict=False)

    canonical_text = get_canonical_text({"text": clean_text, "content": payload.get("content")})
    canonical: Dict[str, Any] = {
        "subscription_id": _stringify(payload.get("subscription_id") or payload.get("subscriptionId")),
        "profile_id": _stringify(payload.get("profile_id") or payload.get("profileId")),
        "document_id": _stringify(payload.get("document_id") or payload.get("documentId")),
        "profile_name": _stringify(payload.get("profile_name")),
        "text": clean_text,
        "canonical_text": canonical_text,
        "canonical_text_len": len(canonical_text) if canonical_text else 0,
        "canonical_token_count": token_count(canonical_text) if canonical_text else 0,
        "text_data": {
            "clean": clean_text,
            "raw": raw_text if raw_text and raw_text != clean_text else None,
        },
        "source": {
            "name": _stringify(payload.get("source_name") or payload.get("source_file") or payload.get("filename")),
            "uri": _stringify(payload.get("source_uri")),
        },
        "document": {
            "type": _stringify(payload.get("document_type") or payload.get("doc_type")),
            "domain": _stringify(payload.get("doc_domain")),
            "ingestion_source": _stringify(payload.get("ingestion_source") or payload.get("source_type")),
        },
        "section": {
            "id": section_id,
            "title": section_title,
            "path": _section_path_list(section_path),
            "kind": section_kind,
        },
        "chunk": {
            "id": _stringify(payload.get("chunk_id")),
            "index": chunk_index,
            "count": chunk_count,
            "type": chunk_type,
            "role": chunk_role,
            "hash": chunk_hash,
            "size": {"chars": _intify(payload.get("chunk_char_len") or len(clean_text or ""))},
            "links": {
                "prev": _stringify(payload.get("prev_chunk_id")),
                "next": _stringify(payload.get("next_chunk_id")),
            },
            "sentence_complete": bool(payload.get("chunk_sentence_complete", payload.get("sentence_complete", False))),
        },
        "provenance": {
            "page_start": page_start,
            "page_end": page_end,
            "section_title": section_title,
        },
    }

    flattened: Dict[str, Any] = {
        "subscription_id": canonical["subscription_id"],
        "profile_id": canonical["profile_id"],
        "document_id": canonical["document_id"],
        "profile_name": canonical["profile_name"],
        "text": canonical["text"],
        "canonical_text": canonical["canonical_text"],
        "canonical_text_len": canonical["canonical_text_len"],
        "canonical_token_count": canonical["canonical_token_count"],
        "text_clean": clean_text,
        "text_raw": raw_text if raw_text and raw_text != clean_text else None,
        "section_title": section_title,
        "section_path": section_path,
        "section_id": section_id,
        "section_kind": section_kind,
        "page_start": page_start,
        "page_end": page_end,
        "page": page_start,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_id": canonical["chunk"]["id"],
        "prev_chunk_id": canonical["chunk"]["links"]["prev"],
        "next_chunk_id": canonical["chunk"]["links"]["next"],
        "chunk_type": chunk_type,
        "chunk_kind": chunk_role,
        "chunking_mode": _stringify(payload.get("chunking_mode")),
        "chunk_sentence_complete": canonical["chunk"]["sentence_complete"],
        "document_type": canonical["document"]["type"],
        "doc_domain": canonical["document"].get("domain"),
        "ingestion_source": canonical["document"]["ingestion_source"],
    }

    canonical.update(flattened)
    cleaned = _drop_empty(canonical) or {}
    return _ensure_canonical_payload_fields(cleaned)


def normalize_chunk_metadata(
    chunk_metadata: Iterable[Dict[str, Any]],
    *,
    document_id: Optional[str] = None,
    default_doc_type: Optional[str] = None,
    default_chunking_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, meta in enumerate(chunk_metadata or []):
        m = dict(meta) if meta else {}
        section_title = _stringify(m.get("section_title") or m.get("section") or "Untitled Section") or "Untitled Section"
        section_path = _stringify(m.get("section_path") or section_title) or section_title
        page_start = _intify(m.get("page_start") or m.get("page_number") or m.get("page"))
        page_end = _intify(m.get("page_end")) or page_start

        chunk_type = _stringify(m.get("chunk_type")) or "text"
        chunk_kind = normalize_chunk_kind({"chunk_kind": m.get("chunk_kind"), "chunk_type": chunk_type}, strict=False)

        if document_id:
            m["document_id"] = str(document_id)
        if default_doc_type and not m.get("doc_type"):
            m["doc_type"] = default_doc_type
        if default_chunking_mode and not m.get("chunking_mode"):
            m["chunking_mode"] = default_chunking_mode

        m.update(
            {
                "section_title": section_title,
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "page_number": page_start,
                "chunk_index": int(m.get("chunk_index") or idx),
                "chunk_kind": chunk_kind,
            }
        )
        normalized.append(m)
    return normalized


__all__ = ["build_qdrant_payload", "normalize_content", "normalize_payload", "normalize_chunk_metadata"]
