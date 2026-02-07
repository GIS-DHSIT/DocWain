from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Iterable, List, Optional

from src.utils.payload_utils import get_canonical_text, token_count


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


def _listify(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(v) for v in value if v is not None and str(v).strip()]
        return cleaned or None
    return [str(value)] if str(value).strip() else None


def normalize_chunk_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    text = raw.get("text") or raw.get("content") or ""
    canonical_text = get_canonical_text({"text": text, "content": raw.get("content")})
    chunk_hash = hashlib.sha256((canonical_text or "").encode("utf-8")).hexdigest() if canonical_text else None

    payload: Dict[str, Any] = {
        "subscription_id": _stringify(raw.get("subscription_id")),
        "profile_id": _stringify(raw.get("profile_id")),
        "document_id": _stringify(raw.get("document_id")),
        "filename": _stringify(raw.get("filename") or raw.get("source_name")),
        "source_type": _stringify(raw.get("source_type")),
        "domain": _stringify(raw.get("domain")) or "generic",
        "chunk_kind": _stringify(raw.get("chunk_kind")) or "section_chunk",
        "section_path": _listify(raw.get("section_path")) or [],
        "page_range": raw.get("page_range") or [raw.get("page_start"), raw.get("page_end")],
        "anchors": _listify(raw.get("anchors")) or [],
        "entity_types_present": _listify(raw.get("entity_types_present")) or [],
        "checksum_sha256": _stringify(raw.get("checksum_sha256")),
        "source_version": _stringify(raw.get("source_version")),
        "ingest_timestamp": raw.get("ingest_timestamp") or time.time(),
        "chunk_id": _stringify(raw.get("chunk_id")),
        "chunk_hash": chunk_hash,
        "text": canonical_text,
        "canonical_text": canonical_text,
        "canonical_text_len": len(canonical_text) if canonical_text else 0,
        "canonical_token_count": token_count(canonical_text) if canonical_text else 0,
        "text_data": {"clean": canonical_text, "raw": text if text and text != canonical_text else None},
    }

    # Retain legacy keys for backward compatibility
    payload["source_name"] = payload.get("filename")
    payload["doc_domain"] = payload.get("domain")
    payload["section_path"] = payload.get("section_path")
    if payload.get("page_range"):
        payload["page_start"] = _intify(payload["page_range"][0]) if isinstance(payload["page_range"], list) else None
        payload["page_end"] = _intify(payload["page_range"][1]) if isinstance(payload["page_range"], list) else None
        payload["page"] = payload["page_start"]

    return payload


__all__ = ["normalize_chunk_payload"]
