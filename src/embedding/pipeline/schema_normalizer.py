from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text
from src.metadata.normalizer import normalize_chunk_kind
from src.utils.payload_utils import token_count


EMBED_PIPELINE_VERSION = "dwx-2026-02-05"


_DOC_CONNECTOR_MAP = {
    "local": "LOCAL",
    "filesystem": "LOCAL",
    "file": "LOCAL",
    "s3": "S3",
    "aws_s3": "S3",
    "ftp": "FTP",
    "sftp": "FTP",
    "azure_blob": "AZURE_BLOB",
    "azure": "AZURE_BLOB",
    "blob": "AZURE_BLOB",
}


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


def _floatify(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _section_path_list(section_path: Optional[Any]) -> Optional[list[str]]:
    if not section_path:
        return None
    if isinstance(section_path, list):
        parts = [str(part).strip() for part in section_path if str(part).strip()]
        return parts or None
    text = str(section_path)
    parts = [part.strip() for part in text.split(">") if part.strip()]
    return parts or None


def _infer_file_type(source_name: Optional[str]) -> Optional[str]:
    if not source_name:
        return None
    suffix = Path(source_name).suffix.lower().lstrip(".")
    return suffix or None


def _normalize_connector_type(value: Optional[str]) -> str:
    if not value:
        return "LOCAL"
    key = str(value).strip().lower()
    return _DOC_CONNECTOR_MAP.get(key, key.upper())


def _doc_version_hash(raw: Dict[str, Any], *, canonical_text: str) -> str:
    provided = _stringify(raw.get("doc_version_hash") or raw.get("docVersionHash"))
    if provided:
        return provided
    seed = _stringify(raw.get("doc_version_seed")) or _stringify(raw.get("full_text")) or canonical_text
    digest = hashlib.sha1((seed or "").encode("utf-8")).hexdigest()
    return digest[:12]


def normalize_content(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")

    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"([A-Za-z])(\d)", r"\1 \2", normalized)
    normalized = re.sub(r"(\d)([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"([A-Za-z])\(", r"\1 (", normalized)
    normalized = re.sub(r"\s*&\s*", " & ", normalized)
    normalized = re.sub(r"\s*—\s*", " — ", normalized)
    normalized = re.sub(r"(?<=\w)-(?=\w)", " - ", normalized)

    normalized = re.sub(r"(?<!^)(?<!\n)\s*([•●])", r"\n\1", normalized)
    lines = []
    for line in normalized.split("\n"):
        compacted = re.sub(r"[ \t]+", " ", line).strip()
        if compacted:
            lines.append(compacted)
    return "\n".join(lines).strip()


def build_qdrant_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    subscription_id = _stringify(raw.get("subscription_id") or raw.get("subscriptionId") or raw.get("subscription"))
    profile_id = _stringify(raw.get("profile_id") or raw.get("profileId") or raw.get("profile"))
    document_id = _stringify(raw.get("document_id") or raw.get("documentId") or raw.get("doc_id") or raw.get("docId"))
    if not subscription_id or not profile_id or not document_id:
        missing = [name for name, value in (("subscription_id", subscription_id), ("profile_id", profile_id), ("document_id", document_id)) if not value]
        raise ValueError(f"Missing required payload fields: {', '.join(missing)}")

    source_name = _stringify(raw.get("source_name") or raw.get("sourceName"))
    if not source_name:
        source_name = _stringify((raw.get("source") or {}).get("name")) or "unknown"

    connector_type = _normalize_connector_type(
        _stringify(raw.get("connector_type") or raw.get("ingestion_source") or (raw.get("document") or {}).get("ingestion_source"))
    )
    file_type = _stringify(raw.get("file_type")) or _infer_file_type(source_name) or "unknown"
    mime_type = _stringify(raw.get("mime_type") or raw.get("mimeType"))

    doc_domain = _stringify(raw.get("doc_domain") or (raw.get("document") or {}).get("domain")) or "unknown"

    section_id = _stringify(raw.get("section_id") or (raw.get("section") or {}).get("id")) or "unknown"
    section_title = _stringify(raw.get("section_title") or (raw.get("section") or {}).get("title") or "Section")
    section_path = _section_path_list(raw.get("section_path") or (raw.get("section") or {}).get("path") or section_title)
    section_kind = _stringify(raw.get("section_kind") or (raw.get("section") or {}).get("kind")) or "misc"
    section_confidence = _floatify(raw.get("section_confidence") or (raw.get("section") or {}).get("confidence"), default=0.5)
    section_salience = _floatify(raw.get("section_salience") or (raw.get("section") or {}).get("salience"), default=0.5)

    page_start = _intify(raw.get("page_start") or raw.get("page") or raw.get("page_number")) or 0
    page_end = _intify(raw.get("page_end")) or page_start
    chunk_id = _stringify(raw.get("chunk_id") or (raw.get("chunk") or {}).get("id")) or "unknown"
    chunk_index = _intify(raw.get("chunk_index") or (raw.get("chunk") or {}).get("index") or 0) or 0
    chunk_count = _intify(raw.get("chunk_count") or (raw.get("chunk") or {}).get("count")) or 1
    chunk_kind = normalize_chunk_kind(
        {
            "chunk_kind": raw.get("chunk_kind") or (raw.get("chunk") or {}).get("type"),
            "chunk_type": raw.get("chunk_type") or (raw.get("chunk") or {}).get("type"),
        },
        strict=False,
    )
    chunking_mode = _stringify(raw.get("chunking_mode"))

    raw_content = raw.get("content")
    if raw_content is None:
        raw_content = raw.get("text") or raw.get("text_raw") or raw.get("text_clean") or (raw.get("text_data") or {}).get("clean")
    content = normalize_content(raw_content or "")

    canonical_text = _stringify(raw.get("canonical_text")) or content
    canonical_text = normalize_content(canonical_text or content)

    embedding_text = _stringify(raw.get("embedding_text") or raw.get("text_clean") or raw.get("embeddingText"))
    if not embedding_text:
        embedding_text = ensure_embedding_text(content, doc_domain, section_kind)
    if embedding_text.strip() == content.strip():
        embedding_text = ensure_embedding_text(content, doc_domain, section_kind)

    canonical_text_len = len(canonical_text or "")
    canonical_token_count = token_count(canonical_text or "")

    chunk_hash = _stringify(raw.get("hash") or (raw.get("chunk") or {}).get("hash"))
    if not chunk_hash:
        chunk_hash = hashlib.sha256((embedding_text or content).encode("utf-8")).hexdigest()

    detected_language = _stringify(raw.get("detected_language") or raw.get("language"))
    language_confidence = raw.get("language_confidence")
    try:
        if language_confidence is not None:
            language_confidence = _floatify(language_confidence, default=0.0)
    except Exception:
        language_confidence = None
    languages = raw.get("languages")
    if isinstance(languages, str):
        languages = [languages]

    payload: Dict[str, Any] = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": document_id,
        "source_name": source_name,
        "connector_type": connector_type,
        "file_type": file_type,
        "mime_type": mime_type,
        "doc_domain": doc_domain,
        "section_id": section_id,
        "section_title": section_title,
        "section_path": section_path,
        "section_kind": section_kind,
        "section_confidence": max(0.0, min(section_confidence, 1.0)),
        "section_salience": max(0.0, min(section_salience, 1.0)),
        "page": page_start,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_kind": chunk_kind,
        "chunking_mode": chunking_mode,
        "hash": chunk_hash,
        "content": content,
        "canonical_text": canonical_text,
        "embedding_text": embedding_text,
        "canonical_text_len": canonical_text_len,
        "canonical_token_count": canonical_token_count,
        "doc_version_hash": _doc_version_hash(raw, canonical_text=canonical_text),
        "embed_pipeline_version": _stringify(raw.get("embed_pipeline_version")) or EMBED_PIPELINE_VERSION,
        "detected_language": detected_language,
        "language_confidence": language_confidence,
        "languages": languages,
    }

    anchors = raw.get("anchors")
    if anchors:
        payload["anchors"] = anchors

    payload["source"] = {"name": source_name, "uri": _stringify(raw.get("source_uri"))}
    payload["document"] = {
        "type": _stringify(raw.get("document_type") or raw.get("doc_type")),
        "domain": doc_domain,
        "ingestion_source": connector_type,
    }
    payload["section"] = {
        "id": section_id,
        "title": section_title,
        "path": section_path,
        "kind": section_kind,
    }
    payload["chunk"] = {
        "id": chunk_id,
        "index": chunk_index,
        "count": chunk_count,
        "type": chunk_kind,
    }
    payload["provenance"] = {
        "page_start": page_start,
        "page_end": page_end,
        "section_title": section_title,
    }

    return {k: v for k, v in payload.items() if v is not None}


__all__ = ["EMBED_PIPELINE_VERSION", "build_qdrant_payload", "normalize_content"]
