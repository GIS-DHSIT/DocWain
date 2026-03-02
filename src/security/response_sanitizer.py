"""User-facing response sanitization helpers.

Logging may retain internal identifiers, but API responses and chat answers
should not leak document IDs, point IDs, blob names, or similar references.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable

_REDACTED = "[REDACTED]"

# Identifier-like patterns to mask.
_ID_PATTERNS: Iterable[re.Pattern[str]] = (
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b[0-9a-fA-F]{24}\b"),
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),
    re.compile(r"ObjectId\([^\)]*\)", re.IGNORECASE),
)

# Lines containing these markers are removed entirely.
_INTERNAL_LINE_RE = re.compile(
    r"\b(internal|document_id|chunk_id|point_id|system id|reference id|request id|blob|container|\.pkl|\.pickle)\b",
    re.IGNORECASE,
)

# Keys that should not be exposed in user responses.
_INTERNAL_KEYS = {
    "_id",
    "document_id",
    "documentId",
    "doc_id",
    "doc_ids",
    "document_ids",
    "chunk_id",
    "chunk_ids",
    "section_id",
    "point_id",
    "profile_id",
    "subscription_id",
    "request_id",
    "collection",
    "blob_name",
    "source_doc_ids",
    "system_id",
    "reference_id",
    "container",
    "score",
    "vector_score",
    "lexical_score",
    "relevance_score",
    "hit_count",
    "hits",
    "num_sources",
    "retrieval_stats",
    "source_id",
    "target_document_id",
    "target_document_ids",
}


def sanitize_user_text(text: str) -> str:
    """Mask identifier-like substrings and drop internal reference lines."""
    if not text:
        return ""

    safe_lines = []
    for line in str(text).splitlines():
        if _INTERNAL_LINE_RE.search(line):
            continue
        masked = line
        for pattern in _ID_PATTERNS:
            masked = pattern.sub(_REDACTED, masked)
        # Mask explicit id labels even when values are not matched above.
        masked = re.sub(r"\b(document_id|chunk_id|point_id|system id|reference id)\b", "id", masked, flags=re.IGNORECASE)
        safe_lines.append(masked)

    return "\n".join(safe_lines).strip()


def sanitize_user_payload(payload: Any) -> Any:
    """Recursively sanitize payloads destined for user-facing responses."""
    if isinstance(payload, str):
        return sanitize_user_text(payload)
    if isinstance(payload, list):
        return [sanitize_user_payload(item) for item in payload]
    if isinstance(payload, dict):
        clean: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in _INTERNAL_KEYS:
                continue
            clean[key] = sanitize_user_payload(value)
        return clean
    return payload
