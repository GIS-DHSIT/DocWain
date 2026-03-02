from __future__ import annotations

import re

FALLBACK_ANSWER = "Not enough information in the documents to answer that."

# Matches C0/C1 control characters EXCEPT \t (0x09), \n (0x0a), \r (0x0d).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Stray 'n' from escaped newlines: "HANAn Sales" → "HANA Sales"
_STRAY_N_RE = re.compile(r"(?<=[a-zA-Z0-9.,:;)>])\s*\bn\s+(?=[A-Z])")
_LEADING_N_RE = re.compile(r"(?m)^\s*n\b\s+")

# Metadata key fragments that leak through rendering
_METADATA_LEAK_RE = re.compile(
    r"\b(?:section_id|chunk_type|chunk_kind|section_kind|section_title"
    r"|page_start|page_end|start_page|end_page"
    r"|embed_pipeline_version|canonical_json|embedding_text|canonical_text"
    r"|doc_domain|doc_type|document_type"
    r"|subscription_id|profile_id|document_id"
    r"|layout_confidence|ocr_confidence)\s*[:=]?\s*\S+",
    re.IGNORECASE,
)

# Re-join hyphens broken by normalize_content: "self - employed" → "self-employed"
_BROKEN_HYPHEN_RE = re.compile(r"(\w) - (\w)")


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub("", str(text))
    # Re-join broken compound-word hyphens
    cleaned = _BROKEN_HYPHEN_RE.sub(r"\1-\2", cleaned)
    # Strip stray 'n' artifacts
    cleaned = _STRAY_N_RE.sub(" ", cleaned)
    cleaned = _LEADING_N_RE.sub("", cleaned)
    # Remove metadata key leaks
    cleaned = _METADATA_LEAK_RE.sub("", cleaned)
    # Normalize whitespace per line
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned or ""


def sanitize(text: str) -> str:
    return sanitize_text(text)
