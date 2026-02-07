from __future__ import annotations

import re


FALLBACK_ANSWER = "Not enough information in the documents to answer that."

_BANNED_LABELS = [
    "Understanding & Scope",
    "Answer:",
    "Evidence & Gaps",
    "Files used",
    "Mixed:",
    "Invoice Summary",
]
_BANNED_TOKENS = {label.lower() for label in _BANNED_LABELS}


def sanitize_text(text: str) -> str:
    if not text:
        return FALLBACK_ANSWER
    cleaned = str(text)
    cleaned = cleaned.replace("B)||||", " ")
    cleaned = re.sub(r"\|\|\|\|+", " ", cleaned)
    cleaned = re.sub(r"<\|.*?\|>", " ", cleaned)
    cleaned = re.sub(r"\[\[.*?\]\]", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"(?i)\bcontext\s*=\s*\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(
        r"(?i)\b(document_id|doc_id|docid|section_id|chunk_id|chunk_hash|subscription_id|profile_id)\s*[:=]\s*[A-Za-z0-9_-]+\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\bchunk[_-]?[0-9a-f]{6,}\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\btool(?:_call|_trace)?\s*[:=]\s*\S+", " ", cleaned)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    kept: list[str] = []
    source_line = None
    for line in lines:
        lowered = line.lower()
        if "context=[" in lowered or "tool_trace" in lowered or "tool_call" in lowered:
            continue
        if any(lowered.startswith(token.lower()) for token in _BANNED_LABELS):
            continue
        if any(token in lowered for token in _BANNED_TOKENS):
            continue
        if re.match(r"^source\s*:", line, flags=re.IGNORECASE):
            if source_line is None:
                source_line = line
            continue
        kept.append(line)

    body = "\n".join(kept)
    body = re.sub(r"[ \t]+", " ", body).strip()
    if not body:
        body = FALLBACK_ANSWER
    if source_line:
        return f"{body}\n{source_line}"
    return body


def sanitize(text: str) -> str:
    return sanitize_text(text)
