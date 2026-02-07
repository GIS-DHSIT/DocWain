from __future__ import annotations

FALLBACK_ANSWER = "Not enough information in the documents to answer that."


def sanitize_text(text: str) -> str:
    if not text:
        return FALLBACK_ANSWER
    cleaned = "\n".join(line.strip() for line in str(text).splitlines() if line.strip())
    return cleaned or FALLBACK_ANSWER


def sanitize(text: str) -> str:
    return sanitize_text(text)
