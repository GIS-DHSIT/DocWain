from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.mode.execution_mode import ExecutionMode


@dataclass
class ExecutionResult:
    answer: Dict[str, Any]
    mode: ExecutionMode
    debug: Dict[str, Any]
    stream: Optional[Iterable[str]] = None


def _dedup_ocr_chars(text: str) -> str:
    """Fix OCR doubled-character artifacts like 'DDeessccrriippttiioonn' → 'Description'."""
    import re
    def _fix_word(m: "re.Match") -> str:
        word = m.group(0)
        # Only de-duplicate if most chars in the word are doubled pairs
        pairs = 0
        i = 0
        while i < len(word) - 1:
            if word[i] == word[i + 1]:
                pairs += 1
                i += 2
            else:
                i += 1
        # At least 3 doubled pairs and > 60% of chars are in pairs
        if pairs >= 3 and (pairs * 2) / max(len(word), 1) > 0.6:
            result = []
            i = 0
            while i < len(word):
                result.append(word[i])
                if i + 1 < len(word) and word[i] == word[i + 1]:
                    i += 2
                else:
                    i += 1
            return "".join(result)
        return word
    return re.sub(r"\b[A-Za-z]{6,}\b", _fix_word, text)


def _sanitize_response_text(text: str) -> str:
    """Strip internal placeholders and AI disclaimers from the final response."""
    import re
    if not text:
        return text
    # Fix OCR doubled-character artifacts (e.g., "AAmmoouunntt" → "Amount")
    text = _dedup_ocr_chars(text)
    # Strip extraction placeholders
    text = re.sub(r"Not explicitly mentioned(?:\s+in\s+documents)?\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"MISSING_REASON", "", text)
    # Strip AI disclaimers (but not job titles like "AI Engineer")
    text = re.sub(r"\bAs an AI(?:\s+(?:language\s+model|assistant|model|chatbot|system))\b[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAs a language model\b[^.]*\.", "", text, flags=re.IGNORECASE)
    # Strip internal references
    text = re.sub(r"\[SOURCE[^\]]*\]", "", text, flags=re.IGNORECASE)
    # Strip leaked internal metadata tags (e.g. [META]{"grounded": true, ...})
    text = re.sub(r"\n*\[META\]\{[^}]*\}", "", text)
    return text


def normalize_answer(answer: Any) -> Dict[str, Any]:
    """Ensure downstream consumers get a consistent answer structure."""
    if isinstance(answer, dict):
        meta = {}
        raw_meta = answer.get("metadata")
        if isinstance(raw_meta, dict):
            meta.update(raw_meta)
        response_text = answer.get("response") or answer.get("answer") or ""
        if isinstance(response_text, str):
            response_text = _sanitize_response_text(response_text)
        # Guard: never return an empty response to the user.
        if isinstance(response_text, str) and not response_text.strip():
            response_text = "I couldn't find relevant information in the uploaded documents to answer that. Try rephrasing your question or check that the right documents are uploaded."
        # Extract media from answer or from metadata.media
        media = answer.get("media")
        if media is None and isinstance(meta.get("media"), list):
            media = meta.pop("media")

        structured = {
            "response": response_text,
            "sources": answer.get("sources", []),
            "grounded": answer.get("grounded", False),
            "context_found": answer.get("context_found", False),
            "metadata": meta,
        }
        if media:
            structured["media"] = media
        for k, v in answer.items():
            if k in {"response", "answer", "sources", "metadata", "media"}:
                continue
            structured["metadata"][k] = v
        structured["metadata"] = structured.get("metadata") or {}
        return structured
    return {
        "response": str(answer),
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": {},
    }


def chunk_text_stream(text: str, chunk_size: int = 256) -> Iterable[str]:
    if not text:
        yield ""
        return
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]


def chunk_text_stream_with_metadata(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 256,
) -> Iterable[str]:
    """Stream text chunks followed by a final JSON metadata event.

    The final chunk is a JSON line prefixed with '\\n\\n[META]' containing
    grounding signals: grounded, confidence, evidence_coverage, sources.
    Clients can detect this sentinel and parse the metadata.
    """
    import json as _json

    if not text:
        yield ""
    else:
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            if end < len(text):
                # Find last good break point within chunk
                chunk = text[pos:end]
                # Prefer breaking at: newline > sentence end > table row boundary > space
                last_nl = chunk.rfind('\n')
                last_sent = max(chunk.rfind('. '), chunk.rfind('.\n'),
                                chunk.rfind('? '), chunk.rfind('! '))
                last_pipe_nl = chunk.rfind('|\n')  # Table row boundary

                best_break = max(last_nl, last_sent, last_pipe_nl)
                if best_break > chunk_size * 0.4:  # At least 40% of target size
                    end = pos + best_break + 1

            chunk_text = text[pos:end]
            if chunk_text:  # Skip empty chunks
                yield chunk_text
            pos = end

    # Internal metadata is returned via the response structure (headers / JSON),
    # NOT appended to the streamed text.  Emitting [META] tags in the text stream
    # leaks internal data to end users.
