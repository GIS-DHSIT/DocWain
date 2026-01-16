from __future__ import annotations

import re
from typing import Dict, Optional

from fastapi import UploadFile

from src.tools.base import ToolError
from src.tools.common.http_client import fetch_text
from src.tools.common.io_limits import MAX_TEXT_BYTES, coerce_text, decode_base64, enforce_limit


def _strip_html(text: str) -> str:
    """Lightweight tag stripper to avoid pulling in heavy deps."""
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\\1>)", " ", text)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned.strip()


async def extract_text_from_url(url: str, *, max_bytes: int = MAX_TEXT_BYTES) -> Dict[str, str]:
    fetched = await fetch_text(url, max_bytes=max_bytes)
    text = _strip_html(fetched.get("content", ""))
    return {"text": text, "source_url": fetched.get("url", url)}


def extract_text_from_upload(upload: UploadFile, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if upload is None:
        raise ToolError("Upload is required", code="missing_upload")
    raw = upload.file.read(max_bytes + 1)
    enforce_limit(len(raw), max_bytes, "upload")
    return coerce_text(raw, fallback="")


def extract_text_from_base64(value: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    raw = decode_base64(value, max_bytes=max_bytes, label="text_payload")
    return coerce_text(raw, fallback="")


def sanitize_text(text: Optional[str], *, max_chars: int = 4000) -> str:
    if not text:
        return ""
    trimmed = text.strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars]
    return trimmed

