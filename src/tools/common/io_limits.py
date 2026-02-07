from __future__ import annotations

import base64
from typing import Iterable

from fastapi import UploadFile

from src.tools.base import ToolError

MAX_AUDIO_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_TEXT_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_BINARY_BYTES = 12 * 1024 * 1024  # 12 MB

ALLOWED_AUDIO_MIME = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/webm",
}

ALLOWED_ARCHIVES = {"application/zip", "application/x-zip-compressed"}


def enforce_limit(length: int, max_bytes: int, label: str) -> None:
    if length > max_bytes:
        raise ToolError(f"{label} exceeds maximum size of {max_bytes // (1024 * 1024)}MB", code="payload_too_large")


def validate_upload(
    upload: UploadFile,
    *,
    allowed_mime: Iterable[str],
    max_bytes: int,
    label: str,
) -> bytes:
    if upload is None:
        raise ToolError(f"{label} is required", code="missing_upload")
    if upload.content_type not in allowed_mime:
        raise ToolError(f"Unsupported content type: {upload.content_type}", code="unsupported_media_type")

    data = upload.file.read(max_bytes + 1)
    enforce_limit(len(data), max_bytes, label)
    return data


def decode_base64(content: str, *, max_bytes: int, label: str) -> bytes:
    try:
        raw = base64.b64decode(content, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Invalid base64 content for {label}: {exc}", code="invalid_base64") from exc
    enforce_limit(len(raw), max_bytes, label)
    return raw


def coerce_text(data: bytes, *, fallback: str = "") -> str:
    try:
        return data.decode("utf-8")
    except Exception:
        try:
            return data.decode("latin-1")
        except Exception:
            return fallback
