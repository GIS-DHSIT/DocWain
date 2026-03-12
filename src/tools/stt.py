from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Header, UploadFile
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.http_client import fetch_bytes
from src.tools.common.io_limits import ALLOWED_AUDIO_MIME, MAX_AUDIO_BYTES, decode_base64, enforce_limit, validate_upload

logger = get_logger(__name__)

router = APIRouter(prefix="/stt", tags=["Tools-STT"])

_WHISPER_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import whisper  # type: ignore

    _WHISPER_AVAILABLE = True
except Exception:  # noqa: BLE001
    whisper = None  # type: ignore

class TranscriptionRequest(BaseModel):
    audio_url: Optional[str] = Field(default=None, description="Public URL to audio file")
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio bytes")
    language: Optional[str] = Field(default=None, description="Optional language hint")

async def _resolve_audio_bytes(
    *,
    upload: UploadFile | None,
    req: TranscriptionRequest,
) -> bytes:
    if upload:
        return validate_upload(upload, allowed_mime=ALLOWED_AUDIO_MIME, max_bytes=MAX_AUDIO_BYTES, label="audio_file")
    if req.audio_base64:
        return decode_base64(req.audio_base64, max_bytes=MAX_AUDIO_BYTES, label="audio_base64")
    if req.audio_url:
        data = await fetch_bytes(req.audio_url, max_bytes=MAX_AUDIO_BYTES)
        enforce_limit(len(data), MAX_AUDIO_BYTES, "audio_url")
        return data
    raise ToolError("audio_file, audio_url, or audio_base64 is required", code="missing_audio")

def _transcribe_locally(temp_path: str, language: Optional[str]) -> Dict[str, Any]:
    if not _WHISPER_AVAILABLE:
        raise ToolError("Whisper backend is not installed in this deployment", code="backend_unavailable", status_code=501)
    try:
        model_size = os.getenv("TOOLS_WHISPER_MODEL", "base")
        model = whisper.load_model(model_size)  # type: ignore[attr-defined]
        result = model.transcribe(temp_path, language=language)  # type: ignore[attr-defined]
        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "language": result.get("language") or language,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Whisper transcription failed: %s", exc, exc_info=True)
        raise ToolError("Transcription failed", code="transcription_failed", status_code=500) from exc

@register_tool("stt")
async def stt_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Handler used by /api/tools/run. Expects payload keys: audio_base64 or audio_url.
    """
    req = TranscriptionRequest(**(payload.get("input") or {}))
    dummy_upload: UploadFile | None = None
    audio_bytes = await _resolve_audio_bytes(upload=dummy_upload, req=req)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        transcription = _transcribe_locally(tmp_path, req.language)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    sources = [{"type": "tool", "id": correlation_id or "", "title": "stt_audio", "metadata": {"bytes": len(audio_bytes)}}]
    return {
        "result": transcription,
        "sources": sources,
        "context_found": True,
        "grounded": True,
    }

@router.post("/transcribe")
async def transcribe(
    request: TranscriptionRequest,
    audio_file: UploadFile | None = File(None),
    x_correlation_id: str | None = Header(None),
):
    cid = generate_correlation_id(x_correlation_id)
    try:
        audio_bytes = await _resolve_audio_bytes(upload=audio_file, req=request)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            transcription = _transcribe_locally(tmp_path, request.language)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        sources = [
            {"type": "tool", "id": cid, "title": "stt_audio", "metadata": {"bytes": len(audio_bytes)}},
        ]
        return standard_response(
            "stt",
            grounded=True,
            context_found=bool(transcription.get("text")),
            result=transcription,
            sources=sources,
            correlation_id=cid,
            warnings=[],
        )
    except ToolError as exc:
        return standard_response(
            "stt",
            status="error",
            grounded=False,
            context_found=False,
            result={},
            sources=[],
            correlation_id=cid,
            warnings=[],
            error=exc.as_dict(),
        )

