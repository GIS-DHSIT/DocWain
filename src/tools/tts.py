from __future__ import annotations

import io
from src.utils.logging_utils import get_logger
import math
import os
import tempfile
import wave
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response

logger = get_logger(__name__)

router = APIRouter(prefix="/tts", tags=["Tools-TTS"])

_PYTTSX3_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import pyttsx3  # type: ignore

    _PYTTSX3_AVAILABLE = True
except Exception:  # noqa: BLE001
    pyttsx3 = None  # type: ignore

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice id if supported by backend")
    format: str = Field(default="wav", pattern="^(mp3|wav)$")
    rate: Optional[int] = Field(default=None, ge=80, le=240)

def _generate_tone(text: str, *, sample_rate: int = 16000) -> bytes:
    """
    Lightweight deterministic waveform generator used as a safety fallback when
    no TTS backend is available. Encodes characters into a simple sine pattern.
    """
    duration_sec = min(4.0, 0.04 * len(text))
    frames = int(sample_rate * duration_sec)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(frames):
            # Encode text variability into the tone so output is deterministic per text
            freq = 440 + (hash(text) % 220)
            value = int(32767 * 0.2 * math.sin(2 * math.pi * freq * (i / sample_rate)))
            wav_file.writeframes(value.to_bytes(2, "little", signed=True))
    return buffer.getvalue()

def _speak_with_pyttsx3(request: SpeakRequest) -> Tuple[bytes, list[str]]:
    warnings: list[str] = []
    if not _PYTTSX3_AVAILABLE:
        raise ToolError("pyttsx3 is not installed", code="backend_unavailable", status_code=501)

    engine = pyttsx3.init()  # type: ignore[attr-defined]
    if request.voice:
        try:
            engine.setProperty("voice", request.voice)
        except Exception:
            warnings.append("Requested voice not available; using default voice.")
    if request.rate:
        try:
            engine.setProperty("rate", int(request.rate))
        except Exception:
            warnings.append("Requested rate not applied; using default rate.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    try:
        engine.save_to_file(request.text, tmp_path)  # type: ignore[attr-defined]
        engine.runAndWait()  # type: ignore[attr-defined]
        with open(tmp_path, "rb") as handle:
            data = handle.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return data, warnings

def _synthesize(request: SpeakRequest) -> Tuple[bytes, str, list[str]]:
    fmt = request.format or "wav"
    if _PYTTSX3_AVAILABLE and fmt == "wav":
        audio, warnings = _speak_with_pyttsx3(request)
        return audio, "pyttsx3", warnings

    warnings = ["Using fallback tone synthesis; install pyttsx3 for natural speech."]
    audio = _generate_tone(request.text)
    if fmt == "mp3":
        warnings.append("MP3 requested but fallback uses WAV encoding.")
    return audio, "fallback", warnings

@register_tool("tts")
async def tts_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = SpeakRequest(**(payload.get("input") or {}))
    audio_bytes, backend, warnings = _synthesize(req)
    return {
        "result": {
            "bytes": len(audio_bytes),
            "format": "wav" if req.format == "mp3" and backend == "fallback" else req.format,
            "backend": backend,
        },
        "sources": [
            {"type": "tool", "id": correlation_id or "", "title": "tts_audio", "metadata": {"backend": backend}}
        ],
        "grounded": True,
        "context_found": True,
        "warnings": warnings,
        "audio_bytes": audio_bytes,
    }

@router.post("/speak")
async def speak(request: SpeakRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    try:
        audio_bytes, backend, warnings = _synthesize(request)
        media_type = "audio/wav" if request.format == "wav" or backend == "fallback" else "audio/mpeg"
        headers = {
            "X-Correlation-ID": cid,
            "X-TTS-Backend": backend,
        }
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type, headers=headers)
    except ToolError as exc:
        return standard_response(
            "tts",
            status="error",
            grounded=False,
            context_found=False,
            sources=[],
            result={},
            warnings=[],
            error=exc.as_dict(),
            correlation_id=cid,
        )

