from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translator", tags=["Tools-Translator"])

_ARGOS_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from argostranslate import translate as _argos_translate  # type: ignore

    _ARGOS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _argos_translate = None  # type: ignore


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_lang: str = Field(..., description="Target language code")
    source_lang: Optional[str] = Field(default=None, description="Optional source language code")


def _argos_translate_text(text: str, source: Optional[str], target: str) -> str:
    if not _ARGOS_AVAILABLE:
        raise ToolError("Argos Translate is not installed", code="backend_unavailable", status_code=501)
    try:
        available = _argos_translate.get_installed_languages()  # type: ignore[attr-defined]
        source_lang = next((lang for lang in available if source and lang.code == source), available[0])
        target_lang = next((lang for lang in available if lang.code == target), None)
        if not target_lang:
            raise ToolError(f"Target language '{target}' is not installed", code="unsupported_language")
        translation = source_lang.get_translation(target_lang)  # type: ignore[call-arg]
        return translation.translate(text)  # type: ignore[attr-defined]
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Argos translation failed: %s", exc)
        raise ToolError("Translation failed", code="translation_failed") from exc


def _fallback_translate(text: str, target: str) -> str:
    """Deterministic offline fallback to keep endpoint responsive."""
    return f"[{target}] {text}"


def _translate_text(request: TranslateRequest) -> Dict[str, Any]:
    translated = _fallback_translate(request.text, request.target_lang)
    warnings: list[str] = []
    try:
        translated = _argos_translate_text(request.text, request.source_lang, request.target_lang)
    except ToolError as exc:
        warnings.append(str(exc))
    return {
        "translated_text": translated,
        "target_lang": request.target_lang,
        "detected_lang": request.source_lang or "unknown",
        "warnings": warnings,
    }


@register_tool("translator")
async def translator_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = TranslateRequest(**(payload.get("input") or payload))
    result = _translate_text(req)
    sources = [build_source_record("tool", correlation_id or "translator", title="translator")]
    return {
        "result": result,
        "sources": sources,
        "context_found": True,
        "grounded": True,
        "warnings": result.get("warnings", []),
    }


@router.post("/translate")
async def translate(request: TranslateRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    try:
        result = _translate_text(request)
        sources = [build_source_record("tool", cid, title="translator")]
        return standard_response(
            "translator",
            grounded=True,
            context_found=True,
            result=result,
            sources=sources,
            warnings=result.get("warnings", []),
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "translator",
            status="error",
            grounded=False,
            context_found=False,
            result={},
            sources=[],
            warnings=[],
            error=exc.as_dict(),
            correlation_id=cid,
        )

