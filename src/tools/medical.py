from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.api.config import Config
from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.safety import add_disclaimer, collect_warnings
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/medical", tags=["Tools-Medical"])


class MedicalSummaryRequest(BaseModel):
    text: str = Field(..., description="Medical note text")
    redact: bool = Field(default=False, description="Redact detected identifiers")


class MedicalImageRequest(BaseModel):
    image_reference: Optional[str] = Field(default=None, description="URL or identifier of the image")
    notes: Optional[str] = None


def _extract_entities(text: str) -> List[Dict[str, Any]]:
    meds = re.findall(r"\b([A-Za-z]+)\s+(\d+\s?mg)", text)
    entities = []
    for med, dosage in meds:
        entities.append({"type": "medication", "value": med, "dosage": dosage})
    return entities


def _summarize_medical(text: str, redact: bool) -> Dict[str, Any]:
    cleaned = sanitize_text(text, max_chars=3200)
    summary = cleaned[:400] if cleaned else "No content provided."
    entities = _extract_entities(cleaned)
    if redact:
        cleaned = re.sub(r"[A-Z][a-z]+\s[A-Z][a-z]+", "[REDACTED]", cleaned)
    summary = add_disclaimer(summary, domain="medical")
    return {"summary": summary, "entities": entities, "redacted_text": cleaned if redact else None}


def _ensure_domain_enabled() -> None:
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise ToolError(
            "Medical-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
            code="deprecated",
            status_code=410,
        )


@register_tool("medical")
async def medical_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    _ensure_domain_enabled()
    req = MedicalSummaryRequest(**(payload.get("input") or payload))
    result = _summarize_medical(req.text, req.redact)
    sources = [build_source_record("tool", correlation_id or "medical", title="medical")]
    return {
        "result": result,
        "sources": sources,
        "grounded": True,
        "context_found": True,
        "warnings": collect_warnings("medical"),
    }


@router.post("/summarize")
async def summarize(request: MedicalSummaryRequest, x_correlation_id: str | None = Header(None)):
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Medical-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
        )
    cid = generate_correlation_id(x_correlation_id)
    result = _summarize_medical(request.text, request.redact)
    sources = [build_source_record("tool", cid, title="medical")]
    return standard_response(
        "medical",
        grounded=True,
        context_found=True,
        result=result,
        sources=sources,
        warnings=collect_warnings("medical"),
        correlation_id=cid,
    )


@router.post("/image-analyze")
async def image_analyze(request: MedicalImageRequest, x_correlation_id: str | None = Header(None)):
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Medical-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
        )
    cid = generate_correlation_id(x_correlation_id)
    note = request.notes or "Medical image analysis is not available in this environment."
    result = {
        "summary": add_disclaimer(note, domain="medical"),
        "image_reference": request.image_reference,
    }
    sources = [build_source_record("tool", cid, title="medical_image")]
    return standard_response(
        "medical",
        grounded=True,
        context_found=False,
        result=result,
        sources=sources,
        warnings=collect_warnings("medical"),
        correlation_id=cid,
    )
