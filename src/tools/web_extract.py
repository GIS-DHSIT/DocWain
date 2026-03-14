from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.http_client import fetch_text
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/web", tags=["Tools-Web"])

class WebExtractRequest(BaseModel):
    url: str = Field(..., description="URL to fetch")
    max_chars: int = Field(default=4000, ge=100, le=12000)

class WebAnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    max_chars: int = Field(default=4000, ge=100, le=12000)

async def _extract(url: str, max_chars: int) -> Dict[str, Any]:
    fetched = await fetch_text(url, max_bytes=max_chars * 4)
    text = sanitize_text(fetched.get("content", ""), max_chars=max_chars)
    return {"text": text, "url": fetched.get("url", url), "bytes": fetched.get("bytes")}

def _summarize_text(text: str, max_chars: int) -> Dict[str, Any]:
    cleaned = sanitize_text(text, max_chars=max_chars)
    summary = cleaned[:500] if cleaned else ""
    return {"summary": summary, "length": len(cleaned)}

@register_tool("web_extract")
async def web_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    input_payload = payload.get("input") or payload
    url = input_payload.get("url")
    if not url:
        return {
            "result": {"error": "url is required"},
            "sources": [],
            "grounded": False,
            "context_found": False,
        }
    extracted = await _extract(url, input_payload.get("max_chars", 4000))
    sources = [build_source_record("url", extracted.get("url", url), title=extracted.get("url", url))]
    return {"result": extracted, "sources": sources, "grounded": True, "context_found": True}

@router.post("/extract")
async def extract(request: WebExtractRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    extracted = await _extract(request.url, request.max_chars)
    sources = [build_source_record("url", extracted.get("url", request.url), title=request.url)]
    return standard_response(
        "web_extract",
        grounded=True,
        context_found=True,
        result=extracted,
        sources=sources,
        correlation_id=cid,
    )

@router.post("/analyze")
async def analyze(request: WebAnalyzeRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    text = request.text or ""
    sources = []
    if request.url:
        extracted = await _extract(request.url, request.max_chars)
        text = extracted.get("text", "")
        sources.append(build_source_record("url", extracted.get("url", request.url), title=request.url))
    summary = _summarize_text(text, request.max_chars)
    return standard_response(
        "web_extract",
        grounded=True,
        context_found=bool(text),
        result=summary,
        sources=sources,
        correlation_id=cid,
    )

