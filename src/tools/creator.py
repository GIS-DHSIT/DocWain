from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/creator", tags=["Tools-Creator"])


class CreatorRequest(BaseModel):
    content_type: str = Field(..., pattern="^(summary|blog|sop|faq|slide_outline)$")
    tone: str = Field(default="neutral")
    length: str = Field(default="medium", description="rough size hint: short|medium|long")
    context: Optional[Dict[str, Any]] = None
    text: Optional[str] = Field(default=None, description="Reference text to ground generation")


def _build_outline(content: str) -> List[str]:
    if not content:
        return []
    sentences = [s.strip() for s in content.split(".") if s.strip()]
    return sentences[:6]


def _generate_content(req: CreatorRequest) -> Dict[str, Any]:
    reference = sanitize_text(req.text or "Provided context", max_chars=2400)
    header = f"{req.content_type.replace('_', ' ').title()} in a {req.tone} tone ({req.length})"
    outline = _build_outline(reference)
    body = f"{header}: {reference}"
    faqs = []
    if req.content_type == "faq":
        faqs = [{"q": f"What about {idx + 1}?", "a": item} for idx, item in enumerate(outline[:5])]
    return {
        "header": header,
        "outline": outline,
        "content": body,
        "faqs": faqs,
    }


@register_tool("creator")
async def creator_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = CreatorRequest(**(payload.get("input") or payload))
    result = _generate_content(req)
    sources = [build_source_record("tool", correlation_id or "creator", title=req.content_type)]
    return {"result": result, "sources": sources, "grounded": True, "context_found": True}


@router.post("/generate")
async def generate(request: CreatorRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    result = _generate_content(request)
    sources = [build_source_record("tool", cid, title=request.content_type)]
    return standard_response(
        "creator",
        grounded=True,
        context_found=True,
        result=result,
        sources=sources,
        warnings=[],
        correlation_id=cid,
    )
