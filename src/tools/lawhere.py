from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.safety import LEGAL_DISCLAIMER, add_disclaimer
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lawhere", tags=["Tools-LawHere"])


class LawHereRequest(BaseModel):
    text: str = Field(..., description="Legal text to analyze")
    profile_type: Optional[str] = Field(default=None, description="Legal profile or domain")


def _extract_clauses(text: str) -> List[str]:
    matches = re.findall(r"(shall|must|will)\s+[^\.]+", text, flags=re.IGNORECASE)
    return [m.strip() for m in matches[:6]]


def _analyze(request: LawHereRequest) -> Dict[str, Any]:
    cleaned = sanitize_text(request.text, max_chars=3200)
    clauses = _extract_clauses(cleaned)
    risks = [clause for clause in clauses if "liability" in clause.lower() or "indemn" in clause.lower()]
    obligations = clauses
    summary = cleaned[:400] if cleaned else "No content provided."
    summary = add_disclaimer(summary, domain="legal")
    return {
        "summary": summary,
        "key_clauses": clauses,
        "risks": risks,
        "obligations": obligations,
        "profile_type": request.profile_type,
    }


@register_tool("lawhere")
async def lawhere_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = LawHereRequest(**(payload.get("input") or payload))
    result = _analyze(req)
    sources = [build_source_record("tool", correlation_id or "lawhere", title="lawhere")]
    return {"result": result, "sources": sources, "grounded": True, "context_found": True, "warnings": [LEGAL_DISCLAIMER]}


@router.post("/analyze")
async def analyze(request: LawHereRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    result = _analyze(request)
    sources = [build_source_record("tool", cid, title="lawhere")]
    return standard_response(
        "lawhere",
        grounded=True,
        context_found=True,
        result=result,
        sources=sources,
        warnings=[LEGAL_DISCLAIMER],
        correlation_id=cid,
    )
