from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resumes", tags=["Tools-Resumes"])


class ResumeRequest(BaseModel):
    text: str = Field(..., description="Resume content to parse")


def _extract_section(text: str, header: str) -> str:
    pattern = rf"{header}[:\-]\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _parse_resume(text: str) -> Dict[str, Any]:
    cleaned = sanitize_text(text, max_chars=4500)
    contact = _extract_section(cleaned, "contact") or "Not provided"
    summary = _extract_section(cleaned, "summary") or cleaned[:200]
    skills = re.findall(r"skills?:\s*([^\n]+)", cleaned, flags=re.IGNORECASE)
    experience = re.findall(r"(\d{4}.+)", cleaned)
    education = re.findall(r"(Bachelor|Master|PhD)[^\n]+", cleaned, flags=re.IGNORECASE)
    warnings = []
    if "202" not in cleaned:
        warnings.append("Timeline not detected; please verify employment dates.")
    return {
        "contact": contact,
        "summary": summary,
        "experience": experience[:5],
        "education": education[:5],
        "skills": skills[0].split(",") if skills else [],
        "warnings": warnings,
        "ats_hints": ["Use consistent date formats", "Include measurable outcomes"],
    }


@register_tool("resumes")
async def resumes_handler(payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
    req = ResumeRequest(**(payload.get("input") or payload))
    parsed = _parse_resume(req.text)
    sources = [build_source_record("tool", correlation_id or "resumes", title="resume")]
    return {"result": parsed, "sources": sources, "grounded": True, "context_found": True, "warnings": parsed.get("warnings", [])}


@router.post("/analyze")
async def analyze(request: ResumeRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    parsed = _parse_resume(request.text)
    sources = [build_source_record("tool", cid, title="resume")]
    return standard_response(
        "resumes",
        grounded=True,
        context_found=True,
        result=parsed,
        sources=sources,
        warnings=parsed.get("warnings", []),
        correlation_id=cid,
    )
