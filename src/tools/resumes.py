from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from src.api.config import Config
from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.base import ToolError
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/resumes", tags=["Tools-Resumes"])

class ResumeRequest(BaseModel):
    text: str = Field(..., description="Resume content to parse")

# ── JSON Schema for LLM extraction ─────────────────────────────────

_JSON_SCHEMA = """{
  "name": "", "email": "", "phone": "", "linkedin": "",
  "summary": "",
  "skills": [],
  "experience": [{"company": "", "role": "", "dates": "", "achievements": []}],
  "education": [{"degree": "", "institution": "", "year": "", "gpa": ""}],
  "certifications": [],
  "years_of_experience": ""
}"""

_EXPECTED_FIELDS = ["name", "skills", "experience", "education", "summary"]

# ── LLM extraction ─────────────────────────────────────────────────

def _llm_extract(text: str, query: str) -> Optional[Dict[str, Any]]:
    """LLM-powered resume extraction. Returns None on failure."""
    try:
        from src.tools.llm_tools import build_extraction_prompt, tool_generate_structured
        prompt = build_extraction_prompt("resumes", text, query, _JSON_SCHEMA)
        return tool_generate_structured(prompt, domain="hr")
    except Exception as exc:
        logger.debug("Resume LLM extraction failed: %s", exc)
        return None

def _normalize_llm_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM output into the expected result shape."""
    skills = raw.get("skills", [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    experience = raw.get("experience", [])
    if isinstance(experience, str):
        experience = [experience]

    education = raw.get("education", [])
    if isinstance(education, str):
        education = [education]

    certifications = raw.get("certifications", [])
    if isinstance(certifications, str):
        certifications = [c.strip() for c in certifications.split(",") if c.strip()]

    contact_parts = []
    for key in ("name", "email", "phone", "linkedin"):
        val = raw.get(key, "")
        if val:
            contact_parts.append(f"{key.title()}: {val}")
    contact = "; ".join(contact_parts) if contact_parts else "Not provided"

    summary = raw.get("summary", "")
    warnings: List[str] = []

    return {
        "contact": contact,
        "summary": summary,
        "experience": experience[:10],
        "education": education[:5],
        "skills": skills,
        "certifications": certifications,
        "years_of_experience": raw.get("years_of_experience", ""),
        "name": raw.get("name", ""),
        "email": raw.get("email", ""),
        "phone": raw.get("phone", ""),
        "linkedin": raw.get("linkedin", ""),
        "warnings": warnings,
        "ats_hints": ["Use consistent date formats", "Include measurable outcomes"],
    }

# ── Regex fallback ──────────────────────────────────────────────────

def _extract_section(text: str, header: str) -> str:
    pattern = rf"{header}[:\-]\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _regex_parse_resume(text: str) -> Dict[str, Any]:
    cleaned = sanitize_text(text, max_chars=4500)
    contact = _extract_section(cleaned, "contact") or "Not provided"
    summary = _extract_section(cleaned, "summary") or cleaned[:200]
    skills = re.findall(r"skills?:\s*([^\n]+)", cleaned, flags=re.IGNORECASE)
    experience = re.findall(r"(\d{4}.+)", cleaned)
    education = re.findall(r"(Bachelor|Master|PhD)[^\n]+", cleaned, flags=re.IGNORECASE)
    warnings: List[str] = []
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

# ── Unified parse ───────────────────────────────────────────────────

def _parse_resume(text: str, query: str = "") -> Dict[str, Any]:
    """Parse a resume using LLM first, falling back to regex.

    For large text inputs (multi-document concatenation), skip LLM to avoid
    blocking the event loop for minutes. The regex extractor handles large
    inputs efficiently.
    """
    from src.tools.llm_tools import score_tool_response

    # Skip LLM for large inputs — the context is too big for reliable extraction
    # and will block the async event loop (sync ollama call).
    _MAX_LLM_INPUT = 6000  # ~1500 tokens
    llm_result = None
    if len(text) <= _MAX_LLM_INPUT:
        llm_result = _llm_extract(text, query)

    if llm_result:
        result = _normalize_llm_result(llm_result)
        iq = score_tool_response(result, domain="hr", expected_fields=_EXPECTED_FIELDS, source="llm")
    else:
        result = _regex_parse_resume(text)
        iq = score_tool_response(result, domain="hr", expected_fields=_EXPECTED_FIELDS, source="regex")

    result["iq_score"] = iq.as_dict()
    try:
        from src.api.config import Config
        if getattr(Config, "Agents", None) and Config.Agents.RESUMES_INTERNET_ENABLED:
            result = _web_enrich_resume(result)
    except Exception as exc:
        logger.debug("Web enrichment skipped: %s", exc)
    return result

def _ensure_domain_enabled() -> None:
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise ToolError(
            "Resume-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
            code="deprecated",
            status_code=410,
        )

def _render_parsed(parsed: Dict[str, Any]) -> str:
    """Build a human-readable text summary from parsed resume fields."""
    parts: List[str] = []
    name = parsed.get("name", "")
    if name:
        parts.append(f"**{name}**")
    contact_items = []
    for key in ("email", "phone", "linkedin"):
        val = parsed.get(key, "")
        if val:
            contact_items.append(f"{key.title()}: {val}")
    if contact_items:
        parts.append("Contact: " + "; ".join(contact_items))
    summary = parsed.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary[:300]}")
    skills = parsed.get("skills", [])
    if skills:
        parts.append(f"Skills: {', '.join(skills[:15])}")
    experience = parsed.get("experience", [])
    if experience:
        exp_lines = []
        for exp in experience[:5]:
            if isinstance(exp, dict):
                exp_lines.append(f"- {exp.get('role', '')} at {exp.get('company', '')} ({exp.get('dates', '')})")
            elif isinstance(exp, str):
                exp_lines.append(f"- {exp[:150]}")
        if exp_lines:
            parts.append("Experience:\n" + "\n".join(exp_lines))
    education = parsed.get("education", [])
    if education:
        edu_lines = []
        for edu in education[:3]:
            if isinstance(edu, dict):
                edu_lines.append(f"- {edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('year', '')})")
            elif isinstance(edu, str):
                edu_lines.append(f"- {edu[:150]}")
        if edu_lines:
            parts.append("Education:\n" + "\n".join(edu_lines))
    yoe = parsed.get("years_of_experience", "")
    if yoe:
        parts.append(f"Years of Experience: {yoe}")
    web_insights = parsed.get("web_insights", [])
    if web_insights:
        insight_lines = ["**Web Insights:**"]
        for wi in web_insights[:3]:
            wi_type = wi.get("type", "")
            if wi_type == "linkedin_verification":
                insight_lines.append(f"- LinkedIn: [{wi.get('title', 'Profile')}]({wi.get('url', '')})")
            elif wi_type == "certification_lookup":
                insight_lines.append(f"- Cert: {wi.get('certification', '')} — {wi.get('snippet', '')[:100]}")
        if len(insight_lines) > 1:
            parts.append("\n".join(insight_lines))
    return "\n".join(parts)

def _web_enrich_resume(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich parsed resume with web-sourced insights (LinkedIn, certs)."""
    web_insights: List[Dict[str, str]] = []
    try:
        from src.tools.web_search import search_web
    except Exception:
        parsed["web_insights"] = []
        return parsed

    name = parsed.get("name", "")
    if name and len(name) > 3:
        try:
            results = search_web(f"site:linkedin.com {name}", max_results=2, timeout=8)
            if results:
                for r in results[:1]:
                    web_insights.append({
                        "type": "linkedin_verification",
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", "")[:200],
                    })
        except Exception as exc:
            logger.debug("LinkedIn lookup failed for %s: %s", name, exc)

    certifications = parsed.get("certifications", [])
    for cert in certifications[:2]:
        cert_name = cert if isinstance(cert, str) else str(cert)
        if not cert_name or len(cert_name) < 3:
            continue
        try:
            results = search_web(f"{cert_name} certification validity", max_results=1, timeout=8)
            if results:
                web_insights.append({
                    "type": "certification_lookup",
                    "certification": cert_name,
                    "title": results[0].get("title", ""),
                    "url": results[0].get("url", ""),
                    "snippet": results[0].get("snippet", "")[:200],
                })
        except Exception as exc:
            logger.debug("Cert lookup failed for %s: %s", cert_name, exc)

    parsed["web_insights"] = web_insights
    return parsed

@register_tool("resumes")
async def resumes_handler(payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
    _ensure_domain_enabled()
    req = ResumeRequest(**(payload.get("input") or payload))
    query = (payload.get("input") or {}).get("query", "")
    parsed = _parse_resume(req.text, query)
    rendered = _render_parsed(parsed)
    sources = [build_source_record("tool", correlation_id or "resumes", title="resume")]
    return {"result": {**parsed, "rendered": rendered}, "sources": sources, "grounded": True, "context_found": True, "warnings": parsed.get("warnings", [])}

@router.post("/analyze")
async def analyze(request: ResumeRequest, x_correlation_id: str | None = Header(None)):
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Resume-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
        )
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
