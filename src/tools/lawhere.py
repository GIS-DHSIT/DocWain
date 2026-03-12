from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.api.config import Config
from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.safety import LEGAL_DISCLAIMER, add_disclaimer
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/lawhere", tags=["Tools-LawHere"])

class LawHereRequest(BaseModel):
    text: str = Field(..., description="Legal text to analyze")
    query: Optional[str] = Field(default=None, description="User query for focused analysis")
    profile_type: Optional[str] = Field(default=None, description="Legal profile or domain")

# ── JSON Schema for LLM extraction ─────────────────────────────────

_JSON_SCHEMA = """{
  "parties": [{"name": "", "role": ""}],
  "obligations": [{"party": "", "clause": "", "type": ""}],
  "rights": [{"party": "", "clause": ""}],
  "conditions": [],
  "termination_provisions": "",
  "liability": "",
  "governing_law": "",
  "jurisdiction": "",
  "legal_system": "",
  "key_dates": [],
  "risk_assessment": [],
  "summary": ""
}"""

_EXPECTED_FIELDS = ["parties", "obligations", "risks", "summary"]

# ── Jurisdiction detection ────────────────────────────────────────────

_JURISDICTION_PATTERNS: Dict[str, List[str]] = {
    "US": [r"\bU\.?S\.?\b", r"\bUnited States\b", r"\bFederal\s+(?:Court|Law|Register)\b",
           r"\bU\.?S\.?\s+Code\b", r"\bCFR\b", r"\bFICA\b", r"\bHIPAA\b", r"\bADA\b"],
    "UK": [r"\bUnited Kingdom\b", r"\bEngland and Wales\b", r"\bHMRC\b", r"\bCompanies Act\b",
           r"\bNHS\b", r"\bCommon Law\b", r"\bHigh Court\b", r"\bCrown Court\b"],
    "IN": [r"\bIndia\b", r"\bIndian\s+(?:Contract|Penal|Companies)\s+Act\b",
           r"\bSupreme Court of India\b", r"\bNCLT\b", r"\bSEBI\b", r"\bRBI\b"],
    "EU": [r"\bEuropean Union\b", r"\bGDPR\b", r"\bEU\s+Directive\b", r"\bECJ\b",
           r"\bEuropean Court\b", r"\bRegulation\s+\(EU\)\b"],
    "AU": [r"\bAustralia\b", r"\bAustralian\s+(?:Securities|Consumer|Competition)\b",
           r"\bASIC\b", r"\bACCC\b", r"\bFair Work Act\b"],
}

_JURISDICTION_CONTEXT: Dict[str, str] = {
    "US": "United States legal system (federal + state). Key frameworks: UCC, FRCP, CFR. Court hierarchy: District -> Circuit -> Supreme.",
    "UK": "English common law system. Key frameworks: Companies Act, SRA, CPR. Court hierarchy: Magistrates -> Crown/County -> High Court -> Court of Appeal -> Supreme Court.",
    "IN": "Indian legal system based on common law. Key frameworks: Indian Contract Act, Companies Act 2013, CPC/CrPC. Court hierarchy: District -> High Court -> Supreme Court.",
    "EU": "European Union regulatory framework. Key instruments: Regulations (directly applicable), Directives (transposed into national law). GDPR, Consumer Rights Directive.",
    "AU": "Australian legal system (federal + state). Key frameworks: Corporations Act, ACL, Fair Work Act. Court hierarchy: Magistrates -> Federal/State -> High Court.",
}

def _detect_legal_jurisdiction(text: str) -> Optional[str]:
    """Detect the legal jurisdiction from document text using regex patterns."""
    if not text:
        return None
    scores: Dict[str, int] = {}
    for country, patterns in _JURISDICTION_PATTERNS.items():
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, text[:5000], re.IGNORECASE))
        if count > 0:
            scores[country] = count
    if not scores:
        return None
    return max(scores, key=scores.get)

# ── LLM extraction ─────────────────────────────────────────────────

def _llm_extract(text: str, query: str = "", jurisdiction: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """LLM-powered legal document analysis with jurisdiction context. Returns None on failure."""
    try:
        from src.tools.llm_tools import build_extraction_prompt, tool_generate_structured
        prompt = build_extraction_prompt("lawhere", text, query, _JSON_SCHEMA)
        if jurisdiction and jurisdiction in _JURISDICTION_CONTEXT:
            prompt = f"JURISDICTION CONTEXT: {_JURISDICTION_CONTEXT[jurisdiction]}\n\n{prompt}"
        return tool_generate_structured(prompt, domain="legal")
    except Exception as exc:
        logger.debug("Legal LLM extraction failed: %s", exc)
        return None

def _normalize_llm_result(raw: Dict[str, Any], profile_type: Optional[str] = None) -> Dict[str, Any]:
    """Normalize LLM output into the expected legal result shape."""
    parties = raw.get("parties", [])
    obligations = raw.get("obligations", [])
    rights = raw.get("rights", [])
    conditions = raw.get("conditions", [])
    risk_assessment = raw.get("risk_assessment", [])
    summary = raw.get("summary", "")

    # Build key_clauses for backward compatibility
    key_clauses: List[str] = []
    for ob in obligations:
        if isinstance(ob, dict) and ob.get("clause"):
            key_clauses.append(ob["clause"])
        elif isinstance(ob, str):
            key_clauses.append(ob)

    # Build risks for backward compatibility
    risks: List[str] = []
    for r in risk_assessment:
        if isinstance(r, dict):
            risks.append(r.get("description", str(r)))
        elif isinstance(r, str):
            risks.append(r)
    # Also check obligations for liability/indemnity mentions
    for clause in key_clauses:
        if "liability" in clause.lower() or "indemn" in clause.lower():
            if clause not in risks:
                risks.append(clause)

    return {
        "summary": summary,
        "key_clauses": key_clauses[:10],
        "risks": risks,
        "obligations": key_clauses[:10],
        "parties": parties,
        "rights": rights,
        "conditions": conditions,
        "termination_provisions": raw.get("termination_provisions", ""),
        "liability": raw.get("liability", ""),
        "governing_law": raw.get("governing_law", ""),
        "jurisdiction": raw.get("jurisdiction", ""),
        "legal_system": raw.get("legal_system", ""),
        "key_dates": raw.get("key_dates", []),
        "risk_assessment": risk_assessment,
        "profile_type": profile_type,
    }

# ── Regex fallback ──────────────────────────────────────────────────

def _extract_clauses(text: str) -> List[str]:
    matches = re.findall(r"(shall|must|will)\s+[^\.]+", text, flags=re.IGNORECASE)
    return [m.strip() for m in matches[:6]]

def _regex_analyze(text: str, profile_type: Optional[str] = None) -> Dict[str, Any]:
    cleaned = sanitize_text(text, max_chars=4000)
    clauses = _extract_clauses(cleaned)

    # Also extract key-value provisions (e.g., "Coverage: Third-Party Liability")
    kv_provisions = re.findall(r'([A-Z][A-Za-z\s]{3,30}):\s*(.{5,120})', cleaned)
    for label, value in kv_provisions[:6]:
        clause_text = f"{label.strip()}: {value.strip()}"
        if clause_text not in clauses:
            clauses.append(clause_text)

    risks = [clause for clause in clauses if any(
        kw in clause.lower() for kw in ("liability", "indemn", "exclusion", "limit", "deductible", "risk")
    )]
    summary = cleaned[:600] if cleaned else "No content provided."
    return {
        "summary": summary,
        "key_clauses": clauses[:10],
        "risks": risks,
        "obligations": clauses[:10],
        "profile_type": profile_type,
    }

# ── Unified analysis ───────────────────────────────────────────────

def _analyze(request: LawHereRequest) -> Dict[str, Any]:
    """Analyze legal text using LLM first, falling back to regex."""
    from src.tools.llm_tools import score_tool_response

    jurisdiction = _detect_legal_jurisdiction(request.text)
    llm_result = _llm_extract(request.text, query=request.query or "", jurisdiction=jurisdiction)
    if llm_result:
        result = _normalize_llm_result(llm_result, request.profile_type)
        iq = score_tool_response(result, domain="legal", expected_fields=_EXPECTED_FIELDS, source="llm")
    else:
        result = _regex_analyze(request.text, request.profile_type)
        iq = score_tool_response(result, domain="legal", expected_fields=_EXPECTED_FIELDS, source="regex")

    result["iq_score"] = iq.as_dict()
    if jurisdiction:
        result["jurisdiction"] = jurisdiction
        result["legal_system"] = _JURISDICTION_CONTEXT.get(jurisdiction, "")
    result["summary"] = add_disclaimer(result.get("summary", ""), domain="legal")

    # Build rendered text for pipeline pre-rendering (avoids JSON serialization)
    rendered_parts: List[str] = []
    if result.get("summary"):
        rendered_parts.append(f"**Summary:** {result['summary']}")
    clauses = result.get("key_clauses") or []
    if clauses:
        rendered_parts.append("**Key Provisions:**")
        for c in clauses[:8]:
            if isinstance(c, str) and c.strip():
                rendered_parts.append(f"- {c.strip()}")
    risks = result.get("risks") or []
    if risks:
        rendered_parts.append("**Risks/Liabilities:**")
        for r in risks[:5]:
            if isinstance(r, str) and r.strip():
                rendered_parts.append(f"- {r.strip()}")
    parties = result.get("parties") or []
    if parties:
        rendered_parts.append("**Parties:**")
        for p in parties[:5]:
            if isinstance(p, dict):
                rendered_parts.append(f"- {p.get('name', '')} ({p.get('role', '')})")
            elif isinstance(p, str):
                rendered_parts.append(f"- {p}")
    if result.get("jurisdiction"):
        rendered_parts.append(f"**Jurisdiction:** {result['jurisdiction']} — {result.get('legal_system', '')[:150]}")
    if rendered_parts:
        # Prepend analytical overview with intel signal words
        populated = sum(
            1 for k in ("summary", "key_clauses", "risks", "parties")
            if result.get(k)
        )
        if populated > 0:
            overview = (
                f"**Overview:** Analyzed the document and extracted a total of "
                f"{populated} legal/policy data categories. Here is a summary of the "
                f"findings across the available evidence:"
            )
            rendered_parts.insert(0, overview)
        result["rendered"] = "\n\n".join(rendered_parts)
    return result

def _ensure_domain_enabled() -> None:
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise ToolError(
            "Legal-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
            code="deprecated",
            status_code=410,
        )

@register_tool("lawhere")
async def lawhere_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    _ensure_domain_enabled()
    raw = payload.get("input") or payload
    # Pipeline invocation sends {query, chunks, text} — adapt to LawHereRequest
    if isinstance(raw, dict) and "query" in raw and "text" in raw:
        raw = {"text": raw["text"], "query": raw["query"], "profile_type": raw.get("profile_type")}
    req = LawHereRequest(**raw)
    result = _analyze(req)
    sources = [build_source_record("tool", correlation_id or "lawhere", title="lawhere")]
    return {"result": result, "sources": sources, "grounded": True, "context_found": True, "warnings": [LEGAL_DISCLAIMER]}

@router.post("/analyze")
async def analyze(request: LawHereRequest, x_correlation_id: str | None = Header(None)):
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Legal-specific tooling is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
        )
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
