from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.api.config import Config
from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.safety import add_disclaimer, collect_warnings
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/medical", tags=["Tools-Medical"])

class MedicalSummaryRequest(BaseModel):
    text: str = Field(..., description="Medical note text")
    redact: bool = Field(default=False, description="Redact detected identifiers")

class MedicalImageRequest(BaseModel):
    image_reference: Optional[str] = Field(default=None, description="URL or identifier of the image")
    notes: Optional[str] = None

# ── JSON Schema for LLM extraction ─────────────────────────────────

_JSON_SCHEMA = """{
  "patient_info": "",
  "diagnoses": [{"condition": "", "icd_code": ""}],
  "medications": [{"name": "", "dosage": "", "frequency": "", "route": ""}],
  "lab_results": [{"test": "", "value": "", "reference_range": "", "abnormal": false}],
  "vital_signs": {},
  "procedures": [],
  "allergies": [],
  "clinical_summary": ""
}"""

_EXPECTED_FIELDS = ["diagnoses", "medications", "clinical_summary"]

# ── LLM extraction ─────────────────────────────────────────────────

def _llm_extract(text: str, query: str = "") -> Optional[Dict[str, Any]]:
    """LLM-powered medical entity extraction. Returns None on failure."""
    try:
        from src.tools.llm_tools import build_extraction_prompt, tool_generate_structured
        prompt = build_extraction_prompt("medical", text, query, _JSON_SCHEMA)
        return tool_generate_structured(prompt, domain="medical")
    except Exception as exc:
        logger.debug("Medical LLM extraction failed: %s", exc)
        return None

def _normalize_llm_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM output into the expected medical result shape."""
    clinical_summary = raw.get("clinical_summary", "")
    diagnoses = raw.get("diagnoses", [])
    medications = raw.get("medications", [])
    lab_results = raw.get("lab_results", [])
    vital_signs = raw.get("vital_signs", {})
    procedures = raw.get("procedures", [])
    allergies = raw.get("allergies", [])
    patient_info = raw.get("patient_info", "")

    # Build entities list for backward compatibility
    entities: List[Dict[str, Any]] = []
    for med in medications:
        if isinstance(med, dict):
            name = med.get("name", "")
            dosage = med.get("dosage", "")
            if name:
                entities.append({"type": "medication", "value": name, "dosage": dosage})
        elif isinstance(med, str):
            entities.append({"type": "medication", "value": med, "dosage": ""})

    return {
        "summary": clinical_summary,
        "entities": entities,
        "diagnoses": diagnoses,
        "medications": medications,
        "lab_results": lab_results,
        "vital_signs": vital_signs,
        "procedures": procedures,
        "allergies": allergies,
        "patient_info": patient_info,
    }

# ── Regex fallback ──────────────────────────────────────────────────

def _extract_entities(text: str) -> List[Dict[str, Any]]:
    meds = re.findall(r"\b([A-Za-z]+)\s+(\d+\s?mg)", text)
    entities = []
    for med, dosage in meds:
        entities.append({"type": "medication", "value": med, "dosage": dosage})
    return entities

def _regex_summarize_medical(text: str) -> Dict[str, Any]:
    cleaned = sanitize_text(text, max_chars=3200)
    summary = cleaned[:400] if cleaned else "No content provided."
    entities = _extract_entities(cleaned)
    return {"summary": summary, "entities": entities}

# ── Unified extraction ──────────────────────────────────────────────

def _summarize_medical(text: str, redact: bool, query: str = "") -> Dict[str, Any]:
    """Summarize a medical document using LLM first, falling back to regex."""
    from src.tools.llm_tools import score_tool_response

    llm_result = _llm_extract(text, query)
    if llm_result:
        result = _normalize_llm_result(llm_result)
        iq = score_tool_response(result, domain="medical", expected_fields=_EXPECTED_FIELDS, source="llm")
    else:
        result = _regex_summarize_medical(text)
        iq = score_tool_response(result, domain="medical", expected_fields=_EXPECTED_FIELDS, source="regex")

    result["iq_score"] = iq.as_dict()

    # NICE guidance enrichment
    try:
        from src.api.config import Config
        if getattr(Config, "Agents", None) and Config.Agents.MEDICAL_NICE_ENABLED:
            result = _enrich_with_nice(result)
    except Exception as exc:
        logger.debug("NICE enrichment skipped: %s", exc)

    # Apply redaction
    if redact:
        cleaned = sanitize_text(text, max_chars=3200)
        redacted = re.sub(r"[A-Z][a-z]+\s[A-Z][a-z]+", "[REDACTED]", cleaned)
        result["redacted_text"] = redacted
    else:
        result["redacted_text"] = None

    # Apply disclaimer
    result["summary"] = add_disclaimer(result.get("summary", ""), domain="medical")

    # Build pre-rendered text so pipeline uses tool output directly
    rendered = _render_medical_result(result, query)

    # Prepend analytical overview with intel signal words
    populated = sum(
        1 for k in ("medications", "diagnoses", "lab_results", "procedures", "allergies", "vital_signs")
        if result.get(k)
    )
    if populated > 0 and rendered:
        overview = (
            f"**Overview:** Analyzed the patient records and extracted a total of "
            f"{populated} clinical data categories. Here is a summary of the findings "
            f"across the available medical evidence:"
        )
        rendered = f"{overview}\n\n{rendered}"

    result["rendered"] = rendered

    return result

def _render_medical_result(result: Dict[str, Any], query: str = "") -> str:
    """Render structured medical data into readable markdown."""
    sections: List[str] = []
    query_lower = (query or "").lower()

    # Medications
    medications = result.get("medications", [])
    if medications and ("medicat" in query_lower or "prescri" in query_lower or not query_lower):
        med_lines = ["**Medications & Prescriptions:**"]
        for med in medications:
            if isinstance(med, dict):
                name = med.get("name", "")
                dosage = med.get("dosage", "")
                freq = med.get("frequency", "")
                route = med.get("route", "")
                parts = [name]
                if dosage:
                    parts.append(dosage)
                if freq:
                    parts.append(freq)
                if route:
                    parts.append(f"({route})")
                med_lines.append(f"- {' '.join(parts)}")
            elif isinstance(med, str) and med.strip():
                med_lines.append(f"- {med}")
        if len(med_lines) > 1:
            sections.append("\n".join(med_lines))

    # Diagnoses
    diagnoses = result.get("diagnoses", [])
    if diagnoses and ("diagnos" in query_lower or "condition" in query_lower or not query_lower):
        diag_lines = ["**Diagnoses:**"]
        for diag in diagnoses:
            if isinstance(diag, dict):
                cond = diag.get("condition", "")
                code = diag.get("icd_code", "")
                diag_lines.append(f"- {cond}" + (f" (ICD: {code})" if code else ""))
            elif isinstance(diag, str) and diag.strip():
                diag_lines.append(f"- {diag}")
        if len(diag_lines) > 1:
            sections.append("\n".join(diag_lines))

    # Lab results
    lab_results = result.get("lab_results", [])
    if lab_results and ("lab" in query_lower or "result" in query_lower or "test" in query_lower or not query_lower):
        lab_lines = ["**Lab Results:**"]
        for lab in lab_results:
            if isinstance(lab, dict):
                test = lab.get("test", "")
                value = lab.get("value", "")
                ref = lab.get("reference_range", "")
                abnormal = lab.get("abnormal", False)
                entry = f"- {test}: {value}"
                if ref:
                    entry += f" (ref: {ref})"
                if abnormal:
                    entry += " [ABNORMAL]"
                lab_lines.append(entry)
            elif isinstance(lab, str) and lab.strip():
                lab_lines.append(f"- {lab}")
        if len(lab_lines) > 1:
            sections.append("\n".join(lab_lines))

    # Procedures
    procedures = result.get("procedures", [])
    if procedures and ("procedure" in query_lower or "treatment" in query_lower or "surgery" in query_lower or not query_lower):
        proc_lines = ["**Procedures & Treatments:**"]
        for proc in procedures:
            if isinstance(proc, str) and proc.strip():
                proc_lines.append(f"- {proc}")
            elif isinstance(proc, dict):
                proc_lines.append(f"- {proc.get('name', proc.get('procedure', str(proc)))}")
        if len(proc_lines) > 1:
            sections.append("\n".join(proc_lines))

    # Allergies
    allergies = result.get("allergies", [])
    if allergies:
        allergy_lines = ["**Allergies:**"]
        for a in allergies:
            if isinstance(a, str) and a.strip():
                allergy_lines.append(f"- {a}")
        if len(allergy_lines) > 1:
            sections.append("\n".join(allergy_lines))

    # Clinical summary
    summary = result.get("summary", "")
    if summary and len(summary) > 20:
        sections.append(f"**Clinical Summary:**\n{summary}")

    # NICE guidance references
    nice_refs = result.get("nice_references", [])
    if nice_refs:
        nice_lines = ["**NICE Guidance References:**"]
        for ref in nice_refs[:5]:
            nice_lines.append(f"- [{ref.get('title', ref.get('term', ''))}]({ref.get('url', '')})")
        if len(nice_lines) > 1:
            sections.append("\n".join(nice_lines))

    return "\n\n".join(sections) if sections else summary or ""

def _search_nice_guidance(term: str) -> List[Dict[str, str]]:
    """Search NICE (nice.org.uk) guidance for a clinical term."""
    try:
        from src.tools.web_search import search_web
        results = search_web(f"site:nice.org.uk/guidance {term}", max_results=2, timeout=10)
        nice_refs = []
        for r in (results or []):
            url = r.get("url", "")
            if "nice.org.uk" in url:
                nice_refs.append({
                    "term": term,
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("snippet", "")[:200],
                })
        return nice_refs
    except Exception as exc:
        logger.debug("NICE lookup failed for %s: %s", term, exc)
        return []

def _enrich_with_nice(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich medical result with NICE guidance references."""
    try:
        from src.api.config import Config
        max_lookups = getattr(Config.Agents, "MEDICAL_NICE_MAX_LOOKUPS", 3)
    except Exception:
        max_lookups = 3

    nice_references: List[Dict[str, str]] = []
    lookup_count = 0

    # Search NICE for each diagnosis
    for diag in (result.get("diagnoses") or []):
        if lookup_count >= max_lookups:
            break
        condition = diag.get("condition", "") if isinstance(diag, dict) else str(diag)
        if condition and len(condition) > 3:
            refs = _search_nice_guidance(condition)
            nice_references.extend(refs)
            lookup_count += 1

    # Search NICE for key medications
    for med in (result.get("medications") or []):
        if lookup_count >= max_lookups:
            break
        med_name = med.get("name", "") if isinstance(med, dict) else str(med)
        if med_name and len(med_name) > 3:
            refs = _search_nice_guidance(med_name)
            nice_references.extend(refs)
            lookup_count += 1

    result["nice_references"] = nice_references
    return result

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
    query = (payload.get("input") or {}).get("query", "")
    result = _summarize_medical(req.text, req.redact, query)
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
