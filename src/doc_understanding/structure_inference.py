from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Access attribute or dict key — supports both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

_ENTITY_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE),
    "DATE": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    "AMOUNT": re.compile(r"\b(?:USD|EUR|GBP|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b", re.IGNORECASE),
    "PHONE": re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"),
}

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")

def _numeric_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / max(1, len(text))

def _entity_density(text: str) -> float:
    if not text:
        return 0.0
    hits = 0
    for pattern in _ENTITY_PATTERNS.values():
        hits += len(pattern.findall(text))
    word_count = max(1, len(re.findall(r"\w+", text)))
    return hits / word_count

def _list_density(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    bullets = sum(1 for ln in lines if _BULLET_RE.match(ln))
    return bullets / max(1, len(lines))

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "header_meta": ["header", "title page", "cover", "metadata", "document info"],
    "contact_info": ["contact", "address", "phone", "email", "location", "reach us"],
    "summary_like": ["summary", "overview", "abstract", "executive summary", "introduction", "highlights"],
    "experience_history": ["experience", "employment", "work history", "career", "positions held", "professional background"],
    "education_training": ["education", "training", "certification", "qualification", "academic", "degree", "courses"],
    "financial_data": ["financial", "revenue", "profit", "loss", "balance sheet", "income", "expense", "budget"],
    "legal_terms": ["terms", "conditions", "liability", "warranty", "indemnity", "agreement", "clause", "governing law"],
    "clinical_data": ["diagnosis", "treatment", "medication", "patient", "clinical", "lab results", "vitals", "prognosis"],
    "technical_specs": ["specifications", "technical", "architecture", "system requirements", "configuration", "api"],
    "appendix": ["appendix", "annex", "attachment", "exhibit", "addendum", "supplementary"],
    "requirements_like": ["requirements", "spec", "details", "scope", "objectives", "deliverables"],
    "transactional": ["total due", "amount due", "balance due", "invoice number", "invoice date", "payment terms", "subtotal", "unit price", "net amount"],
}

def _infer_section_role(text: str, title: str, numeric_ratio: float, list_density: float) -> str:
    lowered_title = (title or "").lower()
    lowered_text = (text or "").lower()
    word_count = len(re.findall(r"\w+", text or ""))

    # Score each role by keyword hits in title (weight 3) + text (weight 1)
    best_role = "descriptive"
    best_score = 0.0
    for role, keywords in _ROLE_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            if kw in lowered_title:
                score += 3.0
            if kw in lowered_text:
                score += 1.0
        if score > best_score:
            best_score = score
            best_role = role

    # Structural heuristics as tiebreakers when keyword score is low
    if best_score < 2.0:
        if numeric_ratio > 0.15:
            return "transactional"
        if list_density > 0.25:
            return "list_heavy"
        if word_count < 30:
            return "header_meta"
        return "descriptive"

    return best_role

def _section_importance_score(
    *,
    word_count: int,
    numeric_ratio: float,
    entity_density: float,
    list_density: float,
    position_index: int,
    total_sections: int,
) -> float:
    length_score = _clamp(word_count / 200.0)
    numeric_score = _clamp(numeric_ratio * 8.0)
    entity_score = _clamp(entity_density * 6.0)
    list_score = _clamp(list_density * 3.0)
    if total_sections <= 1:
        position_score = 1.0
    else:
        position_score = 1.0 - (position_index / (total_sections - 1)) * 0.3
    weighted = (
        0.32 * length_score
        + 0.22 * entity_score
        + 0.18 * numeric_score
        + 0.12 * list_score
        + 0.16 * position_score
    )
    return round(_clamp(weighted), 4)

def _element_importance_score(text: str, base: float = 0.5) -> float:
    if not text:
        return round(base, 4)
    numeric_score = _clamp(_numeric_ratio(text) * 8.0)
    length_score = _clamp(len(text) / 400.0)
    return round(_clamp(0.6 * base + 0.25 * numeric_score + 0.15 * length_score), 4)

def _extract_sections(extracted: Any) -> List[Dict[str, Any]]:
    sections = list(_get(extracted, "sections", []) or [])
    if sections:
        return [
            {
                "title": _get(sec, "title", "Untitled Section") or "Untitled Section",
                "text": _get(sec, "text", "") or "",
                "start_page": _get(sec, "start_page"),
                "end_page": _get(sec, "end_page"),
            }
            for sec in sections
        ]
    full_text = _get(extracted, "full_text", "") or ""
    if full_text:
        return [{"title": "Document", "text": full_text, "start_page": None, "end_page": None}]
    return []

def infer_structure(extracted: Any) -> Dict[str, Any]:
    sections = _extract_sections(extracted)
    total_sections = max(1, len(sections))
    section_results: List[Dict[str, Any]] = []
    for idx, section in enumerate(sections):
        text = section.get("text", "") or ""
        title = section.get("title", "Untitled Section")
        word_count = len(re.findall(r"\w+", text))
        numeric_ratio = _numeric_ratio(text)
        entity_density = _entity_density(text)
        list_density = _list_density(text)
        role = _infer_section_role(text, title, numeric_ratio, list_density)
        importance = _section_importance_score(
            word_count=word_count,
            numeric_ratio=numeric_ratio,
            entity_density=entity_density,
            list_density=list_density,
            position_index=idx,
            total_sections=total_sections,
        )
        section_results.append(
            {
                "section_title": title,
                "page_start": section.get("start_page"),
                "page_end": section.get("end_page"),
                "section_importance_score": importance,
                "inferred_section_role": role,
                "numeric_ratio": round(numeric_ratio, 4),
                "entity_density": round(entity_density, 4),
                "list_density": round(list_density, 4),
                "word_count": word_count,
            }
        )

    tables_payload: List[Dict[str, Any]] = []
    for table in list(_get(extracted, "tables", []) or []):
        text = _get(table, "text", "") or _get(table, "csv", "") or ""
        csv_text = _get(table, "csv", "") or ""
        rows = [ln for ln in csv_text.splitlines() if ln.strip()] if csv_text else []
        tables_payload.append(
            {
                "page": _get(table, "page"),
                "row_count": len(rows),
                "element_importance_score": _element_importance_score(text, base=0.65),
                "numeric_ratio": round(_numeric_ratio(text), 4),
            }
        )

    images_payload: List[Dict[str, Any]] = []
    for figure in list(_get(extracted, "figures", []) or []):
        caption = _get(figure, "caption", "") or ""
        images_payload.append(
            {
                "page": _get(figure, "page"),
                "caption_present": bool(caption.strip()),
                "element_importance_score": _element_importance_score(caption, base=0.55),
            }
        )

    full_text = _get(extracted, "full_text", "") or ""
    doc_density = {
        "numeric_ratio": round(_numeric_ratio(full_text), 4),
        "entity_density": round(_entity_density(full_text), 4),
    }

    return {
        "sections": section_results,
        "tables": tables_payload,
        "images": images_payload,
        "document_density": doc_density,
    }

__all__ = ["infer_structure"]
