from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.intelligence.domain_indexer import infer_domain


@dataclass
class RoutePlan:
    task_type: str
    domain_hint: str
    scope: str
    output_format: str
    target_person: Optional[str] = None
    target_document: Optional[str] = None
    section_focus: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "domain_hint": self.domain_hint,
            "scope": self.scope,
            "output_format": self.output_format,
            "target_person": self.target_person,
            "target_document": self.target_document,
            "section_focus": self.section_focus,
        }


RANKING_CUES = {"top", "rank", "best", "experienced", "most", "compare", "compare_rank"}
GENERATE_CUES = {"cover letter", "email", "proposal"}
SUMMARY_CUES = {"summarize", "summary", "overview", "recap"}
STRUCTURED_CUES = {"extract", "list", "fields", "json", "table"}

SECTION_FOCUS_CUES: Dict[str, List[str]] = {
    "certifications": ["certification", "certifications", "certified", "license"],
    "skills_technical": ["skills", "technical", "programming", "languages", "tech stack"],
    "skills_functional": ["functional", "competency", "competencies"],
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "degree", "university", "college"],
    "projects": ["project", "projects"],
    "financial_summary": ["total", "amount due", "balance", "subtotal"],
    "terms_conditions": ["terms", "conditions", "due date", "payment"],
    "parties_addresses": ["vendor", "supplier", "customer", "bill to", "ship to"],
    "line_items": ["line items", "items", "qty", "quantity", "unit price"],
    "transactions": ["transactions", "transaction", "debit", "credit"],
    "diagnoses_procedures": ["diagnosis", "procedure", "treatment"],
    "medications": ["medications", "prescription", "rx"],
    "lab_results": ["lab", "lab results", "test results"],
}


def _detect_person(query: str) -> Optional[str]:
    match = re.search(r"\bfor\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", query)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})'s\s+resume\b", query)
    if match:
        return match.group(1).strip()
    return None


def _detect_output_format(query: str, task_type: str) -> str:
    lowered = query.lower()
    if "json" in lowered:
        return "json"
    if "table" in lowered or task_type == "compare_rank":
        return "table"
    if task_type == "generate_document":
        return "cover_letter"
    if task_type == "summarize":
        return "bullets"
    return "free_text"


def _detect_task_type(query: str) -> str:
    lowered = query.lower()
    if any(cue in lowered for cue in GENERATE_CUES):
        return "generate_document"
    if any(cue in lowered for cue in RANKING_CUES):
        return "compare_rank"
    if any(cue in lowered for cue in SUMMARY_CUES):
        return "summarize"
    if any(cue in lowered for cue in STRUCTURED_CUES):
        return "extract_structured"
    return "qa"


def _detect_scope(query: str, session_state: Dict[str, Any], catalog: Dict[str, Any]) -> str:
    lowered = (query or "").lower()
    if "all profiles" in lowered or "all subscriptions" in lowered:
        return "subscription_all_profiles"
    if "all documents" in lowered or "across documents" in lowered or "across the profile" in lowered:
        return "profile_all_docs"
    if "this document" in lowered or "current document" in lowered:
        return "current_document"
    if len((query or "").split()) <= 2:
        if session_state.get("active_document_id"):
            return "current_document"
        if session_state.get("active_profile_id"):
            return "current_profile"
        if catalog.get("profile_id"):
            return "current_profile"
    return "profile_all_docs"


def _detect_section_focus(query: str) -> List[str]:
    lowered = (query or "").lower()
    focus: List[str] = []
    for section_kind, cues in SECTION_FOCUS_CUES.items():
        for cue in cues:
            if cue in lowered:
                focus.append(section_kind)
                break
    return focus


def _infer_domain_from_catalog(catalog: Dict[str, Any]) -> str:
    dominant = catalog.get("dominant_domains") or {}
    if not dominant:
        return "unknown"
    best = sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)
    return best[0][0] if best else "unknown"


def _infer_domain_from_query(query: str, session_state: Dict[str, Any], catalog: Dict[str, Any]) -> str:
    inferred = infer_domain(query)
    if inferred not in {"unknown", "mixed"}:
        return inferred
    if session_state.get("active_domain"):
        return session_state["active_domain"]
    return _infer_domain_from_catalog(catalog)


def auto_route(
    query: str,
    session_state: Optional[Dict[str, Any]],
    catalog: Optional[Dict[str, Any]],
    entities_cache: Optional[Dict[str, Any]],
) -> RoutePlan:
    session_state = session_state or {}
    catalog = catalog or {}
    _ = entities_cache

    task_type = _detect_task_type(query)
    scope = _detect_scope(query, session_state, catalog)
    domain_hint = _infer_domain_from_query(query, session_state, catalog)
    output_format = _detect_output_format(query, task_type)
    section_focus = _detect_section_focus(query)

    target_person = _detect_person(query)
    target_document = None

    if catalog.get("documents"):
        for doc in catalog.get("documents") or []:
            name = doc.get("source_name")
            if name and name.lower() in query.lower():
                target_document = doc.get("document_id")
                break

    if len((query or "").split()) <= 2:
        task_type = "summarize"
        output_format = "bullets"
        if session_state.get("active_domain"):
            domain_hint = session_state["active_domain"]
        elif not domain_hint:
            domain_hint = _infer_domain_from_catalog(catalog)

    return RoutePlan(
        task_type=task_type,
        domain_hint=domain_hint or "unknown",
        scope=scope,
        output_format=output_format,
        target_person=target_person,
        target_document=target_document,
        section_focus=section_focus,
    )


__all__ = ["RoutePlan", "auto_route"]
