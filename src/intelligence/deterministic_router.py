from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.intelligence.domain_indexer import infer_domain


_GREETING_RE = re.compile(r"^\s*(hi|hii+|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b", re.IGNORECASE)
_RANK_RE = re.compile(
    r"\b(rank|ranking|top|best|most|better|highest|strongest|most\s+experienced|well\s+experienced)\b",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(r"\b(compare|comparison|difference|different|vs|versus)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"\b(summarize|summarise|summary|overview|recap|highlights)\b", re.IGNORECASE)
_GENERATE_RE = re.compile(r"\b(generate|draft|create|write|compose)\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\b(list|show|which|filter)\b", re.IGNORECASE)
_EXTRACT_RE = re.compile(r"\b(extract|identify|find|pull)\b", re.IGNORECASE)
_JSON_RE = re.compile(r"\bjson\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\btable\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _detect_output_format(query: str, task_type: str) -> str:
    if _JSON_RE.search(query):
        return "json"
    if _TABLE_RE.search(query) or task_type == "rank":
        return "table"
    if task_type == "generate":
        return "cover_letter"
    if task_type == "summarize":
        return "bullets"
    if task_type == "compare":
        return "sections"
    return "free_text"


def _detect_task_type(query: str) -> str:
    if not query:
        return "summarize"
    if _GREETING_RE.match(query):
        return "greet"
    if _GENERATE_RE.search(query):
        return "generate"
    if _COMPARE_RE.search(query):
        return "compare"
    if _RANK_RE.search(query):
        return "rank"
    if _SUMMARY_RE.search(query):
        return "summarize"
    if _EXTRACT_RE.search(query):
        return "extract"
    if _LIST_RE.search(query):
        return "list"
    return "qa"


def _infer_domain(query: str, session_state: Dict[str, Any], catalog: Dict[str, Any]) -> str:
    inferred = infer_domain(query)
    if inferred not in {"unknown", "generic"}:
        return inferred
    if session_state.get("active_domain"):
        return session_state["active_domain"]
    dominant = catalog.get("dominant_domains") or {}
    if dominant:
        return sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return inferred or "unknown"


def _extract_person_from_query(query: str) -> Optional[str]:
    match = re.search(r"\b(?:of|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", query or "")
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(?:about|regarding)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", query or "")
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(?:patient|vendor)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", query or "")
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})'s\b", query or "")
    if match:
        return match.group(1).strip()
    return None


def _match_entity_cache(query: str, entities_cache: Dict[str, Any]) -> tuple[Optional[str], List[str]]:
    if not entities_cache:
        return None, []
    query_norm = _normalize(query)
    for ent in entities_cache.get("entities") or []:
        ent_type = str(ent.get("type") or "").upper()
        if ent_type and ent_type not in {"PERSON", "ORG", "ORGANIZATION", "VENDOR", "PATIENT"}:
            continue
        value = ent.get("value") or ""
        aliases = ent.get("aliases") or []
        candidates = [value] + [a for a in aliases if a]
        for candidate in candidates:
            if not candidate:
                continue
            if _normalize(candidate) in query_norm:
                doc_ids = [str(d) for d in (ent.get("document_ids") or []) if d]
                return str(value or candidate), doc_ids
    return None, []


def _match_document_names(query: str, catalog: Dict[str, Any]) -> List[str]:
    docs = catalog.get("documents") or []
    if not docs:
        return []
    query_norm = _normalize(query)
    matched: List[str] = []
    for doc in docs:
        source_name = doc.get("source_name") or ""
        doc_id = doc.get("document_id")
        if not source_name or not doc_id:
            continue
        base = source_name.lower()
        base_no_ext = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
        base_tokens = base_no_ext.replace("_", " ").replace("-", " ")
        if base in query_norm or base_no_ext in query_norm or base_tokens in query_norm:
            matched.append(str(doc_id))
    return matched


def _detect_section_focus(query: str, domain_hint: str) -> List[str]:
    lowered = _normalize(query)
    focus: List[str] = []

    contact_terms = ["contact", "address", "location", "email", "phone", "mobile"]
    if _contains_any(lowered, contact_terms):
        focus.append("identity_contact")

    if domain_hint == "resume":
        if _contains_any(lowered, ["skills", "skill", "tech stack", "technologies", "tools"]):
            focus.extend(["skills_technical", "tools_technologies", "skills_functional"])
        if _contains_any(lowered, ["education", "degree", "university", "college"]):
            focus.append("education")
        if _contains_any(lowered, ["experience", "work history", "employment"]):
            focus.append("experience")
        if _contains_any(lowered, ["project", "projects", "portfolio"]):
            focus.append("projects")
        if _contains_any(lowered, ["certification", "certifications", "license"]):
            focus.append("certifications")
        if _contains_any(lowered, ["summary", "objective", "profile"]):
            focus.append("summary_objective")

    if domain_hint == "medical":
        if _contains_any(lowered, ["patient details", "patient info", "patient's details", "patient details"]):
            focus.extend(["identity_contact", "diagnoses_procedures", "medications", "notes"])
        if _contains_any(lowered, ["diagnosis", "procedure", "diagnoses"]):
            focus.append("diagnoses_procedures")
        if _contains_any(lowered, ["medication", "medications", "prescription", "rx"]):
            focus.append("medications")
        if _contains_any(lowered, ["lab", "lab results", "test results"]):
            focus.append("lab_results")
        if _contains_any(lowered, ["notes", "remarks", "doctor's notes", "doctor notes", "physician notes"]):
            focus.append("notes")

    if domain_hint in {"invoice", "purchase_order"}:
        if _contains_any(lowered, ["line item", "line items", "items", "qty", "quantity"]):
            focus.extend(["line_items", "tables"])
        if _contains_any(lowered, ["invoice number", "invoice date", "invoice"]):
            focus.append("invoice_metadata")
        if _contains_any(lowered, ["total", "amount due", "balance due", "subtotal"]):
            focus.append("financial_summary")
        if _contains_any(lowered, ["terms", "payment", "due date"]):
            focus.append("terms_conditions")
        if _contains_any(lowered, ["bill to", "ship to", "vendor", "supplier", "customer"]):
            focus.append("parties_addresses")

    if domain_hint == "tax":
        if _contains_any(lowered, ["taxpayer", "assessee", "taxpayer identity"]):
            focus.append("taxpayer_identity")
        if _contains_any(lowered, ["id", "ids", "pan", "ein", "ssn", "tin"]):
            focus.append("ids")
        if _contains_any(lowered, ["total", "totals", "tax liability", "amount due"]):
            focus.append("totals")
        if _contains_any(lowered, ["deduction", "deductions", "exemption"]):
            focus.append("deductions")
        if _contains_any(lowered, ["ay", "fy", "assessment year", "fiscal year", "tax year"]):
            focus.append("ay_fy")
        if _contains_any(lowered, ["payment", "payments", "refund"]):
            focus.append("payments")

    if domain_hint == "bank_statement":
        if _contains_any(lowered, ["account", "routing", "account number"]):
            focus.append("account_identity")
        if _contains_any(lowered, ["transactions", "transaction", "debit", "credit"]):
            focus.append("transactions")
        if _contains_any(lowered, ["balance", "opening balance", "closing balance"]):
            focus.append("balances")
        if _contains_any(lowered, ["fee", "fees", "charges", "interest"]):
            focus.append("fees")

    seen = set()
    return [f for f in focus if not (f in seen or seen.add(f))]


def _detect_scope(query: str, session_state: Dict[str, Any], target_doc_ids: List[str]) -> str:
    lowered = _normalize(query)
    if target_doc_ids:
        return "current_document"
    if "this document" in lowered or "current document" in lowered:
        return "current_document"
    return "profile_all_docs"


@dataclass
class DeterministicRoute:
    task_type: str
    domain_hint: str
    scope: str
    output_format: str
    section_focus: List[str] = field(default_factory=list)
    target_person: Optional[str] = None
    target_document_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "domain_hint": self.domain_hint,
            "scope": self.scope,
            "output_format": self.output_format,
            "section_focus": self.section_focus,
            "target_person": self.target_person,
            "target_document_ids": self.target_document_ids,
            "reasons": self.reasons,
        }


def route_query(
    query: str,
    session_state: Optional[Dict[str, Any]],
    catalog: Optional[Dict[str, Any]],
    entities_cache: Optional[Dict[str, Any]] = None,
) -> DeterministicRoute:
    session_state = session_state or {}
    catalog = catalog or {}
    entities_cache = entities_cache or {}

    task_type = _detect_task_type(query)
    domain_hint = _infer_domain(query, session_state, catalog)
    section_focus = _detect_section_focus(query, domain_hint)
    output_format = _detect_output_format(query, task_type)

    target_person = _extract_person_from_query(query)
    cache_person, target_doc_ids = _match_entity_cache(query, entities_cache)
    if not target_person and cache_person:
        target_person = cache_person

    explicit_doc_ids = _match_document_names(query, catalog)
    if explicit_doc_ids:
        target_doc_ids = list(set(target_doc_ids + explicit_doc_ids))

    scope = _detect_scope(query, session_state, target_doc_ids)
    reasons = []
    if target_doc_ids:
        reasons.append("entity_cache_match")
    if section_focus:
        reasons.append("section_focus_match")

    return DeterministicRoute(
        task_type=task_type,
        domain_hint=domain_hint or "unknown",
        scope=scope,
        output_format=output_format,
        section_focus=section_focus,
        target_person=target_person,
        target_document_ids=target_doc_ids,
        reasons=reasons,
    )


__all__ = ["DeterministicRoute", "route_query"]
