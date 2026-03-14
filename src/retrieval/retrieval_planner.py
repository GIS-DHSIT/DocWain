from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import os
import re
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.intelligence.domain_indexer import infer_domain
from src.prompting.retrieval_planner import build_retrieval_planner_prompt

logger = get_logger(__name__)

_INTENTS = {"greet", "extract", "summarize", "compare", "rank", "generate", "list_filter"}
_DOMAINS = {
    "resume",
    "medical",
    "invoice",
    "tax",
    "bank_statement",
    "purchase_order",
    "generic",
}
_DOMAIN_ALIASES = {
    "bank": "bank_statement",
    "bankstatement": "bank_statement",
    "bank-statement": "bank_statement",
    "purchaseorder": "purchase_order",
    "purchase-order": "purchase_order",
    "po": "purchase_order",
    "cv": "resume",
}

_INTERNAL_ID_RE = re.compile(
    r"\b([0-9a-fA-F]{24}|[0-9a-fA-F]{32,}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})\b"
)

def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except Exception:
            return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def _normalize_domain(value: Optional[str]) -> str:
    if not value:
        return "generic"
    cleaned = str(value).strip().lower().replace(" ", "_")
    cleaned = _DOMAIN_ALIASES.get(cleaned, cleaned)
    return cleaned if cleaned in _DOMAINS else "generic"

def _normalize_intent(value: Optional[str]) -> str:
    if not value:
        return "extract"
    cleaned = str(value).strip().lower().replace(" ", "_")
    if cleaned in {"list", "list_filter"}:
        return "list_filter"
    if cleaned in {"summary", "summarize", "summarization"}:
        return "summarize"
    if cleaned in {"comparison", "compare"}:
        return "compare"
    if cleaned in {"ranking", "rank"}:
        return "rank"
    if cleaned in {"generate", "draft", "create", "write"}:
        return "generate"
    if cleaned in {"greet", "hello", "hi"}:
        return "greet"
    if cleaned in {"extract", "find", "identify"}:
        return "extract"
    return "extract" if cleaned not in _INTENTS else cleaned

def _detect_intent(query: str) -> str:
    lowered = (query or "").lower()
    if re.match(r"^\s*(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b", lowered):
        return "greet"
    if any(token in lowered for token in ("compare", "comparison", "vs", "versus")):
        return "compare"
    if any(token in lowered for token in ("rank", "ranking", "top", "best", "most")):
        return "rank"
    if any(token in lowered for token in ("summarize", "summary", "overview", "recap", "highlights")):
        return "summarize"
    if any(token in lowered for token in ("generate", "draft", "create", "write", "compose")):
        return "generate"
    if any(token in lowered for token in ("list", "show", "filter")):
        return "list_filter"
    if any(token in lowered for token in ("extract", "identify", "find", "pull")):
        return "extract"
    return "extract"

def _extract_person_name(query: str) -> Optional[str]:
    if not query:
        return None
    patterns = [
        r"\b(?:education|skills?|experience|background|qualification|details?)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\'’]s\s+(?:education|skills?|experience)",
        r"\btell\s+me\s+about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"\b(?:about|for|regarding)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"\b(?:patient|vendor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def _preferred_sections(domain: str, intent: str) -> List[str]:
    if domain == "resume":
        sections = [
            "identity",
            "summary_objective",
            "experience",
            "skills",
            "education",
            "certifications",
            "projects",
        ]
    elif domain == "medical":
        sections = [
            "identity_contact",
            "encounter",
            "diagnoses_procedures",
            "medications",
            "lab_results",
            "notes",
        ]
    elif domain == "invoice":
        sections = [
            "parties_addresses",
            "invoice_metadata",
            "financial_summary",
            "line_items",
            "terms_conditions",
        ]
    elif domain == "tax":
        sections = [
            "taxpayer_identity",
            "tax_year",
            "income",
            "deductions",
            "payments",
            "totals",
        ]
    elif domain == "bank_statement":
        sections = [
            "account_identity",
            "period",
            "balances",
            "transactions",
            "fees",
        ]
    elif domain == "purchase_order":
        sections = [
            "parties_addresses",
            "po_metadata",
            "items",
            "terms_conditions",
            "approvals",
        ]
    else:
        sections = ["identity", "summary", "key_facts", "notable_items"]

    if intent in {"compare", "rank"} and "comparison" not in sections:
        sections.append("comparison")
    return sections

def _response_sections(domain: str, intent: str) -> List[str]:
    if domain == "resume":
        sections = [
            "Candidate overview",
            "Experience summary",
            "Skills (technical/functional)",
            "Education",
            "Certifications",
            "Projects/Achievements",
            "Highlights",
        ]
    elif domain == "medical":
        sections = [
            "Identity & demographics",
            "Encounter/visit",
            "Diagnoses/findings",
            "Medications/allergies",
            "Procedures/tests",
            "Doctor notes summary",
        ]
    elif domain == "invoice":
        sections = [
            "Vendor & recipient",
            "Invoice metadata",
            "Financial summary",
            "Line items summary",
            "Payment terms",
            "Flags/anomalies",
        ]
    elif domain == "tax":
        sections = [
            "Taxpayer identity",
            "Tax year",
            "Income",
            "Deductions",
            "Tax payable/refund",
            "Filing status",
            "Flags",
        ]
    elif domain == "bank_statement":
        sections = [
            "Account identity",
            "Period",
            "Opening/closing balances",
            "Transaction summary",
            "Large/flagged transactions",
            "Notes",
        ]
    elif domain == "purchase_order":
        sections = [
            "Buyer/supplier",
            "PO metadata",
            "Items summary",
            "Delivery/payment terms",
            "Approvals/status",
            "Flags",
        ]
    else:
        sections = [
            "Identity",
            "Key facts",
            "Summary",
            "Notable items",
            "Gaps",
        ]

    if intent in {"compare", "rank"} and "Comparison/Ranking" not in sections:
        sections.append("Comparison/Ranking")
    return sections

def _sanitize_file_name(value: str) -> Optional[str]:
    if not value:
        return None
    base = os.path.basename(str(value)).strip()
    if not base:
        return None
    if _INTERNAL_ID_RE.search(base):
        return None
    return base

def _build_evidence_requests(
    *,
    target_files: List[str],
    available_documents: List[Dict[str, Any]],
    sections: List[str],
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    files = target_files[:] if target_files else [doc.get("file_name") for doc in available_documents or []]
    for name in files:
        safe_name = _sanitize_file_name(name)
        if not safe_name:
            continue
        requests.append(
            {
                "file_name": safe_name,
                "preferred_sections": sections,
                "max_excerpts": 6,
                "excerpt_chars": 600,
                "require_page_numbers": True,
            }
        )
    return requests

def _build_default_plan(
    *,
    user_query: str,
    subscription_id: str,
    profile_id: str,
    target_document_name: Optional[str],
    available_documents: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    intent = _detect_intent(user_query)
    inferred_domain = _normalize_domain(infer_domain(user_query))
    name = _extract_person_name(user_query)
    target_files: List[str] = []
    if target_document_name:
        target_files = [_sanitize_file_name(target_document_name) or target_document_name]

    sections = _preferred_sections(inferred_domain, intent)
    evidence_requests = _build_evidence_requests(
        target_files=target_files,
        available_documents=available_documents or [],
        sections=sections,
    )
    scope_level = "document" if target_files else "profile"

    return {
        "intent": intent,
        "domain": inferred_domain,
        "scope": {
            "level": scope_level,
            "target_files": [f for f in target_files if f],
            "rules": [
                "Scope restricted to subscription_id + profile_id",
                "Never cross-profile retrieval",
                "Never include internal IDs in the plan",
            ],
        },
        "name_query": {
            "enabled": bool(name),
            "name": name if name else None,
            "resolution_strategy": ["exact", "case_insensitive", "fuzzy_filename", "fuzzy_text"],
            "must_restrict_to_matching_files": bool(name),
        },
        "retrieval": {
            "filters": {
                "subscription_id": "REQUIRED",
                "profile_id": "REQUIRED",
                "doc_domain": inferred_domain if inferred_domain != "generic" else "optional",
            },
            "top_k": 40,
            "rerank": True,
            "context_budget_chars": 25000,
            "must_not_fallback_to_unfiltered": True,
        },
        "evidence_requests": evidence_requests,
        "response_template": {
            "understanding_line": "Planned retrieval and response grounded in the current profile scope.",
            "sections": _response_sections(inferred_domain, intent),
            "include_evidence_and_gaps": True,
            "optional_next_hint": True,
        },
    }

def _coerce_plan(
    payload: Dict[str, Any],
    *,
    user_query: str,
    subscription_id: str,
    profile_id: str,
    target_document_name: Optional[str],
    available_documents: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    fallback = _build_default_plan(
        user_query=user_query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        target_document_name=target_document_name,
        available_documents=available_documents,
    )
    if not isinstance(payload, dict):
        return fallback

    intent = _normalize_intent(payload.get("intent")) or fallback["intent"]
    domain = _normalize_domain(payload.get("domain")) or fallback["domain"]

    scope_payload = payload.get("scope") or {}
    scope_level = str(scope_payload.get("level") or fallback["scope"]["level"]).lower()
    scope_level = "document" if scope_level == "document" else "profile"
    target_files = [
        _sanitize_file_name(name)
        for name in (scope_payload.get("target_files") or [])
        if _sanitize_file_name(name)
    ]

    name_payload = payload.get("name_query") or {}
    name_value = name_payload.get("name")
    if name_value:
        name_value = str(name_value).strip() or None
    name_enabled = bool(name_payload.get("enabled")) or bool(name_value)
    resolution_strategy = [
        strat
        for strat in (name_payload.get("resolution_strategy") or [])
        if str(strat) in {"exact", "case_insensitive", "fuzzy_filename", "fuzzy_text"}
    ]
    if not resolution_strategy:
        resolution_strategy = ["exact", "case_insensitive", "fuzzy_filename", "fuzzy_text"]

    retrieval_payload = payload.get("retrieval") or {}
    filters_payload = retrieval_payload.get("filters") or {}
    doc_domain = _normalize_domain(filters_payload.get("doc_domain")) if filters_payload.get("doc_domain") else ""
    doc_domain = doc_domain if doc_domain and doc_domain != "generic" else "optional"
    try:
        top_k = int(retrieval_payload.get("top_k") or fallback["retrieval"]["top_k"])
    except Exception:
        top_k = fallback["retrieval"]["top_k"]
    top_k = max(5, min(top_k, 100))
    try:
        context_budget = int(retrieval_payload.get("context_budget_chars") or fallback["retrieval"]["context_budget_chars"])
    except Exception:
        context_budget = fallback["retrieval"]["context_budget_chars"]

    sections = _preferred_sections(domain, intent)
    evidence_requests = payload.get("evidence_requests")
    if not isinstance(evidence_requests, list) or not evidence_requests:
        evidence_requests = _build_evidence_requests(
            target_files=target_files,
            available_documents=available_documents or [],
            sections=sections,
        )

    response_template = payload.get("response_template") or {}
    response_sections = response_template.get("sections")
    if not isinstance(response_sections, list) or not response_sections:
        response_sections = _response_sections(domain, intent)

    return {
        "intent": intent,
        "domain": domain,
        "scope": {
            "level": scope_level,
            "target_files": target_files,
            "rules": scope_payload.get("rules") or fallback["scope"]["rules"],
        },
        "name_query": {
            "enabled": name_enabled,
            "name": name_value,
            "resolution_strategy": resolution_strategy,
            "must_restrict_to_matching_files": bool(name_payload.get("must_restrict_to_matching_files")) or bool(name_value),
        },
        "retrieval": {
            "filters": {
                "subscription_id": "REQUIRED",
                "profile_id": "REQUIRED",
                "doc_domain": doc_domain,
            },
            "top_k": top_k,
            "rerank": bool(retrieval_payload.get("rerank", True)),
            "context_budget_chars": context_budget,
            "must_not_fallback_to_unfiltered": bool(
                retrieval_payload.get("must_not_fallback_to_unfiltered", True)
            ),
        },
        "evidence_requests": evidence_requests,
        "response_template": {
            "understanding_line": response_template.get("understanding_line")
            or fallback["response_template"]["understanding_line"],
            "sections": response_sections,
            "include_evidence_and_gaps": bool(
                response_template.get("include_evidence_and_gaps", True)
            ),
            "optional_next_hint": bool(response_template.get("optional_next_hint", True)),
        },
    }

class RetrievalPlanner:
    def __init__(self, llm_client: Optional[Any] = None, *, enabled: Optional[bool] = None) -> None:
        self.llm_client = llm_client
        if enabled is None:
            enabled = bool(getattr(Config.Retrieval, "RETRIEVAL_PLANNER_ENABLED", True))
        self.enabled = enabled

    def plan(
        self,
        *,
        user_query: str,
        subscription_id: str,
        profile_id: str,
        target_document_name: Optional[str] = None,
        available_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not profile_id:
            return _build_default_plan(
                user_query=user_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                target_document_name=target_document_name,
                available_documents=available_documents,
            )

        if not self.enabled or not self.llm_client:
            return _build_default_plan(
                user_query=user_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                target_document_name=target_document_name,
                available_documents=available_documents,
            )

        prompt = build_retrieval_planner_prompt(
            user_query=user_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            target_document_name=target_document_name,
            available_documents=available_documents,
        )

        max_retries = int(getattr(Config.Retrieval, "RETRIEVAL_PLANNER_MAX_RETRIES", 2))
        backoff = float(getattr(Config.Retrieval, "RETRIEVAL_PLANNER_BACKOFF", 0.4))

        try:
            response = self.llm_client.generate(prompt, max_retries=max_retries, backoff=backoff)
            payload = _extract_json(response)
            if payload:
                return _coerce_plan(
                    payload,
                    user_query=user_query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    target_document_name=target_document_name,
                    available_documents=available_documents,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Retrieval planner failed: %s", exc)

        return _build_default_plan(
            user_query=user_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            target_document_name=target_document_name,
            available_documents=available_documents,
        )

__all__ = ["RetrievalPlanner"]
