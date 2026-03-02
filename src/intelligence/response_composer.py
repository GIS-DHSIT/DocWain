from __future__ import annotations

from typing import Any, Dict, List, Optional


_DOMAIN_PROMPTS = {
    "resume": [
        "Summarize the candidate's experience from the resumes in this profile.",
        "List the technical skills mentioned in the resumes.",
    ],
    "invoice": [
        "What is the total amount due on the latest invoice?",
        "List the line items and amounts on the invoice.",
    ],
    "purchase_order": [
        "List the line items and quantities in the purchase order.",
        "What are the payment terms on the purchase order?",
    ],
    "medical": [
        "Summarize the patient's diagnoses and medications.",
        "Pull the doctor’s notes from the medical records.",
    ],
    "tax": [
        "What is the total tax due on this return?",
        "List the deductions reported in these tax documents.",
    ],
    "bank_statement": [
        "List the most recent transactions on the statement.",
        "What are the opening and closing balances?",
    ],
}


def _top_domains(catalog: Dict[str, Any], *, limit: int = 3) -> List[str]:
    dominant = catalog.get("dominant_domains") or {}
    ranked = sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)
    domains = [name for name, _count in ranked if name]
    return domains[:limit]


def build_greeting_response(catalog: Optional[Dict[str, Any]] = None) -> str:
    try:
        from src.intelligence.conversational_nlp import generate_conversational_response
        resp = generate_conversational_response("hi", catalog=catalog)
        if resp and resp.text:
            return resp.text
    except Exception:
        pass
    return (
        "Hi! I am DocWain-Agent - a document intelligence assistant. "
        "I answer questions using the documents in your current profile."
    )


def generate_ack(
    *,
    query: str,
    task_type: str,
    domain_hint: Optional[str] = None,
    section_focus: Optional[List[str]] = None,
    target_entity: Optional[str] = None,
) -> str:
    _ = (query, task_type, domain_hint, section_focus, target_entity)
    return ""


def compose_task_response(
    *,
    response_text: str,
    route_plan: Any,
    query: str,
) -> str:
    _ = generate_ack(
        query=query,
        task_type=route_plan.task_type,
        domain_hint=route_plan.domain_hint,
        section_focus=route_plan.section_focus,
        target_entity=route_plan.target_person,
    )
    return response_text or ""


__all__ = ["build_greeting_response", "generate_ack", "compose_task_response"]
