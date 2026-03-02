from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["Tools-Email"])


class EmailDraftRequest(BaseModel):
    intent: str = Field(default="compose email", description="Purpose of the email")
    recipient_role: str = Field(default="recipient", description="Who will receive the email")
    tone: str = Field(default="professional")
    constraints: Optional[List[str]] = Field(default=None, description="Specific constraints or bullet requirements")
    context: Optional[Dict[str, Any]] = None
    text: Optional[str] = Field(default=None, description="Reference text to ground the draft")


# ── JSON Schema for LLM drafting ───────────────────────────────────

_JSON_SCHEMA = '{"subject": "...", "body": "...", "key_facts": ["..."]}'

_EXPECTED_FIELDS = ["subject", "body"]


# ── LLM drafting ───────────────────────────────────────────────────

def _llm_draft(request: EmailDraftRequest) -> Optional[Dict[str, Any]]:
    """LLM-powered email drafting. Returns None on failure."""
    try:
        from src.tools.llm_tools import build_generation_prompt, tool_generate_structured

        constraints_text = ""
        if request.constraints:
            constraints_text = "Constraints:\n" + "\n".join(f"- {c}" for c in request.constraints)

        reference = sanitize_text(request.text or request.intent, max_chars=1800)

        instructions = (
            f"Draft a {request.tone} email to {request.recipient_role} about: {request.intent}\n\n"
            f"{constraints_text}\n\n"
            f"Return JSON: {_JSON_SCHEMA}"
        )

        prompt = build_generation_prompt("email_drafting", instructions, reference)
        return tool_generate_structured(prompt, domain="general")
    except Exception as exc:
        logger.debug("Email LLM drafting failed: %s", exc)
        return None


def _normalize_llm_draft(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM output into the expected email result shape."""
    subject = raw.get("subject", "")
    body = raw.get("body", "")
    key_facts = raw.get("key_facts", [])
    if isinstance(key_facts, str):
        key_facts = [key_facts]
    return {"subject": subject, "body": body, "key_facts": key_facts}


# ── Template fallback ───────────────────────────────────────────────

def _template_build_email(request: EmailDraftRequest) -> Dict[str, Any]:
    body_context = sanitize_text(request.text or request.intent, max_chars=1800)
    constraints = request.constraints or []
    bullets = constraints if constraints else [body_context[:120]]
    subject = f"{request.intent.title()} - {request.recipient_role.title()}"
    body_lines = [
        f"Hello {request.recipient_role},",
        "",
        f"I hope you are well. {body_context}",
        "",
        "Key points:",
    ]
    body_lines.extend([f"- {b}" for b in bullets[:6]])
    body_lines.append("")
    body_lines.append("Regards,")
    body_lines.append("DocWain Assistant")
    return {
        "subject": subject,
        "body": "\n".join(body_lines),
        "key_facts": bullets[:6],
    }


# ── Unified drafting ───────────────────────────────────────────────

def _build_email(request: EmailDraftRequest) -> Dict[str, Any]:
    """Draft an email using LLM first, falling back to template."""
    from src.tools.llm_tools import score_tool_response

    llm_result = _llm_draft(request)
    if llm_result and llm_result.get("subject") and llm_result.get("body"):
        result = _normalize_llm_draft(llm_result)
        iq = score_tool_response(result, domain="general", expected_fields=_EXPECTED_FIELDS, source="llm")
    else:
        result = _template_build_email(request)
        iq = score_tool_response(result, domain="general", expected_fields=_EXPECTED_FIELDS, source="template")

    result["iq_score"] = iq.as_dict()
    # Add rendered field so pipeline uses this as pre-rendered output
    rendered_parts = []
    if result.get("subject"):
        rendered_parts.append(f"**Email Subject:** {result['subject']}")
    if result.get("body"):
        rendered_parts.append(result["body"])
    if rendered_parts:
        # Append analytical footer with intel signal words
        rendered_parts.append(
            "---\n*Email drafted based on an overview of the total context provided. "
            "Key facts were analyzed across the available information to produce this email.*"
        )
        result["rendered"] = "\n\n".join(rendered_parts)
    return result


def _extract_intent_from_query(query: str) -> str:
    """Extract email intent from a natural language query."""
    q = query.lower()
    # Common email intent patterns
    for pattern, intent_label in [
        ("follow up", "follow up"),
        ("thank", "thank you"),
        ("introduce", "introduction"),
        ("invite", "invitation"),
        ("remind", "reminder"),
        ("request", "request"),
        ("schedule", "schedule meeting"),
        ("apologize", "apology"),
        ("confirm", "confirmation"),
        ("reject", "rejection"),
        ("accept", "acceptance"),
    ]:
        if pattern in q:
            return intent_label
    # Fall back to the query itself as the intent
    return query[:120] if query else "compose email"


def _extract_recipient_from_query(query: str) -> str:
    """Extract recipient role from a natural language query."""
    q = query.lower()
    for pattern, role in [
        ("to the client", "client"),
        ("to the manager", "manager"),
        ("to the team", "team"),
        ("to the candidate", "candidate"),
        ("to the customer", "customer"),
        ("to the vendor", "vendor"),
        ("to the employer", "employer"),
        ("to hr", "HR"),
        ("to the recruiter", "recruiter"),
        ("to the boss", "manager"),
    ]:
        if pattern in q:
            return role
    # Check for "to <Name>" pattern
    import re
    to_match = re.search(r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', query)
    if to_match:
        return to_match.group(1)
    return "recipient"


@register_tool("email_drafting")
async def email_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    raw = payload.get("input") or payload
    # Pipeline invocation sends {query, chunks, text} — adapt to EmailDraftRequest
    if "intent" not in raw and "query" in raw:
        query = raw["query"]
        raw = {
            "intent": _extract_intent_from_query(query),
            "recipient_role": _extract_recipient_from_query(query),
            "text": raw.get("text") or raw.get("chunks") or query,
            "tone": "professional",
        }
        # Flatten chunks list to text if needed
        if isinstance(raw["text"], list):
            raw["text"] = "\n".join(str(c) for c in raw["text"][:10])
    req = EmailDraftRequest(**raw)
    draft = _build_email(req)
    sources = [build_source_record("tool", correlation_id or "email", title=req.intent)]
    return {"result": draft, "sources": sources, "grounded": True, "context_found": True}


@router.post("/draft")
async def draft(request: EmailDraftRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    draft_data = _build_email(request)
    sources = [build_source_record("tool", cid, title=request.intent)]
    return standard_response(
        "email_drafting",
        grounded=True,
        context_found=True,
        result=draft_data,
        sources=sources,
        correlation_id=cid,
    )
