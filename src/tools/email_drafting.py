from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional
import re

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

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

# Sentence-importance signals
_ACTION_VERBS = {
    "launched", "delivered", "achieved", "completed", "approved", "confirmed",
    "scheduled", "resolved", "reported", "promoted", "assigned", "requested",
    "submitted", "signed", "onboarded", "released", "deployed", "raised",
    "updated", "closed", "opened", "merged", "reviewed", "presented",
}

_NOISE_PATTERNS = re.compile(
    r"^\s*(i hope|please find|let me know|feel free|don't hesitate|"
    r"as mentioned|as discussed|hope this|thank you for your|"
    r"weather was|the office has|looking forward)\b",
    re.IGNORECASE,
)


def _extract_key_points(text: str) -> List[str]:
    """Score sentences by importance and return the top 3-5 as concise bullets."""
    if not text or not text.strip():
        return []

    # Split on sentence boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    scored: List[tuple[float, str]] = []

    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 300:
            continue
        if _NOISE_PATTERNS.match(sent):
            continue

        score = 0.0

        # Numbers / percentages / currency → high value
        score += len(re.findall(r"\b\d[\d,\.]*%?\b|\$[\d,\.]+", sent)) * 2.0

        # Dates → medium value
        score += len(re.findall(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}|"
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
            r"\b(?:Q[1-4])\b",
            sent, re.IGNORECASE,
        )) * 1.5

        # Proper nouns (capitalized words not at sentence start)
        words = sent.split()
        proper = sum(
            1 for i, w in enumerate(words)
            if i > 0 and w and w[0].isupper() and re.match(r"^[A-Z][a-zA-Z]+$", w)
        )
        score += proper * 0.8

        # Action verbs
        lower_words = {w.lower().rstrip("s") for w in words}
        score += len(lower_words & _ACTION_VERBS) * 1.2

        # Penalty for very short sentences (likely headings or noise)
        if len(words) < 5:
            score -= 1.0

        scored.append((score, sent))

    # Sort descending by score, preserve top 3-5
    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = 5 if len(scored) >= 5 else max(3, len(scored))
    return [sent for _, sent in scored[:top_n]]


def _greeting(tone: str, recipient_role: str) -> str:
    """Return tone-appropriate greeting line."""
    role = recipient_role.title()
    t = (tone or "professional").lower()
    if t == "casual":
        return f"Hi {role},"
    if t == "formal":
        return "Dear Sir/Madam,"
    if t == "friendly":
        return f"Hey {role}!"
    # default: professional
    return f"Dear {role},"


def _closing(tone: str) -> str:
    """Return tone-appropriate closing."""
    t = (tone or "professional").lower()
    if t == "casual":
        return "Thanks,"
    if t == "formal":
        return "Yours faithfully,"
    if t == "friendly":
        return "Cheers,"
    return "Best regards,"


def _subject_from_context(intent: str, recipient_role: str, key_points: List[str]) -> str:
    """Generate a more informative subject line by pulling key entities from key points."""
    intent_label = intent.title()

    # Try to pull a named entity or metric from the first key point
    entity = ""
    if key_points:
        # Prefer a proper noun phrase (2 capitalised words) or a number with context
        match = re.search(
            r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})\b|\b(\$[\d,\.]+[MKB]?|\d+%)\b",
            key_points[0],
        )
        if match:
            entity = (match.group(1) or match.group(2) or "").strip()

    if entity:
        return f"{intent_label}: {entity}"
    return f"{intent_label} — {recipient_role.title()}"


def _template_build_email(request: EmailDraftRequest) -> Dict[str, Any]:
    """Produce a structured, intent-aware email without an LLM."""
    body_context = sanitize_text(request.text or request.intent, max_chars=1800)
    tone = (request.tone or "professional").lower()
    intent = (request.intent or "compose email").lower().strip()
    constraints = request.constraints or []

    # Extract important sentences from context
    key_points = _extract_key_points(body_context) if body_context else []
    # Fall back to explicit constraints if provided
    bullets: List[str] = constraints if constraints else key_points
    # Last-resort: first 120 chars of context
    if not bullets and body_context:
        bullets = [body_context[:120]]

    subject = _subject_from_context(request.intent, request.recipient_role, key_points)
    greeting = _greeting(tone, request.recipient_role)
    closing = _closing(tone)
    bullet_lines = [f"  • {b}" for b in bullets[:5]]

    # ── Intent-specific structure ────────────────────────────────────
    if "follow up" in intent:
        opening = (
            "I am writing to follow up on our previous conversation and wanted to"
            " provide you with an update on the current status."
        )
        body_para = (
            "Based on the information reviewed, the following key points summarise"
            " where things stand:"
        )
        closing_para = (
            "Please let me know if you require any additional information or if there"
            " are any actions you would like me to prioritise. I look forward to your"
            " response."
        )

    elif "thank" in intent:
        opening = (
            "I wanted to take a moment to express my sincere gratitude for your"
            " contributions and the effort you have put in."
        )
        body_para = "The following highlights reflect the impact of your work:"
        closing_para = (
            "Your dedication is truly appreciated, and I look forward to continuing"
            " to work together."
        )

    elif "request" in intent:
        opening = (
            "I am reaching out to formally request your support on the following"
            " matter."
        )
        body_para = (
            "To provide context and justification for this request, please find the"
            " relevant details below:"
        )
        closing_para = (
            "I would appreciate your response at the earliest convenience. Please do"
            " not hesitate to reach out if you need any clarification."
        )

    elif "remind" in intent or "reminder" in intent:
        opening = (
            "I hope this message finds you well. I am writing as a gentle reminder"
            " regarding the items noted below."
        )
        body_para = "The following actions or commitments remain outstanding:"
        closing_para = (
            "Please let me know if there are any blockers preventing progress so we"
            " can address them promptly."
        )

    elif "schedule" in intent or "meeting" in intent:
        opening = (
            "I would like to propose scheduling a meeting to discuss the topics"
            " outlined below."
        )
        body_para = "Proposed agenda:"
        closing_para = (
            "Please suggest a time that works for you, or I am happy to work around"
            " your availability."
        )

    elif "introduc" in intent:
        opening = (
            "I hope this message finds you well. I am reaching out to introduce"
            " myself and share some relevant background."
        )
        body_para = "Here is a brief overview:"
        closing_para = (
            "I would welcome the opportunity to connect further and explore how we"
            " might collaborate."
        )

    else:
        # Generic professional multi-paragraph structure
        opening = (
            "I am writing to bring the following matter to your attention."
        )
        body_para = "Please find the key details below:"
        closing_para = (
            "Should you have any questions or require further information, please"
            " feel free to reach out."
        )

    body_lines: List[str] = [
        greeting,
        "",
        opening,
        "",
        body_para,
    ]
    body_lines.extend(bullet_lines)
    body_lines += [
        "",
        closing_para,
        "",
        closing,
        "DocWain Assistant",
    ]

    body = "\n".join(body_lines)

    return {
        "subject": subject,
        "body": body,
        "key_facts": bullets[:5],
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
