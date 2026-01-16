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
    intent: str = Field(..., description="Purpose of the email")
    recipient_role: str = Field(..., description="Who will receive the email")
    tone: str = Field(default="professional")
    constraints: Optional[List[str]] = Field(default=None, description="Specific constraints or bullet requirements")
    context: Optional[Dict[str, Any]] = None
    text: Optional[str] = Field(default=None, description="Reference text to ground the draft")


def _build_email(request: EmailDraftRequest) -> Dict[str, Any]:
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


@register_tool("email_drafting")
async def email_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = EmailDraftRequest(**(payload.get("input") or payload))
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

