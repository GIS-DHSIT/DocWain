from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/creator", tags=["Tools-Creator"])


class CreatorRequest(BaseModel):
    content_type: str = Field(..., pattern="^(summary|blog|sop|faq|slide_outline)$")
    tone: str = Field(default="neutral")
    length: str = Field(default="medium", description="rough size hint: short|medium|long")
    context: Optional[Dict[str, Any]] = None
    text: Optional[str] = Field(default=None, description="Reference text to ground generation")


# ── Type-specific instructions ──────────────────────────────────────

_TYPE_INSTRUCTIONS = {
    "summary": "Write a concise summary with key takeaways.",
    "blog": "Write a blog post with introduction, body sections, and conclusion.",
    "sop": "Write a standard operating procedure with numbered steps.",
    "faq": "Generate 5-8 FAQ questions and answers as JSON: [{\"q\": \"...\", \"a\": \"...\"}]",
    "slide_outline": "Create a slide deck outline with title + bullet points per slide.",
}

_EXPECTED_FIELDS = ["content"]


# ── LLM generation ──────────────────────────────────────────────────

def _llm_generate(content_type: str, tone: str, length: str, reference: str) -> Optional[Dict[str, Any]]:
    """LLM-powered content generation. Returns None on failure."""
    try:
        from src.tools.llm_tools import build_generation_prompt, tool_generate, tool_generate_structured

        type_instruction = _TYPE_INSTRUCTIONS.get(content_type, "Generate the requested content.")
        instructions = (
            f"Generate a {content_type.replace('_', ' ')} with these specifications:\n"
            f"- Tone: {tone}\n"
            f"- Length: {length}\n"
            f"- Ground ALL content in the reference material.\n\n"
            f"{type_instruction}"
        )
        prompt = build_generation_prompt("creator", instructions, reference)

        if content_type == "faq":
            result = tool_generate_structured(prompt, domain="creative")
            if result:
                faqs = result.get("faqs") or result.get("questions") or []
                if isinstance(result, dict) and not faqs:
                    # LLM may return a list at top level wrapped in a key
                    for key, val in result.items():
                        if isinstance(val, list) and val:
                            faqs = val
                            break
                return {"content": "", "faqs": faqs}
            return None

        text = tool_generate(prompt, domain="creative")
        if text:
            return {"content": text}
        return None
    except Exception as exc:
        logger.debug("Creator LLM generation failed: %s", exc)
        return None


# ── Template fallback ───────────────────────────────────────────────

def _build_outline(content: str) -> List[str]:
    if not content:
        return []
    sentences = [s.strip() for s in content.split(".") if s.strip()]
    return sentences[:6]


def _template_generate(req: CreatorRequest) -> Dict[str, Any]:
    reference = sanitize_text(req.text or "Provided context", max_chars=2400)
    header = f"{req.content_type.replace('_', ' ').title()} in a {req.tone} tone ({req.length})"
    outline = _build_outline(reference)
    body = f"{header}: {reference}"
    faqs: List[Dict[str, str]] = []
    if req.content_type == "faq":
        faqs = [{"q": f"What about {idx + 1}?", "a": item} for idx, item in enumerate(outline[:5])]
    return {
        "header": header,
        "outline": outline,
        "content": body,
        "faqs": faqs,
    }


# ── Unified generation ──────────────────────────────────────────────

def _generate_content(req: CreatorRequest) -> Dict[str, Any]:
    """Generate content using LLM first, falling back to template."""
    from src.tools.llm_tools import score_tool_response

    reference = sanitize_text(req.text or "Provided context", max_chars=2400)
    llm_result = _llm_generate(req.content_type, req.tone, req.length, reference)

    if llm_result and llm_result.get("content") or (llm_result and llm_result.get("faqs")):
        result = llm_result
        # Ensure all expected keys exist
        result.setdefault("header", f"{req.content_type.replace('_', ' ').title()} in a {req.tone} tone ({req.length})")
        result.setdefault("outline", [])
        result.setdefault("content", "")
        result.setdefault("faqs", [])
        iq = score_tool_response(result, domain="general", expected_fields=_EXPECTED_FIELDS, source="llm")
    else:
        result = _template_generate(req)
        iq = score_tool_response(result, domain="general", expected_fields=_EXPECTED_FIELDS, source="template")

    result["iq_score"] = iq.as_dict()
    return result


@register_tool("creator")
async def creator_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = CreatorRequest(**(payload.get("input") or payload))
    result = _generate_content(req)
    sources = [build_source_record("tool", correlation_id or "creator", title=req.content_type)]
    return {"result": result, "sources": sources, "grounded": True, "context_found": True}


@router.post("/generate")
async def generate(request: CreatorRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    result = _generate_content(request)
    sources = [build_source_record("tool", cid, title=request.content_type)]
    return standard_response(
        "creator",
        grounded=True,
        context_found=True,
        result=result,
        sources=sources,
        warnings=[],
        correlation_id=cid,
    )
