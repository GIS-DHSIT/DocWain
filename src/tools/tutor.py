from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tutor", tags=["Tools-Tutor"])


class TutorRequest(BaseModel):
    topic: Optional[str] = Field(default=None, description="Topic of interest")
    learning_level: str = Field(default="beginner", pattern="^(beginner|intermediate|advanced)$")
    text: Optional[str] = Field(default=None, description="Raw reference text")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Context metadata or sources")


def _derive_key_points(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return parts[:5] or [text[:120]]


def _build_quiz(points: List[str]) -> List[Dict[str, str]]:
    quiz = []
    for idx, point in enumerate(points[:5], start=1):
        quiz.append(
            {
                "question": f"Q{idx}: What does this mean? {point}",
                "answer": f"It refers to: {point}",
            }
        )
    return quiz


def _build_lesson(req: TutorRequest) -> Dict[str, Any]:
    topic = req.topic or "document topic"
    reference_text = sanitize_text(req.text or topic, max_chars=2000)
    key_points = _derive_key_points(reference_text)
    examples = [f"Example: {point}" for point in key_points[:3]]
    quiz = _build_quiz(key_points)
    explanation = f"{topic} for {req.learning_level} learners: {reference_text}"

    return {
        "topic": topic,
        "learning_level": req.learning_level,
        "explanation": explanation,
        "key_points": key_points,
        "examples": examples,
        "quiz": quiz,
    }


@register_tool("tutor")
async def tutor_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = TutorRequest(**(payload.get("input") or payload))
    lesson = _build_lesson(req)
    sources = [build_source_record("tool", correlation_id or "tutor", title=req.topic or "tutor")]
    return {
        "result": lesson,
        "sources": sources,
        "grounded": True,
        "context_found": bool(req.text or req.topic),
    }


@router.post("/lesson")
async def lesson(request: TutorRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    lesson_content = _build_lesson(request)
    sources = [build_source_record("tool", cid, title=request.topic or "tutor")]
    return standard_response(
        "tutor",
        grounded=True,
        context_found=True,
        result=lesson_content,
        sources=sources,
        warnings=[],
        correlation_id=cid,
    )

