from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/tutor", tags=["Tools-Tutor"])

class TutorRequest(BaseModel):
    topic: Optional[str] = Field(default=None, description="Topic of interest")
    learning_level: str = Field(default="beginner", pattern="^(beginner|intermediate|advanced)$")
    text: Optional[str] = Field(default=None, description="Raw reference text")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Context metadata or sources")

def _derive_key_points(text: str) -> List[str]:
    """Derive key points using LLM first, regex fallback."""
    if text and len(text) > 50:
        try:
            from src.tools.llm_tools import tool_generate
            prompt = (
                f"Extract 3-5 key learning points from this text. "
                f"Return ONLY a numbered list, one point per line.\n\n{text[:2000]}"
            )
            result = tool_generate(prompt, domain="education")
            if result and len(result) > 20:
                points = [p.strip().lstrip("0123456789.-) ") for p in result.split("\n") if p.strip()]
                if len(points) >= 2:
                    return points[:5]
        except Exception:
            pass
    # Regex fallback
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return parts[:5] or [text[:120]]

def _build_quiz(points: List[str]) -> List[Dict[str, str]]:
    """Build quiz questions using LLM first, template fallback."""
    if points and len(points) >= 2:
        try:
            from src.tools.llm_tools import tool_generate
            material = "\n".join(f"- {p}" for p in points[:5])
            prompt = (
                f"Create {min(len(points), 5)} quiz questions based on these key points. "
                f"For each, provide a question and a brief correct answer.\n"
                f"Format: Q1: [question]\\nA1: [answer]\\n\\n\n\nKey points:\n{material}"
            )
            result = tool_generate(prompt, domain="education")
            if result and len(result) > 30:
                quiz = []
                pairs = re.findall(r'Q\d+:\s*(.+?)(?:\n|$)\s*A\d+:\s*(.+?)(?:\n|$)', result, re.IGNORECASE)
                for q, a in pairs[:5]:
                    quiz.append({"question": q.strip(), "answer": a.strip()})
                if quiz:
                    return quiz
        except Exception:
            pass
    # Template fallback
    quiz = []
    for idx, point in enumerate(points[:5], start=1):
        quiz.append({"question": f"Q{idx}: What does this mean? {point}", "answer": f"It refers to: {point}"})
    return quiz

def _llm_build_lesson(topic: str, text: str, level: str) -> Optional[Dict[str, Any]]:
    """LLM-powered lesson builder. Returns None on failure."""
    try:
        from src.tools.llm_tools import tool_generate
        prompt = (
            f"Create a structured lesson on '{topic}' at the {level} level.\n\n"
            f"Use the following source material:\n{text[:2000]}\n\n"
            f"Structure:\n"
            f"1. Key Learning Points (3-5 bullet points)\n"
            f"2. Detailed Explanation with examples from the material\n"
            f"3. Quiz (3 questions with answers)\n\n"
            f"Make it engaging and educational."
        )
        result = tool_generate(prompt, domain="education")
        if not result or len(result) < 50:
            return None
        key_points = _derive_key_points(text)
        quiz = _build_quiz(key_points)
        return {
            "topic": topic,
            "learning_level": level,
            "explanation": result,
            "key_points": key_points,
            "examples": [f"Example: {p}" for p in key_points[:3]],
            "quiz": quiz,
            "rendered": _render_lesson(topic, level, key_points, quiz, llm_explanation=result),
        }
    except Exception as exc:
        logger.debug("LLM lesson build failed: %s", exc)
        return None

def _render_lesson(topic: str, level: str, key_points: List[str], quiz: List[Dict[str, str]], llm_explanation: str = "") -> str:
    """Render lesson into markdown for pipeline pre-rendering."""
    parts: List[str] = []
    parts.append(f"**Lesson: {topic}** (Level: {level})")
    if key_points:
        parts.append("**Key Points:**")
        for kp in key_points[:5]:
            parts.append(f"- {kp}")
    if llm_explanation:
        parts.append(f"\n**Explanation:**\n{llm_explanation[:1500]}")
    if quiz:
        parts.append("\n**Review Questions:**")
        for q in quiz[:3]:
            parts.append(f"- {q.get('question', '')}")
    return "\n".join(parts)

def _build_lesson(req: TutorRequest) -> Dict[str, Any]:
    """Build a lesson using LLM-powered generation with regex fallback."""
    topic = req.topic or "document topic"
    reference_text = sanitize_text(req.text or topic, max_chars=2000)

    # Try LLM-powered lesson generation
    llm_lesson = _llm_build_lesson(topic, reference_text, req.learning_level)
    if llm_lesson:
        return llm_lesson

    # Fallback to template-based
    key_points = _derive_key_points(reference_text)
    examples = [f"Example: {point}" for point in key_points[:3]]
    quiz = _build_quiz(key_points)
    explanation = f"{topic} for {req.learning_level} learners: {reference_text}"
    rendered = _render_lesson(topic, req.learning_level, key_points, quiz)

    return {
        "topic": topic,
        "learning_level": req.learning_level,
        "explanation": explanation,
        "key_points": key_points,
        "examples": examples,
        "quiz": quiz,
        "rendered": rendered,
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

