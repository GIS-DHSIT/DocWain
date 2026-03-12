"""Screening-as-Tools Bridge.

Exposes screening capabilities as ToolRegistry entries so the RAG pipeline
and gateway can invoke screening tools the same way as regular tools.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional

from src.tools.base import register_tool

logger = get_logger(__name__)

def _get_engine():
    from src.screening.engine import ScreeningEngine
    return ScreeningEngine()

@register_tool("screen_pii")
def screen_pii_tool(payload: Dict[str, Any], *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Screen text for PII sensitivity."""
    text = (payload.get("input") or {}).get("text", "")
    if not text:
        return {"result": {}, "warnings": ["No text provided"]}
    engine = _get_engine()
    result_dict = engine.evaluate(text=text, doc_type=(payload.get("options") or {}).get("doc_type"))
    return {"result": result_dict, "sources": [], "grounded": True}

@register_tool("screen_ai_authorship")
def screen_ai_authorship_tool(payload: Dict[str, Any], *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Detect AI-generated content."""
    text = (payload.get("input") or {}).get("text", "")
    if not text:
        return {"result": {}, "warnings": ["No text provided"]}
    engine = _get_engine()
    result_dict = engine.evaluate(text=text, doc_type=(payload.get("options") or {}).get("doc_type"))
    return {"result": result_dict, "sources": [], "grounded": True}

@register_tool("screen_resume")
def screen_resume_tool(payload: Dict[str, Any], *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Run resume screening analysis on text."""
    text = (payload.get("input") or {}).get("text", "")
    doc_id = (payload.get("input") or {}).get("doc_id")
    if not text:
        return {"result": {}, "warnings": ["No text provided"]}
    engine = _get_engine()
    result = engine.resume_analysis_from_text(
        text=text,
        doc_id=doc_id,
        doc_type="RESUME",
        internet_enabled_override=(payload.get("options") or {}).get("internet_enabled"),
    )
    # Serialize pydantic model if needed
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif hasattr(result, "dict"):
        result = result.dict()
    return {"result": result, "sources": [], "grounded": True}

@register_tool("screen_readability")
def screen_readability_tool(payload: Dict[str, Any], *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Score text readability and style quality."""
    text = (payload.get("input") or {}).get("text", "")
    if not text:
        return {"result": {}, "warnings": ["No text provided"]}
    engine = _get_engine()
    result_dict = engine.evaluate(text=text, doc_type=(payload.get("options") or {}).get("doc_type"))
    return {"result": result_dict, "sources": [], "grounded": True}
