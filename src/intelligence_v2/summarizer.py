"""Document summarizer -- LLM-powered deep analysis at ingestion time."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.intelligence_v2.prompts import build_analysis_prompt

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Structured output from LLM document analysis."""

    document_type: str = "unknown"
    language: str = "en"
    summary: str = ""
    section_summaries: Dict[str, str] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    answerable_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from LLM output.

    Handles:
    - Bare JSON
    - Markdown ```json ... ``` fenced blocks
    - First { ... } brace match as last resort
    """
    # 1. Try markdown fenced block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Try bare JSON parse
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Brace match fallback
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def _fallback_result(text: str, filename: str, doc_type: str) -> AnalysisResult:
    """Minimal result when LLM analysis fails."""
    return AnalysisResult(
        document_type=doc_type or "unknown",
        language="en",
        summary=text[:500],
        section_summaries={},
        entities=[],
        facts=[],
        relationships=[],
        answerable_topics=[],
    )


class DocumentSummarizer:
    """Analyse documents via LLM and return structured metadata."""

    def __init__(self, llm_gateway: Any) -> None:
        self.llm = llm_gateway

    def analyze(
        self, text: str, filename: str, doc_type: str = "general"
    ) -> AnalysisResult:
        """Run deep LLM analysis on document text.

        Returns an :class:`AnalysisResult` -- never raises on LLM or
        parsing failures; falls back to a minimal result instead.
        """
        prompt = build_analysis_prompt(text, filename, doc_type)

        try:
            # Use think=False for analysis: thinking tokens consume from
            # num_predict budget, leaving no room for the JSON response.
            # Keep default num_ctx (8192) and truncate text to fit.
            raw_response, _meta = self.llm.generate_with_metadata(
                prompt, think=False, temperature=0.1, max_tokens=4096,
            )
            logger.info(
                "[SUMMARIZER] LLM response length=%d for %s, first 500 chars: %s",
                len(raw_response), filename, raw_response[:500],
            )
        except Exception:
            logger.warning(
                "LLM call failed for %s; returning fallback result", filename,
                exc_info=True,
            )
            return _fallback_result(text, filename, doc_type)

        parsed = _extract_json(raw_response)
        if parsed is None:
            logger.warning(
                "Could not parse JSON from LLM response for %s; returning fallback. "
                "Raw response (first 1000 chars): %s",
                filename, raw_response[:1000],
            )
            return _fallback_result(text, filename, doc_type)

        return AnalysisResult(
            document_type=parsed.get("document_type", doc_type),
            language=parsed.get("language", "en"),
            summary=parsed.get("summary", text[:500]),
            section_summaries=parsed.get("section_summaries", {}),
            entities=parsed.get("entities", []),
            facts=parsed.get("facts", []),
            relationships=parsed.get("relationships", []),
            answerable_topics=parsed.get("answerable_topics", []),
        )
