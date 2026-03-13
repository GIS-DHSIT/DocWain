from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClarificationDecision:
    should_clarify: bool
    question: str = ""
    reason: str = ""


class ClarificationAgent:
    """Detect underspecified queries and craft clarification questions."""

    _PRONOUNS = {"this", "that", "it", "they", "those", "these", "he", "she", "him", "her", "them"}

    def decide(
        self,
        query: str,
        *,
        conversation_context: str = "",
        query_analysis: Optional[Dict[str, Any]] = None,
        profile_context: Optional[Dict[str, Any]] = None,
    ) -> ClarificationDecision:
        logger.debug("decide called with query_len=%s, has_context=%s, has_analysis=%s",
                     len(query or ""), bool(conversation_context), bool(query_analysis))
        cleaned = re.sub(r"\s+", " ", (query or "").strip())
        if not cleaned:
            logger.debug("decide returning should_clarify=True reason=empty_query")
            return ClarificationDecision(True, "What would you like me to look up in your documents?")

        tokens = cleaned.lower().split()
        is_short = len(tokens) <= 3
        has_pronoun = any(tok in self._PRONOUNS for tok in tokens)
        has_context = bool(conversation_context and conversation_context.strip())

        if (is_short or has_pronoun) and not has_context:
            logger.debug("decide returning should_clarify=True reason=query_too_vague")
            return ClarificationDecision(
                True,
                self._build_question(cleaned, profile_context or {}),
                "query_too_vague",
            )

        if query_analysis:
            intent = str(query_analysis.get("intent") or "")
            metadata = query_analysis.get("metadata_filters") or {}
            if intent in {"comparison", "summary"} and not metadata and len(tokens) < 6:
                logger.debug("decide returning should_clarify=True reason=needs_scope intent=%s", intent)
                return ClarificationDecision(
                    True,
                    self._build_question(cleaned, profile_context or {}),
                    "needs_scope",
                )

        logger.debug("decide returning should_clarify=False")
        return ClarificationDecision(False)

    @staticmethod
    def _build_question(query: str, profile_context: Dict[str, Any]) -> str:
        hints = profile_context.get("hints") or []
        if hints:
            options = ", ".join(str(h) for h in hints[:4])
            return (
                f"I want to make sure I focus on the right material. "
                f"Which document or section should I use for \"{query}\"? "
                f"For example: {options}."
            )
        return (
            f"To answer \"{query}\", which document, section, or page range should I focus on?"
        )
