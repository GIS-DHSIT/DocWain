"""
IntelligentGenerator — single LLM call with Qwen3 thinking mode for complex queries.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.intelligence.understand import UnderstandResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_FORMAT_INSTRUCTIONS = {
    "table": "Present as markdown table with appropriate columns, cite sources per row",
    "bullets": "Present as bulleted list, each with citation, most important first",
    "sections": "Organize with ## headers, bullets within, end with synthesis",
    "numbered": "Numbered list in sequential order",
    "prose": "Clear paragraphs, lead with answer, then evidence",
}

_COMPLEXITY_BASE_TOKENS = {
    "simple": 1024,
    "moderate": 2048,
    "complex": 3072,
}


@dataclass
class GeneratedResponse:
    text: str
    thinking: Optional[str] = None
    citations_found: int = 0
    token_usage: Dict = field(default_factory=dict)


class IntelligentGenerator:
    """Generates answers from evidence using a single LLM call, with optional
    Qwen3 thinking mode for complex queries."""

    def __init__(self, llm_gateway):
        self._llm = llm_gateway

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        evidence: List[Dict],
        understanding: UnderstandResult,
        conversation_history: str = "",
    ) -> GeneratedResponse:
        """Build a prompt from evidence + understanding and call the LLM."""

        evidence_block = self._build_evidence_block(evidence)
        format_instructions = self._build_format_instructions(understanding)
        sub_intent_instructions = self._build_sub_intent_instructions(understanding)
        token_budget = self._token_budget(understanding)

        # ----- system prompt -----
        system_parts = [
            "You are a precise document-intelligence assistant.",
            "Answer the user query using ONLY the evidence below.",
            "Cite evidence as [SOURCE-N] inline.",
            "If the evidence does not support an answer, say so explicitly.",
            "",
            "--- EVIDENCE ---",
            evidence_block,
            "--- END EVIDENCE ---",
        ]

        if format_instructions:
            system_parts.append("")
            system_parts.append(f"FORMAT: {format_instructions}")

        if sub_intent_instructions:
            system_parts.append("")
            system_parts.append(sub_intent_instructions)

        domain_hints = getattr(understanding, "domain_hints", None)
        if domain_hints:
            relevant_fields = getattr(domain_hints, "relevant_fields", None) if not isinstance(domain_hints, dict) else domain_hints.get("relevant_fields")
            terminology = getattr(domain_hints, "terminology_context", None) if not isinstance(domain_hints, dict) else domain_hints.get("terminology_context")
            if relevant_fields:
                system_parts.append(f"\nRelevant fields: {', '.join(relevant_fields)}")
            if terminology:
                system_parts.append(f"Terminology context: {terminology}")

        system_prompt = "\n".join(system_parts)

        # ----- user message -----
        user_content = understanding.resolved_query or query
        if conversation_history:
            user_content = (
                f"Conversation so far:\n{conversation_history}\n\n"
                f"Current question: {user_content}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # ----- call LLM -----
        use_thinking = bool(getattr(understanding, "thinking_required", False))
        options = {"max_tokens": token_budget}

        try:
            raw_text, metadata = self._llm.chat_with_metadata(
                messages, options=options, thinking=use_thinking
            )
        except Exception:
            logger.exception("chat_with_metadata failed, falling back to generate")
            raw_text, metadata = self._llm.generate_with_metadata(
                system_prompt + "\n\n" + user_content, options=options
            )

        # ----- parse response -----
        thinking_text = metadata.get("thinking") if metadata else None
        token_usage = metadata.get("usage", {}) if metadata else {}
        citations_found = len(set(re.findall(r"\[SOURCE-\d+\]", raw_text)))

        logger.info(
            "generation complete — citations=%d, thinking=%s, tokens=%s",
            citations_found,
            thinking_text is not None,
            token_usage,
        )

        return GeneratedResponse(
            text=raw_text,
            thinking=thinking_text,
            citations_found=citations_found,
            token_usage=token_usage,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_evidence_block(self, evidence: List[Dict]) -> str:
        """Format evidence as numbered [SOURCE-N] entries."""
        if not evidence:
            return "(no evidence provided)"

        lines = []
        for idx, ev in enumerate(evidence, start=1):
            source = ev.get("source_name", "unknown")
            page = ev.get("page", "?")
            section = ev.get("section", "")
            score = ev.get("score", 0.0)
            text = ev.get("text", "")
            header = f"[SOURCE-{idx}] {source} | page {page}"
            if section:
                header += f" | {section}"
            header += f" (score {score:.2f})"
            lines.append(header)
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _build_format_instructions(self, understanding: UnderstandResult) -> str:
        """Return format guidance based on output_format."""
        fmt = getattr(understanding, "output_format", "prose") or "prose"
        return _FORMAT_INSTRUCTIONS.get(fmt, _FORMAT_INSTRUCTIONS["prose"])

    def _build_sub_intent_instructions(self, understanding: UnderstandResult) -> str:
        """If multi-part query, add Part 1/2/... instructions."""
        sub_intents = getattr(understanding, "sub_intents", None) or []
        if len(sub_intents) <= 1:
            return ""
        parts = []
        for i, intent in enumerate(sub_intents, start=1):
            parts.append(f"Part {i}: {intent}")
        return "Address each part:\n" + "\n".join(parts)

    def _token_budget(self, understanding: UnderstandResult) -> int:
        """Compute a dynamic token budget based on query complexity."""
        complexity = getattr(understanding, "complexity", "moderate") or "moderate"
        base = _COMPLEXITY_BASE_TOKENS.get(complexity, 2048)

        sub_intents = getattr(understanding, "sub_intents", None) or []
        extra = max(0, len(sub_intents) - 1)
        budget = base + 512 * extra

        if getattr(understanding, "thinking_required", False):
            budget = int(budget * 1.4)

        return min(budget, 4096)
