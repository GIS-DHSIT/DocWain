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
    "table": (
        "Present data in a clean markdown table.\n"
        "- Use | column | headers | with alignment\n"
        "- One data point per row, no merged cells\n"
        "- Bold key values: **$9,000.00**, **Jessica Jones**\n"
        "- Add a brief summary sentence above the table"
    ),
    "bullets": (
        "Present as a structured bulleted list.\n"
        "- Lead with a one-line summary sentence\n"
        "- Group related bullets under **bold category headers**\n"
        "- Each bullet: **Label:** value or description\n"
        "- Bold key names, amounts, dates, entities\n"
        "- Most important items first"
    ),
    "sections": (
        "Organize the response with clear visual hierarchy.\n"
        "- Start with a one-line executive summary\n"
        "- Use ## for major sections, ### for subsections\n"
        "- Within sections, use bullet points with **bold labels**\n"
        "- Format: **Field Name:** extracted value or insight\n"
        "- Bold all key values: names, amounts, dates, identifiers\n"
        "- Use markdown tables for tabular data (line items, comparisons)\n"
        "- Never leave headers as plain text — always use ## or ###\n"
        "- Keep bullets self-contained — each makes sense alone\n"
        "- End with a brief synthesis or key takeaway if appropriate"
    ),
    "numbered": (
        "Present as a numbered list.\n"
        "- Each item: **Label** — description with **bold key values**\n"
        "- Sequential order, one point per number\n"
        "- Brief summary before the list"
    ),
    "prose": (
        "Write clear, structured paragraphs.\n"
        "- Lead with the direct answer in the first sentence\n"
        "- Bold key values: **$9,000.00**, **Jessica Jones**, **Document 0522**\n"
        "- Use short paragraphs (2-3 sentences each)\n"
        "- For any tabular data, use a markdown table instead of inline text"
    ),
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
            "You are a senior document-intelligence analyst providing structured, actionable insights.",
            "",
            "RULES:",
            "1. Answer ONLY from the evidence below — never fabricate data.",
            "2. Lead with a one-sentence executive summary.",
            "3. Use proper markdown: ## headers, ### subheaders, **bold** key values.",
            "4. Bold ALL extracted values: names, amounts, dates, IDs, entities.",
            "5. Use tables for line items, comparisons, or structured data.",
            "6. Keep each bullet self-contained and informative.",
            "7. If evidence is insufficient, state what is missing.",
            "8. Do NOT include [SOURCE-N] tags in the response.",
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

        return min(budget, 8192)
