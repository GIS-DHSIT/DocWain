"""Reasoner — the REASON step of the DocWain Core Agent pipeline.

Calls the LLM with evidence and returns a grounded answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.generation.prompts import build_reason_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReasonerResult:
    """Outcome of a single REASON step."""

    text: str
    sources: List[Dict[str, Any]]
    grounded: bool
    thinking: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token budget lookup
# ---------------------------------------------------------------------------

_BASE_TOKENS: Dict[str, int] = {
    "lookup": 512,
    "extract": 1024,
    "list": 1024,
    "summarize": 2048,
    "overview": 3072,
    "compare": 2048,
    "investigate": 2048,
    "aggregate": 1024,
}

# ---------------------------------------------------------------------------
# Number extraction regex — shared by grounding check
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    \$[\d,]+(?:\.\d+)?    |   # dollar amounts
    \d{1,3}(?:,\d{3})+       |   # comma-separated integers
    \d+\.\d+                  |   # decimals
    \d+%                      |   # percentages
    \b\d{2,}\b                    # bare integers ≥ 2 digits
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Reasoner class
# ---------------------------------------------------------------------------


class Reasoner:
    """Generates an LLM answer from ranked evidence."""

    def __init__(self, llm_gateway: Any) -> None:
        self._llm = llm_gateway

    # -- public API ---------------------------------------------------------

    def reason(
        self,
        query: str,
        task_type: str,
        output_format: str,
        evidence: List[Dict[str, Any]],
        doc_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        use_thinking: bool = False,
        profile_domain: str = "",
        kg_context: str = "",
    ) -> ReasonerResult:
        """Run the REASON step: prompt the LLM and return a grounded result."""

        system_msg = build_system_prompt(
            profile_domain=profile_domain,
            kg_context=kg_context,
        )
        user_msg = build_reason_prompt(
            query=query,
            task_type=task_type,
            output_format=output_format,
            evidence=evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
        )

        max_tokens = self._compute_token_budget(
            task_type, len(evidence), use_thinking,
        )

        try:
            text, metadata = self._llm.generate_with_metadata(
                user_msg,
                system=system_msg,
                think=use_thinking,
                temperature=0.2,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.exception("LLM generation failed during REASON step")
            return ReasonerResult(
                text="Unable to generate an answer due to a processing error.",
                sources=self._extract_sources(evidence),
                grounded=False,
            )

        thinking_text = metadata.get("thinking") if metadata else None
        usage = metadata.get("usage", {}) if metadata else {}

        sources = self._extract_sources(evidence)
        grounded = self._check_grounding(text, evidence)

        return ReasonerResult(
            text=text,
            sources=sources,
            grounded=grounded,
            thinking=thinking_text,
            usage=usage,
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _extract_sources(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a source list from evidence dicts."""
        sources: List[Dict[str, Any]] = []
        for item in evidence:
            sources.append({
                "source_name": item.get("source_name", "unknown"),
                "page": item.get("page"),
                "section": item.get("section"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "source_index": item.get("source_index"),
                "score": item.get("score"),
                "excerpt": (item.get("text", "")[:200]
                            if item.get("text") else None),
            })
        return sources

    def _check_grounding(
        self, answer: str, evidence: List[Dict[str, Any]]
    ) -> bool:
        """Evidence-anchored grounding check.

        Verifies that the answer's key claims are derivable from the provided
        evidence. Uses semantic overlap rather than strict verbatim matching,
        allowing expert-level synthesis and reasoning while catching fabrication.

        Returns False only when the answer introduces substantial content
        that cannot be traced to any evidence.
        """
        if not evidence:
            return False

        evidence_text = " ".join(item.get("text", "") for item in evidence)
        evidence_lower = evidence_text.lower()

        # Short or empty answer — consider grounded if we have evidence
        if len(answer.strip()) < 20:
            return True

        # Check that numeric claims in the answer are traceable to evidence
        answer_numbers = set(_NUMBER_RE.findall(answer))
        if answer_numbers:
            evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
            ungrounded_nums = answer_numbers - evidence_numbers
            # Allow up to 25% ungrounded numbers (expert may compute ratios,
            # percentages, or reformat values)
            if len(ungrounded_nums) / len(answer_numbers) > 0.25:
                logger.debug(
                    "[Reasoner] Grounding: %d/%d numbers ungrounded",
                    len(ungrounded_nums), len(answer_numbers),
                )
                return False

        # Check semantic overlap: answer sentences should have word overlap
        # with evidence. An expert synthesizes but doesn't invent topics.
        answer_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', answer)
        )
        evidence_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', evidence_text)
        )

        if answer_words:
            overlap = len(answer_words & evidence_words) / len(answer_words)
            # At least 40% of answer's significant words should appear in evidence
            # This is generous — allows expert vocabulary and reasoning language
            if overlap < 0.40:
                logger.debug(
                    "[Reasoner] Grounding: word overlap %.2f below threshold",
                    overlap,
                )
                return False

        return True

    @staticmethod
    def _compute_token_budget(
        task_type: str, evidence_count: int, thinking: bool
    ) -> int:
        """Compute a max-tokens budget based on task and context."""
        base = _BASE_TOKENS.get(task_type, 512)

        # Scale up for richer evidence sets
        if evidence_count > 10:
            base = int(base * 1.3)
        elif evidence_count > 5:
            base = int(base * 1.15)

        if thinking:
            base = int(base * 1.5)

        return min(base, 2048)
