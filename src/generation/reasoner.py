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
    "lookup": 256,
    "extract": 512,
    "list": 768,
    "summarize": 1024,
    "compare": 1024,
    "investigate": 1024,
    "aggregate": 768,
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
    ) -> ReasonerResult:
        """Run the REASON step: prompt the LLM and return a grounded result."""

        system_msg = build_system_prompt()
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
        """Lightweight deterministic grounding check (no LLM call).

        Returns *False* when the answer contains numbers that do not appear
        anywhere in the evidence text.  Up to 20 % ungrounded numbers are
        tolerated (they could be computed values like totals).
        """
        if not evidence:
            return False

        answer_numbers = set(_NUMBER_RE.findall(answer))
        if not answer_numbers:
            # No numeric claims → grounded by default
            return True

        evidence_text = " ".join(item.get("text", "") for item in evidence)
        evidence_numbers = set(_NUMBER_RE.findall(evidence_text))

        ungrounded = answer_numbers - evidence_numbers
        if not answer_numbers:
            return True

        ratio = len(ungrounded) / len(answer_numbers)
        return ratio <= 0.20

    @staticmethod
    def _compute_token_budget(
        task_type: str, evidence_count: int, thinking: bool
    ) -> int:
        """Compute a max-tokens budget based on task and context."""
        base = _BASE_TOKENS.get(task_type, 512)

        if evidence_count > 10:
            base = int(base * 1.3)

        if thinking:
            base = int(base * 1.5)

        return min(base, 1536)
