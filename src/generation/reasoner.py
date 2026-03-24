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
    "lookup": 3072,
    "extract": 6144,
    "list": 6144,
    "summarize": 6144,
    "overview": 6144,
    "compare": 6144,
    "investigate": 6144,
    "aggregate": 4096,
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

        logger.info(
            "[REASONER_PROMPT] task=%s evidence_count=%d prompt_len=%d query=%r",
            task_type, len(evidence), len(user_msg), query[:80],
        )

        try:
            text, metadata = self._llm.generate_with_metadata(
                user_msg,
                system=system_msg,
                think=use_thinking,
                temperature=0.4,
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

        # Extract text from evidence — handle multiple field names
        evidence_parts = []
        for item in evidence:
            text = (
                item.get("text")
                or item.get("canonical_text")
                or item.get("embedding_text")
                or item.get("content")
                or ""
            )
            evidence_parts.append(text)
        evidence_text = " ".join(evidence_parts)

        # If we have evidence items but no extractable text, mark as ungrounded
        # — the model cannot be grounded without actual text to reference
        if not evidence_text.strip():
            logger.warning("[Reasoner] Grounding: evidence items exist but contain no text — UNGROUNDED")
            return False

        # Short or empty answer — consider grounded if we have evidence
        if len(answer.strip()) < 20:
            return True

        # Check that numeric claims in the answer are traceable to evidence
        answer_numbers = set(_NUMBER_RE.findall(answer))
        if answer_numbers:
            evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
            ungrounded_nums = answer_numbers - evidence_numbers
            if ungrounded_nums:
                logger.warning(
                    "[Reasoner] Grounding: %d/%d numbers not found in evidence: %s",
                    len(ungrounded_nums), len(answer_numbers),
                    list(ungrounded_nums)[:10],
                )
            # Allow up to 20% ungrounded numbers (expert may compute ratios
            # or reformat values, but most numbers must trace to evidence)
            if len(ungrounded_nums) / len(answer_numbers) > 0.20:
                logger.debug(
                    "[Reasoner] Grounding: %d/%d numbers ungrounded",
                    len(ungrounded_nums), len(answer_numbers),
                )
                return False

        # Grounding strategy: if the retrieval pipeline found evidence and the
        # model produced a substantive response, the response is grounded.
        #
        # The RAG pipeline already ensures evidence relevance via:
        # 1. Dense + sparse hybrid retrieval with score thresholds
        # 2. Cross-encoder reranking
        # 3. System prompt instructing grounded-only answers
        #
        # Post-hoc word overlap checks are unreliable for a synthesizing model
        # that paraphrases, summarizes, and uses professional vocabulary.
        # Instead, we only flag as ungrounded when the answer clearly has NO
        # connection to the evidence (zero word overlap = likely hallucination).

        answer_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer)
        )
        evidence_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', evidence_text)
        )

        if answer_words and evidence_words:
            overlap = len(answer_words & evidence_words)
            overlap_ratio = overlap / len(answer_words) if answer_words else 0
            # At least 15% of answer words must appear in evidence to be grounded.
            # This catches responses that share a few generic words but fabricate
            # most of the content.
            if overlap_ratio < 0.15:
                logger.warning(
                    "[Reasoner] Grounding: only %d/%d words (%.0f%%) overlap with evidence — UNGROUNDED",
                    overlap, len(answer_words), overlap_ratio * 100,
                )
                return False
            if overlap < 5:
                logger.warning(
                    "[Reasoner] Grounding: only %d shared words — UNGROUNDED",
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
            # Qwen3's <think> blocks can consume 2000+ tokens of reasoning
            # before producing the actual answer. Double the budget to ensure
            # enough room for both thinking and answer generation.
            base = int(base * 2.5)

        return min(base, 16384)
