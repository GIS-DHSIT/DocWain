"""
VerificationGate + Verifier — lightweight claim-checking against evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.intelligence.understand import UnderstandResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    supported: bool
    unsupported_claims: List[str] = field(default_factory=list)
    fix_suggestion: Optional[str] = None


class VerificationGate:
    """Decides whether a generated response needs explicit verification."""

    def needs_verification(
        self,
        evidence: List[Dict],
        response: str,
        understanding: UnderstandResult,
    ) -> bool:
        citation_count = len(set(re.findall(r"\[SOURCE-\d+\]", response)))
        scores = [ev.get("score", 0.0) for ev in evidence]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        doc_ids = {ev.get("source_name") or ev.get("chunk_id", idx) for idx, ev in enumerate(evidence)}
        doc_count = len(doc_ids)
        complexity = getattr(understanding, "complexity", "moderate") or "moderate"

        # Skip fast-path cases
        if complexity == "simple" and citation_count >= 1 and avg_score > 0.7:
            logger.debug("verification skipped: simple + cited + high score")
            return False
        if complexity == "moderate" and citation_count >= 2 and avg_score > 0.6:
            logger.debug("verification skipped: moderate + well-cited + decent score")
            return False

        # Verify if risky
        if doc_count > 3:
            logger.info("verification required: %d source documents", doc_count)
            return True
        if avg_score < 0.4:
            logger.info("verification required: low avg score %.2f", avg_score)
            return True
        if citation_count == 0:
            logger.info("verification required: zero citations")
            return True

        # Default: no verification needed
        return False


class Verifier:
    """Makes one LLM call to check claims in a response against evidence."""

    def __init__(self, llm_gateway):
        self._llm = llm_gateway

    def verify(self, response: str, evidence: List[Dict]) -> VerificationResult:
        """Check each factual claim in *response* against *evidence*."""

        evidence_text = "\n\n".join(
            f"[SOURCE-{i}] {ev.get('text', '')}" for i, ev in enumerate(evidence, 1)
        )

        prompt = (
            "You are a fact-checking assistant.\n\n"
            "EVIDENCE:\n"
            f"{evidence_text}\n\n"
            "RESPONSE TO VERIFY:\n"
            f"{response}\n\n"
            "List any claims in the response that are NOT supported by the evidence.\n"
            "Output format — one claim per line prefixed with '- '.\n"
            "If every claim is supported, output exactly: ALL_SUPPORTED"
        )

        try:
            raw = self._llm.generate(prompt, max_tokens=1024)
        except Exception:
            logger.exception("verification LLM call failed, assuming supported")
            return VerificationResult(supported=True)

        raw = raw.strip()

        if "ALL_SUPPORTED" in raw:
            logger.info("verification passed: all claims supported")
            return VerificationResult(supported=True)

        unsupported = [
            line.lstrip("- ").strip()
            for line in raw.splitlines()
            if line.strip().startswith("-")
        ]

        if not unsupported:
            # Could not parse any claims — treat as supported
            return VerificationResult(supported=True)

        suggestion = (
            "Remove or rephrase unsupported claims."
            if len(unsupported) <= 2
            else "Consider regenerating the response with stricter grounding."
        )

        logger.warning(
            "verification found %d unsupported claims", len(unsupported)
        )
        return VerificationResult(
            supported=False,
            unsupported_claims=unsupported,
            fix_suggestion=suggestion,
        )

    def handle_failure(
        self, response: str, verification: VerificationResult
    ) -> Optional[str]:
        """Attempt to fix a response that failed verification.

        Returns:
            Cleaned response if fixable (<=2 unsupported claims),
            None if the caller should regenerate (>2 unsupported claims).
        """
        if verification.supported:
            return response

        if len(verification.unsupported_claims) > 2:
            logger.info(
                "too many unsupported claims (%d), signalling regeneration",
                len(verification.unsupported_claims),
            )
            return None

        # Strip sentences that contain unsupported claims
        cleaned_lines = []
        sentences = re.split(r"(?<=[.!?])\s+", response)
        for sentence in sentences:
            dominated = any(
                claim.lower() in sentence.lower()
                for claim in verification.unsupported_claims
            )
            if not dominated:
                cleaned_lines.append(sentence)

        cleaned = " ".join(cleaned_lines).strip()
        if not cleaned:
            return None

        logger.info("stripped %d unsupported sentences", len(sentences) - len(cleaned_lines))
        return cleaned
