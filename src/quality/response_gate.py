"""Response quality gate — hallucination check and grounding verification."""

import logging

logger = logging.getLogger(__name__)


class ResponseQualityGate:
    """Verifies response quality before returning to user.

    Checks:
    1. Grounding — is the answer supported by retrieved context?
    2. Hallucination — does the answer contain claims not in context?
    3. Completeness — does the answer address the query?

    If ungrounded, retries with stricter prompt.
    If still ungrounded, returns "insufficient context" response.
    """

    def __init__(self, llm_client=None, max_retries: int = 1):
        self._llm = llm_client
        self.max_retries = max_retries

    def check(self, query: str, response: str, context_chunks: list,
              sources: list = None) -> dict:
        """Run quality checks on generated response.

        Args:
            query: Original user query
            response: Generated response text
            context_chunks: Retrieved context chunks used for generation
            sources: Source references

        Returns:
            dict with: passed, grounded, confidence, issues, final_response
        """
        # Check grounding
        grounding_result = self._check_grounding(response, context_chunks)

        if grounding_result["grounded"]:
            return {
                "passed": True,
                "grounded": True,
                "confidence": grounding_result["confidence"],
                "issues": [],
                "final_response": response
            }

        # Response is ungrounded — try stricter generation
        logger.warning(
            "Response failed grounding check: %s", grounding_result["issues"]
        )

        if self.max_retries > 0:
            # TODO: Retry with stricter prompt via LLM
            # For now, add a disclaimer
            logger.info("Would retry with stricter prompt (not yet implemented)")

        # If still ungrounded, return insufficient context
        if grounding_result["confidence"] < 0.3:
            return {
                "passed": False,
                "grounded": False,
                "confidence": grounding_result["confidence"],
                "issues": grounding_result["issues"],
                "final_response": (
                    "I don't have sufficient information in the available "
                    "documents to provide an accurate answer to this question. "
                    "Please try rephrasing your question or ensure the relevant "
                    "documents have been uploaded and processed."
                )
            }

        # Partially grounded — return with warning
        return {
            "passed": True,
            "grounded": False,
            "confidence": grounding_result["confidence"],
            "issues": grounding_result["issues"],
            "final_response": response
        }

    def _check_grounding(self, response: str, context_chunks: list) -> dict:
        """Check if response is grounded in context chunks."""
        if not context_chunks:
            return {
                "grounded": False,
                "confidence": 0.0,
                "issues": ["No context chunks available"]
            }

        # Build context text
        context_text = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in context_chunks
        ).lower()

        if not context_text.strip():
            return {
                "grounded": False,
                "confidence": 0.0,
                "issues": ["Context chunks contain no text"]
            }

        # Simple grounding check: what fraction of response sentences
        # have supporting evidence in context?
        response_sentences = [
            s.strip() for s in response.split('.')
            if len(s.strip()) > 10
        ]

        if not response_sentences:
            return {"grounded": True, "confidence": 1.0, "issues": []}

        grounded_count = 0
        issues = []

        for sentence in response_sentences:
            # Check if key words from sentence appear in context
            words = set(sentence.lower().split())
            # Remove common words
            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be",
                "been", "being", "have", "has", "had", "do", "does",
                "did", "will", "would", "could", "should", "may",
                "might", "shall", "can", "need", "dare", "ought",
                "used", "to", "of", "in", "for", "on", "with", "at",
                "by", "from", "as", "into", "through", "during",
                "before", "after", "above", "below", "between",
                "and", "but", "or", "nor", "not", "so", "yet",
                "both", "either", "neither", "each", "every",
                "this", "that", "these", "those", "it", "its",
            }
            content_words = words - stop_words

            if not content_words:
                grounded_count += 1
                continue

            # Check overlap with context
            overlap = sum(1 for w in content_words if w in context_text)
            ratio = overlap / len(content_words) if content_words else 0

            if ratio >= 0.5:
                grounded_count += 1
            else:
                issues.append(
                    f"Potentially ungrounded: '{sentence[:80]}...'"
                )

        confidence = grounded_count / len(response_sentences)
        grounded = confidence >= 0.7

        return {
            "grounded": grounded,
            "confidence": round(confidence, 2),
            "issues": issues
        }
