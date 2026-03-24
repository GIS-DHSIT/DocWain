"""
N-gram entropy and repetition heuristics.

Lower normalized entropy and higher repetition are treated as indicators of
AI-generated or heavily templated text.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from ..models import ScreeningContext
from .base import (
    ToolResult,
    ScreeningTool,
    clamp,
    normalized_entropy,
    safe_mean,
    tokenize_words,
)


class NgramEntropyTool(ScreeningTool):
    """Compute bigram/trigram entropy and repetition signals."""

    name = "ngram_entropy"
    category = "Authenticity & Originality"
    default_weight = 0.1
    tool_version = "0.1"

    def _repetition_score(self, unigram_counts: Counter[str], bigram_counts: Counter[str]) -> float:
        total_tokens = sum(unigram_counts.values()) or 1
        total_bigrams = sum(bigram_counts.values()) or 1

        # Dominance of the top unigram token.
        top_unigram_ratio = max(unigram_counts.values(), default=0) / total_tokens

        # Repetition of bigrams (counts beyond the first occurrence).
        repeated_bigram_ratio = sum(count - 1 for count in bigram_counts.values() if count > 1) / total_bigrams

        repetition_score = clamp(0.6 * top_unigram_ratio + 0.4 * repeated_bigram_ratio)
        return repetition_score

    def run(self, ctx: ScreeningContext) -> ToolResult:
        text = ctx.text
        tokens = tokenize_words(text)

        # Build n-grams
        bigram_counts = Counter([" ".join(pair) for pair in zip(tokens, tokens[1:])])
        trigram_counts = Counter([" ".join(triple) for triple in zip(tokens, tokens[1:], tokens[2:])])
        unigram_counts: Counter[str] = Counter(tokens)

        bigram_entropy = normalized_entropy(dict(bigram_counts))
        trigram_entropy = normalized_entropy(dict(trigram_counts))
        avg_entropy = safe_mean([bigram_entropy, trigram_entropy])

        entropy_score = clamp(1.0 - avg_entropy)
        repetition_score = self._repetition_score(unigram_counts, bigram_counts)
        combined_score = clamp(0.7 * entropy_score + 0.3 * repetition_score)

        # Build reasons list
        reasons: List[str] = [
            f"Entropy={avg_entropy:.2f} (lower suggests templating); "
            f"repetition score={repetition_score:.2f}."
        ]

        # Actionable intelligence based on score thresholds
        actions: List[str] = ["tag"]
        if combined_score > 0.6:
            reasons.append("High repetition detected - review for templated/generated content.")
            actions.append("flag_for_review")
        elif combined_score < 0.3:
            reasons.append("Text shows natural entropy variation.")
        else:
            reasons.append("Moderate entropy - possible mild templating or structured content.")

        raw_features: Dict[str, float] = {
            "bigram_entropy": bigram_entropy,
            "trigram_entropy": trigram_entropy,
            "avg_entropy": avg_entropy,
            "entropy_score": entropy_score,
            "repetition_score": repetition_score,
            "top_unigram_ratio": max(unigram_counts.values(), default=0) / (sum(unigram_counts.values()) or 1),
            "repeated_bigram_ratio": sum(count - 1 for count in bigram_counts.values() if count > 1)
            / (sum(bigram_counts.values()) or 1),
            "token_count": len(tokens),
        }

        # Evidence spans: top repeated bigrams and trigrams
        evidence_spans: List[dict] = []
        for ngram, count in bigram_counts.most_common(5):
            if count > 1:
                evidence_spans.append({"label": "repeated_bigram", "text": ngram, "count": count})
        for ngram, count in trigram_counts.most_common(3):
            if count > 1:
                evidence_spans.append({"label": "repeated_trigram", "text": ngram, "count": count})

        return self.result(
            ctx,
            combined_score,
            reasons,
            raw_features=raw_features,
            actions=actions,
            evidence_spans=evidence_spans,
        )
