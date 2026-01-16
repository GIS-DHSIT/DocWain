from __future__ import annotations

import re
from collections import Counter
from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp, coefficient_of_variation, normalized_entropy, split_sentences, tokenize_words


class AIAuthorshipTool(ScreeningTool):
    name = "ai_authorship"
    category = "AI Authorship Likelihood"
    default_weight = 0.11
    tool_version = "1.0"

    def run(self, ctx: ScreeningContext):
        words = tokenize_words(ctx.text)
        bigrams = list(zip(words, words[1:]))
        counts = Counter(bigrams)
        entropy = normalized_entropy(counts)
        repetition_ratio = sum(1 for _, c in counts.items() if c > 1) / max(len(counts), 1)

        sentences = split_sentences(ctx.text)
        sentence_lengths = [len(tokenize_words(s)) for s in sentences]
        length_cv = coefficient_of_variation(sentence_lengths)
        punctuation_density = len(re.findall(r"[.,;:!?]", ctx.text)) / max(len(words), 1)

        score_components: List[float] = []
        reasons: List[str] = []

        if entropy < 0.55:
            reasons.append(f"Low n-gram entropy ({entropy:.2f}).")
            score_components.append((0.55 - entropy) * 0.8)
        if repetition_ratio > 0.25:
            reasons.append("Repeated phrasing detected.")
            score_components.append(min(0.4, repetition_ratio))
        if length_cv < 0.35 and len(sentence_lengths) > 3:
            reasons.append("Uniform sentence lengths (stylometry signal).")
            score_components.append(0.25)
        if punctuation_density < 0.03:
            reasons.append("Sparse punctuation may indicate templated text.")
            score_components.append(0.1)

        if not reasons:
            reasons.append("Stylistic signals look human-like.")

        score = clamp(sum(score_components))
        raw_features = {
            "entropy": entropy,
            "repetition_ratio": repetition_ratio,
            "sentence_length_cv": length_cv,
            "punctuation_density": punctuation_density,
            "sentence_count": len(sentences),
            "word_count": len(words),
        }

        actions = ["tag"]
        if score >= 0.66:
            actions.append("warn")

        return self.result(ctx, score, reasons, raw_features=raw_features, actions=actions)
