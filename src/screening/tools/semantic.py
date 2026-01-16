"""
Semantic consistency and formatting/boilerplate heuristics.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set, Tuple

from .base import (
    ToolResult,
    ScreeningTool,
    clamp,
    split_sentences,
    tokenize_words,
)

ENTITY_RE = re.compile(r"\b([A-Z][a-z]+|[A-Z]{2,})\b")
CONTRAST_MARKERS = {
    "however",
    "but",
    "though",
    "although",
    "yet",
    "nevertheless",
    "nonetheless",
    "still",
    "conversely",
    "on the other hand",
}
NEGATIONS = {"not", "never", "no", "none", "n't", "without"}

TRANSITION_PHRASES = [
    "in conclusion",
    "overall",
    "moreover",
    "furthermore",
    "additionally",
    "in summary",
    "to summarize",
    "in closing",
    "thus",
    "therefore",
    "hence",
]

BOILERPLATE_PHRASES = [
    "in conclusion",
    "this highlights",
    "this demonstrates",
    "this shows that",
    "as mentioned earlier",
    "the aforementioned",
    "it is important to note",
    "it should be noted",
    "overall",
]


def _extract_entities(sentence: str) -> Set[str]:
    return set(match[0] for match in ENTITY_RE.findall(sentence))


def _entity_continuity(sentences: List[str]) -> float:
    if len(sentences) < 2:
        return 0.0

    overlaps: List[float] = []
    prev_entities: Set[str] = set()
    for sentence in sentences:
        current_entities = _extract_entities(sentence)
        if prev_entities or current_entities:
            union = prev_entities | current_entities
            overlap = len(prev_entities & current_entities) / max(len(union), 1)
            overlaps.append(overlap)
        prev_entities = current_entities

    if not overlaps:
        return 0.0
    return sum(overlaps) / len(overlaps)


def _repetition_signals(words: List[str], sentences: List[str]) -> Tuple[float, float]:
    # Phrase repetition via 5-grams.
    phrases = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
    phrase_counts = Counter(phrases)
    repeated_phrases = sum(count - 1 for count in phrase_counts.values() if count > 1)
    phrase_repetition_ratio = repeated_phrases / max(len(phrases), 1)

    # Sentence-start repetition (first two tokens).
    starts = [" ".join(tokenize_words(s)[:2]) for s in sentences if s.strip()]
    start_counts = Counter(starts)
    repeated_starts = sum(count - 1 for count in start_counts.values() if count > 1)
    start_repetition_ratio = repeated_starts / max(len(starts), 1)

    return phrase_repetition_ratio, start_repetition_ratio


class SemanticConsistencyTool(ScreeningTool):
    """Entity continuity, contradiction/negation density, and repetition."""

    name = "semantic_consistency"

    def run(self, text: str) -> ToolResult:
        sentences = split_sentences(text)
        words = tokenize_words(text)
        word_count = max(len(words), 1)

        continuity = _entity_continuity(sentences)
        contrast_hits = sum(
            text.lower().count(marker) for marker in CONTRAST_MARKERS
        )
        negation_hits = sum(1 for w in words if w in NEGATIONS)
        phrase_rep, start_rep = _repetition_signals(words, sentences)

        contrast_density = (contrast_hits / word_count) * 1000
        negation_density = (negation_hits / word_count) * 1000

        continuity_score = clamp(1 - continuity)
        contrast_score = clamp(min(contrast_density / 12.0, 1.0))
        negation_score = clamp(min(negation_density / 14.0, 1.0))
        repetition_score = clamp(min((phrase_rep * 5) + (start_rep * 2), 1.0))

        combined_score = clamp(
            0.4 * continuity_score + 0.2 * contrast_score + 0.2 * negation_score + 0.2 * repetition_score
        )

        reason = (
            f"Entity continuity={continuity:.2f}, contrast density={contrast_density:.1f}/1k words, "
            f"negation density={negation_density:.1f}/1k words, repetition={phrase_rep:.2f}."
        )

        raw_features: Dict[str, float] = {
            "entity_continuity": continuity,
            "contrast_density_per_1k": contrast_density,
            "negation_density_per_1k": negation_density,
            "phrase_repetition_ratio": phrase_rep,
            "start_repetition_ratio": start_rep,
            "continuity_score": continuity_score,
            "contrast_score": contrast_score,
            "negation_score": negation_score,
            "repetition_score": repetition_score,
            "word_count": word_count,
        }

        return ToolResult(
            name=self.name,
            score=combined_score,
            reason=reason,
            raw_features=raw_features,
        )


class FormattingHeuristicTool(ScreeningTool):
    """Formatting and boilerplate phrases."""

    name = "formatting"

    def run(self, text: str) -> ToolResult:
        words = tokenize_words(text)
        word_count = max(len(words), 1)
        text_lower = text.lower()

        transition_hits = sum(text_lower.count(p) for p in TRANSITION_PHRASES)
        boilerplate_hits = sum(text_lower.count(p) for p in BOILERPLATE_PHRASES)
        bullet_hits = len(re.findall(r"(^|\n)[\-\u2022\*]\s", text_lower))
        heading_hits = len(re.findall(r"\n\s*[A-Z][A-Za-z0-9 ]{3,}\s*\n", text))

        transition_density = (transition_hits / word_count) * 1000
        boilerplate_density = (boilerplate_hits / word_count) * 1000
        layout_density = (bullet_hits + heading_hits) / max(text.count("\n"), 1)

        transition_score = clamp(min(transition_density / 10.0, 1.0))
        boilerplate_score = clamp(min(boilerplate_density / 6.0, 1.0))
        layout_score = clamp(min(layout_density / 3.0, 1.0))

        combined_score = clamp(0.45 * transition_score + 0.4 * boilerplate_score + 0.15 * layout_score)

        reason = (
            f"Transition density={transition_density:.1f}/1k words, "
            f"boilerplate density={boilerplate_density:.1f}/1k words."
        )

        raw_features: Dict[str, float] = {
            "transition_density_per_1k": transition_density,
            "boilerplate_density_per_1k": boilerplate_density,
            "bullet_density": layout_density,
            "transition_hits": transition_hits,
            "boilerplate_hits": boilerplate_hits,
            "bullet_hits": bullet_hits,
            "heading_hits": heading_hits,
            "transition_score": transition_score,
            "boilerplate_score": boilerplate_score,
            "layout_score": layout_score,
        }

        return ToolResult(
            name=self.name,
            score=combined_score,
            reason=reason,
            raw_features=raw_features,
        )
