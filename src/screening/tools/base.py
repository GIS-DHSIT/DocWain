from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, List, Optional

from ..models import ScreeningContext, ToolResult

# Lightweight regex tokenization to keep runtime low.
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
PASSIVE_PATTERN = re.compile(r"\b(be|is|was|were|been|being|am|are)\s+[A-Za-z]+ed\b", re.IGNORECASE)


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def normalized_entropy(counts: Dict[str, int]) -> float:
    vocab_size = max(len(counts), 1)
    max_entropy = math.log2(vocab_size)
    if max_entropy == 0:
        return 0.0
    return shannon_entropy(counts) / max_entropy


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    if avg == 0:
        return 0.0
    return stdev(values) / avg if len(values) > 1 else 0.0


def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllables += 1
        previous_was_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def flesch_reading_ease(text: str) -> float:
    sentences = split_sentences(text)
    words = tokenize_words(text)
    if not sentences or not words:
        return 0.0
    syllable_count = sum(count_syllables(w) for w in words)
    words_per_sentence = len(words) / max(len(sentences), 1)
    syllables_per_word = syllable_count / max(len(words), 1)
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


@dataclass(init=False)
class ScreeningTool(ABC):
    """Base class for screening tools."""

    name: str
    category: str
    default_weight: float = 0.1
    requires_internet: bool = False
    supported_doc_types: Optional[List[str]] = None
    tool_version: str = "0.1"

    def applies_to(self, doc_type: Optional[str]) -> bool:
        if not self.supported_doc_types:
            return True
        if not doc_type:
            return False
        normalized = doc_type.upper()
        return normalized in {d.upper() for d in self.supported_doc_types}

    def _risk_level(self, score: float) -> str:
        if score >= 0.66:
            return "HIGH"
        if score >= 0.33:
            return "MEDIUM"
        return "LOW"

    def result(
        self,
        ctx: ScreeningContext,
        score: float,
        reasons: List[str],
        raw_features: Optional[Dict[str, object]] = None,
        actions: Optional[List[str]] = None,
        evidence_spans: Optional[List[Dict[str, object]]] = None,
    ) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            category=self.category,
            score_0_1=clamp(score),
            weight=ctx.config.weights.get(self.name, self.default_weight) if ctx.config else self.default_weight,
            risk_level=self._risk_level(score),
            reasons=reasons,
            raw_features=raw_features or {},
            actions=actions or [],
            evidence_spans=evidence_spans or [],
            tool_version=self.tool_version,
        )

    @abstractmethod
    def run(self, ctx: ScreeningContext) -> ToolResult:
        ...
