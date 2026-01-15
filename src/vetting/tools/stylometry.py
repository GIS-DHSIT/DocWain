"""
Stylometric heuristics: sentence length variation, punctuation profile,
approximate POS ratios, and readability oscillation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean, pstdev
from typing import Dict, List, Tuple

from .base import (
    ToolResult,
    VettingTool,
    clamp,
    split_sentences,
    tokenize_words,
)

try:  # Optional dependency
    import spacy
except Exception:  # pragma: no cover - optional path
    spacy = None  # type: ignore

_SPACY_NLP = None


def _get_spacy_model():
    """Lazily initialize spaCy if available."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if not spacy:
        return None
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception:
        try:
            # Blank English is extremely lightweight; only used when installed.
            _SPACY_NLP = spacy.blank("en")
        except Exception:
            _SPACY_NLP = None
    return _SPACY_NLP


VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _estimate_syllables(word: str) -> int:
    """Rudimentary syllable estimator for readability."""
    matches = VOWEL_RE.findall(word)
    if not matches:
        return 1
    count = len(matches)
    if word.endswith(("e", "es")) and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_reading_ease(text: str) -> float:
    """Compute a simple Flesch reading ease score."""
    sentences = split_sentences(text)
    words = tokenize_words(text)
    syllables = sum(_estimate_syllables(w) for w in words)

    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)

    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllables / word_count

    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


def _readability_oscillation(sentences: List[str], window: int = 5) -> float:
    """Standard deviation of readability across sliding windows."""
    if not sentences:
        return 0.0

    scores: List[float] = []
    for start in range(0, len(sentences), max(1, window)):
        chunk = " ".join(sentences[start : start + window])
        scores.append(_flesch_reading_ease(chunk))

    if len(scores) <= 1:
        return 0.0
    return pstdev(scores)


def _sentence_length_stats(sentences: List[str]) -> Tuple[float, float, float]:
    lengths = [len(tokenize_words(s)) for s in sentences if s.strip()]
    if not lengths:
        return 0.0, 0.0, 0.0
    mean_len = mean(lengths)
    std_len = pstdev(lengths) if len(lengths) > 1 else 0.0
    cv = std_len / mean_len if mean_len else 0.0
    return mean_len, std_len, cv


def _punctuation_profile(text: str) -> Dict[str, float]:
    marks = [",", ";", ":", "-", "\u2014", "(", ")", "\"", "'", "!", "?"]
    total_chars = max(len(text), 1)
    counts = {mark: text.count(mark) for mark in marks}
    freqs = {mark: count / total_chars for mark, count in counts.items()}
    return freqs


def _heuristic_pos_ratios(words: List[str]) -> Dict[str, float]:
    """Rough POS approximations without spaCy."""
    endings = {
        "verb": ("ing", "ed", "en"),
        "adverb": ("ly",),
        "adj": ("ous", "ive", "able", "ible", "al"),
    }
    counts = Counter()
    for w in words:
        lw = w.lower()
        if lw in {"the", "a", "an"}:
            counts["det"] += 1
        if any(lw.endswith(suf) for suf in endings["verb"]):
            counts["verb"] += 1
        elif any(lw.endswith(suf) for suf in endings["adverb"]):
            counts["adv"] += 1
        elif any(lw.endswith(suf) for suf in endings["adj"]):
            counts["adj"] += 1
        else:
            counts["noun_like"] += 1

    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _pos_ratios(text: str, words: List[str]) -> Dict[str, float]:
    nlp = _get_spacy_model()
    if not nlp or not getattr(nlp, "pipe_names", []):
        return _heuristic_pos_ratios(words)

    doc = nlp(text)
    counts = Counter(token.pos_ for token in doc if token.pos_)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


class StylometryTool(VettingTool):
    """Stylometric fingerprinting."""

    name = "stylometry"

    def run(self, text: str) -> ToolResult:
        sentences = split_sentences(text)
        words = tokenize_words(text)

        mean_len, std_len, cv = _sentence_length_stats(sentences)
        readability_std = _readability_oscillation(sentences)

        punct_profile = _punctuation_profile(text)
        punct_diversity = len([p for p, c in punct_profile.items() if c > 0]) / max(len(punct_profile), 1)
        punct_density = sum(punct_profile.values())

        pos_ratios = _pos_ratios(text, words)
        pos_entropy = 0.0
        if pos_ratios:
            pos_entropy = -sum(v * math.log2(v) for v in pos_ratios.values() if v > 0)
            pos_entropy = pos_entropy / math.log2(max(len(pos_ratios), 1))

        # Map raw metrics to suspicion scores (higher => more AI-like regularity).
        cv_score = clamp(1 - min(cv, 1.5) / 1.5)
        readability_score = clamp(1 - min(readability_std / 30.0, 1.0))
        punct_score = clamp(1 - punct_diversity)  # more uniform punctuation -> higher score
        pos_score = clamp(1 - pos_entropy)

        combined_score = clamp(0.3 * cv_score + 0.25 * readability_score + 0.2 * punct_score + 0.25 * pos_score)

        reason = (
            f"Sentence length CV={cv:.2f}, readability std={readability_std:.2f}, "
            f"punctuation diversity={punct_diversity:.2f}."
        )

        raw_features: Dict[str, float | Dict[str, float]] = {
            "sentence_length_mean": mean_len,
            "sentence_length_std": std_len,
            "sentence_length_cv": cv,
            "readability_std": readability_std,
            "punctuation_diversity": punct_diversity,
            "punctuation_density": punct_density,
            "punctuation_profile": punct_profile,
            "pos_ratios": pos_ratios,
            "pos_entropy": pos_entropy,
            "cv_score": cv_score,
            "readability_score": readability_score,
        }

        return ToolResult(
            name=self.name,
            score=combined_score,
            reason=reason,
            raw_features=raw_features,
        )
