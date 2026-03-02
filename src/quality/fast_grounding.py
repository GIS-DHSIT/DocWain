from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

_MODAL_VERBS = {
    "should",
    "must",
    "will",
    "may",
    "might",
    "can",
    "could",
    "would",
    "shall",
    "need",
    "needs",
    "required",
    "requires",
}


@dataclass
class GroundingResult:
    supported_ratio: float
    critical_supported_ratio: float
    supported_sentences: List[str]
    unsupported_sentences: List[str]
    sentence_scores: List[float]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    rough = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in rough if s.strip()]


def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in tokens if len(t) > 2]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / max(len(union), 1)


def _sentence_critical(sentence: str) -> bool:
    if not sentence:
        return False
    if re.search(r"\d", sentence):
        return True
    if _ENTITY_RE.search(sentence):
        return True
    lowered = sentence.lower()
    return any(modal in lowered.split() for modal in _MODAL_VERBS)


def _key_terms(sentence: str) -> List[str]:
    digits = re.findall(r"\d+(?:\.\d+)?", sentence)
    entities = [m.group(0) for m in _ENTITY_RE.finditer(sentence)]
    terms = [t.lower() for t in digits + entities if t]
    return list(dict.fromkeys(terms))


def _contains_key_terms(sentence_terms: Sequence[str], chunk_text: str) -> bool:
    if not sentence_terms or not chunk_text:
        return False
    lowered = chunk_text.lower()
    return any(term.lower() in lowered for term in sentence_terms)


def evaluate_grounding(
    answer: str,
    chunk_texts: Sequence[str],
    *,
    support_threshold: float = 0.18,
) -> GroundingResult:
    sentences = _split_sentences(answer)
    if not sentences:
        return GroundingResult(0.0, 0.0, [], [], [])

    chunk_tokens = [_tokenize(text) for text in chunk_texts]
    chunk_lowers = [text.lower() for text in chunk_texts]

    supported_sentences: List[str] = []
    unsupported_sentences: List[str] = []
    scores: List[float] = []

    critical_total = 0
    critical_supported = 0

    for sentence in sentences:
        tokens = _tokenize(sentence)
        if not tokens:
            scores.append(1.0)
            supported_sentences.append(sentence)
            continue
        key_terms = _key_terms(sentence)
        best = 0.0
        for idx, chunk in enumerate(chunk_tokens):
            overlap = _jaccard(tokens, chunk)
            if overlap > best:
                best = overlap
            if overlap >= support_threshold:
                best = max(best, overlap)
                break
        key_match = any(_contains_key_terms(key_terms, chunk_lowers[i]) for i in range(len(chunk_lowers)))
        supported = best >= support_threshold or key_match
        scores.append(best)
        if supported:
            supported_sentences.append(sentence)
        else:
            unsupported_sentences.append(sentence)
        if _sentence_critical(sentence):
            critical_total += 1
            if supported:
                critical_supported += 1

    supported_ratio = len(supported_sentences) / max(len(sentences), 1)
    if critical_total:
        critical_ratio = critical_supported / max(critical_total, 1)
    else:
        critical_ratio = 1.0

    return GroundingResult(
        supported_ratio=round(supported_ratio, 4),
        critical_supported_ratio=round(critical_ratio, 4),
        supported_sentences=supported_sentences,
        unsupported_sentences=unsupported_sentences,
        sentence_scores=scores,
    )
