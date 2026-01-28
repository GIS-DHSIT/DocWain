from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


@dataclass
class RelevanceResult:
    relevance_score: float
    query_overlap: float
    definitive: bool
    low_relevance: bool


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _query_overlap(query: str, chunk_text: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(chunk_text)
    return len(q_tokens & c_tokens) / max(len(q_tokens), 1)


def _is_definitive(answer: str) -> bool:
    lowered = (answer or "").lower()
    hedges = [
        "insufficient",
        "not enough",
        "unclear",
        "cannot determine",
        "can't determine",
        "not provided",
        "no evidence",
        "unknown",
        "might",
        "may",
        "could",
    ]
    return not any(h in lowered for h in hedges)


def evaluate_relevance_with_answer(
    query: str,
    answer: str,
    chunk_texts: Sequence[str],
    *,
    retrieval_confidence: Optional[float] = None,
    low_relevance_threshold: float = 0.18,
) -> RelevanceResult:
    combined = " ".join(chunk_texts)
    overlap = _query_overlap(query, combined)
    relevance = float(retrieval_confidence) if retrieval_confidence is not None else overlap
    definitive = _is_definitive(answer)
    low = (relevance < low_relevance_threshold) and definitive
    return RelevanceResult(
        relevance_score=round(relevance, 4),
        query_overlap=round(overlap, 4),
        definitive=definitive,
        low_relevance=low,
    )
