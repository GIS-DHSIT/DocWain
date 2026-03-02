"""
Response confidence scoring and explainability for DocWain.

Provides an explicit 0-1 confidence score with reasoning trace for every response.
Five weighted scoring dimensions:
  - Evidence coverage (0.30): % of response sentences supported by chunks
  - Source diversity (0.15): unique documents contributing to answer
  - Entity grounding (0.20): named entities in response found in evidence
  - Extraction completeness (0.20): filled fields / expected fields
  - Judge verdict (0.15): pass=1.0, uncertain=0.6, fail=0.2
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")

_VERDICT_SCORES = {"pass": 1.0, "uncertain": 0.6, "fail": 0.2}

_DIMENSION_WEIGHTS = {
    "evidence_coverage": 0.30,
    "source_diversity": 0.15,
    "entity_grounding": 0.20,
    "extraction_completeness": 0.20,
    "judge_verdict": 0.15,
}


@dataclass
class ConfidenceResult:
    """Confidence scoring result with reasoning trace."""

    score: float
    dimensions: Dict[str, float]
    reasoning: List[str]
    level: str  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "dimensions": {k: round(v, 3) for k, v in self.dimensions.items()},
            "reasoning": self.reasoning,
        }


def _tokenize(text: str) -> Set[str]:
    """Extract lowercase tokens from text."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s and len(s.strip()) > 5]


def _extract_entities(text: str) -> Set[str]:
    """Extract named entities (TitleCase multi-word phrases) from text."""
    return {m.group().lower() for m in _ENTITY_RE.finditer(text)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / max(len(union), 1)


def _asymmetric_containment(response_tokens: Set[str], evidence_tokens: Set[str]) -> float:
    """Asymmetric keyword containment: what fraction of response tokens appear in evidence."""
    if not response_tokens:
        return 0.0
    overlap = response_tokens & evidence_tokens
    return len(overlap) / len(response_tokens)


def score_evidence_coverage(
    response: str, chunk_texts: List[str], threshold: float = 0.40
) -> tuple[float, str]:
    """Score what fraction of response sentences are supported by evidence.

    Uses asymmetric containment (response tokens found in evidence) rather
    than Jaccard, which penalises when evidence is much larger than response.
    """
    sentences = _split_sentences(response)
    if not sentences:
        return 0.0, "No sentences to evaluate"

    all_chunk_tokens: Set[str] = set()
    for c in chunk_texts:
        all_chunk_tokens.update(_tokenize(c))

    supported = 0
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        containment = _asymmetric_containment(sent_tokens, all_chunk_tokens)
        if containment >= threshold:
            supported += 1

    ratio = supported / len(sentences)
    reason = f"Evidence coverage: {supported}/{len(sentences)} sentences supported ({ratio:.0%})"
    return ratio, reason


def score_source_diversity(
    sources: List[Dict[str, Any]], max_expected: int = 5
) -> tuple[float, str]:
    """Score how many unique documents contribute to the answer."""
    if not sources:
        return 0.0, "Source diversity: no sources provided"

    unique_docs: Set[str] = set()
    for src in sources:
        doc_id = src.get("document_id") or src.get("doc_id") or src.get("source", "")
        if doc_id:
            unique_docs.add(str(doc_id))

    if not unique_docs:
        unique_docs = {str(i) for i in range(len(sources))}

    count = len(unique_docs)
    score = min(count / max(max_expected, 1), 1.0)
    reason = f"Source diversity: {count} unique document(s) (score {score:.2f})"
    return score, reason


def score_entity_grounding(
    response: str, chunk_texts: List[str]
) -> tuple[float, str]:
    """Score what fraction of named entities in response appear in evidence."""
    response_entities = _extract_entities(response)
    if not response_entities:
        return 1.0, "Entity grounding: no named entities to verify"

    evidence_text = " ".join(chunk_texts)
    evidence_entities = _extract_entities(evidence_text)
    evidence_lower = evidence_text.lower()

    grounded = 0
    for ent in response_entities:
        if ent in evidence_entities or ent in evidence_lower:
            grounded += 1

    ratio = grounded / len(response_entities)
    reason = f"Entity grounding: {grounded}/{len(response_entities)} entities found in evidence ({ratio:.0%})"
    return ratio, reason


def score_extraction_completeness(
    schema: Any, domain: Optional[str] = None
) -> tuple[float, str]:
    """Score how many expected fields were filled in the extraction."""
    if schema is None:
        return 0.5, "Extraction completeness: no schema available (default 0.5)"

    filled = 0
    total = 0

    schema_dict = None
    if hasattr(schema, "to_dict"):
        schema_dict = schema.to_dict()
    elif hasattr(schema, "__dict__"):
        schema_dict = {
            k: v for k, v in schema.__dict__.items() if not k.startswith("_")
        }
    elif isinstance(schema, dict):
        schema_dict = schema

    if not schema_dict:
        return 0.5, "Extraction completeness: could not inspect schema (default 0.5)"

    for key, value in schema_dict.items():
        if key.startswith("_") or key in ("domain", "intent", "doc_domain"):
            continue
        total += 1
        if value is not None and value != "" and value != [] and value != {}:
            if isinstance(value, str) and len(value.strip()) < 3:
                continue
            filled += 1

    if total == 0:
        return 0.5, "Extraction completeness: no fields to evaluate (default 0.5)"

    ratio = filled / total
    reason = f"Extraction completeness: {filled}/{total} fields populated ({ratio:.0%})"
    return ratio, reason


def score_judge_verdict(verdict_status: Optional[str]) -> tuple[float, str]:
    """Convert judge verdict to confidence score."""
    if verdict_status is None:
        return 0.5, "Judge verdict: no verdict available (default 0.5)"

    score = _VERDICT_SCORES.get(verdict_status, 0.5)
    reason = f"Judge verdict: '{verdict_status}' -> {score:.1f}"
    return score, reason


def compute_confidence(
    response: str,
    chunk_texts: List[str],
    sources: List[Dict[str, Any]],
    schema: Any = None,
    verdict_status: Optional[str] = None,
    domain: Optional[str] = None,
    cloud_verified: bool = False,
) -> ConfidenceResult:
    """
    Compute overall confidence score with reasoning trace.

    Returns ConfidenceResult with score 0-1, per-dimension scores, and reasoning.
    """
    dimensions: Dict[str, float] = {}
    reasoning: List[str] = []

    # 1. Evidence coverage (0.30)
    cov_score, cov_reason = score_evidence_coverage(response, chunk_texts)
    dimensions["evidence_coverage"] = cov_score
    reasoning.append(cov_reason)

    # 2. Source diversity (0.15)
    div_score, div_reason = score_source_diversity(sources)
    dimensions["source_diversity"] = div_score
    reasoning.append(div_reason)

    # 3. Entity grounding (0.20)
    ent_score, ent_reason = score_entity_grounding(response, chunk_texts)
    dimensions["entity_grounding"] = ent_score
    reasoning.append(ent_reason)

    # 4. Extraction completeness (0.20)
    ext_score, ext_reason = score_extraction_completeness(schema, domain)
    dimensions["extraction_completeness"] = ext_score
    reasoning.append(ext_reason)

    # 5. Judge verdict (0.15)
    jud_score, jud_reason = score_judge_verdict(verdict_status)
    dimensions["judge_verdict"] = jud_score
    reasoning.append(jud_reason)

    # Weighted sum
    overall = sum(
        dimensions[dim] * weight for dim, weight in _DIMENSION_WEIGHTS.items()
    )

    # Cloud verification bonus: bump confidence when answer was verified by cloud LLM
    if cloud_verified:
        cloud_bonus = 0.05
        overall += cloud_bonus
        dimensions["cloud_verification"] = 1.0
        reasoning.append(f"Cloud verification: +{cloud_bonus:.2f} bonus (verified by cloud LLM)")
    else:
        dimensions["cloud_verification"] = 0.0

    # Low-coverage penalty: reduce confidence when evidence coverage is poor
    if cov_score < 0.5:
        penalty = (0.5 - cov_score) * 0.1
        overall -= penalty
        reasoning.append(f"Low evidence coverage penalty: -{penalty:.3f}")

    overall = max(0.0, min(1.0, overall))

    # Level classification
    if overall >= 0.75:
        level = "high"
    elif overall >= 0.45:
        level = "medium"
    else:
        level = "low"

    reasoning.append(f"Overall confidence: {overall:.3f} ({level})")

    return ConfidenceResult(
        score=overall,
        dimensions=dimensions,
        reasoning=reasoning,
        level=level,
    )
