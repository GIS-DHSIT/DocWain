"""
Response confidence scoring and explainability for DocWain.

Provides an explicit 0-1 confidence score with reasoning trace for every response.
Five weighted scoring dimensions:
  - Evidence coverage (0.20): % of response sentences supported by chunks
  - Source diversity (0.15): unique documents contributing to answer
  - Entity grounding (0.20): named entities in response found in evidence
  - Extraction completeness (0.15): filled fields / expected fields
  - Judge verdict (0.30): pass=1.0, uncertain=0.6, fail=0.2
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# Match proper nouns (TitleCase) and acronyms (2+ uppercase letters)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

_VERDICT_SCORES = {"pass": 1.0, "uncertain": 0.6, "fail": 0.2}

_DIMENSION_WEIGHTS = {
    "evidence_coverage": 0.20,
    "source_diversity": 0.15,
    "entity_grounding": 0.20,
    "extraction_completeness": 0.15,
    "judge_verdict": 0.30,
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

_NUMERIC_PRECISION_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_TRIVIAL_NUMS = frozenset({
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "24", "25", "30", "40", "48", "50", "60", "72", "75", "90", "100",
})

def _tokenize(text: str) -> Set[str]:
    """Extract lowercase tokens from text (includes 2-char domain abbreviations like IV, BP, HR)."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2}

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s and len(s.strip()) > 5]

# TitleCase words that are common labels, not named entities
_ENTITY_FALSE_POSITIVES = frozenset({
    "total", "summary", "experience", "education", "skills",
    "certifications", "achievements", "technical", "functional",
    "contact", "address", "overview", "details", "description",
    "objective", "references", "projects", "languages", "profile",
    "diagnosis", "treatment", "medication", "assessment", "notes",
    "clause", "section", "terms", "conditions", "obligations",
    "payment", "invoice", "amount", "vendor", "balance", "date",
    "coverage", "premium", "exclusions", "deductible", "benefit",
    "name", "role", "position", "department", "company",
    "source", "sources", "primary", "supporting", "additional",
    "key", "strengths", "comparison", "criterion", "result",
    "note", "based", "according", "evidence", "document",
})

def _extract_entities(text: str) -> Set[str]:
    """Extract named entities (TitleCase multi-word phrases + acronyms) from text."""
    entities = set()
    for m in _ENTITY_RE.finditer(text):
        entity = m.group().lower()
        # Filter out common label words that happen to be TitleCase
        words = entity.split()
        if len(words) == 1 and words[0] in _ENTITY_FALSE_POSITIVES:
            continue
        entities.add(entity)
    # Also extract acronyms (CEO, MBA, API, etc.) — common in professional documents
    _generic_acronyms = {"the", "and", "for", "not", "are", "but", "was", "has"}
    acronyms = {m.group().lower() for m in _ACRONYM_RE.finditer(text)
                if m.group().lower() not in _generic_acronyms}
    return entities | acronyms

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

    Adaptive threshold: short factual responses (1-2 sentences) use a lower
    threshold since concise answers naturally share fewer filler words with
    verbose evidence chunks. Long responses (5+ sentences) use a stricter
    threshold since they should draw more heavily from evidence.
    """
    sentences = _split_sentences(response)
    if not sentences:
        return 0.0, "No sentences to evaluate"

    all_chunk_tokens: Set[str] = set()
    for c in chunk_texts:
        all_chunk_tokens.update(_tokenize(c))

    # Adaptive threshold based on response length
    n_sent = len(sentences)
    if n_sent <= 2:
        eff_threshold = max(0.25, threshold - 0.10)  # more lenient for short answers
    elif n_sent >= 5:
        eff_threshold = min(0.55, threshold + 0.05)  # stricter for long answers
    else:
        eff_threshold = threshold

    supported = 0
    _partial_support = 0  # sentences with some evidence but below threshold
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        containment = _asymmetric_containment(sent_tokens, all_chunk_tokens)
        if containment >= eff_threshold:
            supported += 1
        elif containment >= eff_threshold * 0.6:
            _partial_support += 1

    # Partial support contributes 0.5 weight
    effective_supported = supported + _partial_support * 0.5
    ratio = effective_supported / len(sentences)
    reason = f"Evidence coverage: {supported}/{len(sentences)} sentences supported, {_partial_support} partial ({ratio:.0%})"
    return min(ratio, 1.0), reason

def score_source_diversity(
    sources: List[Dict[str, Any]], max_expected: int = 5
) -> tuple[float, str]:
    """Score how many unique documents contribute to the answer."""
    if not sources:
        return 0.0, "Source diversity: no sources provided"

    unique_docs: Set[str] = set()
    for src in sources:
        doc_id = (
            src.get("document_id") or src.get("doc_id") or src.get("source", "")
            or src.get("file_name") or src.get("source_name") or src.get("document_name", "")
        )
        if doc_id:
            unique_docs.add(str(doc_id))

    if not unique_docs:
        unique_docs = {str(i) for i in range(len(sources))}

    count = len(unique_docs)
    score = min(count / max(max_expected, 1), 1.0)
    reason = f"Source diversity: {count} unique document(s) (score {score:.2f})"
    return score, reason

def _fuzzy_entity_match(entity: str, evidence_lower: str, evidence_entities: Set[str]) -> bool:
    """Check if entity appears in evidence, with fuzzy matching for names.

    Handles: exact match, substring match, initial-based match (J. Smith ≈ John Smith),
    and last-name-only match.
    """
    if entity in evidence_entities or entity in evidence_lower:
        return True
    # Last-name match: "smith" in "john smith"
    parts = entity.split()
    if len(parts) >= 2:
        last = parts[-1]
        if last in evidence_lower:
            return True
    # Initial match: "j smith" matches "john smith" or "j. smith"
    if len(parts) == 2 and len(parts[0]) <= 2:
        initial = parts[0].rstrip(".")
        last = parts[1]
        for ent in evidence_entities:
            ent_parts = ent.split()
            if len(ent_parts) >= 2 and ent_parts[-1] == last and ent_parts[0].startswith(initial):
                return True
    return False

def score_entity_grounding(
    response: str, chunk_texts: List[str]
) -> tuple[float, str]:
    """Score what fraction of named entities in response appear in evidence.

    Uses fuzzy matching for names (initials, last-name-only) and includes
    acronym detection.
    """
    response_entities = _extract_entities(response)
    if not response_entities:
        # Short/factual responses (yes/no, numbers, brief answers) are naturally
        # entity-free — don't penalize them even if evidence has entities
        if len(response) < 80:
            return 1.0, "Entity grounding: short factual response, no entities expected"
        # Penalize generic answers when evidence contains specific entities
        all_evidence = " ".join(chunk_texts) if chunk_texts else ""
        evidence_entities = _extract_entities(all_evidence)
        if len(evidence_entities) >= 2:
            return 0.65, "Entity grounding: response lacks specificity despite entity-rich evidence"
        return 1.0, "Entity grounding: no named entities to verify"

    evidence_text = " ".join(chunk_texts)
    evidence_entities = _extract_entities(evidence_text)
    evidence_lower = evidence_text.lower()

    grounded = 0
    for ent in response_entities:
        if _fuzzy_entity_match(ent, evidence_lower, evidence_entities):
            grounded += 1

    ratio = grounded / len(response_entities)
    reason = f"Entity grounding: {grounded}/{len(response_entities)} entities found in evidence ({ratio:.0%})"
    return ratio, reason

# Domain-specific critical fields — if these are missing, penalize harder
_DOMAIN_CRITICAL_FIELDS: Dict[str, set] = {
    "hr": {"name", "skills", "technical_skills", "experience", "education"},
    "medical": {"patient", "diagnosis", "medications", "lab_results"},
    "invoice": {"items", "totals", "amount", "vendor", "date"},
    "legal": {"clauses", "parties", "obligations", "effective_date"},
    "policy": {"coverage", "premium", "policyholder", "exclusions", "effective_date"},
}

def score_extraction_completeness(
    schema: Any, domain: Optional[str] = None
) -> tuple[float, str]:
    """Score how many expected fields were filled in the extraction.

    Domain-aware: critical fields for the detected domain carry heavier weight.
    """
    if schema is None:
        return 0.5, "Extraction completeness: no schema available (default 0.5)"

    filled = 0
    total = 0
    critical_filled = 0
    critical_total = 0

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

    # Get domain-specific critical fields
    critical_keys = _DOMAIN_CRITICAL_FIELDS.get((domain or "").lower(), set())

    for key, value in schema_dict.items():
        if key.startswith("_") or key in ("domain", "intent", "doc_domain"):
            continue
        total += 1
        is_critical = key.lower() in critical_keys or any(
            ck in key.lower() for ck in critical_keys
        )
        if is_critical:
            critical_total += 1

        has_value = (
            value is not None
            and value != ""
            and value != []
            and value != {}
        )
        if has_value:
            if isinstance(value, str) and len(value.strip()) < 3:
                continue
            filled += 1
            if is_critical:
                critical_filled += 1

    if total == 0:
        return 0.5, "Extraction completeness: no fields to evaluate (default 0.5)"

    # Weighted ratio: critical fields count 2x
    if critical_total > 0:
        regular_filled = filled - critical_filled
        regular_total = total - critical_total
        weighted_score = (
            (critical_filled * 2.0 + regular_filled)
            / (critical_total * 2.0 + regular_total)
        )
    else:
        weighted_score = filled / total

    reason = f"Extraction completeness: {filled}/{total} fields ({critical_filled}/{critical_total} critical)"
    return weighted_score, reason

def score_numeric_precision(
    response: str, chunk_texts: List[str]
) -> tuple[float, str]:
    """Score whether numbers in the response are grounded in evidence.

    Fabricated numbers are a strong hallucination signal.  Returns 1.0
    when all response numbers appear in the evidence, and penalizes
    proportionally for ungrounded numbers.
    """
    _NUM_RE = _NUMERIC_PRECISION_RE
    _TRIVIAL = _TRIVIAL_NUMS

    response_nums = {n for n in _NUM_RE.findall(response) if n not in _TRIVIAL}
    if not response_nums:
        return 1.0, "Numeric precision: no significant numbers to verify"

    evidence_text = " ".join(chunk_texts).lower()
    evidence_nums = set(_NUM_RE.findall(evidence_text))

    grounded = sum(1 for n in response_nums if n in evidence_nums)
    ratio = grounded / len(response_nums)
    reason = f"Numeric precision: {grounded}/{len(response_nums)} numbers grounded ({ratio:.0%})"
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

    # 1. Evidence coverage (0.20)
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

    # 4. Extraction completeness (0.15)
    ext_score, ext_reason = score_extraction_completeness(schema, domain)
    dimensions["extraction_completeness"] = ext_score
    reasoning.append(ext_reason)

    # 5. Judge verdict (0.30)
    jud_score, jud_reason = score_judge_verdict(verdict_status)
    dimensions["judge_verdict"] = jud_score
    reasoning.append(jud_reason)

    # 6. Numeric precision (bonus/penalty — not weighted, applied as modifier)
    num_score, num_reason = score_numeric_precision(response, chunk_texts)
    dimensions["numeric_precision"] = num_score
    reasoning.append(num_reason)

    # Weighted sum
    overall = sum(
        dimensions[dim] * weight for dim, weight in _DIMENSION_WEIGHTS.items()
    )

    # Apply numeric precision penalty for fabricated numbers
    if num_score < 0.7:
        num_penalty = (0.7 - num_score) * 0.15
        overall -= num_penalty
        reasoning.append(f"Numeric fabrication penalty: -{num_penalty:.3f}")

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

    # Entity-evidence cross-penalty: if response has many entities but few grounded,
    # and numeric precision is also weak, compound the penalty
    if ent_score < 0.5 and num_score < 0.7:
        cross_penalty = (0.5 - ent_score) * (0.7 - num_score) * 0.2
        overall -= cross_penalty
        reasoning.append(f"Entity+numeric cross-penalty: -{cross_penalty:.3f}")

    # Response length bonus: very short responses (<50 chars) that pass all checks
    # are likely accurate (direct answers), give a small boost
    if len(response.strip()) < 50 and cov_score >= 0.7 and ent_score >= 0.8:
        short_bonus = 0.03
        overall += short_bonus
        reasoning.append(f"Concise accurate response bonus: +{short_bonus:.3f}")

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
