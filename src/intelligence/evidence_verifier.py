"""Evidence verification for LLM knowledge extractions.

Verifies that evidence spans quoted by the LLM actually exist in the source
text. Uses fuzzy string matching to handle minor formatting differences
between the LLM's quote and the original text.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Minimum similarity ratio for evidence to be considered valid
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Minimum evidence length to bother checking (very short quotes are unreliable)
MIN_EVIDENCE_LENGTH = 10


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation variance."""
    t = text.lower().strip()
    t = re.sub(r'\s+', ' ', t)
    # Normalize common punctuation variations
    t = t.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    t = t.replace('\u2013', '-').replace('\u2014', '-')
    return t


def _find_best_match(evidence: str, source: str) -> Tuple[float, int]:
    """Find the best matching substring in source for the given evidence.

    Uses a sliding window approach with SequenceMatcher for efficiency.
    Returns (best_ratio, best_position).
    """
    norm_evidence = _normalize_text(evidence)
    norm_source = _normalize_text(source)

    if not norm_evidence or not norm_source:
        return 0.0, -1

    # Quick check: exact substring match
    if norm_evidence in norm_source:
        return 1.0, norm_source.find(norm_evidence)

    # For short evidence, check if it's a substring with minor differences
    evidence_len = len(norm_evidence)
    if evidence_len < MIN_EVIDENCE_LENGTH:
        return 0.0, -1

    # Sliding window: check windows of similar length in source
    best_ratio = 0.0
    best_pos = -1
    window_size = evidence_len
    # Allow window to be 80%-120% of evidence length
    min_window = max(MIN_EVIDENCE_LENGTH, int(window_size * 0.8))
    max_window = int(window_size * 1.2)

    # Step through source in increments (not every character for performance)
    step = max(1, evidence_len // 4)

    for start in range(0, len(norm_source) - min_window + 1, step):
        for wsize in (window_size, min_window, max_window):
            end = min(start + wsize, len(norm_source))
            candidate = norm_source[start:end]

            ratio = SequenceMatcher(None, norm_evidence, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = start

            # Early exit if we found a very good match
            if best_ratio >= 0.95:
                return best_ratio, best_pos

    return best_ratio, best_pos


def verify_evidence(
    evidence: str,
    source_text: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[bool, float]:
    """Verify that an evidence span exists in the source text.

    Args:
        evidence: The quoted evidence from LLM extraction.
        source_text: The original document text to verify against.
        threshold: Minimum similarity ratio (0.0-1.0).

    Returns:
        Tuple of (is_verified, similarity_score).
    """
    if not evidence or not source_text:
        return False, 0.0

    if len(evidence.strip()) < MIN_EVIDENCE_LENGTH:
        # Very short evidence — can't reliably verify
        return False, 0.0

    ratio, _pos = _find_best_match(evidence, source_text)
    return ratio >= threshold, round(ratio, 3)


def verify_extraction_batch(
    extractions: List[Dict[str, Any]],
    source_text: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Verify a batch of extractions against source text.

    Each extraction dict must have an 'evidence' key.

    Args:
        extractions: List of extraction dicts with 'evidence' field.
        source_text: The original document text.
        threshold: Minimum similarity ratio.

    Returns:
        Tuple of (verified_extractions, rejected_extractions).
    """
    verified = []
    rejected = []

    for item in extractions:
        evidence = item.get("evidence", "")
        is_valid, score = verify_evidence(evidence, source_text, threshold)

        if is_valid:
            item["evidence_score"] = score
            verified.append(item)
        else:
            item["evidence_score"] = score
            item["rejection_reason"] = (
                f"Evidence not found in source (similarity={score:.2f}, "
                f"threshold={threshold})"
            )
            rejected.append(item)

    if rejected:
        logger.debug(
            "[EvidenceVerifier] Rejected %d/%d extractions (threshold=%.2f)",
            len(rejected), len(extractions), threshold,
        )

    return verified, rejected


def verify_knowledge_result(
    result,  # KnowledgeExtractionResult
    source_text: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
):
    """Verify all extractions in a KnowledgeExtractionResult against source text.

    Modifies the result in-place: removes unverified extractions and updates
    the rejected_count.

    Args:
        result: A KnowledgeExtractionResult instance.
        source_text: The original document text.
        threshold: Minimum similarity ratio.

    Returns:
        The modified result (also modified in-place).
    """
    total_rejected = 0

    # Verify entities
    verified_entities = []
    for ent in result.entities:
        is_valid, score = verify_evidence(ent.evidence, source_text, threshold)
        if is_valid:
            verified_entities.append(ent)
        else:
            total_rejected += 1
            logger.debug(
                "[EvidenceVerifier] Rejected entity '%s' (score=%.2f)",
                ent.name, score,
            )
    result.entities = verified_entities

    # Verify facts
    verified_facts = []
    for fact in result.facts:
        is_valid, score = verify_evidence(fact.evidence, source_text, threshold)
        if is_valid:
            verified_facts.append(fact)
        else:
            total_rejected += 1
            logger.debug(
                "[EvidenceVerifier] Rejected fact (score=%.2f): %s",
                score, fact.statement[:80],
            )
    result.facts = verified_facts

    # Verify relationships
    verified_rels = []
    for rel in result.relationships:
        is_valid, score = verify_evidence(rel.evidence, source_text, threshold)
        if is_valid:
            verified_rels.append(rel)
        else:
            total_rejected += 1
    result.relationships = verified_rels

    # Verify claims
    verified_claims = []
    for claim in result.claims:
        is_valid, score = verify_evidence(claim.evidence, source_text, threshold)
        if is_valid:
            verified_claims.append(claim)
        else:
            total_rejected += 1
    result.claims = verified_claims

    result.rejected_count += total_rejected

    logger.info(
        "[EvidenceVerifier] Verified result: entities=%d facts=%d rels=%d "
        "claims=%d rejected=%d",
        len(result.entities), len(result.facts), len(result.relationships),
        len(result.claims), total_rejected,
    )

    return result


__all__ = [
    "verify_evidence",
    "verify_extraction_batch",
    "verify_knowledge_result",
    "DEFAULT_SIMILARITY_THRESHOLD",
]
