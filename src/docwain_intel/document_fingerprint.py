"""Domain-agnostic document fingerprinting.

Computes a multi-dimensional feature vector from extraction results
to auto-tag documents WITHOUT hardcoded domain labels.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import math
import re
import threading
from collections import Counter
from typing import List

from .models import (
    DocumentFingerprint,
    ExtractionResult,
    StructuredDocument,
    UnitType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"[\d$%#]+")

_FORMAL_WORDS = frozenset({
    "herein", "thereof", "pursuant", "whereas", "shall", "hereby",
    "aforementioned", "notwithstanding", "hereunder", "therein",
    "hereafter", "thereto", "heretofore", "whomsoever",
})

_PASSIVE_MARKERS = frozenset({
    "is", "are", "was", "were", "been", "being",
    "be", "am",
})

_PASSIVE_SUFFIX = "ed"

_MAX_TEXT_CHARS = 100_000

_logger = get_logger(__name__)

_spacy_nlp = None
_spacy_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_all_text(doc: StructuredDocument) -> str:
    """Concatenate all unit texts."""
    return " ".join(u.text for u in doc.units if u.text)

def _compute_entity_distribution(extraction: ExtractionResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ent in extraction.entities:
        counts[ent.label] = counts.get(ent.label, 0) + 1
    return counts

def _compute_structure_profile(doc: StructuredDocument) -> dict[str, int]:
    counts: dict[str, int] = {}
    for u in doc.units:
        key = u.unit_type.value  # e.g. "paragraph", "table"
        counts[key] = counts.get(key, 0) + 1
    return counts

def _compute_numeric_density(all_text: str) -> float:
    if not all_text:
        return 0.0
    tokens = all_text.split()
    if not tokens:
        return 0.0
    numeric_count = sum(1 for t in tokens if _NUMERIC_RE.search(t))
    return numeric_count / len(tokens)

def _compute_entity_density(all_text: str, extraction: ExtractionResult) -> float:
    if not all_text:
        return 0.0
    word_count = len(all_text.split())
    if word_count == 0:
        return 0.0
    return len(extraction.entities) / word_count

def _compute_formality(doc: StructuredDocument, all_text: str) -> float:
    if not all_text:
        return 0.5  # neutral default

    # Split into sentences (rough heuristic)
    sentences = re.split(r"[.!?]+", all_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.5

    signals = 0.0
    total_checks = 0

    # 1. Average sentence length
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    total_checks += 1
    if avg_len > 20:
        signals += 1.0
    elif avg_len > 12:
        signals += 0.5

    # 2. Formal vocabulary
    words_lower = set(all_text.lower().split())
    formal_hits = len(words_lower & _FORMAL_WORDS)
    total_checks += 1
    if formal_hits >= 3:
        signals += 1.0
    elif formal_hits >= 1:
        signals += 0.5

    # 3. Passive voice markers (simplified: "was/were/is/are" followed by *ed word)
    tokens = all_text.lower().split()
    passive_count = 0
    for i in range(len(tokens) - 1):
        if tokens[i] in _PASSIVE_MARKERS and tokens[i + 1].endswith(_PASSIVE_SUFFIX):
            passive_count += 1
    total_checks += 1
    if passive_count >= 2:
        signals += 1.0
    elif passive_count >= 1:
        signals += 0.5

    # 4. KV_GROUP units suggest structured/formal content
    kv_count = sum(1 for u in doc.units if u.unit_type == UnitType.KV_GROUP)
    total_checks += 1
    if kv_count >= 2:
        signals += 1.0
    elif kv_count >= 1:
        signals += 0.5

    return min(signals / total_checks, 1.0) if total_checks else 0.5

def _compute_structure_complexity(doc: StructuredDocument) -> float:
    """Shannon entropy of unit type distribution."""
    if not doc.units:
        return 0.0
    counts = Counter(u.unit_type.value for u in doc.units)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def _compute_relational_density(
    doc: StructuredDocument, extraction: ExtractionResult
) -> float:
    unit_count = len(doc.units)
    if unit_count == 0:
        return 0.0
    return len(extraction.facts) / unit_count

def _get_spacy_nlp():
    """Return a shared spaCy model instance using double-check locking."""
    global _spacy_nlp
    if _spacy_nlp is None:
        with _spacy_lock:
            if _spacy_nlp is None:
                import spacy
                _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

def _compute_auto_tags(all_text: str) -> List[str]:
    """Extract top noun-chunk lemmas via spaCy. Returns empty list on failure."""
    if not all_text or not all_text.strip():
        return []
    try:
        nlp = _get_spacy_nlp()

        # Process only first 100k chars to avoid memory issues
        text = all_text
        if len(text) > _MAX_TEXT_CHARS:
            _logger.warning(
                "Text truncated from %d to %d chars for auto-tag extraction",
                len(text),
                _MAX_TEXT_CHARS,
            )
            text = text[:_MAX_TEXT_CHARS]

        doc = nlp(text)
        chunk_counter: Counter = Counter()
        for chunk in doc.noun_chunks:
            lemma = chunk.lemma_.lower().strip()
            # Skip very short or stopword-only chunks
            if len(lemma) <= 2:
                continue
            chunk_counter[lemma] += 1

        # Return top 5 by frequency
        return [lemma for lemma, _ in chunk_counter.most_common(5)]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fingerprint(
    doc: StructuredDocument,
    extraction: ExtractionResult,
) -> DocumentFingerprint:
    """Compute a domain-agnostic document fingerprint.

    Parameters
    ----------
    doc:
        The structured document (units).
    extraction:
        Entity/fact extraction results for the document.

    Returns
    -------
    DocumentFingerprint with all fields populated.
    """
    all_text = _collect_all_text(doc)

    return DocumentFingerprint(
        entity_distribution=_compute_entity_distribution(extraction),
        structure_profile=_compute_structure_profile(doc),
        numeric_density=_compute_numeric_density(all_text),
        entity_density=_compute_entity_density(all_text, extraction),
        formality_score=_compute_formality(doc, all_text),
        structure_complexity=_compute_structure_complexity(doc),
        relational_density=_compute_relational_density(doc, extraction),
        auto_tags=_compute_auto_tags(all_text),
    )
