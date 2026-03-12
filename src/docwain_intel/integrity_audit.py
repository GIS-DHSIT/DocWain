"""Post-ingestion integrity audit for DocWain Intelligence Engine.

Validates that data stored in vector (Qdrant) and graph (Neo4j) backends
matches what was produced by the ingestion pipeline.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .models import ExtractionResult, StructuredDocument

logger = get_logger(__name__)

_PROVENANCE_SAMPLE_SIZE = 3
_PROVENANCE_KEYWORD_OVERLAP_THRESHOLD = 0.4

class IntegrityReport(BaseModel):
    """Result of a post-ingestion integrity audit."""

    document_id: str
    passed: bool = True
    vector_count_match: bool = True
    entity_count_match: bool = True
    fact_count_match: bool = True
    provenance_sample_passed: bool = True
    expected_vectors: int = 0
    actual_vectors: int = 0
    expected_entities: int = 0
    actual_entities: int = 0
    expected_facts: int = 0
    actual_facts: int = 0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

def _keyword_set(text: str) -> set[str]:
    """Extract lowercased non-trivial keywords from text."""
    return {w.lower().strip(".,;:!?\"'()[]") for w in text.split() if len(w) > 2}

def _check_provenance(
    doc: StructuredDocument,
    extraction: ExtractionResult,
    graph_store: Any,
) -> bool:
    """Sample facts from graph_store and verify raw_text traces back to document units."""
    facts = extraction.facts
    if not facts:
        return True

    sample = facts[:_PROVENANCE_SAMPLE_SIZE]
    all_unit_text = " ".join(u.text for u in doc.units)
    all_unit_keywords = _keyword_set(all_unit_text)
    if not all_unit_keywords:
        return True

    for fact in sample:
        stored_raw = graph_store.get_fact_raw_text(fact.fact_id)
        if stored_raw is None:
            return False

        stored_keywords = _keyword_set(stored_raw)
        if not stored_keywords:
            continue

        overlap = len(stored_keywords & all_unit_keywords) / len(stored_keywords)
        if overlap < _PROVENANCE_KEYWORD_OVERLAP_THRESHOLD:
            return False

    return True

def run_integrity_audit(
    doc: StructuredDocument,
    extraction: ExtractionResult,
    *,
    vector_store: Any = None,
    graph_store: Any = None,
) -> IntegrityReport:
    """Run post-ingestion integrity checks.

    Parameters
    ----------
    doc:
        The structured document produced by the ingestion pipeline.
    extraction:
        Entity/fact extraction results for the document.
    vector_store:
        Duck-typed object with ``count_vectors(document_id) -> int``.
    graph_store:
        Duck-typed object with ``count_entities(document_id) -> int``,
        ``count_facts(document_id) -> int``, and
        ``get_fact_raw_text(fact_id) -> str``.

    Returns
    -------
    IntegrityReport
    """
    report = IntegrityReport(document_id=doc.document_id)

    if vector_store is None and graph_store is None:
        report.warnings.append("Storage backends not provided, audit skipped")
        return report

    # --- Vector count check ---
    if vector_store is not None:
        report.expected_vectors = doc.unit_count
        report.actual_vectors = vector_store.count_vectors(doc.document_id)
        report.vector_count_match = report.expected_vectors == report.actual_vectors
        if not report.vector_count_match:
            report.errors.append(
                f"Vector count mismatch: expected {report.expected_vectors}, "
                f"got {report.actual_vectors}"
            )

    # --- Entity count check ---
    if graph_store is not None:
        report.expected_entities = len(extraction.entities)
        report.actual_entities = graph_store.count_entities(doc.document_id)
        report.entity_count_match = report.expected_entities == report.actual_entities
        if not report.entity_count_match:
            report.errors.append(
                f"Entity count mismatch: expected {report.expected_entities}, "
                f"got {report.actual_entities}"
            )

        # --- Fact count check ---
        report.expected_facts = len(extraction.facts)
        report.actual_facts = graph_store.count_facts(doc.document_id)
        report.fact_count_match = report.expected_facts == report.actual_facts
        if not report.fact_count_match:
            report.errors.append(
                f"Fact count mismatch: expected {report.expected_facts}, "
                f"got {report.actual_facts}"
            )

        # --- Provenance sample check ---
        report.provenance_sample_passed = _check_provenance(doc, extraction, graph_store)
        if not report.provenance_sample_passed:
            report.errors.append("Fact provenance check failed: stored raw_text does not trace back to document")

    # --- Overall pass/fail ---
    report.passed = (
        report.vector_count_match
        and report.entity_count_match
        and report.fact_count_match
        and report.provenance_sample_passed
    )

    return report
