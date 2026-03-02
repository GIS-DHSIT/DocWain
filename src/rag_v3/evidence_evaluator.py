"""Evidence sufficiency evaluator: checks whether retrieved chunks contain
enough evidence to answer a query BEFORE running extraction/generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceSufficiency:
    """Result of evidence evaluation."""
    coverage_score: float = 0.0
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    overall_score: float = 0.0
    is_sufficient: bool = False
    missing_entities: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.overall_score == 0.0 and (self.coverage_score or self.relevance_score or self.diversity_score):
            self.overall_score = (
                0.5 * self.coverage_score
                + 0.3 * self.relevance_score
                + 0.2 * self.diversity_score
            )
        if self.overall_score >= 0.35:
            self.is_sufficient = True


def evaluate_evidence(
    query: str,
    chunks: list,
    entity_hints: Optional[List[str]] = None,
    *,
    sufficiency_threshold: float = 0.35,
) -> EvidenceSufficiency:
    """Evaluate whether retrieved chunks are sufficient to answer the query.

    Args:
        query: Original user query
        chunks: Retrieved Chunk objects with .text, .score, .source attributes
        entity_hints: Entity names expected in results (for coverage check)
        sufficiency_threshold: Minimum overall score to consider sufficient
    """
    if not chunks:
        return EvidenceSufficiency(
            coverage_score=0.0,
            relevance_score=0.0,
            diversity_score=0.0,
            overall_score=0.0,
            is_sufficient=False,
            missing_entities=list(entity_hints or []),
        )

    # -- Coverage: check entity and keyword presence --
    entity_hints = entity_hints or []
    all_text = " ".join(getattr(c, "text", "") for c in chunks).lower()
    missing = []
    found_count = 0
    for entity in entity_hints:
        if entity.lower() in all_text:
            found_count += 1
        else:
            missing.append(entity)

    if entity_hints:
        entity_coverage = found_count / len(entity_hints)
    else:
        query_words = {w.lower() for w in query.split() if len(w) > 3}
        if query_words:
            matched = sum(1 for w in query_words if w in all_text)
            entity_coverage = matched / len(query_words)
        else:
            entity_coverage = 1.0

    # -- Relevance: average chunk score --
    scores = [getattr(c, "score", 0.0) for c in chunks if getattr(c, "score", None) is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    relevance = max(0.0, min(1.0, avg_score))

    # -- Diversity: unique documents and sections --
    unique_docs = set()
    unique_sections = set()
    for c in chunks:
        source = getattr(c, "source", None)
        if source and getattr(source, "document_name", None):
            unique_docs.add(source.document_name)
        meta = getattr(c, "meta", None) or {}
        sk = meta.get("section_kind") or meta.get("section_title")
        if sk:
            unique_sections.add(sk)

    doc_diversity = min(1.0, len(unique_docs) / 3.0)
    section_diversity = min(1.0, len(unique_sections) / 4.0)
    diversity = 0.6 * doc_diversity + 0.4 * section_diversity

    # -- Overall --
    overall = 0.5 * entity_coverage + 0.3 * relevance + 0.2 * diversity

    return EvidenceSufficiency(
        coverage_score=entity_coverage,
        relevance_score=relevance,
        diversity_score=diversity,
        overall_score=overall,
        is_sufficient=overall >= sufficiency_threshold,
        missing_entities=missing,
    )
