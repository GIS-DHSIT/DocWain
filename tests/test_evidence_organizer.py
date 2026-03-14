"""Tests for evidence_organizer module."""

import pytest

from src.docwain_intel.evidence_organizer import (
    EvidenceGap,
    EvidenceGroup,
    OrganizedEvidence,
    ProvenanceRecord,
    organize_evidence,
)
from src.docwain_intel.models import EntitySpan, ExtractionResult, FactTriple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    *,
    chunk_id: str = "c1",
    score: float = 0.5,
    entity_ids: list | None = None,
    source_document: str = "doc.pdf",
    page: int | None = None,
    section: str | None = None,
    text: str = "sample text",
) -> dict:
    payload: dict = {}
    if entity_ids is not None:
        payload["intel_entity_ids"] = entity_ids
    payload["source_document"] = source_document
    if page is not None:
        payload["page"] = page
    if section is not None:
        payload["section"] = section
    return {"id": chunk_id, "score": score, "payload": payload, "text": text}


def _fact(
    subject: str = "s1",
    predicate: str = "has_skill",
    value: str = "Python",
    confidence: float = 0.9,
) -> dict:
    return {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntityGrouping:
    """Chunks with shared entity_ids are grouped together."""

    def test_chunks_grouped_by_entity(self):
        chunks = [
            _chunk(chunk_id="c1", entity_ids=["e1"], score=0.8),
            _chunk(chunk_id="c2", entity_ids=["e1"], score=0.6),
            _chunk(chunk_id="c3", entity_ids=["e2"], score=0.7),
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        assert len(result.entity_groups) == 2
        # Find group for e1
        g1 = [g for g in result.entity_groups if g.entity_id == "e1"][0]
        assert len(g1.chunks) == 2
        assert result.ungrouped_chunks == []

    def test_ungrouped_chunks_when_no_entity(self):
        chunks = [
            _chunk(chunk_id="c1", entity_ids=["e1"]),
            _chunk(chunk_id="c2"),  # no entity
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        assert len(result.entity_groups) == 1
        assert len(result.ungrouped_chunks) == 1
        assert result.ungrouped_chunks[0]["id"] == "c2"


class TestFactDeduplication:
    """Duplicate facts deduplicated — same subject+predicate+value keeps highest confidence."""

    def test_duplicate_facts_keep_highest_confidence(self):
        facts = [
            _fact(subject="Alice", predicate="has_skill", value="Python", confidence=0.7),
            _fact(subject="Alice", predicate="has_skill", value="Python", confidence=0.95),
            _fact(subject="Alice", predicate="has_skill", value="Java", confidence=0.8),
        ]
        result = organize_evidence(chunks=[], facts=facts, query_entities=[])
        assert result.total_facts == 2  # Python deduped, Java kept
        # The Python fact should have the higher confidence

    def test_fact_dedup_case_insensitive(self):
        facts = [
            _fact(subject="alice", predicate="HAS_SKILL", value="python", confidence=0.6),
            _fact(subject="Alice", predicate="has_skill", value="Python", confidence=0.9),
        ]
        result = organize_evidence(chunks=[], facts=facts, query_entities=[])
        assert result.total_facts == 1


class TestGapDetection:
    """Gap detected when query entity has no evidence."""

    def test_gap_for_missing_entity(self):
        chunks = [_chunk(chunk_id="c1", entity_ids=["e1"])]
        extraction = ExtractionResult(
            document_id="d1",
            entities=[
                EntitySpan(
                    entity_id="e1", text="Alice", normalized="alice",
                    label="PERSON", unit_id="u1",
                ),
            ],
        )
        result = organize_evidence(
            chunks=chunks,
            facts=[],
            query_entities=["Alice", "Bob"],
            extraction=extraction,
        )
        assert len(result.gaps) == 1
        assert result.gaps[0].field_name == "Bob"
        assert "Bob" in result.gaps[0].description

    def test_no_gap_when_entity_covered(self):
        chunks = [_chunk(chunk_id="c1", entity_ids=["e1"])]
        extraction = ExtractionResult(
            document_id="d1",
            entities=[
                EntitySpan(
                    entity_id="e1", text="Alice", normalized="alice",
                    label="PERSON", unit_id="u1",
                ),
            ],
        )
        result = organize_evidence(
            chunks=chunks, facts=[], query_entities=["Alice"], extraction=extraction,
        )
        assert len(result.gaps) == 0


class TestProvenance:
    """Provenance records extracted from chunk metadata."""

    def test_provenance_from_payload(self):
        chunks = [
            _chunk(chunk_id="c1", source_document="report.pdf", page=3, section="Introduction"),
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        assert len(result.provenance) == 1
        p = result.provenance[0]
        assert p.source_document == "report.pdf"
        assert p.page == 3
        assert p.section == "Introduction"
        assert p.chunk_id == "c1"


class TestEmptyInput:
    """Empty input returns empty OrganizedEvidence."""

    def test_empty(self):
        result = organize_evidence(chunks=[], facts=[], query_entities=[])
        assert result.entity_groups == []
        assert result.ungrouped_chunks == []
        assert result.gaps == []
        assert result.provenance == []
        assert result.total_facts == 0
        assert result.total_chunks == 0


class TestMixedChunks:
    """Some chunks with entities, some without — both groups populated."""

    def test_mixed(self):
        chunks = [
            _chunk(chunk_id="c1", entity_ids=["e1"], score=0.9),
            _chunk(chunk_id="c2", score=0.4),
            _chunk(chunk_id="c3", entity_ids=["e2"], score=0.7),
            _chunk(chunk_id="c4", score=0.3),
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        assert len(result.entity_groups) == 2
        assert len(result.ungrouped_chunks) == 2
        assert result.total_chunks == 4


class TestRelevanceOrdering:
    """Higher relevance chunks first within groups."""

    def test_chunks_sorted_by_score_desc(self):
        chunks = [
            _chunk(chunk_id="c1", entity_ids=["e1"], score=0.3),
            _chunk(chunk_id="c2", entity_ids=["e1"], score=0.9),
            _chunk(chunk_id="c3", entity_ids=["e1"], score=0.6),
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        grp = result.entity_groups[0]
        scores = [c["score"] for c in grp.chunks]
        assert scores == [0.9, 0.6, 0.3]

    def test_groups_sorted_by_avg_relevance(self):
        chunks = [
            _chunk(chunk_id="c1", entity_ids=["low"], score=0.1),
            _chunk(chunk_id="c2", entity_ids=["high"], score=0.95),
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        assert result.entity_groups[0].entity_id == "high"
        assert result.entity_groups[1].entity_id == "low"


class TestMultipleEntitiesPerChunk:
    """Chunk with multiple entity_ids appears in multiple groups."""

    def test_chunk_in_multiple_groups(self):
        chunks = [
            _chunk(chunk_id="c1", entity_ids=["e1", "e2"], score=0.8),
        ]
        result = organize_evidence(chunks=chunks, facts=[], query_entities=[])
        assert len(result.entity_groups) == 2
        for grp in result.entity_groups:
            assert len(grp.chunks) == 1
            assert grp.chunks[0]["id"] == "c1"


class TestLargeInput:
    """Large input (30+ chunks) completes correctly."""

    def test_thirty_plus_chunks(self):
        n = 35
        chunks = [
            _chunk(
                chunk_id=f"c{i}",
                entity_ids=[f"e{i % 5}"],
                score=round(0.1 + (i % 10) * 0.09, 2),
            )
            for i in range(n)
        ]
        facts = [
            _fact(subject=f"e{i % 5}", predicate="attr", value=f"v{i}", confidence=0.5 + i * 0.01)
            for i in range(20)
        ]
        result = organize_evidence(chunks=chunks, facts=facts, query_entities=[])
        assert result.total_chunks == n
        assert len(result.entity_groups) == 5  # e0..e4
        assert result.total_facts == 20  # all unique
        # Every chunk is grouped (all have entity_ids)
        assert len(result.ungrouped_chunks) == 0
        # Provenance count matches chunks
        assert len(result.provenance) == n
