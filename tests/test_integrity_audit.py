"""Tests for post-ingestion integrity audit."""
import pytest
from unittest.mock import MagicMock
from src.docwain_intel.models import (
    SemanticUnit, UnitType, StructuredDocument,
    EntitySpan, FactTriple, ExtractionResult,
)
from src.docwain_intel.integrity_audit import run_integrity_audit, IntegrityReport


def _make_unit(uid, text):
    return SemanticUnit(unit_id=uid, unit_type=UnitType.PARAGRAPH, text=text, page_start=1, page_end=1)


def _make_entity(eid, text, label):
    return EntitySpan(entity_id=eid, text=text, normalized=text.lower(), label=label, unit_id="u1", confidence=0.9, source="spacy")


def _make_fact(fid, subj, pred, obj_val, raw):
    return FactTriple(fact_id=fid, subject_id=subj, predicate=pred, object_value=obj_val, unit_id="u1", raw_text=raw, confidence=0.9, extraction_method="dep")


class TestIntegrityAudit:
    def test_all_checks_pass(self):
        units = [_make_unit("u1", "John works at Google."), _make_unit("u2", "Sarah is a manager.")]
        doc = StructuredDocument(document_id="doc_001", units=units, unit_count=2)
        entities = [_make_entity("e1", "John", "PERSON"), _make_entity("e2", "Google", "ORG")]
        facts = [_make_fact("f1", "e1", "WORKED_AT", "Google", "John works at Google.")]
        extraction = ExtractionResult(document_id="doc_001", entities=entities, facts=facts)

        # Mock storage backends — everything stored correctly
        vector_store = MagicMock()
        vector_store.count_vectors.return_value = 2  # matches unit count
        graph_store = MagicMock()
        graph_store.count_entities.return_value = 2
        graph_store.count_facts.return_value = 1
        graph_store.get_fact_raw_text.return_value = "John works at Google."

        report = run_integrity_audit(doc, extraction, vector_store=vector_store, graph_store=graph_store)
        assert isinstance(report, IntegrityReport)
        assert report.passed
        assert report.vector_count_match
        assert report.entity_count_match
        assert report.fact_count_match

    def test_vector_count_mismatch(self):
        units = [_make_unit("u1", "Hello."), _make_unit("u2", "World.")]
        doc = StructuredDocument(document_id="doc_001", units=units, unit_count=2)
        extraction = ExtractionResult(document_id="doc_001")

        vector_store = MagicMock()
        vector_store.count_vectors.return_value = 1  # should be 2
        graph_store = MagicMock()
        graph_store.count_entities.return_value = 0
        graph_store.count_facts.return_value = 0

        report = run_integrity_audit(doc, extraction, vector_store=vector_store, graph_store=graph_store)
        assert not report.vector_count_match
        assert not report.passed

    def test_entity_count_mismatch(self):
        doc = StructuredDocument(document_id="doc_001", units=[], unit_count=0)
        entities = [_make_entity("e1", "John", "PERSON"), _make_entity("e2", "Google", "ORG")]
        extraction = ExtractionResult(document_id="doc_001", entities=entities)

        vector_store = MagicMock()
        vector_store.count_vectors.return_value = 0
        graph_store = MagicMock()
        graph_store.count_entities.return_value = 1  # should be 2
        graph_store.count_facts.return_value = 0

        report = run_integrity_audit(doc, extraction, vector_store=vector_store, graph_store=graph_store)
        assert not report.entity_count_match

    def test_fact_provenance_check(self):
        units = [_make_unit("u1", "John works at Google as an engineer.")]
        doc = StructuredDocument(document_id="doc_001", units=units, unit_count=1)
        facts = [_make_fact("f1", "e1", "WORKED_AT", "Google", "John works at Google")]
        extraction = ExtractionResult(document_id="doc_001", facts=facts)

        vector_store = MagicMock()
        vector_store.count_vectors.return_value = 1
        graph_store = MagicMock()
        graph_store.count_entities.return_value = 0
        graph_store.count_facts.return_value = 1
        graph_store.get_fact_raw_text.return_value = "John works at Google"

        report = run_integrity_audit(doc, extraction, vector_store=vector_store, graph_store=graph_store)
        assert report.provenance_sample_passed

    def test_provenance_fails_on_mismatch(self):
        units = [_make_unit("u1", "Completely different content.")]
        doc = StructuredDocument(document_id="doc_001", units=units, unit_count=1)
        facts = [_make_fact("f1", "e1", "WORKED_AT", "Google", "John works at Google")]
        extraction = ExtractionResult(document_id="doc_001", facts=facts)

        vector_store = MagicMock()
        vector_store.count_vectors.return_value = 1
        graph_store = MagicMock()
        graph_store.count_entities.return_value = 0
        graph_store.count_facts.return_value = 1
        graph_store.get_fact_raw_text.return_value = "Totally fabricated text not in doc"

        report = run_integrity_audit(doc, extraction, vector_store=vector_store, graph_store=graph_store)
        assert not report.provenance_sample_passed

    def test_no_stores_provided(self):
        """Without stores, audit should still run with warnings."""
        doc = StructuredDocument(document_id="doc_001", units=[], unit_count=0)
        extraction = ExtractionResult(document_id="doc_001")
        report = run_integrity_audit(doc, extraction)
        assert isinstance(report, IntegrityReport)
        # Without stores, can't verify — should mark as not checked
        assert len(report.warnings) >= 1 or report.passed  # either warns or passes vacuously

    def test_empty_extraction(self):
        doc = StructuredDocument(document_id="doc_001", units=[], unit_count=0)
        extraction = ExtractionResult(document_id="doc_001")
        vector_store = MagicMock()
        vector_store.count_vectors.return_value = 0
        graph_store = MagicMock()
        graph_store.count_entities.return_value = 0
        graph_store.count_facts.return_value = 0

        report = run_integrity_audit(doc, extraction, vector_store=vector_store, graph_store=graph_store)
        assert report.passed
