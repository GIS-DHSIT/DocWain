"""Tests for extraction verification and conflict detection."""
import pytest
from src.docwain_intel.models import (
    EntitySpan, FactTriple, ExtractionResult, VerificationResult,
    SemanticUnit, UnitType, StructuredDocument,
)
from src.docwain_intel.verification import verify_extraction


def _make_entity(eid, text, label, unit_id="u1"):
    return EntitySpan(entity_id=eid, text=text, normalized=text.lower(), label=label, unit_id=unit_id, confidence=0.9, source="spacy")


def _make_fact(fid, subj, pred, obj_val, unit_id="u1", raw="raw text"):
    return FactTriple(fact_id=fid, subject_id=subj, predicate=pred, object_value=obj_val, unit_id=unit_id, raw_text=raw, confidence=0.9, extraction_method="dep")


def _make_doc(units):
    return StructuredDocument(document_id="doc_001", units=units, unit_count=len(units))


class TestVerifyExtraction:
    def test_clean_extraction_passes(self):
        entities = [_make_entity("e1", "John Smith", "PERSON")]
        facts = [_make_fact("f1", "e1", "WORKED_AT", "Google")]
        extraction = ExtractionResult(document_id="doc_001", entities=entities, facts=facts)
        units = [SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH, text="John Smith worked at Google.", page_start=1, page_end=1)]
        result = verify_extraction(extraction, _make_doc(units))
        assert isinstance(result, VerificationResult)
        assert result.is_valid
        assert result.quality_score > 0.5

    def test_entity_type_inconsistency(self):
        """Same text labeled as different types should create a conflict."""
        entities = [
            _make_entity("e1", "Apple", "ORG", "u1"),
            _make_entity("e2", "Apple", "PRODUCT", "u2"),
        ]
        extraction = ExtractionResult(document_id="doc_001", entities=entities)
        units = [
            SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH, text="Apple is a company.", page_start=1, page_end=1),
            SemanticUnit(unit_id="u2", unit_type=UnitType.PARAGRAPH, text="Apple is a fruit.", page_start=2, page_end=2),
        ]
        result = verify_extraction(extraction, _make_doc(units))
        type_conflicts = [c for c in result.conflicts if c.conflict_type == "entity_type_inconsistency"]
        assert len(type_conflicts) >= 1

    def test_numeric_conflict_detection(self):
        """Same metric with different values should be flagged."""
        facts = [
            _make_fact("f1", "e1", "SALARY", "$50,000", "u1", "salary is $50,000"),
            _make_fact("f2", "e1", "SALARY", "$55,000", "u2", "salary is $55,000"),
        ]
        extraction = ExtractionResult(document_id="doc_001", facts=facts)
        units = [
            SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH, text="Salary is $50,000.", page_start=1, page_end=1),
            SemanticUnit(unit_id="u2", unit_type=UnitType.PARAGRAPH, text="Salary is $55,000.", page_start=2, page_end=2),
        ]
        result = verify_extraction(extraction, _make_doc(units))
        numeric_conflicts = [c for c in result.conflicts if c.conflict_type == "numeric_mismatch"]
        assert len(numeric_conflicts) >= 1

    def test_duplicate_fact_detection(self):
        """Same subject+predicate with different objects should warn."""
        facts = [
            _make_fact("f1", "e1", "ROLE", "Engineer", "u1", "role: Engineer"),
            _make_fact("f2", "e1", "ROLE", "Manager", "u2", "role: Manager"),
        ]
        extraction = ExtractionResult(document_id="doc_001", facts=facts)
        units = [
            SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH, text="Role: Engineer.", page_start=1, page_end=1),
            SemanticUnit(unit_id="u2", unit_type=UnitType.PARAGRAPH, text="Role: Manager.", page_start=2, page_end=2),
        ]
        result = verify_extraction(extraction, _make_doc(units))
        dup_conflicts = [c for c in result.conflicts if c.conflict_type == "duplicate_predicate"]
        assert len(dup_conflicts) >= 1

    def test_missing_provenance_warning(self):
        """Facts without matching raw_text in source should warn."""
        facts = [_make_fact("f1", "e1", "SKILL", "Python", "u1", "Expert in Python")]
        extraction = ExtractionResult(document_id="doc_001", facts=facts)
        units = [SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH, text="Completely different text.", page_start=1, page_end=1)]
        result = verify_extraction(extraction, _make_doc(units))
        assert len(result.warnings) >= 1

    def test_quality_score_reduced_with_conflicts(self):
        facts = [
            _make_fact("f1", "e1", "SALARY", "$50,000", "u1", "salary is $50,000"),
            _make_fact("f2", "e1", "SALARY", "$55,000", "u2", "salary is $55,000"),
        ]
        extraction = ExtractionResult(document_id="doc_001", facts=facts)
        units = [
            SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH, text="Salary is $50,000.", page_start=1, page_end=1),
            SemanticUnit(unit_id="u2", unit_type=UnitType.PARAGRAPH, text="Salary is $55,000.", page_start=2, page_end=2),
        ]
        result = verify_extraction(extraction, _make_doc(units))
        assert result.quality_score < 1.0

    def test_empty_extraction(self):
        extraction = ExtractionResult(document_id="doc_001")
        result = verify_extraction(extraction, _make_doc([]))
        assert result.is_valid
        assert len(result.conflicts) == 0
