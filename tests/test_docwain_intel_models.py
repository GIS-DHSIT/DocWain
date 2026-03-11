"""Tests for docwain_intel data models."""
import pytest
from src.docwain_intel.models import (
    SemanticUnit, UnitType, EntitySpan, FactTriple,
    DocumentFingerprint, VerificationResult, ConflictRecord,
    StructuredDocument, ExtractionResult,
)


class TestSemanticUnit:
    def test_create_paragraph(self):
        unit = SemanticUnit(
            unit_id="u_001", unit_type=UnitType.PARAGRAPH,
            text="John Smith worked at Google from 2020 to 2023.",
            page_start=1, page_end=1, heading_path=["Experience"], confidence=0.95,
        )
        assert unit.unit_type == UnitType.PARAGRAPH
        assert unit.text == "John Smith worked at Google from 2020 to 2023."
        assert unit.heading_path == ["Experience"]

    def test_create_table(self):
        unit = SemanticUnit(
            unit_id="u_002", unit_type=UnitType.TABLE,
            text="Name | Role | Years\nJohn | Engineer | 5",
            page_start=2, page_end=2,
            table_headers=["Name", "Role", "Years"],
            table_rows=[{"Name": "John", "Role": "Engineer", "Years": "5"}],
        )
        assert unit.unit_type == UnitType.TABLE
        assert len(unit.table_rows) == 1
        assert unit.table_headers == ["Name", "Role", "Years"]

    def test_create_kv_group(self):
        unit = SemanticUnit(
            unit_id="u_003", unit_type=UnitType.KV_GROUP,
            text="Name: John Smith\nEmail: john@example.com",
            page_start=1, page_end=1,
            kv_pairs={"Name": "John Smith", "Email": "john@example.com"},
        )
        assert unit.kv_pairs["Name"] == "John Smith"


class TestEntitySpan:
    def test_create(self):
        span = EntitySpan(
            entity_id="ent_001", text="John Smith", normalized="john smith",
            label="PERSON", unit_id="u_001",
            char_start=0, char_end=10, confidence=0.95, source="spacy",
        )
        assert span.label == "PERSON"
        assert span.normalized == "john smith"


class TestFactTriple:
    def test_create(self):
        fact = FactTriple(
            fact_id="f_001", subject_id="ent_001", predicate="WORKED_AT",
            object_id="ent_002", object_value="Google", unit_id="u_001",
            raw_text="John Smith worked at Google", confidence=0.90,
            extraction_method="dependency_parse",
        )
        assert fact.predicate == "WORKED_AT"
        assert fact.object_value == "Google"

    def test_fact_requires_raw_text(self):
        fact = FactTriple(
            fact_id="f_002", subject_id="ent_001", predicate="HAS_SKILL",
            object_value="Python", unit_id="u_001",
            raw_text="Proficient in Python", confidence=0.85,
            extraction_method="dependency_parse",
        )
        assert fact.raw_text


class TestDocumentFingerprint:
    def test_create(self):
        fp = DocumentFingerprint(
            entity_distribution={"PERSON": 3, "ORG": 2},
            structure_profile={"paragraph": 10, "table": 2},
            numeric_density=0.15, entity_density=0.08,
            formality_score=0.7, auto_tags=["employment", "technology"],
        )
        assert fp.entity_distribution["PERSON"] == 3
        assert "employment" in fp.auto_tags


class TestVerificationResult:
    def test_clean_result(self):
        result = VerificationResult(is_valid=True, conflicts=[], warnings=[], quality_score=0.95)
        assert result.is_valid
        assert result.quality_score == 0.95

    def test_result_with_conflicts(self):
        conflict = ConflictRecord(
            fact_id_1="f_001", fact_id_2="f_002",
            conflict_type="numeric_mismatch",
            description="Salary stated as $50K on page 2 and $55K on page 4",
        )
        result = VerificationResult(is_valid=False, conflicts=[conflict], warnings=[], quality_score=0.4)
        assert not result.is_valid
        assert len(result.conflicts) == 1
