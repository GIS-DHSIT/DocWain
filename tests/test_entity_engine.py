"""Tests for universal entity and fact extraction engine."""
import pytest
from src.docwain_intel.models import SemanticUnit, UnitType, StructuredDocument
from src.docwain_intel.entity_engine import extract_entities_and_facts, ExtractionResult


def _make_doc(units):
    return StructuredDocument(document_id="doc_001", units=units, unit_count=len(units))


class TestEntityExtraction:
    def test_person_extraction(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.PARAGRAPH,
            text="John Smith is a senior software engineer at Google.",
            page_start=1, page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        assert isinstance(result, ExtractionResult)
        person_ents = [e for e in result.entities if e.label == "PERSON"]
        assert len(person_ents) >= 1
        assert any("john" in e.normalized for e in person_ents)

    def test_org_extraction(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.PARAGRAPH,
            text="She worked at Microsoft and Amazon before joining Google.",
            page_start=1, page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        org_ents = [e for e in result.entities if e.label == "ORG"]
        assert len(org_ents) >= 2

    def test_date_money_extraction(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.PARAGRAPH,
            text="The contract dated January 15, 2024 specifies a payment of $50,000.",
            page_start=1, page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        date_ents = [e for e in result.entities if e.label == "DATE"]
        money_ents = [e for e in result.entities if e.label == "MONEY"]
        assert len(date_ents) >= 1
        assert len(money_ents) >= 1

    def test_kv_unit_extracts_structured_entities(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.KV_GROUP,
            text="Name: John Smith\nEmail: john@example.com\nPhone: 555-0123",
            page_start=1, page_end=1,
            kv_pairs={"Name": "John Smith", "Email": "john@example.com", "Phone": "555-0123"},
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        assert len(result.kv_pairs) >= 3

    def test_entity_deduplication(self):
        units = [
            SemanticUnit(unit_id="u1", unit_type=UnitType.PARAGRAPH,
                         text="John Smith joined in 2020.", page_start=1, page_end=1),
            SemanticUnit(unit_id="u2", unit_type=UnitType.PARAGRAPH,
                         text="John Smith was promoted in 2022.", page_start=1, page_end=1),
        ]
        result = extract_entities_and_facts(_make_doc(units))
        john_ents = [e for e in result.entities if "john" in e.normalized and e.label == "PERSON"]
        assert len(john_ents) <= 2

    def test_fact_has_provenance(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.PARAGRAPH,
            text="Sarah Chen managed a team of 15 engineers.",
            page_start=1, page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        for fact in result.facts:
            assert fact.raw_text, f"Fact {fact.fact_id} missing provenance text"
            assert fact.unit_id == "u1"

    def test_table_unit_extracts_row_facts(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.TABLE,
            text="Name | Role | Years\nJohn | Engineer | 5\nSarah | Manager | 3",
            page_start=1, page_end=1,
            table_headers=["Name", "Role", "Years"],
            table_rows=[
                {"Name": "John", "Role": "Engineer", "Years": "5"},
                {"Name": "Sarah", "Role": "Manager", "Years": "3"},
            ],
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        assert len(result.tables_structured) >= 1
        assert len(result.facts) >= 2

    def test_empty_document(self):
        result = extract_entities_and_facts(_make_doc([]))
        assert len(result.entities) == 0
        assert len(result.facts) == 0

    def test_no_hallucinated_entities(self):
        unit = SemanticUnit(
            unit_id="u1", unit_type=UnitType.PARAGRAPH,
            text="The quick brown fox jumps over the lazy dog.",
            page_start=1, page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        for ent in result.entities:
            assert ent.text.lower() in unit.text.lower(), \
                f"Entity '{ent.text}' not found in source text"
