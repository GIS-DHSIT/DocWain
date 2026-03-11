"""Tests for src/docwain_intel/payload_enricher.py."""

import pytest

from src.docwain_intel.models import (
    DocumentFingerprint,
    EntitySpan,
    ExtractionResult,
    SemanticUnit,
    StructuredDocument,
    UnitType,
)
from src.docwain_intel.payload_enricher import (
    batch_enrich,
    enrich_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(entity_id: str, text: str, label: str = "PERSON") -> EntitySpan:
    return EntitySpan(
        entity_id=entity_id,
        text=text,
        normalized=text.lower(),
        label=label,
        unit_id="u1",
    )


def _make_extraction(*entities: EntitySpan, doc_id: str = "doc1") -> ExtractionResult:
    return ExtractionResult(document_id=doc_id, entities=list(entities))


def _make_unit(unit_id: str, text: str, unit_type: UnitType = UnitType.PARAGRAPH) -> SemanticUnit:
    return SemanticUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        text=text,
        page_start=1,
        page_end=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnrichPayload:

    def test_entity_ids_added_when_found(self):
        entity = _make_entity("e1", "Alice")
        extraction = _make_extraction(entity)
        payload = enrich_payload({}, "Alice works at Acme.", extraction)
        assert payload["intel_entity_ids"] == ["e1"]

    def test_entity_labels_collected_unique(self):
        e1 = _make_entity("e1", "Alice", "PERSON")
        e2 = _make_entity("e2", "Bob", "PERSON")
        e3 = _make_entity("e3", "Acme", "ORG")
        extraction = _make_extraction(e1, e2, e3)
        payload = enrich_payload({}, "Alice and Bob work at Acme.", extraction)
        assert sorted(payload["intel_entity_labels"]) == ["ORG", "PERSON"]

    def test_fingerprint_tags_added(self):
        extraction = _make_extraction()
        fp = DocumentFingerprint(auto_tags=["resume", "technical"])
        payload = enrich_payload({}, "some text", extraction, fingerprint=fp)
        assert payload["intel_fingerprint_tags"] == ["resume", "technical"]

    def test_unit_type_detected_by_text_overlap(self):
        extraction = _make_extraction()
        unit = _make_unit("u1", "The quick brown fox jumps over the lazy dog.", UnitType.PARAGRAPH)
        sdoc = StructuredDocument(document_id="doc1", units=[unit])
        payload = enrich_payload(
            {}, "The quick brown fox jumps over the lazy dog.", extraction,
            structured_doc=sdoc,
        )
        assert payload["intel_unit_type"] == "paragraph"

    def test_no_entity_match_empty_ids(self):
        entity = _make_entity("e1", "Zebra")
        extraction = _make_extraction(entity)
        payload = enrich_payload({}, "Alice works at Acme.", extraction)
        assert payload["intel_entity_ids"] == []

    def test_word_boundary_john_does_not_match_johnson(self):
        entity = _make_entity("e1", "John")
        extraction = _make_extraction(entity)
        payload = enrich_payload({}, "Johnson is a common surname.", extraction)
        assert payload["intel_entity_ids"] == []

    def test_word_boundary_john_matches_john(self):
        entity = _make_entity("e1", "John")
        extraction = _make_extraction(entity)
        payload = enrich_payload({}, "John is a common name.", extraction)
        assert payload["intel_entity_ids"] == ["e1"]

    def test_case_insensitive_matching(self):
        entity = _make_entity("e1", "alice")
        extraction = _make_extraction(entity)
        payload = enrich_payload({}, "ALICE works at Acme.", extraction)
        assert payload["intel_entity_ids"] == ["e1"]

    def test_existing_payload_fields_preserved(self):
        entity = _make_entity("e1", "Alice")
        extraction = _make_extraction(entity)
        original = {"document_id": "doc1", "chunk_index": 3}
        payload = enrich_payload(original, "Alice is here.", extraction)
        assert payload["document_id"] == "doc1"
        assert payload["chunk_index"] == 3
        assert "intel_entity_ids" in payload

    def test_existing_intel_keys_not_overwritten(self):
        entity = _make_entity("e1", "Alice")
        extraction = _make_extraction(entity)
        original = {"intel_entity_ids": ["old_id"]}
        payload = enrich_payload(original, "Alice is here.", extraction)
        # Should keep the original value, not overwrite
        assert payload["intel_entity_ids"] == ["old_id"]

    def test_multiple_entities_in_same_chunk(self):
        e1 = _make_entity("e1", "Alice", "PERSON")
        e2 = _make_entity("e2", "Acme", "ORG")
        e3 = _make_entity("e3", "Bob", "PERSON")
        extraction = _make_extraction(e1, e2, e3)
        payload = enrich_payload({}, "Alice and Bob both work at Acme.", extraction)
        assert sorted(payload["intel_entity_ids"]) == ["e1", "e2", "e3"]

    def test_none_fingerprint_gives_empty_tags(self):
        extraction = _make_extraction()
        payload = enrich_payload({}, "text", extraction, fingerprint=None)
        assert payload["intel_fingerprint_tags"] == []

    def test_none_structured_doc_gives_none_unit_type(self):
        extraction = _make_extraction()
        payload = enrich_payload({}, "text", extraction, structured_doc=None)
        assert payload["intel_unit_type"] is None

    def test_unit_type_none_when_no_overlap(self):
        extraction = _make_extraction()
        unit = _make_unit("u1", "Completely different content here.", UnitType.TABLE)
        sdoc = StructuredDocument(document_id="doc1", units=[unit])
        payload = enrich_payload({}, "Xyz 123 nothing in common.", extraction, structured_doc=sdoc)
        assert payload["intel_unit_type"] is None


class TestBatchEnrich:

    def test_batch_processes_multiple_chunks(self):
        e1 = _make_entity("e1", "Alice", "PERSON")
        e2 = _make_entity("e2", "Bob", "PERSON")
        extraction = _make_extraction(e1, e2)

        payloads = [
            ({"idx": 0}, "Alice is an engineer."),
            ({"idx": 1}, "Bob is a designer."),
            ({"idx": 2}, "No entities here at all."),
        ]

        results = batch_enrich(payloads, extraction)
        assert len(results) == 3
        assert results[0]["intel_entity_ids"] == ["e1"]
        assert results[1]["intel_entity_ids"] == ["e2"]
        assert results[2]["intel_entity_ids"] == []

    def test_batch_returns_list_of_dicts(self):
        extraction = _make_extraction()
        payloads = [({}, "a"), ({}, "b")]
        results = batch_enrich(payloads, extraction)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_batch_with_fingerprint_and_structured_doc(self):
        extraction = _make_extraction()
        fp = DocumentFingerprint(auto_tags=["invoice"])
        unit = _make_unit("u1", "Total amount due", UnitType.KV_GROUP)
        sdoc = StructuredDocument(document_id="doc1", units=[unit])

        payloads = [({"idx": 0}, "Total amount due")]
        results = batch_enrich(payloads, extraction, fingerprint=fp, structured_doc=sdoc)
        assert results[0]["intel_fingerprint_tags"] == ["invoice"]
        assert results[0]["intel_unit_type"] == "kv_group"
