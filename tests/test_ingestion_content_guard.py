"""Tests for ingestion content guard: garbage detection, text extraction, hybrid extraction."""
from __future__ import annotations

import pytest

from src.api.dataHandler import _extract_text_from_item
from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
from src.rag_v3.retrieve import (
    _to_chunk,
    _is_metadata_garbage as _is_garbage_retrieve,
    _salvage_text_from_garbage,
    _extract_text_from_repr,
)
from src.rag_v3.types import Chunk, ChunkSource


# ── _extract_text_from_item ──────────────────────────────────────────────


class TestExtractTextFromItem:
    def test_dict_item_returns_text_field(self):
        item = {"text": "Professional Summary", "section_id": "s1", "chunk_type": "text"}
        assert _extract_text_from_item(item) == "Professional Summary"

    def test_dict_item_falls_back_to_content(self):
        item = {"content": "Some content", "section_id": "s1"}
        assert _extract_text_from_item(item) == "Some content"

    def test_dict_item_empty_when_no_text_keys(self):
        item = {"section_id": "s1", "chunk_type": "text"}
        assert _extract_text_from_item(item) == ""

    def test_string_item_passes_through(self):
        assert _extract_text_from_item("hello world") == "hello world"

    def test_object_with_text_attr(self):
        class FakeChunk:
            text = "from attribute"
        assert _extract_text_from_item(FakeChunk()) == "from attribute"


# ── _is_metadata_garbage ─────────────────────────────────────────────────


class TestIsMetadataGarbage:
    def test_detects_stringified_dict_garbage(self):
        garbage = "{'section_id': 'sec-1', 'section_title': 'Introduction', 'page': None, 'chunk_type': 'text', 'text': 'Professional Summary'}"
        assert _is_metadata_garbage(garbage) is True

    def test_passes_clean_text(self):
        clean = "John Doe is a software engineer with 10 years of experience in Python and Java development."
        assert _is_metadata_garbage(clean) is False

    def test_short_text_not_flagged(self):
        assert _is_metadata_garbage("short") is False

    def test_empty_text_not_flagged(self):
        assert _is_metadata_garbage("") is False
        assert _is_metadata_garbage(None) is False

    def test_single_marker_not_flagged(self):
        text = "This text mentions 'chunk_type': something but nothing else suspicious about this content."
        assert _is_metadata_garbage(text) is False

    def test_detects_extracted_document_repr(self):
        garbage = "Extracted Document (full_text='Abhishek Prem Kumar\\n SAP MM Consultant')"
        assert _is_metadata_garbage(garbage) is True

    def test_detects_extracted_document_repr_no_space(self):
        garbage = "ExtractedDocument(full_text='Some resume text here and more content follows')"
        assert _is_metadata_garbage(garbage) is True


# ── _to_chunk garbage detection ──────────────────────────────────────────


class FakePoint:
    def __init__(self, payload, score=0.8):
        self.payload = payload
        self.score = score
        self.id = "p1"


class TestToChunkGarbageDetection:
    def test_skips_garbage_canonical_text_uses_embedding_text(self):
        point = FakePoint(payload={
            "canonical_text": "{'section_id': 'sec-1', 'section_title': 'Skills', 'page': None, 'chunk_type': 'text', 'text': 'Python'}",
            "embedding_text": "[Skills Technical] Python, Java, Docker",
            "source_name": "resume.pdf",
            "page": 1,
            "chunk_id": "c1",
        })
        chunk = _to_chunk(point)
        assert "Python" in chunk.text
        assert "'section_id'" not in chunk.text

    def test_uses_clean_canonical_text(self):
        point = FakePoint(payload={
            "canonical_text": "Python developer with 5 years of experience",
            "source_name": "resume.pdf",
            "page": 1,
            "chunk_id": "c2",
        })
        chunk = _to_chunk(point)
        assert chunk.text == "Python developer with 5 years of experience"

    def test_falls_back_to_embedding_text_when_canonical_is_garbage(self):
        point = FakePoint(payload={
            "canonical_text": "{'section_id': 'sec-1', 'section_title': 'X', 'page': None, 'chunk_type': 'text'}",
            "embedding_text": "[Skills Technical] Section 3: Python, Java, Kubernetes",
            "source_name": "resume.pdf",
            "page": 1,
            "chunk_id": "c3",
        })
        chunk = _to_chunk(point)
        # embedding_text is clean, so _to_chunk uses it directly from the field loop
        assert "Python" in chunk.text
        assert "'section_id'" not in chunk.text

    def test_empty_payload_returns_empty_text(self):
        point = FakePoint(payload={})
        chunk = _to_chunk(point)
        assert chunk.text == ""

    def test_extracts_from_extracted_document_repr(self):
        """canonical_text is ExtractedDocument repr — should extract real full_text."""
        point = FakePoint(payload={
            "canonical_text": "Extracted Document (full_text='Abhishek Prem Kumar\\n SAP MM Consultant | Procurement and Inventory Management')",
            "source_name": "resume.pdf",
            "page": 1,
            "chunk_id": "c4",
        })
        chunk = _to_chunk(point)
        assert "Abhishek Prem Kumar" in chunk.text
        assert "SAP MM Consultant" in chunk.text
        assert not chunk.text.startswith("Extracted Document")

    def test_extracts_real_text_from_repr_when_all_fields_garbage(self):
        """Both canonical and embedding are garbage, salvage path extracts from ExtractedDocument repr."""
        point = FakePoint(payload={
            "canonical_text": "Extracted Document (full_text='Professional Summary\\nResults-oriented Supply Chain Management professional with over 16 years of experience')",
            "embedding_text": "{'section_id': 'abc', 'section_title': 'X', 'page': None, 'chunk_type': 'text'}",
            "source_name": "resume.pdf",
            "page": 1,
            "chunk_id": "c5",
        })
        chunk = _to_chunk(point)
        assert "Supply Chain Management" in chunk.text
        assert not chunk.text.startswith("Extracted Document")


# ── _extract_text_from_repr ───────────────────────────────────────────────


class TestExtractTextFromRepr:
    def test_extracts_full_text_from_repr(self):
        text = "Extracted Document (full_text='Abhishek Prem Kumar\\nSAP MM Consultant')"
        result = _extract_text_from_repr(text)
        assert "Abhishek Prem Kumar" in result
        assert "SAP MM Consultant" in result

    def test_returns_empty_for_non_repr(self):
        assert _extract_text_from_repr("Just normal text") == ""
        assert _extract_text_from_repr("") == ""

    def test_handles_long_repr_with_truncation(self):
        text = "Extracted Document (full_text='A very long resume text with lots of content about skills and experience in Python and Java development across multiple companies')"
        result = _extract_text_from_repr(text)
        assert "skills and experience" in result


# ── section_path list handling ───────────────────────────────────────────


class TestSectionPathListHandling:
    def test_section_path_list_converted_to_string(self):
        from src.embedding.pipeline.schema_normalizer import _section_path_list
        result = _section_path_list(["Skills", "Technical"])
        assert result == ["Skills", "Technical"]

    def test_section_path_string_split(self):
        from src.embedding.pipeline.schema_normalizer import _section_path_list
        result = _section_path_list("Skills > Technical")
        assert result == ["Skills", "Technical"]


# ── _salvage_text_from_garbage ───────────────────────────────────────────


class TestSalvageTextFromGarbage:
    def test_salvages_from_clean_embedding_text(self):
        payload = {
            "canonical_text": "{'section_id': 'sec-1', 'section_title': 'X', 'page': None, 'chunk_type': 'text'}",
            "embedding_text": "[Education] Bachelor of Science in Computer Engineering",
        }
        result = _salvage_text_from_garbage(payload)
        assert "Bachelor of Science" in result

    def test_salvages_from_text_clean(self):
        payload = {
            "text_clean": "Experienced Python developer",
        }
        result = _salvage_text_from_garbage(payload)
        assert result == "Experienced Python developer"

    def test_salvages_from_extracted_document_repr(self):
        payload = {
            "canonical_text": "Extracted Document (full_text='John Doe is a senior engineer with 10 years experience in Python and cloud infrastructure.')",
            "embedding_text": "[Skills] section_title X, section_id abc, chunk_type text",
        }
        result = _salvage_text_from_garbage(payload)
        assert "John Doe" in result
        assert "senior engineer" in result

    def test_returns_empty_when_all_garbage(self):
        garbage = "{'section_id': 'sec-1', 'section_title': 'X', 'page': None, 'chunk_type': 'text'}"
        payload = {
            "canonical_text": garbage,
            "embedding_text": garbage,
            "text": garbage,
        }
        result = _salvage_text_from_garbage(payload)
        assert result == ""
