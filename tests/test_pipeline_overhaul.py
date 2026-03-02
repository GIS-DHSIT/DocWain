"""Tests for the comprehensive pipeline overhaul:
screening-in-pickle, doc classification, raw text sanitization, repr safety, clean content flow."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ── Screening pickle update ────────────────────────────────────────────


class TestScreeningPickleUpdate:
    @patch("src.api.content_store.save_extracted_pickle")
    @patch("src.api.content_store.load_extracted_pickle")
    def test_update_pickle_with_screening(self, mock_load, mock_save):
        from src.api.screening_service import _update_pickle_with_screening

        existing = {"raw": {"doc.pdf": {}}, "structured": {}, "intelligence": None}
        mock_load.return_value = existing
        report = {"status": "passed", "risk_level": "LOW", "report": {"score": 80}}

        _update_pickle_with_screening("doc123", report)

        mock_save.assert_called_once()
        saved = mock_save.call_args[0][1]
        assert "screening" in saved
        assert saved["screening"]["status"] == "passed"
        assert saved["raw"] == {"doc.pdf": {}}

    @patch("src.api.content_store.save_extracted_pickle")
    @patch("src.api.content_store.load_extracted_pickle")
    def test_screening_wraps_non_dict_pickle(self, mock_load, mock_save):
        from src.api.screening_service import _update_pickle_with_screening

        mock_load.return_value = "raw string content"
        report = {"status": "passed"}

        _update_pickle_with_screening("doc456", report)

        saved = mock_save.call_args[0][1]
        assert saved["raw"] == "raw string content"
        assert saved["screening"]["status"] == "passed"

    @patch("src.api.content_store.load_extracted_pickle", side_effect=ValueError("not found"))
    def test_screening_update_handles_missing_pickle(self, mock_load):
        from src.api.screening_service import _update_pickle_with_screening

        # Should not raise — just log a warning
        _update_pickle_with_screening("missing_doc", {"status": "passed"})


# ── Document classification in pickle ──────────────────────────────────


class TestDocumentClassificationInPickle:
    def test_extract_classification_from_structured_dict(self):
        from src.api.extraction_service import _extract_classification_from_structured

        structured = {
            "resume.pdf": {
                "document_type": "RESUME",
                "document_classification": {"domain": "hr", "confidence": 0.95},
            }
        }
        result = _extract_classification_from_structured(structured)
        assert result["document_type"] == "RESUME"
        assert result["domain"] == "hr"
        assert result["confidence"] == 0.95
        assert result["filename"] == "resume.pdf"

    def test_classification_defaults_to_generic(self):
        from src.api.extraction_service import _extract_classification_from_structured

        result = _extract_classification_from_structured({})
        assert result["document_type"] == "GENERIC"
        assert result["domain"] == "generic"

    def test_classification_from_none(self):
        from src.api.extraction_service import _extract_classification_from_structured

        result = _extract_classification_from_structured(None)
        assert result["document_type"] == "GENERIC"

    def test_classification_missing_nested_fields(self):
        from src.api.extraction_service import _extract_classification_from_structured

        structured = {"invoice.pdf": {"document_type": "INVOICE"}}
        result = _extract_classification_from_structured(structured)
        assert result["document_type"] == "INVOICE"
        assert result["domain"] == "generic"
        assert result["confidence"] == 0.0


# ── Sanitize raw text fields ──────────────────────────────────────────


class TestSanitizeRawTextFields:
    def test_garbage_full_text_recovered_from_sections(self):
        from src.api.extraction_service import _sanitize_raw_text_fields

        docs = {
            "resume.pdf": {
                "full_text": "{'section_id': 'sec-1', 'section_title': 'Skills', 'page': None, 'chunk_type': 'text', 'text': 'Python'}",
                "sections": [
                    {"text": "Python developer with 5 years experience"},
                    {"text": "Skilled in Django and Flask"},
                ],
            }
        }
        result = _sanitize_raw_text_fields(docs)
        assert "Python developer" in result["resume.pdf"]["full_text"]
        assert "'section_id'" not in result["resume.pdf"]["full_text"]

    def test_garbage_texts_list_cleaned(self):
        from src.api.extraction_service import _sanitize_raw_text_fields

        docs = {
            "doc.pdf": {
                "texts": [
                    {"text": "Clean text from dict", "section_id": "s1"},
                    "Already clean string",
                    "{'section_id': 'sec-1', 'section_title': 'X', 'page': None, 'chunk_type': 'text'}",
                ],
            }
        }
        result = _sanitize_raw_text_fields(docs)
        texts = result["doc.pdf"]["texts"]
        assert "Clean text from dict" in texts
        assert "Already clean string" in texts
        # Garbage string should be removed
        assert not any("'section_id'" in t and "'chunk_type'" in t for t in texts)

    def test_clean_text_unchanged(self):
        from src.api.extraction_service import _sanitize_raw_text_fields

        docs = {
            "doc.pdf": {
                "full_text": "John Doe is a software engineer with extensive experience.",
                "texts": ["Hello world", "Python developer"],
            }
        }
        result = _sanitize_raw_text_fields(docs)
        assert result["doc.pdf"]["full_text"] == "John Doe is a software engineer with extensive experience."
        assert result["doc.pdf"]["texts"] == ["Hello world", "Python developer"]

    def test_non_dict_passthrough(self):
        from src.api.extraction_service import _sanitize_raw_text_fields

        assert _sanitize_raw_text_fields("string") == "string"
        assert _sanitize_raw_text_fields(None) is None


# ── Normalize structured payload safety ────────────────────────────────


class TestNormalizeStructuredPayloadSafety:
    def test_object_with_raw_text_attr_used(self):
        from src.api.embedding_service import _normalize_structured_payload

        class FakeStructured:
            raw_text = "This is the actual document content about Python programming."

        result = _normalize_structured_payload({"doc.pdf": FakeStructured()})
        assert "doc.pdf" in result
        assert "Python programming" in result["doc.pdf"]["full_text"]

    def test_garbage_str_repr_salvaged(self):
        from src.api.embedding_service import _normalize_structured_payload

        class GarbageRepr:
            def __str__(self):
                return "ExtractedDocument(full_text='some text here that is long enough to trigger detection')"

        result = _normalize_structured_payload({"doc.pdf": GarbageRepr()})
        # Should salvage the real text from ExtractedDocument repr
        assert "doc.pdf" in result
        texts = result["doc.pdf"].get("texts", [])
        assert len(texts) == 1
        assert "some text here" in texts[0]
        assert "ExtractedDocument" not in texts[0]

    def test_dict_value_processes_normally(self):
        from src.api.embedding_service import _normalize_structured_payload

        structured = {
            "doc.pdf": {
                "raw_text": "Clean document text about software engineering.",
                "sections": [
                    {"content": "Clean document text about software engineering.", "start_page": 1, "end_page": 1}
                ],
                "document_type": "GENERIC",
            }
        }
        result = _normalize_structured_payload(structured)
        assert "doc.pdf" in result
        assert "software engineering" in result["doc.pdf"]["full_text"]

    def test_doc_domain_in_normalized_output(self):
        from src.api.embedding_service import _normalize_structured_payload

        structured = {
            "resume.pdf": {
                "raw_text": "John Doe software engineer with 10 years experience",
                "sections": [
                    {"content": "John Doe software engineer with 10 years experience", "start_page": 1, "end_page": 1}
                ],
                "document_type": "RESUME",
                "document_classification": {"domain": "hr", "confidence": 0.9},
            }
        }
        result = _normalize_structured_payload(structured)
        assert result["resume.pdf"]["doc_domain"] == "hr"
        assert result["resume.pdf"]["doc_type"] == "RESUME"


# ── End-to-end clean content flow ─────────────────────────────────────


class TestEndToEndCleanContent:
    def test_clean_text_reaches_qdrant_payload(self):
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub123",
            "profile_id": "prof456",
            "document_id": "doc789",
            "source_name": "resume.pdf",
            "canonical_text": "Senior Python developer with expertise in Django, FastAPI, and cloud infrastructure.",
            "section_title": "Professional Summary",
            "page_start": 1,
            "chunk_id": "c1",
            "chunk_index": 0,
        }
        payload = build_qdrant_payload(raw)
        assert "Python developer" in payload["canonical_text"]
        assert "'section_id'" not in payload["canonical_text"]
        assert payload["embedding_text"]
        assert "'chunk_type'" not in payload["embedding_text"]

    def test_garbage_canonical_text_replaced(self):
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub123",
            "profile_id": "prof456",
            "document_id": "doc789",
            "source_name": "resume.pdf",
            "canonical_text": "{'section_id': 'sec-1', 'section_title': 'X', 'page': None, 'chunk_type': 'text'}",
            "content": "Real document content about software engineering and Python.",
            "section_title": "Content",
            "chunk_id": "c1",
        }
        payload = build_qdrant_payload(raw)
        # canonical_text should fall back to content, not contain garbage
        assert "'section_id'" not in payload["canonical_text"]

    def test_doc_domain_classified_for_resume(self):
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub123",
            "profile_id": "prof456",
            "document_id": "doc789",
            "source_name": "resume.pdf",
            "canonical_text": "Senior Python developer with expertise in Django and cloud infrastructure. 10 years professional experience managing teams.",
            "section_title": "Professional Summary",
            "chunk_id": "c1",
            "doc_domain": "resume",
        }
        payload = build_qdrant_payload(raw)
        assert payload["doc_domain"] == "resume"

    def test_doc_domain_flows_from_metadata(self):
        """doc_domain passed in raw metadata should be preserved in payload."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub123",
            "profile_id": "prof456",
            "document_id": "doc789",
            "source_name": "invoice.pdf",
            "canonical_text": "Invoice #12345 dated 2026-01-15 for consulting services rendered.",
            "section_title": "Invoice",
            "chunk_id": "c1",
            "doc_domain": "invoice",
        }
        payload = build_qdrant_payload(raw)
        assert payload["doc_domain"] == "invoice"
