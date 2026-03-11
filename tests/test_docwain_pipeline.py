"""Tests for the 6-stage document intelligence pipeline."""
import pytest
from unittest.mock import MagicMock, patch
from src.docwain_intel.models import (
    Block, Page, ExtractedDocumentJSON, DocumentStatus,
)
from src.docwain_intel.intel_pipeline import process_document, ProcessingResult, PipelineStage


class TestProcessDocument:
    def _make_raw_doc(self):
        """Create a minimal ExtractedDocumentJSON for testing."""
        blocks = [
            Block(block_id="b1", type="paragraph", text="John Smith is a senior software engineer at Google.", page_number=1, reading_order=0),
            Block(block_id="b2", type="paragraph", text="He has 8 years of experience in Python and distributed systems.", page_number=1, reading_order=1),
            Block(block_id="b3", type="key_value", text="Email: john@example.com", key="Email", value="john@example.com", page_number=1, reading_order=2),
        ]
        pages = [Page(page_number=1, blocks=blocks)]
        return ExtractedDocumentJSON(document_id="doc_001", pages=pages)

    def test_full_pipeline_runs(self):
        raw_doc = self._make_raw_doc()
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert isinstance(result, ProcessingResult)
        assert result.document_id == "doc_001"
        assert result.stage_reached in (s.value for s in PipelineStage)

    def test_structured_document_produced(self):
        raw_doc = self._make_raw_doc()
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.structured_doc is not None
        assert len(result.structured_doc.units) >= 1

    def test_entities_extracted(self):
        raw_doc = self._make_raw_doc()
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.extraction is not None
        assert len(result.extraction.entities) >= 1

    def test_verification_included(self):
        raw_doc = self._make_raw_doc()
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.verification is not None

    def test_fingerprint_computed(self):
        raw_doc = self._make_raw_doc()
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.fingerprint is not None

    def test_empty_document(self):
        raw_doc = ExtractedDocumentJSON(document_id="doc_empty", pages=[])
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_empty",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.document_id == "doc_empty"
        assert result.structured_doc is not None
        assert len(result.structured_doc.units) == 0

    def test_stage_timing_recorded(self):
        raw_doc = self._make_raw_doc()
        result = process_document(
            extracted_doc=raw_doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert len(result.stage_timings) >= 1
        for stage, duration in result.stage_timings.items():
            assert duration >= 0.0

    def test_error_captured_on_failure(self):
        """If a stage raises, the error should be captured, not crash."""
        raw_doc = self._make_raw_doc()
        with patch("src.docwain_intel.intel_pipeline.parse_structure", side_effect=ValueError("test error")):
            result = process_document(
                extracted_doc=raw_doc,
                document_id="doc_001",
                subscription_id="sub_001",
                profile_id="prof_001",
            )
        assert result.error is not None
        assert "test error" in result.error
        assert result.stage_reached == PipelineStage.FAILED.value
