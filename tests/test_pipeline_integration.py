"""Tests for pipeline integration with upload flow."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.docwain_intel.intel_pipeline import process_document, ProcessingResult, PipelineStage
from src.docwain_intel.models import Block, Page, ExtractedDocumentJSON


class TestPipelineIntegration:
    def test_process_document_returns_result(self):
        """Basic smoke test: pipeline runs end-to-end."""
        blocks = [
            Block(block_id="b1", type="paragraph", text="John Smith works at Google as a senior engineer.", page_number=1, reading_order=0),
        ]
        pages = [Page(page_number=1, blocks=blocks)]
        doc = ExtractedDocumentJSON(document_id="doc_001", pages=pages)

        result = process_document(
            extracted_doc=doc,
            document_id="doc_001",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert isinstance(result, ProcessingResult)
        assert result.stage_reached != PipelineStage.FAILED.value
        assert result.structured_doc is not None
        assert result.extraction is not None
        assert result.fingerprint is not None
        assert result.verification is not None

    def test_intel_pipeline_hook_exists(self):
        """Verify the integration hook function exists and is callable."""
        from src.docwain_intel.integration import run_intel_pipeline_hook
        assert callable(run_intel_pipeline_hook)

    def test_integration_hook_with_extracted_doc(self):
        """The hook should accept an ExtractedDocumentJSON and return ProcessingResult."""
        from src.docwain_intel.integration import run_intel_pipeline_hook

        blocks = [
            Block(block_id="b1", type="paragraph", text="Contract value is $50,000 for services rendered.", page_number=1, reading_order=0),
        ]
        pages = [Page(page_number=1, blocks=blocks)]
        doc = ExtractedDocumentJSON(document_id="doc_002", pages=pages)

        with patch("src.docwain_intel.integration.INTEL_PIPELINE_ENABLED", True):
            result = run_intel_pipeline_hook(
                extracted_doc=doc,
                document_id="doc_002",
                subscription_id="sub_001",
                profile_id="prof_001",
            )
        assert isinstance(result, ProcessingResult)
        assert result.document_id == "doc_002"

    def test_integration_hook_feature_flag(self):
        """When feature flag is disabled, hook returns None."""
        from src.docwain_intel.integration import run_intel_pipeline_hook

        blocks = [Block(block_id="b1", type="paragraph", text="Test.", page_number=1, reading_order=0)]
        doc = ExtractedDocumentJSON(document_id="doc_003", pages=[Page(page_number=1, blocks=blocks)])

        with patch("src.docwain_intel.integration.INTEL_PIPELINE_ENABLED", False):
            result = run_intel_pipeline_hook(
                extracted_doc=doc,
                document_id="doc_003",
                subscription_id="sub_001",
                profile_id="prof_001",
            )
        assert result is None

    def test_integration_hook_error_handling(self):
        """Errors in the pipeline should not crash the upload flow."""
        from src.docwain_intel.integration import run_intel_pipeline_hook

        doc = ExtractedDocumentJSON(document_id="doc_004", pages=[])

        with patch("src.docwain_intel.integration.INTEL_PIPELINE_ENABLED", True), \
             patch("src.docwain_intel.integration.process_document", side_effect=RuntimeError("test crash")):
            result = run_intel_pipeline_hook(
                extracted_doc=doc,
                document_id="doc_004",
                subscription_id="sub_001",
                profile_id="prof_001",
            )
        # Should return None or a failed result, NOT raise
        assert result is None or (isinstance(result, ProcessingResult) and result.error is not None)

    def test_query_routing_integration(self):
        """Verify query router is accessible from integration module."""
        from src.docwain_intel.integration import route_and_assemble
        assert callable(route_and_assemble)
