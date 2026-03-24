"""End-to-end smoke test for Phase 2 (Tables + OCR + KG)."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, StructuredTableResult, OCRPatch,
)
from src.visual_intelligence.page_renderer import PageImage


def _make_pdf_bytes():
    import fitz
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 72), "t")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.asyncio
async def test_full_pipeline_table_and_ocr():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.sections = []
        extracted.tables = [MagicMock(page=1, csv="a,b\n1,2")]
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.4)]
        extracted.full_text = "Garblod toxt on page 1"
        extracted.metrics = {"total_pages": 1}

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.4,
                          image_ratio=0.7, has_tables=True, has_forms=True),
        ])

        mock_img = Image.new("RGB", (2550, 3300), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=2550, height=3300, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 50, 2400, 1500), label="text", confidence=0.93, page=1),
            VisualRegion(bbox=(100, 1520, 2400, 2800), label="table", confidence=0.89, page=1),
        ])

        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[
            StructuredTableResult(page=1, bbox=(100, 1520, 2400, 2800),
                                 headers=["Name", "Amount"], rows=[["Rent", "$1200"]],
                                 spans=[], confidence=0.87),
        ])

        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=OCRPatch(
            page=1, bbox=(100, 50, 2400, 1500),
            original_text="Garblod toxt",
            enhanced_text="Garbled text",
            original_confidence=0.4,
            enhanced_confidence=0.92,
            method="trocr_printed",
        ))

        mock_kg = MagicMock()
        orch.kg_enricher = mock_kg

        result = await orch.enrich(
            "doc-e2e-p2", extracted, _make_pdf_bytes(),
            subscription_id="sub-1", profile_id="prof-1",
        )

        orch.dit_detector.detect.assert_called_once()
        orch.table_detector.extract.assert_called_once()
        orch.trocr_enhancer.enhance_region.assert_called()
        mock_kg.enqueue_enrichment.assert_called_once()
        assert result.metrics.get("visual_intelligence_applied") is True
