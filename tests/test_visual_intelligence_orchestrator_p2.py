"""Phase 2 orchestrator tests — Table Transformer + TrOCR + KG wiring."""
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
async def test_orchestrator_runs_table_transformer_on_table_regions():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.6)]
        extracted.sections = []
        extracted.full_text = "text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.3, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 200, 500, 600), label="table", confidence=0.9, page=1),
        ])

        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[
            StructuredTableResult(page=1, bbox=(100, 200, 500, 600),
                                 headers=["A", "B"], rows=[["1", "2"]], spans=[], confidence=0.88),
        ])

        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.kg_enricher = MagicMock()

        result = await orch.enrich("doc-p2", extracted, _make_pdf_bytes())
        orch.table_detector.extract.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_fires_kg_enrichment():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = []
        extracted.sections = []
        extracted.full_text = "text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.9, page=1),
        ])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)

        mock_kg = MagicMock()
        orch.kg_enricher = mock_kg

        result = await orch.enrich(
            "doc-kg", extracted, _make_pdf_bytes(),
            subscription_id="sub-1", profile_id="prof-1",
        )
        mock_kg.enqueue_enrichment.assert_called_once()
