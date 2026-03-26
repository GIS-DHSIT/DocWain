"""End-to-end smoke test for Phase 3 (full visual intelligence pipeline)."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, StructuredTableResult, OCRPatch, KVPair,
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
async def test_full_pipeline_all_models():
    """Tier 2 page: DiT + Table Transformer + TrOCR + LayoutLMv3 + KG."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.sections = []
        extracted.tables = [MagicMock(page=1, csv="a,b\n1,2")]
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.3)]
        extracted.full_text = "Garblod Invoyse No 12345"
        extracted.metrics = {"total_pages": 1}

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.3,
                          image_ratio=0.8, has_tables=True, has_forms=True),
        ])

        mock_img = Image.new("RGB", (2550, 3300), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=2550, height=3300, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 50, 2400, 800), label="text", confidence=0.93, page=1),
            VisualRegion(bbox=(100, 850, 2400, 2500), label="table", confidence=0.91, page=1),
        ])

        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[
            StructuredTableResult(page=1, bbox=(100, 850, 2400, 2500),
                                 headers=["Item", "Amount"],
                                 rows=[["Service", "$500"], ["Tax", "$50"]],
                                 spans=[], confidence=0.87),
        ])

        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=OCRPatch(
            page=1, bbox=(100, 50, 2400, 800),
            original_text="Garblod Invoyse",
            enhanced_text="Garbled Invoice",
            original_confidence=0.3,
            enhanced_confidence=0.91,
            method="trocr_printed",
        ))

        orch.layoutlmv3_extractor = MagicMock()
        orch.layoutlmv3_extractor.extract = MagicMock(return_value=[
            KVPair(key="Invoice No", value="12345", confidence=0.88, page=1, source="layoutlmv3"),
        ])

        mock_kg = MagicMock()
        orch.kg_enricher = mock_kg

        result = await orch.enrich(
            "doc-full", extracted, _make_pdf_bytes(),
            subscription_id="sub-1", profile_id="prof-1",
        )

        # All models invoked
        orch.dit_detector.detect.assert_called_once()
        orch.table_detector.extract.assert_called_once()
        orch.trocr_enhancer.enhance_region.assert_called()
        orch.layoutlmv3_extractor.extract.assert_called_once()
        mock_kg.enqueue_enrichment.assert_called_once()

        # Provenance
        assert result.metrics.get("visual_intelligence_applied") is True
        models_used = result.metrics.get("visual_intelligence_models", [])
        assert "dit" in models_used
        assert "layoutlmv3" in models_used


@pytest.mark.asyncio
async def test_graceful_degradation_all_models_unavailable():
    """If all models return empty, existing pipeline continues unchanged."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.3)]
        extracted.sections = []
        extracted.full_text = "original text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.3,
                          image_ratio=0.8, has_tables=True, has_forms=True),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        # All models return empty
        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.layoutlmv3_extractor = MagicMock()
        orch.layoutlmv3_extractor.extract = MagicMock(return_value=[])
        orch.kg_enricher = MagicMock()

        result = await orch.enrich("doc-degrade", extracted, _make_pdf_bytes())
        # Document returned without crash
        assert result.full_text == "original text"
