"""Phase 3 orchestrator tests — LayoutLMv3 on Tier 2, skipped on Tier 1."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, KVPair,
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
async def test_orchestrator_runs_layoutlmv3_on_tier2():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.4)]
        extracted.sections = []
        extracted.full_text = "Invoice No 12345"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.4,
                          image_ratio=0.8, has_tables=False, has_forms=True),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(10, 20, 700, 900), label="text", confidence=0.9, page=1),
        ])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)

        orch.layoutlmv3_extractor = MagicMock()
        orch.layoutlmv3_extractor.extract = MagicMock(return_value=[
            KVPair(key="Invoice No", value="12345", confidence=0.85, page=1, source="layoutlmv3"),
        ])

        orch.kg_enricher = MagicMock()

        result = await orch.enrich("doc-p3", extracted, _make_pdf_bytes(),
                                    subscription_id="sub-1", profile_id="prof-1")
        orch.layoutlmv3_extractor.extract.assert_called_once()
        assert "layoutlmv3" in result.metrics.get("visual_intelligence_models", [])


@pytest.mark.asyncio
async def test_orchestrator_skips_layoutlmv3_on_tier1():
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
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.78,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.layoutlmv3_extractor = MagicMock()
        orch.kg_enricher = MagicMock()

        result = await orch.enrich("doc-skip-lm", extracted, _make_pdf_bytes())
        orch.layoutlmv3_extractor.extract.assert_not_called()
