"""End-to-end smoke test for Visual Intelligence Layer (Phase 1)."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, VisualEnrichmentResult,
)
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_full_pipeline_tier1_page():
    """Simulate full pipeline: scorer → renderer → DiT → merger."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.sections = []
        extracted.tables = []
        extracted.figures = []
        extracted.full_text = "Sample document text"
        extracted.metrics = {"total_pages": 1}

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (2550, 3300), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=2550, height=3300, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 50, 2400, 200), label="title", confidence=0.96, page=1),
            VisualRegion(bbox=(100, 220, 2400, 1500), label="text", confidence=0.93, page=1),
            VisualRegion(bbox=(100, 1520, 2400, 2800), label="table", confidence=0.89, page=1),
        ])

        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "test")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-e2e", extracted, pdf_bytes)

        assert result.metrics.get("visual_intelligence_applied") is True
        assert "dit" in result.metrics.get("visual_intelligence_models", [])
        assert result.metrics.get("visual_intelligence_time_ms", 0) > 0
        assert len(result.metrics.get("visual_layout_regions", [])) == 3


@pytest.mark.asyncio
async def test_full_pipeline_all_skip():
    """All pages Tier 0 — pipeline exits early with no overhead."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 3}
        extracted.tables = []
        extracted.figures = []
        extracted.sections = []

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=i, tier=Tier.SKIP, ocr_confidence=0.95,
                          image_ratio=0.0, has_tables=False, has_forms=False)
            for i in range(1, 4)
        ])

        result = await orch.enrich("doc-skip", extracted, b"fake")
        assert result is extracted
