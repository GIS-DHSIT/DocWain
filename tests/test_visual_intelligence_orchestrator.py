import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.asyncio
async def test_orchestrator_skips_when_disabled():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=False):
        orch = VisualIntelligenceOrchestrator()
        extracted = MagicMock()
        result = await orch.enrich("doc-1", extracted, b"fake-pdf")
        assert result is extracted


@pytest.mark.asyncio
async def test_orchestrator_skips_all_tier0_pages():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    from src.visual_intelligence.datatypes import PageComplexity, Tier
    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()
        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.SKIP, ocr_confidence=0.95,
                          image_ratio=0.0, has_tables=False, has_forms=False),
        ])
        extracted = MagicMock()
        extracted.metrics = {}
        result = await orch.enrich("doc-1", extracted, b"fake-pdf")
        assert result is extracted


@pytest.mark.asyncio
async def test_orchestrator_processes_tier1_pages():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    from src.visual_intelligence.datatypes import (
        PageComplexity, Tier, VisualRegion, VisualEnrichmentResult,
    )
    from src.visual_intelligence.page_renderer import PageImage
    from PIL import Image

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()
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
        extracted = MagicMock()
        extracted.metrics = {}
        orch.merger.merge = MagicMock(return_value=extracted)
        result = await orch.enrich("doc-1", extracted, b"fake-pdf")
        orch.dit_detector.detect.assert_called_once()
        orch.merger.merge.assert_called_once()
