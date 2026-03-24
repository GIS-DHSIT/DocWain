"""Batch processing and model warmup tests."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import Tier, VisualRegion, PageComplexity
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_concurrent_page_processing():
    """Multiple pages should be processed."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 4}
        extracted.tables = []
        extracted.figures = []
        extracted.sections = []
        extracted.full_text = "text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=i, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False)
            for i in range(1, 5)
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=i, image=mock_img, width=800, height=1000, dpi=300)
            for i in range(1, 5)
        ])

        call_count = 0

        def mock_detect(img, page):
            nonlocal call_count
            call_count += 1
            return [VisualRegion(bbox=(10, 20, 300, 400), label="text",
                                confidence=0.9, page=page)]

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = mock_detect
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.kg_enricher = MagicMock()

        import fitz
        doc = fitz.open()
        for _ in range(4):
            doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-batch", extracted, pdf_bytes)
        assert call_count == 4


def test_model_warmup():
    """Model pool should support pre-warming models."""
    from src.visual_intelligence.model_pool import ModelPool

    pool = ModelPool()
    with patch.object(pool, "load_model", return_value=MagicMock()) as mock_load:
        pool.warmup(["dit", "table_det"])
        assert mock_load.call_count == 2
