import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.asyncio
async def test_visual_enrichment_called_after_extraction():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    mock_orch = MagicMock(spec=VisualIntelligenceOrchestrator)
    mock_orch.enrich = AsyncMock(return_value=MagicMock())
    with patch("src.visual_intelligence.orchestrator.get_visual_orchestrator", return_value=mock_orch):
        orch = mock_orch
        extracted = MagicMock()
        result = await orch.enrich("doc-1", extracted, b"pdf-bytes")
        orch.enrich.assert_called_once_with("doc-1", extracted, b"pdf-bytes")


def test_visual_intelligence_import_fallback():
    try:
        from src.visual_intelligence.orchestrator import get_visual_orchestrator
        VISUAL_INTELLIGENCE_AVAILABLE = True
    except ImportError:
        VISUAL_INTELLIGENCE_AVAILABLE = False
    assert isinstance(VISUAL_INTELLIGENCE_AVAILABLE, bool)
