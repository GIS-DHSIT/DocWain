import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from src.visual_intelligence.datatypes import OCRPatch


def test_trocr_enhancer_creation():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
    with patch("src.visual_intelligence.model_pool.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        enhancer = TrOCREnhancer()
        assert enhancer is not None


def test_trocr_enhance_low_confidence_region():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    mock_pool.load_model.return_value = MagicMock()
    mock_pool.load_processor.return_value = MagicMock()
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        enhancer = TrOCREnhancer()
        with patch.object(enhancer, "_run_trocr", return_value=("Enhanced text here", 0.92)):
            img = Image.new("RGB", (200, 50), "white")
            result = enhancer.enhance_region(
                image=img, page=1, bbox=(10, 20, 210, 70),
                original_text="Enhancod toxt hore", original_confidence=0.45,
            )
            assert isinstance(result, OCRPatch)
            assert result.enhanced_text == "Enhanced text here"
            assert result.enhanced_confidence == 0.92
            assert result.method == "trocr_printed"


def test_trocr_skips_high_confidence():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        enhancer = TrOCREnhancer()
        img = Image.new("RGB", (200, 50), "white")
        result = enhancer.enhance_region(
            image=img, page=1, bbox=(10, 20, 210, 70),
            original_text="Good text", original_confidence=0.95,
        )
        assert result is None


def test_trocr_unavailable_returns_none():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False
    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        enhancer = TrOCREnhancer()
        img = Image.new("RGB", (200, 50), "white")
        result = enhancer.enhance_region(
            image=img, page=1, bbox=(0, 0, 200, 50),
            original_text="bad", original_confidence=0.3,
        )
        assert result is None
