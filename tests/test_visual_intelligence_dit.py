import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from src.visual_intelligence.datatypes import VisualRegion


def test_dit_detector_creation():
    from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
    with patch("src.visual_intelligence.models.dit_layout.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        detector = DiTLayoutDetector()
        assert detector is not None


def test_dit_detect_returns_regions():
    from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
    mock_pool = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = MagicMock()
    mock_output.pred_boxes = MagicMock()
    mock_model.return_value = mock_output
    mock_model.config.id2label = {0: "text", 1: "title", 2: "table", 3: "figure"}
    mock_pool.load_model.return_value = mock_model
    mock_pool.load_processor.return_value = mock_processor
    mock_pool.is_available.return_value = True
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.models.dit_layout.get_model_pool", return_value=mock_pool):
        with patch.object(DiTLayoutDetector, "_postprocess") as mock_pp:
            mock_pp.return_value = [
                VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.95, page=1),
                VisualRegion(bbox=(10, 420, 300, 500), label="table", confidence=0.88, page=1),
            ]
            detector = DiTLayoutDetector()
            img = Image.new("RGB", (800, 1000), "white")
            regions = detector.detect(img, page_number=1)
            assert len(regions) == 2
            assert regions[0].label == "text"
            assert regions[1].label == "table"


def test_dit_unavailable_returns_empty():
    from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False
    with patch("src.visual_intelligence.models.dit_layout.get_model_pool", return_value=mock_pool):
        detector = DiTLayoutDetector()
        img = Image.new("RGB", (800, 1000), "white")
        regions = detector.detect(img, page_number=1)
        assert regions == []
