import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from src.visual_intelligence.datatypes import StructuredTableResult


def test_table_detector_creation():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    with patch("src.visual_intelligence.models.table_transformer.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        detector = TableTransformerDetector()
        assert detector is not None


def test_table_detect_returns_structured_results():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    mock_pool.load_model.return_value = MagicMock()
    mock_pool.load_processor.return_value = MagicMock()
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.models.table_transformer.get_model_pool", return_value=mock_pool):
        detector = TableTransformerDetector()
        with patch.object(detector, "_detect_tables", return_value=[
            {"bbox": (100, 200, 500, 600), "confidence": 0.92},
        ]):
            with patch.object(detector, "_recognize_structure", return_value=StructuredTableResult(
                page=1, bbox=(100, 200, 500, 600),
                headers=["Col1", "Col2"], rows=[["A", "1"], ["B", "2"]],
                spans=[], confidence=0.88,
            )):
                img = Image.new("RGB", (800, 1000), "white")
                tables = detector.extract(img, page_number=1)
                assert len(tables) == 1
                assert tables[0].headers == ["Col1", "Col2"]
                assert len(tables[0].rows) == 2


def test_table_unavailable_returns_empty():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False
    with patch("src.visual_intelligence.models.table_transformer.get_model_pool", return_value=mock_pool):
        detector = TableTransformerDetector()
        img = Image.new("RGB", (800, 1000), "white")
        tables = detector.extract(img, page_number=1)
        assert tables == []


def test_cells_to_grid():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    detector = TableTransformerDetector.__new__(TableTransformerDetector)
    cells = [
        {"bbox": (100, 200, 250, 240), "row": 0, "col": 0, "text": "Name"},
        {"bbox": (260, 200, 400, 240), "row": 0, "col": 1, "text": "Value"},
        {"bbox": (100, 250, 250, 290), "row": 1, "col": 0, "text": "A"},
        {"bbox": (260, 250, 400, 290), "row": 1, "col": 1, "text": "1"},
    ]
    headers, rows = detector._cells_to_grid(cells)
    assert headers == ["Name", "Value"]
    assert rows == [["A", "1"]]
