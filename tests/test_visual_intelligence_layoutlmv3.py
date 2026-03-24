import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from src.visual_intelligence.datatypes import KVPair


def test_layoutlmv3_creation():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    with patch("src.visual_intelligence.model_pool.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        extractor = LayoutLMv3Extractor()
        assert extractor is not None


def test_layoutlmv3_extracts_kv_pairs():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    mock_pool.load_model.return_value = MagicMock()
    mock_pool.load_processor.return_value = MagicMock()
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        extractor = LayoutLMv3Extractor()
        with patch.object(extractor, "_run_inference", return_value={
            "token_labels": [
                {"text": "Invoice No", "label": "question", "bbox": (10, 20, 100, 40)},
                {"text": "12345", "label": "answer", "bbox": (110, 20, 200, 40)},
                {"text": "Date", "label": "question", "bbox": (10, 50, 60, 70)},
                {"text": "2026-01-15", "label": "answer", "bbox": (70, 50, 180, 70)},
            ],
        }):
            img = Image.new("RGB", (800, 1000), "white")
            words = ["Invoice", "No", "12345", "Date", "2026-01-15"]
            boxes = [(10, 20, 60, 40), (65, 20, 100, 40), (110, 20, 200, 40),
                     (10, 50, 60, 70), (70, 50, 180, 70)]
            kv_pairs = extractor.extract(img, words, boxes, page_number=1)
            assert len(kv_pairs) == 2
            assert kv_pairs[0].key == "Invoice No"
            assert kv_pairs[0].value == "12345"
            assert kv_pairs[1].key == "Date"
            assert kv_pairs[1].value == "2026-01-15"


def test_layoutlmv3_unavailable_returns_empty():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False
    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        extractor = LayoutLMv3Extractor()
        img = Image.new("RGB", (800, 1000), "white")
        kv_pairs = extractor.extract(img, [], [], page_number=1)
        assert kv_pairs == []


def test_pair_question_answer():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    extractor = LayoutLMv3Extractor.__new__(LayoutLMv3Extractor)
    tokens = [
        {"text": "Name", "label": "question", "bbox": (10, 10, 50, 20)},
        {"text": "John Doe", "label": "answer", "bbox": (60, 10, 150, 20)},
        {"text": "Some text", "label": "other", "bbox": (10, 30, 100, 40)},
        {"text": "Total", "label": "question", "bbox": (10, 50, 50, 60)},
        {"text": "$500", "label": "answer", "bbox": (60, 50, 100, 60)},
    ]
    pairs = extractor._pair_question_answer(tokens, page=1)
    assert len(pairs) == 2
    assert pairs[0].key == "Name"
    assert pairs[0].value == "John Doe"
    assert pairs[1].key == "Total"
    assert pairs[1].value == "$500"
