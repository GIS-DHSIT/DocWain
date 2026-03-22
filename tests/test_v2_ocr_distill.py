import json, pytest
from pathlib import Path
from unittest.mock import MagicMock


class TestOCRDistiller:
    def test_distill_result_format(self):
        from src.finetune.v2.ocr_distill import OCRDistillResult
        r = OCRDistillResult(image_path="/tmp/test.png", ocr_text="Hello world", confidence=0.95)
        assert r.image_path == "/tmp/test.png"
        assert r.ocr_text == "Hello world"

    def test_convert_to_sft_pair(self):
        from src.finetune.v2.ocr_distill import OCRDistillResult
        r = OCRDistillResult(image_path="/tmp/test.png", ocr_text="Table header: Revenue", confidence=0.9)
        pair = r.to_sft_pair()
        assert "messages" in pair
        assert "<image>" in pair["messages"][1]["content"]
        assert "Table header: Revenue" in pair["messages"][2]["content"]

    def test_filter_low_confidence(self):
        from src.finetune.v2.ocr_distill import filter_results
        results = [MagicMock(confidence=0.95), MagicMock(confidence=0.3), MagicMock(confidence=0.85)]
        filtered = filter_results(results, min_confidence=0.7)
        assert len(filtered) == 2
