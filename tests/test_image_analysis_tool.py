from __future__ import annotations

import asyncio
import base64
import io

import pytest
from PIL import Image

from src.tools.base import ToolError
from src.tools.image_analysis import OCRCandidate, image_analysis_handler


def _sample_png_base64() -> str:
    image = Image.new("RGB", (12, 12), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_image_analysis_handler_extracts_text_and_insights(monkeypatch):
    mock_text = "Invoice Number: INV-001\nTotal Amount: $5200\nEmail: finance@example.com"
    best = OCRCandidate(engine="easyocr", variant="original", text=mock_text, confidence=92.6)

    def _fake_extract(image, *, ocr_engine, high_accuracy):
        return best, [best]

    monkeypatch.setattr("src.tools.image_analysis._extract_text_high_accuracy", _fake_extract)

    result = asyncio.run(
        image_analysis_handler(
            {
                "input": {
                    "image_base64": _sample_png_base64(),
                    "query": "What is the total amount?",
                }
            },
            correlation_id="cid-test",
        )
    )

    assert result["grounded"] is True
    assert result["context_found"] is True
    assert result["result"]["text"] == mock_text
    assert result["result"]["ocr"]["engine"] == "easyocr"
    assert result["result"]["insights"]["document_type"] == "invoice"
    assert result["result"]["insights"]["entities"]["emails"] == ["finance@example.com"]
    assert any("Total Amount: $5200" in line for line in result["result"]["insights"]["query_hits"])


def test_image_analysis_handler_requires_image():
    with pytest.raises(ToolError) as exc:
        asyncio.run(image_analysis_handler({"input": {"query": "extract text"}}))
    assert exc.value.code == "missing_image"


def test_image_analysis_handler_warns_on_low_confidence(monkeypatch):
    best = OCRCandidate(engine="pytesseract", variant="original", text="blurred", confidence=32.0)

    def _fake_extract(image, *, ocr_engine, high_accuracy):
        return best, [best]

    monkeypatch.setattr("src.tools.image_analysis._extract_text_high_accuracy", _fake_extract)

    result = asyncio.run(
        image_analysis_handler(
            {
                "input": {
                    "image_base64": _sample_png_base64(),
                    "ocr_engine": "auto",
                    "min_confidence": 60,
                }
            }
        )
    )

    assert any("Low OCR confidence" in warning for warning in result["warnings"])
    assert any("EasyOCR output was unavailable" in warning for warning in result["warnings"])
