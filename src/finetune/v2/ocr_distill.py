"""glm-ocr distillation pipeline.

Distils OCR outputs from GLM-4V (via Ollama) into SFT training pairs
for the DocWain unified vision model.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OCRDistillResult:
    """Single OCR distillation result from a document image."""

    image_path: str
    ocr_text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----- conversion helpers -----

    def to_sft_pair(self) -> Dict[str, Any]:
        """Convert this result into a chat-format SFT training pair.

        Returns a dict with a ``messages`` list containing system, user
        (with ``<image>`` token), and assistant turns.
        """
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are DocWain, an expert document intelligence assistant. "
                        "Extract all visible text from the provided document image accurately."
                    ),
                },
                {
                    "role": "user",
                    "content": "<image>\nExtract all text from this document image.",
                },
                {
                    "role": "assistant",
                    "content": self.ocr_text,
                },
            ],
            "metadata": {
                "image_path": self.image_path,
                "confidence": self.confidence,
                **self.metadata,
            },
        }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_results(
    results: Sequence[OCRDistillResult],
    min_confidence: float = 0.7,
) -> List[OCRDistillResult]:
    """Keep only results that meet the minimum confidence threshold."""
    return [r for r in results if r.confidence >= min_confidence]


# ---------------------------------------------------------------------------
# Ollama / GLM-4V integration
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_glm_ocr(
    image_path: str,
    model: str = "glm-4v",
    ollama_base_url: str = "http://localhost:11434",
    *,
    prompt: str = "Extract all visible text from this document image. Return only the extracted text.",
    timeout: int = 120,
) -> OCRDistillResult:
    """Call Ollama with a base64-encoded image and return an OCRDistillResult.

    Parameters
    ----------
    image_path:
        Path to the image file on disk.
    model:
        Ollama model tag to use (default ``glm-4v``).
    ollama_base_url:
        Ollama API base URL.
    prompt:
        The text prompt sent alongside the image.
    timeout:
        Request timeout in seconds.
    """
    import httpx

    b64_image = _encode_image(image_path)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
    }

    response = httpx.post(
        f"{ollama_base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    ocr_text = data.get("response", "").strip()
    # Heuristic confidence: longer responses from a VLM are generally more
    # complete; very short ones may indicate a failed read.
    confidence = min(1.0, len(ocr_text) / 200) if ocr_text else 0.0

    logger.info(
        "GLM-OCR for %s: %d chars, confidence=%.2f",
        image_path,
        len(ocr_text),
        confidence,
    )

    return OCRDistillResult(
        image_path=image_path,
        ocr_text=ocr_text,
        confidence=confidence,
        metadata={"model": model, "raw_response_keys": list(data.keys())},
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def distill_batch(
    image_paths: Sequence[str],
    *,
    model: str = "glm-4v",
    ollama_base_url: str = "http://localhost:11434",
    min_confidence: float = 0.7,
    output_path: Optional[str] = None,
) -> List[OCRDistillResult]:
    """Run GLM-OCR over a batch of images and optionally write SFT pairs.

    Parameters
    ----------
    image_paths:
        Iterable of image file paths.
    model:
        Ollama model tag.
    ollama_base_url:
        Ollama API base URL.
    min_confidence:
        Minimum confidence to keep a result.
    output_path:
        If provided, filtered SFT pairs are written as JSONL to this path.

    Returns
    -------
    List of *filtered* ``OCRDistillResult`` objects.
    """
    raw_results: List[OCRDistillResult] = []
    for img in image_paths:
        try:
            result = run_glm_ocr(
                img, model=model, ollama_base_url=ollama_base_url
            )
            raw_results.append(result)
        except Exception:
            logger.exception("Failed to process image %s", img)

    filtered = filter_results(raw_results, min_confidence=min_confidence)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in filtered:
                fh.write(json.dumps(r.to_sft_pair(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d SFT pairs to %s", len(filtered), output_path)

    return filtered
