"""TrOCR-based enhancer for low-confidence OCR text regions.

Uses Microsoft TrOCR (printed and handwritten variants) to re-run OCR on
image regions where the original OCR confidence falls below
``ENHANCEMENT_THRESHOLD``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.visual_intelligence.datatypes import OCRPatch
from src.visual_intelligence import model_pool as _model_pool_mod

logger = logging.getLogger(__name__)

ENHANCEMENT_THRESHOLD = 0.70


class TrOCREnhancer:
    """Enhanced OCR for low-confidence text regions using TrOCR."""

    def __init__(self) -> None:
        self._pool = _model_pool_mod.get_model_pool()
        self.trocr_printed_model = None
        self.trocr_printed_processor = None
        self.trocr_handwritten_model = None
        self.trocr_handwritten_processor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_printed(self) -> bool:
        """Load the printed TrOCR model pair. Returns True on success."""
        if self.trocr_printed_model is not None and self.trocr_printed_processor is not None:
            return True
        if not self._pool.is_available("trocr_printed"):
            return False
        self.trocr_printed_model = self._pool.load_model("trocr_printed")
        self.trocr_printed_processor = self._pool.load_processor("trocr_printed")
        return self.trocr_printed_model is not None and self.trocr_printed_processor is not None

    def _ensure_handwritten(self) -> bool:
        """Load the handwritten TrOCR model pair. Returns True on success."""
        if self.trocr_handwritten_model is not None and self.trocr_handwritten_processor is not None:
            return True
        if not self._pool.is_available("trocr_handwritten"):
            return False
        self.trocr_handwritten_model = self._pool.load_model("trocr_handwritten")
        self.trocr_handwritten_processor = self._pool.load_processor("trocr_handwritten")
        return self.trocr_handwritten_model is not None and self.trocr_handwritten_processor is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance_region(
        self,
        image,
        page: int,
        bbox: Tuple[float, float, float, float],
        original_text: str,
        original_confidence: float,
        is_handwritten: bool = False,
    ) -> Optional[OCRPatch]:
        """Re-run OCR on a region if confidence is below threshold.

        Returns an ``OCRPatch`` when TrOCR produces a higher-confidence
        result, or ``None`` if enhancement is not needed or not possible.
        """
        if original_confidence >= ENHANCEMENT_THRESHOLD:
            return None

        # Ensure the appropriate model is available
        if is_handwritten:
            if not self._ensure_handwritten():
                return None
        else:
            if not self._ensure_printed():
                return None

        try:
            enhanced_text, enhanced_confidence = self._run_trocr(image, is_handwritten=is_handwritten)
        except Exception:
            logger.exception("TrOCR enhancement failed for page %d bbox %s", page, bbox)
            return None

        if enhanced_confidence <= original_confidence:
            return None

        method = "trocr_handwritten" if is_handwritten else "trocr_printed"
        return OCRPatch(
            page=page,
            bbox=bbox,
            original_text=original_text,
            enhanced_text=enhanced_text,
            original_confidence=original_confidence,
            enhanced_confidence=enhanced_confidence,
            method=method,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_trocr(self, image, is_handwritten: bool = False) -> Tuple[str, float]:
        """Run TrOCR inference on *image* and return (text, confidence)."""
        import torch

        # Ensure RGB
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        if is_handwritten:
            model = self.trocr_handwritten_model
            processor = self.trocr_handwritten_processor
            model_key = "trocr_handwritten"
        else:
            model = self.trocr_printed_model
            processor = self.trocr_printed_processor
            model_key = "trocr_printed"

        device = self._pool.get_device(model_key)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Estimate confidence from generation scores
        confidence = self._estimate_confidence(outputs.scores)

        return text, confidence

    @staticmethod
    def _estimate_confidence(scores) -> float:
        """Estimate average token-level confidence from generation scores."""
        import torch

        if not scores:
            return 0.0

        confidences = []
        for score in scores:
            probs = torch.softmax(score, dim=-1)
            max_prob = probs.max(dim=-1).values.mean().item()
            confidences.append(max_prob)

        return sum(confidences) / len(confidences) if confidences else 0.0
