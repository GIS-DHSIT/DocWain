"""DiT layout detection wrapper.

Uses Microsoft DiT (Document Image Transformer) fine-tuned on PubLayNet
to detect document layout regions such as text blocks, titles, tables,
figures, and lists.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from PIL import Image

from src.visual_intelligence.datatypes import VisualRegion
from src.visual_intelligence.model_pool import get_model_pool

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.7


class DiTLayoutDetector:
    """Detect document layout regions using DiT object detection."""

    _MODEL_KEY = "dit"

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image, page_number: int) -> List[VisualRegion]:
        """Run layout detection on *image* and return detected regions.

        Returns an empty list when the model is unavailable or inference
        fails for any reason.
        """
        try:
            pool = get_model_pool()

            if not pool.is_available(self._MODEL_KEY):
                logger.debug("DiT model is not available; skipping detection.")
                return []

            if not self._ensure_loaded():
                return []

            # Prepare inputs
            inputs = self._processor(images=image, return_tensors="pt")
            device = pool.get_device(self._MODEL_KEY)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process
            target_sizes = torch.tensor(
                [[image.height, image.width]], dtype=torch.float32,
            )
            results = self._processor.post_process_object_detection(
                outputs, threshold=MIN_CONFIDENCE, target_sizes=target_sizes,
            )

            return self._postprocess(results, page_number)

        except Exception:
            logger.warning(
                "DiT layout detection failed for page %s", page_number,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load model and processor from the model pool.

        Returns ``True`` if both are ready, ``False`` otherwise.
        """
        if self._model is not None and self._processor is not None:
            return True

        try:
            pool = get_model_pool()
            self._model = pool.load_model(self._MODEL_KEY)
            self._processor = pool.load_processor(self._MODEL_KEY)

            if self._model is None or self._processor is None:
                logger.warning("DiT model or processor could not be loaded.")
                return False

            return True
        except Exception:
            logger.warning("Failed to load DiT model.", exc_info=True)
            return False

    def _postprocess(
        self,
        results: List[Dict[str, Any]],
        page_number: int,
    ) -> List[VisualRegion]:
        """Convert raw detection outputs to a sorted list of VisualRegion."""
        regions: List[VisualRegion] = []
        id2label = self._model.config.id2label

        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            for score, label_id, box in zip(scores, labels, boxes):
                label_name = id2label.get(
                    label_id.item() if hasattr(label_id, "item") else int(label_id),
                    f"class_{label_id}",
                )
                bbox = tuple(
                    b.item() if hasattr(b, "item") else float(b) for b in box
                )
                regions.append(
                    VisualRegion(
                        bbox=bbox,
                        label=label_name,
                        confidence=float(
                            score.item() if hasattr(score, "item") else score,
                        ),
                        page=page_number,
                    )
                )

        # Sort by vertical position (top of bounding box).
        regions.sort(key=lambda r: r.bbox[1])
        return regions
