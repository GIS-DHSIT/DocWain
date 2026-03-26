"""Thread-safe, lazy-loading model pool for visual intelligence models.

Provides automatic download, GPU/CPU routing, and graceful degradation
when a model cannot be loaded.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default HuggingFace model IDs
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "dit": "microsoft/dit-large-finetuned-publaynet",
    "table_det": "microsoft/table-transformer-detection",
    "table_str": "microsoft/table-transformer-structure-recognition",
    "trocr_printed": "microsoft/trocr-base-printed",
    "trocr_handwritten": "microsoft/trocr-base-handwritten",
    "layoutlmv3": "microsoft/layoutlmv3-base",
}

# Models that benefit from GPU acceleration.
GPU_PREFERRED: Set[str] = {"dit", "table_det", "table_str", "trocr_printed", "trocr_handwritten", "layoutlmv3"}

# Object detection models that need AutoModelForObjectDetection.
_OBJECT_DETECTION_KEYS: Set[str] = {"dit", "table_det", "table_str"}


class ModelPool:
    """Lazy-loading, thread-safe pool for visual intelligence models."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}
        self.disabled_models: Set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, model_key: str) -> bool:
        """Return True if the model is not disabled."""
        return model_key not in self.disabled_models

    def get_device(self, model_key: str) -> "torch.device":
        """Choose the best device for *model_key*."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; cannot determine device.")

        if model_key in GPU_PREFERRED and torch.cuda.is_available():
            # Honour Config.VisualIntelligence.GPU_DEVICE when available.
            try:
                from src.api.config import Config
                gpu_device = getattr(Config.VisualIntelligence, "GPU_DEVICE", "cuda:0")
            except Exception:
                gpu_device = "cuda:0"
            return torch.device(gpu_device)
        return torch.device("cpu")

    def load_model(self, model_key: str) -> Optional[Any]:
        """Load a model through 3 stages: cache -> download -> warn+skip.

        Returns the model object or ``None`` if loading fails entirely.
        """
        with self._lock:
            # Stage 0: already cached in-process
            if model_key in self._models:
                return self._models[model_key]

            if not self.is_available(model_key):
                return None

            # Stage 1: try loading from HF cache
            model = self._try_load(model_key)

            # Stage 2: force download
            if model is None:
                logger.info("Model %s not in cache, attempting download...", model_key)
                model = self._try_install(model_key)

            # Stage 3: give up gracefully
            if model is None:
                logger.warning(
                    "Model %s could not be loaded or installed — disabling.",
                    model_key,
                )
                self.disabled_models.add(model_key)
                return None

            self._models[model_key] = model
            return model

    def load_processor(self, model_key: str) -> Optional[Any]:
        """Load the processor / feature extractor for *model_key*."""
        with self._lock:
            if model_key in self._processors:
                return self._processors[model_key]

            if not self.is_available(model_key):
                return None

            processor = self._try_load_processor(model_key)
            if processor is None:
                logger.warning(
                    "Processor for %s could not be loaded.", model_key,
                )
                return None

            self._processors[model_key] = processor
            return processor

    def unload(self, model_key: str) -> None:
        """Remove a model and its processor from the pool."""
        with self._lock:
            self._models.pop(model_key, None)
            self._processors.pop(model_key, None)

    def unload_all(self) -> None:
        """Remove all models and processors from the pool."""
        with self._lock:
            self._models.clear()
            self._processors.clear()

    def warmup(self, model_keys: List[str]) -> None:
        """Pre-load models to avoid cold-start latency."""
        for key in model_keys:
            if self.is_available(key):
                self.load_model(key)
                logger.info("Warmed up model '%s'", key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_id(self, model_key: str) -> str:
        """Return the HuggingFace model ID for *model_key*.

        Checks ``Config.VisualIntelligence`` attributes first, then falls
        back to ``MODEL_REGISTRY``.
        """
        config_attr_map = {
            "dit": "DIT_MODEL",
            "table_det": "TABLE_DET_MODEL",
            "table_str": "TABLE_STR_MODEL",
            "trocr_printed": "TROCR_PRINTED_MODEL",
            "trocr_handwritten": "TROCR_HANDWRITTEN_MODEL",
            "layoutlmv3": "LAYOUTLMV3_MODEL",
        }

        attr_name = config_attr_map.get(model_key)
        if attr_name:
            try:
                from src.api.config import Config
                value = getattr(Config.VisualIntelligence, attr_name, None)
                if value:
                    return value
            except Exception:
                pass

        return MODEL_REGISTRY.get(model_key, model_key)

    def _try_load(self, model_key: str) -> Optional[Any]:
        """Try loading the model from the local HuggingFace cache."""
        try:
            from transformers import AutoModel, AutoModelForObjectDetection

            model_id = self._resolve_model_id(model_key)
            device = self.get_device(model_key)

            if model_key in _OBJECT_DETECTION_KEYS:
                model = AutoModelForObjectDetection.from_pretrained(
                    model_id, local_files_only=True,
                )
            else:
                model = AutoModel.from_pretrained(
                    model_id, local_files_only=True,
                )

            model = model.to(device).eval()
            logger.info("Loaded %s (%s) from cache on %s", model_key, model_id, device)
            return model
        except Exception:
            return None

    def _try_install(self, model_key: str) -> Optional[Any]:
        """Force-download the model from HuggingFace Hub."""
        try:
            from transformers import AutoModel, AutoModelForObjectDetection

            model_id = self._resolve_model_id(model_key)
            device = self.get_device(model_key)

            if model_key in _OBJECT_DETECTION_KEYS:
                model = AutoModelForObjectDetection.from_pretrained(
                    model_id, force_download=True,
                )
            else:
                model = AutoModel.from_pretrained(
                    model_id, force_download=True,
                )

            model = model.to(device).eval()
            logger.info("Downloaded and loaded %s (%s) on %s", model_key, model_id, device)
            return model
        except Exception:
            logger.exception("Failed to download model %s", model_key)
            return None

    def _try_load_processor(self, model_key: str) -> Optional[Any]:
        """Load the image processor or feature extractor for *model_key*."""
        model_id = self._resolve_model_id(model_key)

        # Try AutoImageProcessor first (newer API), then AutoFeatureExtractor.
        try:
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(model_id)
            logger.info("Loaded processor (AutoImageProcessor) for %s", model_key)
            return processor
        except Exception:
            pass

        try:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(model_id)
            logger.info("Loaded processor (AutoFeatureExtractor) for %s", model_key)
            return processor
        except Exception:
            logger.exception("Failed to load processor for %s", model_key)
            return None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_pool_instance: Optional[ModelPool] = None
_pool_lock = threading.Lock()


def get_model_pool() -> ModelPool:
    """Return the global ``ModelPool`` singleton (thread-safe)."""
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = ModelPool()
    return _pool_instance
