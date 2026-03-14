"""Vision OCR client wrapping Ollama's multimodal model for image text extraction.

Uses ``glm-ocr:latest`` (or configurable model) to extract text from:
- Content images embedded in PDFs (charts, diagrams, tables-as-images)
- Full scanned pages when traditional OCR underperforms

Thread-safe singleton via ``get_vision_ocr_client()``.
"""

from __future__ import annotations

import io
from src.utils.logging_utils import get_logger
import threading
from typing import Any, Optional, Tuple

logger = get_logger(__name__)

_PAGE_PROMPT = (
    "Extract ALL text from this scanned document page. "
    "Preserve the original layout, formatting, and structure as closely as possible. "
    "Return ONLY the extracted text, no commentary."
)

_CONTENT_PROMPT = (
    "Extract ALL text, labels, numbers, and data from this image. "
    "If it is a chart or diagram, describe the data values and axis labels. "
    "If it is a table, preserve the row and column structure. "
    "Return ONLY the extracted text, no commentary."
)

# ── Specialized analysis prompts for MoE vision sub-agent ──────────

_CHART_ANALYSIS_PROMPT = (
    "Analyze this chart/graph image in detail:\n"
    "1. Chart type (bar, line, pie, scatter, area, etc.)\n"
    "2. Title and axis labels\n"
    "3. ALL data values and series — extract every number visible\n"
    "4. Trends, patterns, and outliers\n"
    "5. Key insights and takeaways\n"
    "Format as structured analysis with data tables where applicable."
)

_TABLE_ANALYSIS_PROMPT = (
    "Extract and analyze this table image:\n"
    "1. Reproduce the COMPLETE table in markdown format\n"
    "2. Column headers and row labels\n"
    "3. ALL cell values — preserve exact numbers and text\n"
    "4. Calculated totals, averages, or aggregates if visible\n"
    "5. Data quality notes (missing cells, unclear values marked [unclear])\n"
    "Return the markdown table first, then analysis."
)

_DIAGRAM_ANALYSIS_PROMPT = (
    "Analyze this diagram/flowchart image:\n"
    "1. Diagram type (flowchart, process flow, org chart, architecture, etc.)\n"
    "2. ALL nodes/boxes with their labels\n"
    "3. ALL connections/arrows with their labels and directions\n"
    "4. Decision points and branches\n"
    "5. Overall process description in plain language\n"
    "Return structured analysis with node list and flow sequence."
)

_PHOTO_ANALYSIS_PROMPT = (
    "Describe this photograph/image in detail:\n"
    "1. Main subject and scene description\n"
    "2. Key objects, people, or elements visible\n"
    "3. Any text, labels, signs, or identifiers in the image\n"
    "4. Context clues (location, time, setting)\n"
    "5. Quality assessment (resolution, clarity, lighting)\n"
    "Provide comprehensive description suitable for document indexing."
)

_ANALYSIS_PROMPTS = {
    "chart": _CHART_ANALYSIS_PROMPT,
    "table": _TABLE_ANALYSIS_PROMPT,
    "diagram": _DIAGRAM_ANALYSIS_PROMPT,
    "photo": _PHOTO_ANALYSIS_PROMPT,
    "general": _CONTENT_PROMPT,
}

class VisionOCRClient:
    """Ollama-based vision OCR using the ``images`` parameter."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        from src.api.config import Config

        self.model_name: str = model_name or getattr(
            getattr(Config, "VisionOCR", None), "MODEL", "glm-ocr:latest"
        )
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Lazy check whether the vision model is pulled in Ollama."""
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                import ollama

                ollama.show(self.model_name)
                self._available = True
            except Exception:
                self._available = False
            return self._available

    # ------------------------------------------------------------------
    # OCR entry points
    # ------------------------------------------------------------------

    def ocr_image(self, image: Any, prompt: Optional[str] = None) -> Tuple[str, Optional[float]]:
        """OCR a content image (chart, diagram, table screenshot).

        Parameters
        ----------
        image : PIL.Image.Image or bytes
            The image to process.
        prompt : str, optional
            Custom prompt; defaults to ``_CONTENT_PROMPT``.

        Returns
        -------
        (text, confidence)
            Extracted text and estimated confidence (0-100), or ``("", None)``
            on failure.
        """
        return self._run(image, prompt or _CONTENT_PROMPT)

    def ocr_page_image(self, page_image: Any) -> Tuple[str, Optional[float]]:
        """OCR a full scanned page rendered as an image.

        Parameters
        ----------
        page_image : PIL.Image.Image or bytes
            Rendered page image.

        Returns
        -------
        (text, confidence)
        """
        return self._run(page_image, _PAGE_PROMPT)

    def analyze_image(
        self,
        image: Any,
        analysis_type: str = "general",
        custom_prompt: Optional[str] = None,
    ) -> Tuple[str, Optional[float]]:
        """Perform specialized visual analysis using glm-ocr.

        Parameters
        ----------
        image : PIL.Image.Image or bytes
            The image to analyze.
        analysis_type : str
            One of "chart", "table", "diagram", "photo", "general".
        custom_prompt : str, optional
            Override the built-in prompt for this analysis type.

        Returns
        -------
        (analysis_text, confidence)
        """
        prompt = custom_prompt or _ANALYSIS_PROMPTS.get(analysis_type, _CONTENT_PROMPT)
        return self._run(image, prompt)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, image: Any, prompt: str) -> Tuple[str, Optional[float]]:
        if not self.is_available():
            return "", None
        try:
            png_bytes = self._to_png_bytes(image)
            if not png_bytes:
                return "", None

            import ollama

            resp = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[png_bytes],
                options={"temperature": 0.1, "num_predict": 4096},
            )
            text = (resp.get("response") or "").strip()
            confidence = self._estimate_confidence(text)
            return text, confidence
        except Exception as exc:
            logger.debug("Vision OCR failed: %s", exc)
            return "", None

    @staticmethod
    def _to_png_bytes(image: Any) -> Optional[bytes]:
        """Convert PIL Image or raw bytes to PNG bytes."""
        if isinstance(image, (bytes, bytearray)):
            return bytes(image)
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            logger.debug("Image → PNG conversion failed: %s", exc)
            return None

    @staticmethod
    def _estimate_confidence(text: str) -> Optional[float]:
        if not text:
            return None
        if len(text) > 20:
            return 85.0
        return 50.0

# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_client: Optional[VisionOCRClient] = None
_client_lock = threading.Lock()

def get_vision_ocr_client() -> Optional[VisionOCRClient]:
    """Return (or create) the global VisionOCRClient singleton."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        from src.api.config import Config

        cfg = getattr(Config, "VisionOCR", None)
        if cfg and not getattr(cfg, "ENABLED", True):
            return None
        _client = VisionOCRClient()
        return _client

def set_vision_ocr_client(client: Optional[VisionOCRClient]) -> None:
    """Override the singleton (used by tests)."""
    global _client
    _client = client
