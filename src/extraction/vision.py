"""Vision extraction pipeline — OCR and visual analysis via Ollama glm-ocr."""

import logging

logger = logging.getLogger(__name__)


class VisionExtractor:
    """Extracts content from visual elements using glm-ocr.

    Handles: OCR for scanned pages, diagram/chart extraction, table images.
    """

    def __init__(self, ollama_host: str = None, model: str = "glm-ocr"):
        self.ollama_host = ollama_host or "http://localhost:11434"
        self.model = model

    def extract(self, document_bytes: bytes, file_type: str,
                page_images: list = None) -> dict:
        """Run vision extraction.

        Returns dict with: ocr_text, diagrams, charts, table_images, scanned_text
        """
        # TODO: Implement vision extraction via glm-ocr
        # 1. Convert document pages to images
        # 2. Detect visual elements
        # 3. OCR scanned pages
        # 4. Extract chart data
        # 5. Extract table data from table images

        logger.info("Vision extraction called (stub)")
        return {
            "ocr_text": "",
            "diagrams": [],
            "charts": [],
            "table_images": [],
            "scanned_text": ""
        }
