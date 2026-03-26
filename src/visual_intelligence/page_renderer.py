"""Page Renderer — converts PDF pages to PIL images using PyMuPDF."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]
    warnings.warn(
        "PyMuPDF (fitz) is not installed. PageRenderer will not be able to render PDFs. "
        "Install with: pip install PyMuPDF",
        stacklevel=2,
    )


@dataclass
class PageImage:
    """A rendered page image with metadata."""

    page_number: int  # 1-indexed
    image: Image.Image
    width: int
    height: int
    dpi: int


class PageRenderer:
    """Renders PDF pages as PIL images using PyMuPDF (fitz)."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def render(
        self,
        pdf_bytes: bytes,
        page_numbers: Optional[List[int]] = None,
    ) -> List[PageImage]:
        """Render PDF pages to PIL images.

        Args:
            pdf_bytes: Raw PDF file bytes.
            page_numbers: Optional list of 1-indexed page numbers to render.
                          If *None*, all pages are rendered.

        Returns:
            List of ``PageImage`` instances, one per rendered page.
            Returns an empty list on non-PDF input or if fitz is unavailable.
        """
        if fitz is None:
            logger.warning("PyMuPDF not available — cannot render pages.")
            return []

        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            # Determine which pages to render (convert 1-indexed → 0-indexed).
            if page_numbers is not None:
                indices = [p - 1 for p in page_numbers if 0 <= p - 1 < len(doc)]
            else:
                indices = list(range(len(doc)))

            results: List[PageImage] = []
            for idx in indices:
                page = doc[idx]
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                results.append(
                    PageImage(
                        page_number=idx + 1,
                        image=img,
                        width=pix.width,
                        height=pix.height,
                        dpi=self.dpi,
                    )
                )

            return results
        except Exception:
            logger.debug("Failed to render PDF pages — input may not be a valid PDF.", exc_info=True)
            return []
        finally:
            if doc is not None:
                doc.close()
