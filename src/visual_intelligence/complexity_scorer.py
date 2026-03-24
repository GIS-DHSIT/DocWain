"""Adaptive complexity scoring — gates pages into processing tiers."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.visual_intelligence.datatypes import PageComplexity, Tier

logger = logging.getLogger(__name__)


class ComplexityScorer:
    """Score page complexity and assign processing tiers.

    Thresholds control when a page is escalated from SKIP → LIGHT → FULL:
      - ocr_high: below this, the page needs at least LIGHT processing.
      - ocr_low:  below this, the page needs FULL processing.
      - text_ratio_threshold: pages above this ratio are pure text (SKIP).
    """

    def __init__(
        self,
        ocr_high: float = 0.85,
        ocr_low: float = 0.70,
        text_ratio_threshold: float = 0.95,
    ) -> None:
        self.ocr_high = ocr_high
        self.ocr_low = ocr_low
        self.text_ratio_threshold = text_ratio_threshold

    # ------------------------------------------------------------------
    # Per-page scoring
    # ------------------------------------------------------------------

    def score_page(
        self,
        page: int,
        ocr_confidence: float = 1.0,
        image_ratio: float = 0.0,
        has_tables: bool = False,
        has_forms: bool = False,
        block_types: Optional[Dict[str, int]] = None,
    ) -> PageComplexity:
        """Evaluate a single page and return its complexity assessment.

        Tier assignment (first matching rule wins):
          FULL  — ocr_confidence < ocr_low
                  OR (has_forms AND image_ratio > 0.3)
                  OR image_ratio > 0.7
          LIGHT — has_tables
                  OR image_ratio > 0.1
                  OR ocr_confidence < ocr_high
          SKIP  — everything else
        """
        signals: Dict[str, Any] = {
            "ocr_confidence": ocr_confidence,
            "image_ratio": image_ratio,
            "has_tables": has_tables,
            "has_forms": has_forms,
            "block_types": block_types or {},
        }

        # --- Tier FULL checks ---
        if ocr_confidence < self.ocr_low:
            tier = Tier.FULL
            signals["reason"] = "ocr_confidence below ocr_low threshold"
        elif has_forms and image_ratio > 0.3:
            tier = Tier.FULL
            signals["reason"] = "form with significant image content"
        elif image_ratio > 0.7:
            tier = Tier.FULL
            signals["reason"] = "image_ratio exceeds 0.7"
        # --- Tier LIGHT checks ---
        elif has_tables:
            tier = Tier.LIGHT
            signals["reason"] = "page contains tables"
        elif image_ratio > 0.1:
            tier = Tier.LIGHT
            signals["reason"] = "image_ratio exceeds 0.1"
        elif ocr_confidence < self.ocr_high:
            tier = Tier.LIGHT
            signals["reason"] = "ocr_confidence below ocr_high threshold"
        # --- Tier SKIP ---
        else:
            tier = Tier.SKIP
            signals["reason"] = "simple text page"

        return PageComplexity(
            page=page,
            tier=tier,
            ocr_confidence=ocr_confidence,
            image_ratio=image_ratio,
            has_tables=has_tables,
            has_forms=has_forms,
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Document-level scoring
    # ------------------------------------------------------------------

    def score_document(self, extracted_doc: Any) -> List[PageComplexity]:
        """Score every page in an extracted document.

        Reads ``extracted_doc.metrics["total_pages"]`` for page count, then
        aggregates per-page signals from ``extracted_doc.tables`` and
        ``extracted_doc.figures``.
        """
        total_pages: int = extracted_doc.metrics.get("total_pages", 0)
        if total_pages == 0:
            logger.warning("score_document called with 0 total_pages")
            return []

        # Collect per-page signals from tables and figures
        table_pages: set = set()
        figure_signals: Dict[int, List[float]] = defaultdict(list)

        for table in getattr(extracted_doc, "tables", []):
            table_pages.add(table.page)

        for figure in getattr(extracted_doc, "figures", []):
            ocr_conf = getattr(figure, "ocr_confidence", 1.0)
            figure_signals[figure.page].append(ocr_conf)

        results: List[PageComplexity] = []
        for page_num in range(1, total_pages + 1):
            has_tables = page_num in table_pages
            has_figures = page_num in figure_signals

            # Use the minimum OCR confidence from figures on this page
            if figure_signals[page_num]:
                ocr_confidence = min(figure_signals[page_num])
            else:
                ocr_confidence = 1.0

            # Estimate image ratio from presence of figures
            image_ratio = 0.3 if has_figures else 0.0

            complexity = self.score_page(
                page=page_num,
                ocr_confidence=ocr_confidence,
                image_ratio=image_ratio,
                has_tables=has_tables,
                has_forms=False,
            )
            results.append(complexity)

        return results
