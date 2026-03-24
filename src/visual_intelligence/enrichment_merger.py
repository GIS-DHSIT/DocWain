"""Confidence-based merger that reconciles visual layer results with existing extraction.

Key principle: NEVER delete existing data, only add or upgrade.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.visual_intelligence.datatypes import (
    KVPair,
    OCRPatch,
    StructuredTableResult,
    VisualEnrichmentResult,
    VisualRegion,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_MARGIN = 0.15
LAYOUT_CONFIDENCE_MIN = 0.7


class EnrichmentMerger:
    """Merges visual intelligence enrichments into an extracted document.

    All merge operations are additive -- existing data is never removed.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, extracted_doc: Any, visual_result: VisualEnrichmentResult) -> Any:
        """Merge *visual_result* into *extracted_doc* and return the doc.

        If the visual result contains no enrichments the document is returned
        unchanged (aside from provenance tags).
        """
        has_enrichments = (
            visual_result.regions
            or visual_result.tables
            or visual_result.ocr_patches
            or visual_result.kv_pairs
        )

        if not has_enrichments:
            self._tag_provenance(extracted_doc, visual_result)
            return extracted_doc

        self._merge_layout(extracted_doc, visual_result.regions)
        self._merge_tables(extracted_doc, visual_result.tables)
        self._merge_ocr(extracted_doc, visual_result.ocr_patches)
        self._merge_kv_pairs(extracted_doc, visual_result.kv_pairs)
        self._tag_provenance(extracted_doc, visual_result)

        return extracted_doc

    # ------------------------------------------------------------------
    # Layout regions
    # ------------------------------------------------------------------

    def _merge_layout(self, doc: Any, regions: List[VisualRegion]) -> None:
        if not regions:
            return

        high_conf = [r for r in regions if r.confidence >= LAYOUT_CONFIDENCE_MIN]
        if not high_conf:
            return

        # Add figure regions that don't already exist in doc.figures
        figure_regions = [r for r in high_conf if r.label == "figure"]
        if figure_regions and hasattr(doc, "figures"):
            existing_pages = {getattr(f, "page", None) for f in doc.figures}
            for region in figure_regions:
                if region.page not in existing_pages:
                    figure = self._make_figure(region)
                    if figure is not None:
                        doc.figures.append(figure)
                        existing_pages.add(region.page)

        # Persist all high-confidence regions in metrics
        if not hasattr(doc, "metrics"):
            doc.metrics = {}
        doc.metrics["visual_layout_regions"] = [
            {
                "bbox": r.bbox,
                "label": r.label,
                "confidence": r.confidence,
                "page": r.page,
                "source": r.source,
            }
            for r in high_conf
        ]

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _merge_tables(self, doc: Any, tables: List[StructuredTableResult]) -> None:
        if not tables:
            return

        for table_result in tables:
            matched = self._find_table_by_page(doc, table_result.page)
            if matched is not None:
                # Upgrade existing table with structured data
                matched.structured = {
                    "headers": table_result.headers,
                    "rows": table_result.rows,
                    "spans": table_result.spans,
                    "confidence": table_result.confidence,
                    "source": "visual_intelligence",
                }
            else:
                # Create a new table entry from the structured result
                new_table = self._make_table(table_result)
                if new_table is not None and hasattr(doc, "tables"):
                    doc.tables.append(new_table)

    # ------------------------------------------------------------------
    # OCR patches
    # ------------------------------------------------------------------

    def _merge_ocr(self, doc: Any, ocr_patches: Dict[int, List[OCRPatch]]) -> None:
        if not ocr_patches:
            return

        patches_applied = 0
        full_text: str = getattr(doc, "full_text", None) or ""

        for _page, patches in ocr_patches.items():
            for patch in patches:
                margin = patch.enhanced_confidence - patch.original_confidence
                if margin < OCR_CONFIDENCE_MARGIN:
                    continue
                if patch.original_text and patch.original_text in full_text:
                    full_text = full_text.replace(
                        patch.original_text, patch.enhanced_text, 1
                    )
                    patches_applied += 1

        if patches_applied:
            doc.full_text = full_text

        if not hasattr(doc, "metrics"):
            doc.metrics = {}
        doc.metrics["ocr_patches_applied"] = patches_applied

    # ------------------------------------------------------------------
    # Key-value pairs
    # ------------------------------------------------------------------

    def _merge_kv_pairs(self, doc: Any, kv_pairs: List[KVPair]) -> None:
        if not kv_pairs:
            return

        if not hasattr(doc, "metrics"):
            doc.metrics = {}
        doc.metrics["visual_kv_pairs"] = [
            {
                "key": kv.key,
                "value": kv.value,
                "confidence": kv.confidence,
                "page": kv.page,
                "source": kv.source,
            }
            for kv in kv_pairs
        ]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _tag_provenance(self, doc: Any, result: VisualEnrichmentResult) -> None:
        if not hasattr(doc, "metrics"):
            doc.metrics = {}

        doc.metrics["visual_intelligence_applied"] = True
        doc.metrics["visual_intelligence_models"] = result.models_used
        doc.metrics["visual_intelligence_time_ms"] = result.processing_time_ms

        if result.errors:
            doc.metrics["visual_intelligence_errors"] = result.errors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_table_by_page(doc: Any, page: int) -> Optional[Any]:
        """Find the first existing table on *page*."""
        for table in getattr(doc, "tables", []):
            if getattr(table, "page", None) == page:
                return table
        return None

    @staticmethod
    def _make_figure(region: VisualRegion) -> Optional[Any]:
        """Create a Figure from a visual region, if the model is available."""
        try:
            from src.api.pipeline_models import Figure
        except ImportError:
            logger.debug("pipeline_models.Figure not available; skipping figure creation")
            return None

        return Figure(
            page=region.page,
            caption=f"[Detected {region.label}]",
            ocr_method="visual_intelligence",
            ocr_confidence=region.confidence,
        )

    @staticmethod
    def _make_table(result: StructuredTableResult) -> Optional[Any]:
        """Create a Table from a StructuredTableResult, if the model is available."""
        try:
            from src.api.pipeline_models import Table
        except ImportError:
            logger.debug("pipeline_models.Table not available; skipping table creation")
            return None

        # Build a CSV representation
        buf = io.StringIO()
        writer = csv.writer(buf)
        if result.headers:
            writer.writerow(result.headers)
        for row in result.rows:
            writer.writerow(row)
        csv_text = buf.getvalue().strip()

        return Table(
            page=result.page,
            text=csv_text,
            csv=csv_text,
            structured={
                "headers": result.headers,
                "rows": result.rows,
                "spans": result.spans,
                "confidence": result.confidence,
                "source": "visual_intelligence",
            },
        )
