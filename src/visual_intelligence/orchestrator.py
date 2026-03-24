"""Visual Intelligence Orchestrator — coordinates the full visual enrichment pipeline.

Pipeline flow:
    score_document → filter pages → render → detect (DiT) → merge
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from src.visual_intelligence.datatypes import (
    Tier,
    VisualEnrichmentResult,
    VisualRegion,
    StructuredTableResult,
    OCRPatch,
    KVPair,
)
from src.visual_intelligence.page_renderer import PageImage

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="visual-intel")


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

def _is_enabled() -> bool:
    """Check whether the visual intelligence layer is enabled via config."""
    try:
        from src.api.config import Config
        value = Config.VisualIntelligence.ENABLED
        return str(value).lower() in {"1", "true", "yes", "on"}
    except (ImportError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class VisualIntelligenceOrchestrator:
    """Coordinates complexity scoring, page rendering, layout detection, and merging."""

    def __init__(self) -> None:
        from src.visual_intelligence.complexity_scorer import ComplexityScorer
        from src.visual_intelligence.page_renderer import PageRenderer
        from src.visual_intelligence.enrichment_merger import EnrichmentMerger

        self.scorer = ComplexityScorer()
        self.renderer = PageRenderer()
        self.merger = EnrichmentMerger()
        self._dit_detector: Optional[Any] = None
        self._table_detector: Optional[Any] = None
        self._trocr_enhancer: Optional[Any] = None
        self._layoutlmv3_extractor: Optional[Any] = None
        self._kg_enricher: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lazy DiT detector with setter for testing
    # ------------------------------------------------------------------

    @property
    def dit_detector(self) -> Any:
        """Lazily create the DiT layout detector on first access."""
        if self._dit_detector is None:
            from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
            self._dit_detector = DiTLayoutDetector()
        return self._dit_detector

    @dit_detector.setter
    def dit_detector(self, value: Any) -> None:
        self._dit_detector = value

    @property
    def table_detector(self) -> Any:
        if self._table_detector is None:
            from src.visual_intelligence.models.table_transformer import TableTransformerDetector
            self._table_detector = TableTransformerDetector()
        return self._table_detector

    @table_detector.setter
    def table_detector(self, value: Any) -> None:
        self._table_detector = value

    @property
    def trocr_enhancer(self) -> Any:
        if self._trocr_enhancer is None:
            from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
            self._trocr_enhancer = TrOCREnhancer()
        return self._trocr_enhancer

    @trocr_enhancer.setter
    def trocr_enhancer(self, value: Any) -> None:
        self._trocr_enhancer = value

    @property
    def layoutlmv3_extractor(self) -> Any:
        if self._layoutlmv3_extractor is None:
            from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
            self._layoutlmv3_extractor = LayoutLMv3Extractor()
        return self._layoutlmv3_extractor

    @layoutlmv3_extractor.setter
    def layoutlmv3_extractor(self, value: Any) -> None:
        self._layoutlmv3_extractor = value

    @property
    def kg_enricher(self) -> Any:
        if self._kg_enricher is None:
            from src.visual_intelligence.kg_enricher import VisualKGEnricher
            self._kg_enricher = VisualKGEnricher()
        return self._kg_enricher

    @kg_enricher.setter
    def kg_enricher(self, value: Any) -> None:
        self._kg_enricher = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich(
        self,
        doc_id: str,
        extracted_doc: Any,
        file_bytes: bytes,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Any:
        """Run the full visual intelligence pipeline on *extracted_doc*.

        Returns the (possibly enriched) document. On any failure the
        original document is returned unchanged.
        """
        if not _is_enabled():
            return extracted_doc

        try:
            t0 = time.perf_counter()
            loop = asyncio.get_event_loop()

            # 1. Score complexity per page
            complexities = self.scorer.score_document(extracted_doc)

            # 2. Filter pages that need processing
            needed = [c for c in complexities if c.tier != Tier.SKIP]
            if not needed:
                logger.debug("doc=%s — all pages SKIP tier, no visual processing needed", doc_id)
                return extracted_doc

            page_numbers = [c.page for c in needed]
            tier_map = {c.page: c.tier for c in needed}

            # 3. Render needed pages (blocking → executor)
            page_images: List[PageImage] = await loop.run_in_executor(
                _executor,
                lambda: self.renderer.render(file_bytes, page_numbers=page_numbers),
            )

            # 4. Process each page (blocking → executor)
            all_regions: List[VisualRegion] = []
            all_tables: List[StructuredTableResult] = []
            all_ocr_patches: dict = {}
            all_kv_pairs: List[KVPair] = []
            models_used: List[str] = []

            for page_img in page_images:
                tier = tier_map.get(page_img.page_number, Tier.LIGHT)
                page_result = await loop.run_in_executor(
                    _executor,
                    lambda pi=page_img, t=tier: self._process_page(pi, t),
                )
                all_regions.extend(page_result.get("regions", []))
                all_tables.extend(page_result.get("tables", []))
                if page_result.get("ocr_patches"):
                    page_num = page_img.page_number
                    all_ocr_patches.setdefault(page_num, []).extend(page_result["ocr_patches"])
                all_kv_pairs.extend(page_result.get("kv_pairs", []))

            if all_regions:
                models_used.append("dit")
            if all_tables:
                models_used.append("table_transformer")
            if all_ocr_patches:
                models_used.append("trocr")
            if all_kv_pairs:
                models_used.append("layoutlmv3")

            # 5. Build enrichment result
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = VisualEnrichmentResult(
                doc_id=doc_id,
                regions=all_regions,
                tables=all_tables,
                ocr_patches=all_ocr_patches,
                kv_pairs=all_kv_pairs,
                page_complexities=complexities,
                processing_time_ms=elapsed_ms,
                models_used=models_used,
            )

            # 6. Merge into extracted doc
            enriched = self.merger.merge(extracted_doc, result)

            # 7. Fire KG enrichment (async, non-blocking)
            if subscription_id and profile_id:
                try:
                    self.kg_enricher.enqueue_enrichment(
                        doc_id, subscription_id, profile_id, result,
                    )
                except Exception:
                    logger.debug("KG enrichment failed for doc=%s", doc_id, exc_info=True)

            logger.info(
                "doc=%s — visual intelligence completed in %.0fms, %d regions, %d tables across %d pages",
                doc_id, elapsed_ms, len(all_regions), len(all_tables), len(needed),
            )
            return enriched

        except Exception:
            logger.warning(
                "Visual intelligence failed for doc=%s — returning original document",
                doc_id,
                exc_info=True,
            )
            return extracted_doc

    # ------------------------------------------------------------------
    # Per-page processing
    # ------------------------------------------------------------------

    def _process_page(self, page_img: PageImage, tier: Tier) -> dict:
        """Process a single page image according to its tier.

        LIGHT+: DiT layout detection + Table Transformer
        FULL:   + TrOCR on low-confidence regions
        """
        regions: List[VisualRegion] = []
        tables: List[StructuredTableResult] = []
        ocr_patches: List[OCRPatch] = []
        kv_pairs: List[KVPair] = []

        if tier >= Tier.LIGHT:
            # DiT layout detection
            dit_regions = self.dit_detector.detect(page_img.image, page_img.page_number)
            regions.extend(dit_regions)

            # Table Transformer on table regions
            table_regions = [
                {"bbox": r.bbox, "confidence": r.confidence}
                for r in dit_regions if r.label == "table"
            ]
            if table_regions:
                extracted_tables = self.table_detector.extract(
                    page_img.image, page_img.page_number,
                    table_regions=table_regions,
                )
                tables.extend(extracted_tables)

        if tier >= Tier.FULL:
            # TrOCR on low-confidence text regions
            for region in regions:
                if region.label in ("text", "title"):
                    patch = self.trocr_enhancer.enhance_region(
                        image=page_img.image.crop(
                            (int(region.bbox[0]), int(region.bbox[1]),
                             int(region.bbox[2]), int(region.bbox[3]))
                        ),
                        page=page_img.page_number,
                        bbox=region.bbox,
                        original_text="",  # placeholder — actual text from extraction
                        original_confidence=0.5,  # default for visual-only regions
                    )
                    if patch:
                        ocr_patches.append(patch)

            # LayoutLMv3 for semantic KIE (key information extraction)
            try:
                extracted_kv = self.layoutlmv3_extractor.extract(
                    page_img.image, words=[], boxes=[],
                    page_number=page_img.page_number,
                )
                kv_pairs.extend(extracted_kv)
            except Exception:
                logger.debug("LayoutLMv3 failed on page %d", page_img.page_number, exc_info=True)

        return {
            "regions": regions,
            "tables": tables,
            "ocr_patches": ocr_patches,
            "kv_pairs": kv_pairs,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[VisualIntelligenceOrchestrator] = None


def get_visual_orchestrator() -> VisualIntelligenceOrchestrator:
    """Return the singleton orchestrator instance."""
    global _instance
    if _instance is None:
        _instance = VisualIntelligenceOrchestrator()
    return _instance
