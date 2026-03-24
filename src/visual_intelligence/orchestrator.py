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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich(
        self,
        doc_id: str,
        extracted_doc: Any,
        file_bytes: bytes,
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
            for page_img in page_images:
                tier = tier_map.get(page_img.page_number, Tier.LIGHT)
                regions = await loop.run_in_executor(
                    _executor,
                    lambda pi=page_img, t=tier: self._process_page(pi, t),
                )
                all_regions.extend(regions)

            # 5. Build enrichment result
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = VisualEnrichmentResult(
                doc_id=doc_id,
                regions=all_regions,
                page_complexities=complexities,
                processing_time_ms=elapsed_ms,
                models_used=["dit"] if all_regions else [],
            )

            # 6. Merge into extracted doc
            enriched = self.merger.merge(extracted_doc, result)

            logger.info(
                "doc=%s — visual intelligence completed in %.0fms, %d regions detected across %d pages",
                doc_id, elapsed_ms, len(all_regions), len(needed),
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

    def _process_page(self, page_img: PageImage, tier: Tier) -> List[VisualRegion]:
        """Process a single page image according to its tier.

        LIGHT and above: run DiT layout detection.
        """
        regions: List[VisualRegion] = []

        if tier >= Tier.LIGHT:
            dit_regions = self.dit_detector.detect(page_img.image, page_img.page_number)
            regions.extend(dit_regions)

        return regions


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
