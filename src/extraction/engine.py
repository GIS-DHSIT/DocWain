"""Extraction engine — orchestrates three-model parallel extraction."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.structural import StructuralExtractor
from src.extraction.semantic import SemanticExtractor
from src.extraction.vision import VisionExtractor
from src.extraction.merger import ExtractionMerger
from src.extraction.models import ExtractionResult

logger = logging.getLogger(__name__)


class ExtractionEngine:
    """Orchestrates parallel extraction using three model pipelines."""

    def __init__(self, triton_url: str = None, ollama_host: str = None):
        self.structural = StructuralExtractor(triton_url=triton_url)
        self.semantic = SemanticExtractor(ollama_host=ollama_host)
        self.vision = VisionExtractor(ollama_host=ollama_host)
        self.merger = ExtractionMerger()

    def extract(self, document_id: str, subscription_id: str, profile_id: str,
                document_bytes: bytes, file_type: str,
                text_content: str = None) -> ExtractionResult:
        """Run three-model extraction in parallel, then merge results."""
        structural_result = {}
        semantic_result = {}
        vision_result = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.structural.extract, document_bytes, file_type
                ): "structural",
                executor.submit(
                    self.semantic.extract, text_content or "", file_type
                ): "semantic",
                executor.submit(
                    self.vision.extract, document_bytes, file_type
                ): "vision",
            }

            for future in as_completed(futures):
                pipeline_name = futures[future]
                try:
                    result = future.result(timeout=600)
                    if pipeline_name == "structural":
                        structural_result = result
                    elif pipeline_name == "semantic":
                        semantic_result = result
                    elif pipeline_name == "vision":
                        vision_result = result
                    logger.info(f"{pipeline_name} extraction completed for {document_id}")
                except Exception as e:
                    logger.error(f"{pipeline_name} extraction failed for {document_id}: {e}")

        merged = self.merger.merge(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            structural=structural_result,
            semantic=semantic_result,
            vision=vision_result,
        )

        logger.info(
            f"Extraction complete for {document_id}: "
            f"{len(merged.entities)} entities, "
            f"{len(merged.tables)} tables, "
            f"confidence={merged.metadata.get('extraction_confidence', 0):.2f}"
        )

        return merged
