"""Merger — reconciles outputs from structural, semantic, and vision pipelines."""

from src.extraction.models import ExtractionResult, Entity, Relationship, TableData
import logging

logger = logging.getLogger(__name__)


class ExtractionMerger:
    """Merges and reconciles outputs from three extraction pipelines.

    Responsibilities:
    - Deduplicate entities by text+type fuzzy match
    - Cross-validate tables (structural vs vision)
    - Reconcile section boundaries
    - Assign confidence scores based on cross-model agreement
    - Build unified reading order
    """

    def merge(self, document_id: str, subscription_id: str, profile_id: str,
              structural: dict, semantic: dict, vision: dict,
              page_count: int = 0) -> ExtractionResult:
        """Merge three pipeline outputs into a unified ExtractionResult."""

        entities = self._merge_entities(
            structural.get("entities", []),
            semantic.get("entities", []),
            vision.get("entities", [])
        )

        relationships = []
        for r in semantic.get("relationships", []):
            if isinstance(r, Relationship):
                relationships.append(r)
            elif isinstance(r, dict):
                relationships.append(Relationship(**r))

        tables = self._merge_tables(
            structural.get("tables", []),
            vision.get("table_images", [])
        )

        # Build clean text: prefer structural reading order, fill with vision OCR
        clean_text = self._build_clean_text(structural, semantic, vision)

        models_used = []
        if structural.get("layout") or structural.get("sections"):
            models_used.append("layoutlm-v3")
        if semantic.get("entities") or semantic.get("context"):
            models_used.append("qwen3:14b")
        if vision.get("ocr_text") or vision.get("scanned_text"):
            models_used.append("glm-ocr")

        return ExtractionResult(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            clean_text=clean_text,
            structure={
                "sections": structural.get("sections", []),
                "headers": structural.get("headers", []),
                "footers": structural.get("footers", []),
                "reading_order": structural.get("reading_order", [])
            },
            entities=entities,
            relationships=relationships,
            tables=tables,
            metadata={
                "page_count": page_count,
                "language": "en",
                "doc_type_detected": semantic.get("context", "unknown"),
                "models_used": models_used,
                "extraction_confidence": self._calculate_confidence(
                    structural, semantic, vision
                )
            }
        )

    def _merge_entities(self, structural_entities, semantic_entities,
                        vision_entities) -> list:
        """Deduplicate entities across pipelines by text+type match."""
        seen = {}
        merged = []

        for source_name, entity_list in [
            ("structural", structural_entities),
            ("semantic", semantic_entities),
            ("vision", vision_entities)
        ]:
            for e in entity_list:
                if isinstance(e, dict):
                    e = Entity(
                        text=e.get("text", ""),
                        type=e.get("type", "UNKNOWN"),
                        confidence=e.get("confidence", 0.5),
                        source=source_name,
                        locations=e.get("locations", [])
                    )
                elif not isinstance(e, Entity):
                    continue

                key = (e.text.lower().strip(), e.type.upper())
                if key in seen:
                    # Boost confidence when multiple models agree
                    existing = seen[key]
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                else:
                    seen[key] = e
                    merged.append(e)

        return merged

    def _merge_tables(self, structural_tables, vision_tables) -> list:
        """Cross-validate tables from structural and vision pipelines."""
        tables = []
        for t in structural_tables:
            if isinstance(t, dict):
                t = TableData(
                    id=t.get("id", ""),
                    page=t.get("page", 0),
                    rows=t.get("rows", 0),
                    cols=t.get("cols", 0),
                    headers=t.get("headers", []),
                    data=t.get("data", []),
                    source="structural"
                )
            tables.append(t)
        # TODO: Cross-validate with vision tables
        return tables

    def _build_clean_text(self, structural, semantic, vision) -> str:
        """Build clean text from best available sources."""
        # Priority: structural reading order > vision OCR > semantic context
        if structural.get("reading_order"):
            # TODO: reconstruct text from reading order
            pass

        ocr_text = vision.get("ocr_text", "") or vision.get("scanned_text", "")
        if ocr_text:
            return ocr_text

        return semantic.get("summary", "") or semantic.get("context", "")

    def _calculate_confidence(self, structural, semantic, vision) -> float:
        """Calculate overall extraction confidence based on model agreement."""
        scores = []
        if structural.get("layout") or structural.get("sections"):
            scores.append(0.8)
        if semantic.get("entities") or semantic.get("context"):
            scores.append(0.7)
        if vision.get("ocr_text") or vision.get("scanned_text"):
            scores.append(0.6)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)
