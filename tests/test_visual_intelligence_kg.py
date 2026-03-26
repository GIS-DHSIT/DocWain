import pytest
from unittest.mock import MagicMock, patch
from src.visual_intelligence.datatypes import (
    VisualEnrichmentResult, VisualRegion, StructuredTableResult, KVPair,
)


def test_kg_enricher_builds_payload():
    from src.visual_intelligence.kg_enricher import VisualKGEnricher
    enricher = VisualKGEnricher()
    result = VisualEnrichmentResult(doc_id="doc-kg-1")
    result.regions = [
        VisualRegion(bbox=(10, 20, 300, 400), label="table", confidence=0.9, page=1),
        VisualRegion(bbox=(10, 420, 300, 500), label="figure", confidence=0.85, page=1),
    ]
    result.tables = [
        StructuredTableResult(page=1, bbox=(10, 20, 300, 400),
                             headers=["A", "B"], rows=[["1", "2"]], spans=[], confidence=0.88),
    ]
    result.kv_pairs = [
        KVPair(key="Invoice No", value="12345", confidence=0.91, page=1),
    ]
    payload = enricher.build_payload("doc-kg-1", "sub-1", "prof-1", result)
    assert payload["doc_id"] == "doc-kg-1"
    assert len(payload["nodes"]) >= 4
    assert len(payload["edges"]) >= 3


def test_kg_enricher_enqueue_fires():
    from src.visual_intelligence.kg_enricher import VisualKGEnricher
    enricher = VisualKGEnricher()
    result = VisualEnrichmentResult(doc_id="doc-kg-2")
    result.regions = [
        VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.9, page=1),
    ]
    with patch.object(enricher, "_enqueue") as mock_enqueue:
        enricher.enqueue_enrichment("doc-kg-2", "sub-1", "prof-1", result)
        mock_enqueue.assert_called_once()


def test_kg_enricher_empty_result_skips():
    from src.visual_intelligence.kg_enricher import VisualKGEnricher
    enricher = VisualKGEnricher()
    result = VisualEnrichmentResult(doc_id="doc-empty")
    with patch.object(enricher, "_enqueue") as mock_enqueue:
        enricher.enqueue_enrichment("doc-empty", "sub-1", "prof-1", result)
        mock_enqueue.assert_not_called()
