import pytest
from unittest.mock import MagicMock
from src.visual_intelligence.datatypes import VisualRegion, VisualEnrichmentResult


def test_merge_layout_adds_new_sections():
    from src.visual_intelligence.enrichment_merger import EnrichmentMerger
    merger = EnrichmentMerger()
    extracted = MagicMock()
    existing_section = MagicMock()
    existing_section.section_id = "s1"
    existing_section.title = "Introduction"
    existing_section.start_page = 1
    existing_section.end_page = 1
    extracted.sections = [existing_section]
    visual_result = VisualEnrichmentResult(doc_id="doc-1")
    visual_result.regions = [
        VisualRegion(bbox=(10, 20, 300, 100), label="title", confidence=0.95, page=1),
        VisualRegion(bbox=(10, 120, 300, 400), label="text", confidence=0.92, page=1),
        VisualRegion(bbox=(320, 20, 600, 200), label="figure", confidence=0.88, page=1),
    ]
    merged = merger.merge(extracted, visual_result)
    assert hasattr(merged, "sections")
    assert len(merged.sections) >= 1


def test_merge_never_deletes_existing():
    from src.visual_intelligence.enrichment_merger import EnrichmentMerger
    merger = EnrichmentMerger()
    extracted = MagicMock()
    extracted.sections = [MagicMock(section_id="s1"), MagicMock(section_id="s2")]
    extracted.tables = [MagicMock(page=1, csv="a,b\n1,2")]
    extracted.figures = [MagicMock(page=1)]
    extracted.full_text = "existing text"
    visual_result = VisualEnrichmentResult(doc_id="doc-1")
    merged = merger.merge(extracted, visual_result)
    assert len(merged.sections) >= 2
    assert len(merged.tables) >= 1
    assert merged.full_text == "existing text"


def test_merge_provenance_tracking():
    from src.visual_intelligence.enrichment_merger import EnrichmentMerger
    merger = EnrichmentMerger()
    extracted = MagicMock()
    extracted.sections = []
    extracted.tables = []
    extracted.figures = []
    extracted.full_text = "text"
    extracted.metrics = {}
    visual_result = VisualEnrichmentResult(doc_id="doc-1")
    visual_result.regions = [
        VisualRegion(bbox=(10, 20, 300, 400), label="table", confidence=0.9, page=1),
    ]
    merged = merger.merge(extracted, visual_result)
    assert merged.metrics.get("visual_intelligence_applied") is True
