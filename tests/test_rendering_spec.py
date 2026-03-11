"""Tests for the rendering spec generator."""

import pytest

from src.docwain_intel.query_analyzer import QueryGeometry
from src.docwain_intel.evidence_organizer import (
    EvidenceGap,
    EvidenceGroup,
    OrganizedEvidence,
    ProvenanceRecord,
)
from src.docwain_intel.rendering_spec import RenderingSpec, generate_spec


# ---------------------------------------------------------------------------
# Helpers to build test fixtures concisely
# ---------------------------------------------------------------------------

def _geometry(**overrides) -> QueryGeometry:
    defaults = dict(
        query="test query",
        intent_type="narrative",
        expected_entity_count=0,
        granularity=0.5,
        temporal_ordering=False,
        is_comparison=False,
        is_aggregation=False,
        focus_type="process_centric",
        question_word=None,
        requested_attributes=[],
    )
    defaults.update(overrides)
    return QueryGeometry(**defaults)


def _group(entity_id: str, n_facts: int = 0, n_chunks: int = 1, fields: list | None = None) -> EvidenceGroup:
    facts = []
    for i in range(n_facts):
        field_name = fields[i] if fields and i < len(fields) else f"field_{i}"
        facts.append({"predicate": field_name, "value": f"val_{i}", "confidence": 0.9})
    chunks = [{"text": f"chunk_{i}", "score": 0.8} for i in range(n_chunks)]
    return EvidenceGroup(
        entity_id=entity_id,
        entity_text=f"Entity {entity_id}",
        facts=facts,
        chunks=chunks,
        relevance_score=0.8,
    )


def _evidence(
    groups: list[EvidenceGroup] | None = None,
    ungrouped: int = 0,
    gaps: list[EvidenceGap] | None = None,
    total_facts: int | None = None,
    total_chunks: int | None = None,
) -> OrganizedEvidence:
    groups = groups or []
    ungrouped_chunks = [{"text": f"ug_{i}", "score": 0.5} for i in range(ungrouped)]
    computed_facts = sum(len(g.facts) for g in groups) if total_facts is None else total_facts
    computed_chunks = (sum(len(g.chunks) for g in groups) + ungrouped) if total_chunks is None else total_chunks
    return OrganizedEvidence(
        entity_groups=groups,
        ungrouped_chunks=ungrouped_chunks,
        gaps=gaps or [],
        provenance=[ProvenanceRecord(source_document="doc.pdf")],
        total_facts=computed_facts,
        total_chunks=computed_chunks,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleEntityCard:
    """Single entity with 4+ facts should yield 'card' layout."""

    def test_card_layout(self):
        geo = _geometry(focus_type="entity_centric")
        grp = _group("e1", n_facts=5, fields=["name", "email", "phone", "role", "dept"])
        ev = _evidence(groups=[grp])
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "card"
        assert spec.use_headers is True
        assert spec.use_bold_values is True


class TestTableLayout:
    """3+ entities with 2+ shared fields should yield 'table' layout."""

    def test_table_layout(self):
        shared = ["name", "email", "dept"]
        groups = [_group(f"e{i}", n_facts=3, fields=shared) for i in range(4)]
        ev = _evidence(groups=groups)
        geo = _geometry()
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "table"
        assert spec.use_table is True


class TestComparisonLayout:
    """Comparison geometry with 2+ entities should yield 'comparison' layout."""

    def test_comparison(self):
        groups = [_group("a", n_facts=3), _group("b", n_facts=3)]
        ev = _evidence(groups=groups)
        geo = _geometry(is_comparison=True)
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "comparison"
        assert spec.use_table is True
        assert spec.grouping_strategy == "by_entity"


class TestTimelineLayout:
    """Temporal ordering should yield 'timeline' with chronological grouping."""

    def test_timeline(self):
        geo = _geometry(temporal_ordering=True)
        ev = _evidence(groups=[_group("e1", n_facts=2)], ungrouped=3)
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "timeline"
        assert spec.grouping_strategy == "chronological"
        assert spec.use_headers is True


class TestSingleValueLayout:
    """High granularity + single fact → single_value layout."""

    def test_single_value(self):
        grp = _group("e1", n_facts=1, fields=["email"])
        ev = _evidence(groups=[grp])
        geo = _geometry(granularity=0.9)
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "single_value"
        assert spec.include_provenance is False


class TestNarrativeForProse:
    """Prose-only evidence (0 facts, many chunks) → narrative."""

    def test_prose_narrative(self):
        ev = _evidence(ungrouped=10, total_facts=0, total_chunks=10)
        geo = _geometry()
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "narrative"


class TestAggregationSummary:
    """Aggregation flag → summary layout."""

    def test_summary(self):
        geo = _geometry(is_aggregation=True)
        ev = _evidence(groups=[_group("e1", n_facts=5)])
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "summary"
        assert spec.use_headers is True


class TestEnumerativeList:
    """Enumerative intent → list layout."""

    def test_list(self):
        geo = _geometry(intent_type="enumerative")
        ev = _evidence(groups=[_group("e1", n_facts=3)])
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "list"
        assert spec.max_items == 20


class TestDetailLevelTiers:
    """Detail level should follow granularity through all 4 tiers."""

    @pytest.mark.parametrize("granularity,expected", [
        (0.95, "minimal"),
        (0.7, "concise"),
        (0.5, "standard"),
        (0.1, "comprehensive"),
    ])
    def test_detail_tiers(self, granularity, expected):
        geo = _geometry(granularity=granularity)
        ev = _evidence(ungrouped=3, total_facts=0, total_chunks=3)
        spec = generate_spec(geo, ev)

        assert spec.detail_level == expected


class TestFieldOrdering:
    """field_ordering should respect requested_attributes first."""

    def test_requested_attributes_first(self):
        geo = _geometry(requested_attributes=["salary", "title"])
        shared = ["name", "title", "salary"]
        groups = [_group(f"e{i}", n_facts=3, fields=shared) for i in range(2)]
        ev = _evidence(groups=groups)
        spec = generate_spec(geo, ev)

        assert spec.field_ordering[0] == "salary"
        assert spec.field_ordering[1] == "title"
        # "name" should also appear somewhere
        assert "name" in spec.field_ordering


class TestGapsIncluded:
    """Gaps present in evidence → include_gaps=True."""

    def test_gaps_flag(self):
        gaps = [EvidenceGap(field_name="revenue", description="No revenue data found")]
        ev = _evidence(gaps=gaps, ungrouped=2, total_facts=0, total_chunks=2)
        geo = _geometry()
        spec = generate_spec(geo, ev)

        assert spec.include_gaps is True

    def test_no_gaps_flag(self):
        ev = _evidence(ungrouped=2, total_facts=0, total_chunks=2)
        geo = _geometry()
        spec = generate_spec(geo, ev)

        assert spec.include_gaps is False


class TestEmptyEvidence:
    """Empty evidence → sensible defaults: narrative + comprehensive."""

    def test_empty(self):
        ev = OrganizedEvidence()
        geo = _geometry(granularity=0.1)
        spec = generate_spec(geo, ev)

        assert spec.layout_mode == "narrative"
        assert spec.detail_level == "comprehensive"
        assert spec.include_gaps is False
        assert spec.include_provenance is True


class TestComparisonOverridesTable:
    """Comparison with shared fields should choose comparison over table."""

    def test_comparison_priority(self):
        shared = ["name", "email", "dept"]
        groups = [_group(f"e{i}", n_facts=3, fields=shared) for i in range(4)]
        ev = _evidence(groups=groups)
        geo = _geometry(is_comparison=True)
        spec = generate_spec(geo, ev)

        # Comparison takes priority over table even though 4 entities with shared fields.
        assert spec.layout_mode == "comparison"


class TestEntityCentricGrouping:
    """Entity-centric focus → by_entity grouping when not overridden."""

    def test_entity_grouping(self):
        geo = _geometry(focus_type="entity_centric")
        ev = _evidence(ungrouped=5, total_facts=0, total_chunks=5)
        spec = generate_spec(geo, ev)

        assert spec.grouping_strategy == "by_entity"
