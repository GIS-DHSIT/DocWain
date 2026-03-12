"""Rendering spec generator — algorithmically computes output format from data shape + query geometry.

No fixed templates: the spec is derived fresh for every query by composing three
axes of analysis (data shape, query geometry, structural inference).
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
from collections import Counter
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .evidence_organizer import OrganizedEvidence, EvidenceGroup
from .query_analyzer import QueryGeometry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class RenderingSpec(BaseModel):
    layout_mode: str = "narrative"  # single_value, card, table, narrative, list, timeline, comparison, summary
    field_ordering: List[str] = Field(default_factory=list)
    grouping_strategy: str = "flat"  # by_entity, by_attribute, by_document, chronological, flat
    detail_level: str = "standard"  # minimal, concise, standard, comprehensive
    use_headers: bool = False
    use_bold_values: bool = False
    use_table: bool = False
    max_items: Optional[int] = None
    include_provenance: bool = True
    include_gaps: bool = False

# ---------------------------------------------------------------------------
# Axis 1 — Data Shape Analysis
# ---------------------------------------------------------------------------

class _DataShape:
    """Intermediate representation of evidence shape properties."""

    __slots__ = (
        "n_entity_groups", "total_facts", "total_chunks",
        "fact_to_chunk_ratio", "shared_field_count",
        "is_predominantly_prose", "field_frequency",
    )

    def __init__(self, evidence: OrganizedEvidence) -> None:
        self.n_entity_groups: int = len(evidence.entity_groups)
        self.total_facts: int = evidence.total_facts
        self.total_chunks: int = evidence.total_chunks

        # Fact-to-chunk ratio: high = structured KV data, low = prose-heavy.
        if self.total_chunks > 0:
            self.fact_to_chunk_ratio: float = self.total_facts / self.total_chunks
        else:
            self.fact_to_chunk_ratio = 0.0

        # Shared fields across entity groups.
        self.field_frequency: Counter = Counter()
        field_sets: List[Set[str]] = []
        for grp in evidence.entity_groups:
            fields = _extract_field_names(grp)
            field_sets.append(fields)
            self.field_frequency.update(fields)

        if len(field_sets) >= 2:
            common = field_sets[0]
            for fs in field_sets[1:]:
                common = common & fs
            self.shared_field_count: int = len(common)
        else:
            self.shared_field_count = 0

        # Predominantly prose: zero facts but chunks exist.
        self.is_predominantly_prose: bool = (
            self.total_facts == 0 and self.total_chunks > 0
        )

def _extract_field_names(group: EvidenceGroup) -> Set[str]:
    """Extract the set of field/predicate names from an entity group's facts."""
    names: Set[str] = set()
    for fact in group.facts:
        pred = fact.get("predicate") or fact.get("field") or fact.get("key")
        if pred:
            names.add(str(pred).strip().lower())
    return names

# ---------------------------------------------------------------------------
# Axis 2 — Query Geometry helpers
# ---------------------------------------------------------------------------

def _detail_from_granularity(granularity: float) -> str:
    """Map granularity (0.0=detailed .. 1.0=concise) to a detail level label."""
    if granularity > 0.8:
        return "minimal"
    if granularity > 0.6:
        return "concise"
    if granularity > 0.3:
        return "standard"
    return "comprehensive"

def _grouping_from_geometry(geometry: QueryGeometry) -> str:
    """Derive an initial grouping strategy from query geometry."""
    if geometry.temporal_ordering:
        return "chronological"
    if geometry.focus_type == "entity_centric":
        return "by_entity"
    if geometry.focus_type == "attribute_centric":
        return "by_attribute"
    return "flat"

# ---------------------------------------------------------------------------
# Axis 3 — Structural Inference (combines data shape + geometry)
# ---------------------------------------------------------------------------

def _infer_layout(geometry: QueryGeometry, shape: _DataShape) -> str:
    """Derive the best layout mode by composing data shape and query geometry signals."""

    # Comparison takes priority when geometry says comparison AND 2+ entities.
    if geometry.is_comparison and shape.n_entity_groups >= 2:
        return "comparison"

    # Aggregation → summary.
    if geometry.is_aggregation:
        return "summary"

    # Temporal ordering → timeline.
    if geometry.temporal_ordering:
        return "timeline"

    # Enumerative intent → list.
    if geometry.intent_type == "enumerative":
        return "list"

    # 3+ entity groups with 2+ shared fields → table.
    if shape.n_entity_groups >= 3 and shape.shared_field_count >= 2:
        return "table"

    # Single entity with 4+ facts → card.
    if shape.n_entity_groups == 1 and shape.total_facts >= 4:
        return "card"

    # Single fact + high granularity → single_value.
    if shape.total_facts == 1 and geometry.granularity > 0.6:
        return "single_value"

    # Predominantly prose → narrative.
    if shape.is_predominantly_prose:
        return "narrative"

    # Fallback: narrative.
    return "narrative"

# ---------------------------------------------------------------------------
# Field ordering
# ---------------------------------------------------------------------------

def _derive_field_ordering(
    geometry: QueryGeometry,
    shape: _DataShape,
) -> List[str]:
    """Compute field ordering from requested attributes, then by frequency."""
    ordering: List[str] = []
    seen: Set[str] = set()

    # Requested attributes first (preserve user order).
    for attr in geometry.requested_attributes:
        key = attr.strip().lower()
        if key and key not in seen:
            ordering.append(key)
            seen.add(key)

    # Then by frequency across entities (most common first).
    for field, _count in shape.field_frequency.most_common():
        if field not in seen:
            ordering.append(field)
            seen.add(field)

    return ordering

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_spec(
    geometry: QueryGeometry,
    evidence: OrganizedEvidence,
) -> RenderingSpec:
    """Algorithmically derive a RenderingSpec from query geometry and evidence shape.

    Three axes compose to produce the spec:
      1. Data Shape — entity count, fact density, shared fields, prose vs structured.
      2. Query Geometry — granularity, focus type, comparison/aggregation flags.
      3. Structural Inference — combining both to pick layout, grouping, and detail.
    """
    shape = _DataShape(evidence)

    # --- Axis 2: geometry-driven defaults ---
    detail_level = _detail_from_granularity(geometry.granularity)
    grouping_strategy = _grouping_from_geometry(geometry)

    # --- Axis 3: structural inference ---
    layout_mode = _infer_layout(geometry, shape)

    # --- Post-inference adjustments ---

    # Comparison layout forces table and by_entity grouping.
    use_table = layout_mode in ("table", "comparison")
    if layout_mode == "comparison":
        grouping_strategy = "by_entity"

    # Timeline forces chronological grouping (even if geometry didn't set it).
    if layout_mode == "timeline":
        grouping_strategy = "chronological"

    # Headers for multi-entity or structured layouts.
    use_headers = layout_mode in ("card", "table", "comparison", "timeline", "summary")

    # Bold values for card and table-like layouts.
    use_bold_values = layout_mode in ("card", "table", "comparison")

    # Max items: cap list and table layouts.
    max_items: Optional[int] = None
    if layout_mode == "list":
        max_items = 20
    elif layout_mode == "table" and shape.n_entity_groups > 10:
        max_items = 10

    # Provenance: always unless single_value (too small to cite).
    include_provenance = layout_mode != "single_value"

    # Gaps: include when there are actual gaps.
    include_gaps = len(evidence.gaps) > 0

    # Field ordering.
    field_ordering = _derive_field_ordering(geometry, shape)

    return RenderingSpec(
        layout_mode=layout_mode,
        field_ordering=field_ordering,
        grouping_strategy=grouping_strategy,
        detail_level=detail_level,
        use_headers=use_headers,
        use_bold_values=use_bold_values,
        use_table=use_table,
        max_items=max_items,
        include_provenance=include_provenance,
        include_gaps=include_gaps,
    )
