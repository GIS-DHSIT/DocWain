"""Data models for the Visual Intelligence Layer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class Tier(IntEnum):
    """Processing tier for a document page."""

    SKIP = 0
    LIGHT = 1
    FULL = 2


@dataclass
class VisualRegion:
    """A detected visual region on a page (table, figure, heading, etc.)."""

    bbox: Tuple[float, float, float, float]
    label: str
    confidence: float
    page: int
    source: str = "visual_intelligence"


@dataclass
class StructuredTableResult:
    """Extracted table with headers, rows, and optional cell spans."""

    page: int
    bbox: Tuple[float, float, float, float]
    headers: List[str]
    rows: List[List[str]]
    spans: List[Dict]
    confidence: float


@dataclass
class OCRPatch:
    """A region where OCR was re-run to improve text quality."""

    page: int
    bbox: Tuple[float, float, float, float]
    original_text: str
    enhanced_text: str
    original_confidence: float
    enhanced_confidence: float
    method: str


@dataclass
class KVPair:
    """An extracted key-value pair from a form or document."""

    key: str
    value: str
    confidence: float
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = "visual_intelligence"


@dataclass
class PageComplexity:
    """Complexity assessment for a single page, determining processing tier."""

    page: int
    tier: Tier
    ocr_confidence: float
    image_ratio: float
    has_tables: bool
    has_forms: bool
    signals: Dict = field(default_factory=dict)


@dataclass
class VisualEnrichmentResult:
    """Aggregated result from the visual intelligence pipeline for a document."""

    doc_id: str
    regions: List[VisualRegion] = field(default_factory=list)
    tables: List[StructuredTableResult] = field(default_factory=list)
    ocr_patches: Dict[int, List[OCRPatch]] = field(default_factory=dict)
    kv_pairs: List[KVPair] = field(default_factory=list)
    page_complexities: List[PageComplexity] = field(default_factory=list)
    processing_time_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize the result to a plain dictionary."""
        return asdict(self)
