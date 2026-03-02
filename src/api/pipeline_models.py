from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    start_page: int
    end_page: int
    text: str


@dataclass
class Table:
    page: int
    text: str
    csv: Optional[str] = None
    structured: Optional[Any] = None  # StructuredTable when available


@dataclass
class Figure:
    page: int
    caption: str
    ocr_method: Optional[str] = None
    ocr_confidence: Optional[float] = None
    is_diagram: bool = False
    diagram_type: Optional[str] = None
    diagram_structure: Optional[Dict[str, Any]] = None


@dataclass
class ChunkCandidate:
    text: str
    page: Optional[int]
    section_title: str
    section_id: Optional[str]
    chunk_type: str = "text"
    table_meta: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedDocument:
    full_text: str
    sections: List[Section]
    tables: List[Table]
    figures: List[Figure]
    chunk_candidates: List[ChunkCandidate]
    doc_type: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    canonical_json: Dict[str, Any] = field(default_factory=dict)
    doc_quality: Optional[str] = None


@dataclass
class ChunkRecord:
    chunk_id: str
    dense_vector: List[float]
    sparse_vector: Optional[Any]
    payload: Dict[str, Any]
