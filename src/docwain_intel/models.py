from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    EXTRACTED = "EXTRACTED"
    NORMALIZED = "NORMALIZED"
    CHUNKED = "CHUNKED"
    READY_TO_EMBED = "READY_TO_EMBED"
    EMBEDDED = "EMBEDDED"


class DocumentManifest(BaseModel):
    subscription_id: str
    profile_id: str
    document_id: str

    filename: str
    mime_type: Optional[str] = None
    file_type: Optional[str] = None
    size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    checksum_sha256: str

    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    source_version: Optional[str] = None

    ingested_at: Optional[float] = None
    extracted_at: Optional[float] = None
    embedded_at: Optional[float] = None
    status: DocumentStatus = DocumentStatus.UPLOADED


class Block(BaseModel):
    block_id: str
    type: str
    text: Optional[str] = None
    key: Optional[str] = None
    value: Optional[str] = None
    table_ref: Optional[str] = None
    bbox: Optional[List[float]] = None
    page_number: Optional[int] = None
    reading_order: Optional[int] = None


class Page(BaseModel):
    page_number: int
    blocks: List[Block] = Field(default_factory=list)


class Section(BaseModel):
    section_path: List[str]
    content_refs: List[str]
    page_range: List[int]


class Table(BaseModel):
    table_id: str
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None


class Image(BaseModel):
    page_number: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None


class ExtractedDocumentJSON(BaseModel):
    document_id: str
    pages: List[Page] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    images: List[Image] = Field(default_factory=list)
    raw_text: Optional[str] = None


class Entity(BaseModel):
    entity_id: str
    label: str
    text: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    subject_id: str
    predicate: str
    object_value: Optional[str] = None
    object_id: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class EntityFactBundle(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)


class UnitType(str, enum.Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    KV_GROUP = "kv_group"
    FIGURE_CAPTION = "figure_caption"
    FRAGMENT = "fragment"


class SemanticUnit(BaseModel):
    unit_id: str
    unit_type: UnitType
    text: str
    page_start: int
    page_end: int
    heading_path: List[str] = Field(default_factory=list)
    parent_unit_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    table_headers: Optional[List[str]] = None
    table_rows: Optional[List[Dict[str, Any]]] = None
    kv_pairs: Optional[Dict[str, str]] = None
    raw_spans: List[Dict[str, Any]] = Field(default_factory=list)
    is_ocr: bool = False
    is_uncertain: bool = False


class EntitySpan(BaseModel):
    entity_id: str
    text: str
    normalized: str
    label: str
    unit_id: str
    char_start: int = 0
    char_end: int = 0
    confidence: float = 0.0
    source: str = "unknown"
    aliases: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class FactTriple(BaseModel):
    fact_id: str
    subject_id: str
    predicate: str
    object_id: Optional[str] = None
    object_value: Optional[str] = None
    unit_id: str
    raw_text: str
    confidence: float = 0.0
    extraction_method: str = "unknown"
    is_uncertain: bool = False


class ConflictRecord(BaseModel):
    fact_id_1: str
    fact_id_2: str
    conflict_type: str
    description: str


class StructuredDocument(BaseModel):
    document_id: str
    units: List[SemanticUnit] = Field(default_factory=list)
    unit_count: int = 0
    total_chars: int = 0


class ExtractionResult(BaseModel):
    document_id: str
    entities: List[EntitySpan] = Field(default_factory=list)
    facts: List[FactTriple] = Field(default_factory=list)
    tables_structured: List[Dict[str, Any]] = Field(default_factory=list)
    kv_pairs: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentFingerprint(BaseModel):
    entity_distribution: Dict[str, int] = Field(default_factory=dict)
    structure_profile: Dict[str, int] = Field(default_factory=dict)
    topic_vectors: List[List[float]] = Field(default_factory=list)
    numeric_density: float = 0.0
    entity_density: float = 0.0
    formality_score: float = 0.5
    structure_complexity: float = 0.0
    relational_density: float = 0.0
    auto_tags: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    is_valid: bool = True
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    quality_score: float = 1.0


__all__ = [
    "DocumentManifest",
    "DocumentStatus",
    "ExtractedDocumentJSON",
    "Block",
    "Page",
    "Section",
    "Table",
    "Image",
    "Entity",
    "Fact",
    "EntityFactBundle",
    "UnitType",
    "SemanticUnit",
    "EntitySpan",
    "FactTriple",
    "ConflictRecord",
    "StructuredDocument",
    "ExtractionResult",
    "DocumentFingerprint",
    "VerificationResult",
]
