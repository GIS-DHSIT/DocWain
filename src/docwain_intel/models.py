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
]
