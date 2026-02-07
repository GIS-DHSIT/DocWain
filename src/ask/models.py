from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Plan:
    intent: str
    scope: Dict[str, Any]
    query_rewrites: List[str]
    entity_hints: List[str]
    expected_answer_shape: str
    query: str = ""


@dataclass
class EvidenceChunk:
    text: str
    score: float
    metadata: Dict[str, Any]
    file_name: str
    document_id: str
    section_id: str
    page: Optional[int]
    chunk_kind: str
    snippet: str
    snippet_sha: str


@dataclass
class EvidenceQuality:
    quality: str
    reasons: List[str]
    stats: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Plan", "EvidenceChunk", "EvidenceQuality"]
