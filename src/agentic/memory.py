from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AgentMemory:
    """Shared in-memory state passed across agents during a single request."""

    visited_documents: Set[str] = field(default_factory=set)
    visited_sections: Set[str] = field(default_factory=set)
    used_citations: Set[str] = field(default_factory=set)
    unresolved_gaps: List[str] = field(default_factory=list)
    retrieval_notes: List[str] = field(default_factory=list)

    def register_chunk(self, metadata: Dict[str, Any]) -> None:
        doc_id = metadata.get("document_id") or metadata.get("doc_id") or metadata.get("document_name")
        if doc_id:
            self.visited_documents.add(str(doc_id))
        section = metadata.get("section_title") or metadata.get("section")
        if section:
            self.visited_sections.add(str(section))
        logger.debug("register_chunk doc_id=%s, total_docs=%s, total_sections=%s",
                     doc_id, len(self.visited_documents), len(self.visited_sections))

    def note_gap(self, gap: str) -> None:
        cleaned = (gap or "").strip()
        if cleaned:
            self.unresolved_gaps.append(cleaned)
            logger.debug("note_gap total_gaps=%s", len(self.unresolved_gaps))

    def note_retrieval(self, note: str) -> None:
        cleaned = (note or "").strip()
        if cleaned:
            self.retrieval_notes.append(cleaned)
            logger.debug("note_retrieval total_notes=%s", len(self.retrieval_notes))
