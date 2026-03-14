"""
Semantic Payload Builder - Constructs intelligent Qdrant payloads with rich metadata.

Creates payloads that maintain full document context, semantic relationships, and
traceability back to source documents.
"""

import json
from src.utils.logging_utils import get_logger
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = get_logger(__name__)

@dataclass
class SemanticPayload:
    """Rich metadata payload for Qdrant."""
    # Core identifiers
    chunk_id: str
    document_id: str
    document_type: str  # From classifier

    # Content metadata
    section_id: str
    section_type: str  # e.g., "skills", "education", "contact"
    section_title: Optional[str]
    chunk_sequence: int  # Order in section
    total_chunks_in_section: int

    # Document context (enables type-aware retrieval)
    document_classification_confidence: float
    document_domain: Optional[str]

    # Semantic markers
    content_role: str  # "entity_list", "narrative", "structural", "tabular"
    semantic_keywords: List[str]  # Key entities/concepts in chunk
    semantic_type: str  # "name", "skill", "education", "experience", etc.

    # Extraction quality
    extraction_confidence: float
    section_confidence: float

    # Linkage to source
    pickle_reference: str  # Path/ID to pickle file
    extraction_version: str
    extraction_timestamp: str

    # Layout information
    page_number: int
    approximate_position: str  # "header", "body", "footer"

    # Chunk content hints
    chunk_length: int
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant payload."""
        return {
            # Required fields
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_type": self.document_type,

            # Section hierarchy
            "section_id": self.section_id,
            "section_type": self.section_type,
            "section_title": self.section_title,
            "chunk_sequence": self.chunk_sequence,
            "total_chunks_in_section": self.total_chunks_in_section,

            # Document context
            "document_classification_confidence": self.document_classification_confidence,
            "document_domain": self.document_domain,

            # Semantic markers
            "content_role": self.content_role,
            "semantic_keywords": self.semantic_keywords,
            "semantic_type": self.semantic_type,

            # Quality scores
            "extraction_confidence": self.extraction_confidence,
            "section_confidence": self.section_confidence,
            "combined_confidence": (self.extraction_confidence + self.section_confidence) / 2,

            # Lineage tracking
            "pickle_reference": self.pickle_reference,
            "extraction_version": self.extraction_version,
            "extraction_timestamp": self.extraction_timestamp,

            # Layout
            "page_number": self.page_number,
            "approximate_position": self.approximate_position,

            # Content hints
            "chunk_length": self.chunk_length,
            "language": self.language,
        }

class SemanticPayloadBuilder:
    """Build intelligent Qdrant payloads from structured documents."""

    def __init__(self):
        """Initialize builder."""
        self.content_role_mapping = {
            "contact": "entity_list",
            "skills": "entity_list",
            "certifications": "entity_list",
            "education": "mixed",
            "experience": "narrative",
            "summary": "narrative",
            "line_items": "tabular",
            "medications": "entity_list",
            "findings": "narrative",
            "clause": "structural",
        }

        self.semantic_type_mapping = {
            "contact": "contact_information",
            "skills": "technical_skills",
            "certifications": "certifications",
            "education": "education",
            "experience": "work_experience",
            "summary": "professional_summary",
            "line_items": "invoice_items",
            "medications": "medical_prescriptions",
            "findings": "clinical_findings",
            "clause": "legal_clause",
        }

    def build_payload(
        self,
        chunk: Dict[str, Any],
        structured_document: Dict[str, Any],
        section_info: Dict[str, Any],
        chunk_index: int,
        total_chunks_in_section: int,
        pickle_reference: str,
        extraction_timestamp: str
    ) -> SemanticPayload:
        """
        Build semantic payload for a chunk.

        Args:
            chunk: Chunk metadata and content info
            structured_document: Complete structured extraction
            section_info: Section-specific information
            chunk_index: Position of chunk in section
            total_chunks_in_section: Total chunks in this section
            pickle_reference: Reference to pickle file/location
            extraction_timestamp: When extraction occurred

        Returns:
            SemanticPayload ready for Qdrant
        """

        document_id = structured_document.get("document_id")
        document_type = structured_document.get("document_type")
        section_type = section_info.get("section_type")

        # Extract semantic keywords from chunk content
        content = chunk.get("text", "")
        semantic_keywords = self._extract_semantic_keywords(content, section_type)

        # Determine content role and semantic type
        content_role = self.content_role_mapping.get(section_type, "narrative")
        semantic_type = self.semantic_type_mapping.get(section_type, "content")

        # Calculate confidence scores
        section_confidence = section_info.get("confidence", 0.8)
        doc_classification = structured_document.get("document_classification", {})
        extraction_confidence = doc_classification.get("confidence", 0.8)

        # Infer approximate position in document
        approximate_position = self._infer_position(chunk_index, total_chunks_in_section)

        payload = SemanticPayload(
            chunk_id=chunk.get("chunk_id", f"{document_id}_chunk_{chunk_index}"),
            document_id=document_id,
            document_type=document_type,

            section_id=section_info.get("section_id", "unknown"),
            section_type=section_type,
            section_title=section_info.get("title"),
            chunk_sequence=chunk_index,
            total_chunks_in_section=total_chunks_in_section,

            document_classification_confidence=extraction_confidence,
            document_domain=doc_classification.get("domain"),

            content_role=content_role,
            semantic_keywords=semantic_keywords,
            semantic_type=semantic_type,

            extraction_confidence=extraction_confidence,
            section_confidence=section_confidence,

            pickle_reference=pickle_reference,
            extraction_version=structured_document.get("extraction_version", "1.0"),
            extraction_timestamp=extraction_timestamp,

            page_number=chunk.get("page", 1),
            approximate_position=approximate_position,

            chunk_length=len(content),
        )

        return payload

    def _extract_semantic_keywords(self, content: str, section_type: str) -> List[str]:
        """Extract key semantic terms from chunk content."""
        keywords = []

        # Split content into candidate terms
        terms = content.split()

        # Filter based on section type
        if section_type == "skills":
            # Capitalize words are likely skill names
            keywords = [t for t in terms if t and (t[0].isupper() or not t[0].isalpha())][:15]

        elif section_type in ["education", "certifications"]:
            # Look for degree names, institutions, companies
            keywords = [t for t in terms if len(t) > 4 and t[0].isupper()][:12]

        elif section_type == "experience":
            # Extract job titles and company names
            lines = content.split("\n")
            for line in lines[:3]:  # First few lines usually have title/company
                words = line.split()
                keywords.extend([w for w in words if len(w) > 4])

        elif section_type == "contact":
            # Extract email addresses, phone numbers
            import re
            emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", content)
            phones = re.findall(r"\+?\d[\d\s\-()]{6,}", content)
            keywords = emails + phones

        else:
            # Generic: extract capitalized words
            keywords = [t for t in terms if t and t[0].isupper()][:10]

        return list(set(keywords))[:15]  # Return unique terms, max 15

    def _infer_position(self, index: int, total: int) -> str:
        """Infer approximate position in document."""
        if total <= 3:
            if index == 0:
                return "header"
            elif index == total - 1:
                return "footer"
            else:
                return "body"

        position_ratio = index / max(1, total - 1)

        if position_ratio < 0.2:
            return "header"
        elif position_ratio > 0.8:
            return "footer"
        else:
            return "body"

    def build_batched_payloads(
        self,
        chunks: List[Dict[str, Any]],
        structured_document: Dict[str, Any],
        pickle_reference: str,
        extraction_timestamp: str
    ) -> List[Dict[str, Any]]:
        """
        Build payloads for multiple chunks from same document.

        Args:
            chunks: List of chunks (grouped by section ideally)
            structured_document: Complete structured extraction
            pickle_reference: Reference to pickle
            extraction_timestamp: Extraction time

        Returns:
            List of Qdrant-ready payloads
        """

        payloads = []
        sections = structured_document.get("sections", [])
        section_map = {s["section_id"]: s for s in sections}

        # Group chunks by section
        chunks_by_section = {}
        for chunk in chunks:
            section_id = chunk.get("section_id", "unknown")
            if section_id not in chunks_by_section:
                chunks_by_section[section_id] = []
            chunks_by_section[section_id].append(chunk)

        # Build payloads per section
        for section_id, section_chunks in chunks_by_section.items():
            section_info = section_map.get(section_id, {"section_type": "content", "confidence": 0.7})

            for index, chunk in enumerate(section_chunks):
                payload = self.build_payload(
                    chunk=chunk,
                    structured_document=structured_document,
                    section_info=section_info,
                    chunk_index=index,
                    total_chunks_in_section=len(section_chunks),
                    pickle_reference=pickle_reference,
                    extraction_timestamp=extraction_timestamp
                )
                payloads.append(payload.to_dict())

        return payloads

# Singleton instance
_payload_builder = None

def get_semantic_payload_builder() -> SemanticPayloadBuilder:
    """Get or create payload builder singleton."""
    global _payload_builder
    if _payload_builder is None:
        _payload_builder = SemanticPayloadBuilder()
    return _payload_builder

