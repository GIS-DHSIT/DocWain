"""
Structured Extraction Module - Extracts documents into standardized JSON during ingestion.

This module processes documents during extraction phase (NOT embedding phase) to produce
structured JSON that serves as the source-of-truth for all downstream operations.
"""

import json
from src.utils.logging_utils import get_logger
from typing import Dict, Any, List, Optional
from dataclasses import asdict, dataclass
from datetime import datetime

from src.embedding.document_classifier import DocumentType, DocumentClassification, get_document_classifier

logger = get_logger(__name__)

@dataclass
class ExtractedSection:
    """A semantic section from a document."""
    section_id: str
    section_type: str  # e.g., "skills", "education", "experience", "contact"
    title: Optional[str]
    content: str
    start_page: int
    end_page: int
    confidence: float
    key_items: List[str] = None  # Extracted key items (skills, education, etc.)

    def __post_init__(self):
        if self.key_items is None:
            self.key_items = []

@dataclass
class StructuredDocument:
    """Complete structured representation of an extracted document."""
    document_id: str
    original_filename: str
    extraction_timestamp: str
    document_type: str  # DocumentType.value
    document_classification: Dict[str, Any]  # Full classification result

    # Raw content
    raw_text: str
    total_pages: int

    # Semantic sections
    sections: List[Dict[str, Any]]  # List of ExtractedSection as dicts

    # Document-level metadata
    metadata: Dict[str, Any]

    # Extraction quality metrics
    extraction_quality_score: float
    extraction_notes: List[str]

    # Source references for traceability
    extraction_version: str = "1.0"
    extraction_pipeline: str = "structured_extraction_v1"

class StructuredExtractionEngine:
    """Extract documents into structured JSON format."""

    def __init__(self):
        """Initialize extraction engine."""
        self.classifier = get_document_classifier()
        self.extraction_timestamp = datetime.now().isoformat()

    def extract_document(
        self,
        document_id: str,
        text: str,
        filename: str,
        metadata: Optional[Dict] = None,
        page_count: int = 1
    ) -> StructuredDocument:
        """
        Extract a document into structured JSON format.

        Args:
            document_id: Unique document identifier
            text: Raw document text
            filename: Original filename
            metadata: Additional metadata
            page_count: Total pages in document

        Returns:
            StructuredDocument with all extracted data
        """

        logger.info(f"Extracting document {document_id} - {filename}")

        try:
            # Step 1: Classify document
            classification = self.classifier.classify(text, {"filename": filename, **(metadata or {})})

            # Step 2: Extract semantic sections
            sections = self._extract_sections(text, classification)

            # Step 3: Extract document-level metadata
            doc_metadata = self._extract_document_metadata(text, classification, filename)

            # Step 4: Calculate quality score
            quality_score = self._calculate_extraction_quality(sections, len(text))

            # Create structured document
            structured = StructuredDocument(
                document_id=document_id,
                original_filename=filename,
                extraction_timestamp=self.extraction_timestamp,
                document_type=classification.primary_type.value,
                document_classification={
                    "primary_type": classification.primary_type.value,
                    "confidence": classification.confidence,
                    "domain": classification.domain,
                    "secondary_types": [
                        {"type": t.value, "confidence": c}
                        for t, c in (classification.secondary_types or [])
                    ],
                    "key_indicators": classification.key_indicators,
                    "structured_fields": classification.structured_fields
                },
                raw_text=text,
                total_pages=page_count,
                sections=[asdict(s) for s in sections],
                metadata=doc_metadata,
                extraction_quality_score=quality_score,
                extraction_notes=self._generate_extraction_notes(classification, sections)
            )

            logger.info(
                f"Extracted document {document_id}: type={classification.primary_type.value}, "
                f"sections={len(sections)}, quality={quality_score:.2f}"
            )

            return structured

        except Exception as e:
            logger.error(f"Error extracting document {document_id}: {e}", exc_info=True)
            raise

    def _extract_sections(self, text: str, classification: DocumentClassification) -> List[ExtractedSection]:
        """Extract semantic sections from document based on type."""

        doc_type = classification.primary_type

        if doc_type in [DocumentType.RESUME, DocumentType.CV]:
            return self._extract_resume_sections(text)
        elif doc_type == DocumentType.INVOICE:
            return self._extract_invoice_sections(text)
        elif doc_type == DocumentType.MEDICAL_RECORD:
            return self._extract_medical_sections(text)
        elif doc_type == DocumentType.LEGAL_DOCUMENT:
            return self._extract_legal_sections(text)
        else:
            return self._extract_generic_sections(text)

    def _extract_resume_sections(self, text: str) -> List[ExtractedSection]:
        """Extract sections from resume/CV."""
        sections = []
        section_id = 0

        # Define section patterns - expanded for better coverage
        section_patterns = {
            "contact": r"(?:contact|personal|contact info|contact information|personal details|personal info)",
            "summary": r"(?:professional summary|objective|career objective|summary statement|profile|about me|about|overview)",
            "skills": r"(?:technical skills|core competencies|skills|expertise|competencies|proficiencies|technologies)",
            "experience": r"(?:work experience|professional experience|employment|career history|experience|work history|employment history)",
            "education": r"(?:education|academic|qualifications|degree|educational background|academic background)",
            "certifications": r"(?:certification|certifications|credentials|licenses|professional certifications)",
            "achievements": r"(?:achievement|awards|recognition|honor|accomplishments|highlights)",
            "projects": r"(?:projects|project experience|key projects)",
            "languages": r"(?:languages|language proficiency|spoken languages)",
            "references": r"(?:references|referees)",
        }

        lines = text.split("\n")
        current_section = None
        current_content = []
        header_content = []  # Content before first recognized section

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Skip empty lines for section detection
            if not line_lower:
                if current_section:
                    current_content.append(line)
                else:
                    header_content.append(line)
                continue

            # Check if this line starts a new section
            matched_section = None
            for section_type, pattern in section_patterns.items():
                if re.match(f"^.*{pattern}.*$", line_lower, re.IGNORECASE):
                    matched_section = section_type
                    break

            if matched_section:
                # Save header content as contact/header section if we haven't started any section yet
                if current_section is None and header_content:
                    header_text = "\n".join(header_content).strip()
                    if header_text:
                        sections.append(ExtractedSection(
                            section_id=f"section_{section_id}",
                            section_type="header",
                            title="Contact Information",
                            content=header_text,
                            start_page=1,
                            end_page=1,
                            confidence=0.85,
                            key_items=self._extract_key_items("contact", header_text)
                        ))
                        section_id += 1

                # Save previous section
                if current_section and current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        sections.append(ExtractedSection(
                            section_id=f"section_{section_id}",
                            section_type=current_section,
                            title=current_section.title(),
                            content=content_text,
                            start_page=1,
                            end_page=1,
                            confidence=0.95,
                            key_items=self._extract_key_items(current_section, content_text)
                        ))
                        section_id += 1

                current_section = matched_section
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    header_content.append(line)

        # Save last section
        if current_section and current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                sections.append(ExtractedSection(
                    section_id=f"section_{section_id}",
                    section_type=current_section,
                    title=current_section.title(),
                    content=content_text,
                    start_page=1,
                    end_page=1,
                    confidence=0.95,
                    key_items=self._extract_key_items(current_section, content_text)
                ))
                section_id += 1

        # If no sections were detected, include header content
        if not sections and header_content:
            header_text = "\n".join(header_content).strip()
            if header_text:
                sections.append(ExtractedSection(
                    section_id="section_0",
                    section_type="content",
                    title="Resume Content",
                    content=header_text,
                    start_page=1,
                    end_page=1,
                    confidence=0.70,
                ))

        # CRITICAL: If still no sections, use the full text as a single section
        if not sections and text.strip():
            sections.append(ExtractedSection(
                section_id="section_0",
                section_type="content",
                title="Document Content",
                content=text.strip(),
                start_page=1,
                end_page=1,
                confidence=0.60,
            ))

        return sections

    def _extract_invoice_sections(self, text: str) -> List[ExtractedSection]:
        """Extract sections from invoice."""
        sections = []

        # Extract header section
        header_lines = []
        for line in text.split("\n")[:20]:
            if any(x in line.lower() for x in ["invoice", "bill", "receipt"]):
                header_lines = text.split("\n")[:20]
                break

        if header_lines:
            sections.append(ExtractedSection(
                section_id="section_0",
                section_type="header",
                title="Invoice Header",
                content="\n".join(header_lines),
                start_page=1,
                end_page=1,
                confidence=0.95
            ))

        # Extract line items section
        if "$" in text or "total" in text.lower():
            sections.append(ExtractedSection(
                section_id="section_1",
                section_type="line_items",
                title="Line Items",
                content=text,
                start_page=1,
                end_page=1,
                confidence=0.85
            ))

        return sections

    def _extract_medical_sections(self, text: str) -> List[ExtractedSection]:
        """Extract sections from medical records."""
        sections = []

        patterns = {
            "patient_info": r"(?:patient|demographics|identification)",
            "chief_complaint": r"(?:chief complaint|presenting problem|reason for visit)",
            "history": r"(?:history of present illness|hpi|patient history)",
            "physical_exam": r"(?:physical examination|physical exam|exam)",
            "assessment": r"(?:assessment|diagnosis|clinical impression)",
            "plan": r"(?:plan|treatment plan|recommendations)",
            "medications": r"(?:medication|medications|prescriptions|rx)",
            "lab_results": r"(?:lab|laboratory|results|findings)",
        }

        # Simple extraction - split by major sections
        current_section = None
        content_buffer = []
        section_id = 0

        for line in text.split("\n"):
            matched = None
            for section_type, pattern in patterns.items():
                if re.match(f".*{pattern}.*", line.lower(), re.IGNORECASE):
                    matched = section_type
                    break

            if matched and current_section != matched:
                if current_section and content_buffer:
                    sections.append(ExtractedSection(
                        section_id=f"section_{section_id}",
                        section_type=current_section,
                        title=current_section.replace("_", " ").title(),
                        content="\n".join(content_buffer),
                        start_page=1,
                        end_page=1,
                        confidence=0.85
                    ))
                    section_id += 1
                    content_buffer = []
                current_section = matched

            if current_section:
                content_buffer.append(line)

        if current_section and content_buffer:
            sections.append(ExtractedSection(
                section_id=f"section_{section_id}",
                section_type=current_section,
                title=current_section.replace("_", " ").title(),
                content="\n".join(content_buffer),
                start_page=1,
                end_page=1,
                confidence=0.85
            ))

        return sections

    def _extract_legal_sections(self, text: str) -> List[ExtractedSection]:
        """Extract sections from legal documents."""
        sections = []

        # Extract by Article/Section numbers
        import re
        section_pattern = r"(?:Article|Section|Clause)\s+(\d+\.?\d*)"

        matches = list(re.finditer(section_pattern, text, re.IGNORECASE))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_content = text[start:end]

            sections.append(ExtractedSection(
                section_id=f"section_{i}",
                section_type="clause",
                title=match.group(0),
                content=section_content,
                start_page=1,
                end_page=1,
                confidence=0.90
            ))

        return sections

    def _extract_generic_sections(self, text: str) -> List[ExtractedSection]:
        """Extract sections from generic documents."""
        # For generic docs, create one section
        return [ExtractedSection(
            section_id="section_0",
            section_type="content",
            title="Document Content",
            content=text,
            start_page=1,
            end_page=1,
            confidence=0.70
        )]

    def _extract_key_items(self, section_type: str, content: str) -> List[str]:
        """Extract key items from a section based on type."""
        items = []

        # Split by common delimiters
        for delimiter in ["\n", ",", "•", "-", "*"]:
            if delimiter in content:
                parts = content.split(delimiter)
                items = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
                break

        return items[:10]  # Return top 10 items

    def _extract_document_metadata(self, text: str, classification: DocumentClassification, filename: str) -> Dict:
        """Extract document-level metadata."""
        return {
            "filename": filename,
            "document_type": classification.primary_type.value,
            "domain": classification.domain,
            "classification_confidence": classification.confidence,
            "text_length": len(text),
            "line_count": text.count("\n") + 1,
            "has_tables": "table" in text.lower(),
            "has_lists": any(text.strip().startswith(c) for c in ["-", "*", "•"]),
            "has_contact_info": bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)),
            "has_currency": bool(re.search(r"\$\d+", text)),
            "has_dates": bool(re.search(r"\d{1,2}/\d{1,2}/\d{4}", text)),
        }

    def _calculate_extraction_quality(self, sections: List[ExtractedSection], text_length: int) -> float:
        """Calculate extraction quality score."""
        if not sections:
            return 0.0

        # Score based on number and quality of sections
        section_quality = len(sections) / 10  # Normalize

        # Score based on content coverage
        total_extracted = sum(len(s.content) for s in sections)
        coverage = total_extracted / text_length if text_length > 0 else 0

        # Score based on average confidence
        avg_confidence = sum(s.confidence for s in sections) / len(sections) if sections else 0

        # Weighted average
        quality = (section_quality * 0.3) + (coverage * 0.3) + (avg_confidence * 0.4)

        return min(1.0, max(0.0, quality))

    def _generate_extraction_notes(self, classification: DocumentClassification, sections: List[ExtractedSection]) -> List[str]:
        """Generate notes about extraction quality and issues."""
        notes = []

        if classification.confidence < 0.5:
            notes.append(f"Low classification confidence: {classification.confidence:.2f}")

        if not sections:
            notes.append("No sections identified")

        if classification.secondary_types and classification.secondary_types[0][1] > 0.4:
            notes.append(f"Document may also be: {classification.secondary_types[0][0].value}")

        return notes

# Singleton instance
_extraction_engine = None

def get_extraction_engine() -> StructuredExtractionEngine:
    """Get or create extraction engine singleton."""
    global _extraction_engine
    if _extraction_engine is None:
        _extraction_engine = StructuredExtractionEngine()
    return _extraction_engine

# Required imports
import re

