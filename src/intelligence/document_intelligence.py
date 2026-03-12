"""
Document Intelligence Module.

Provides comprehensive document analysis including:
- File metadata extraction (filename, size, pages, language, file type)
- Structured content extraction with section hierarchy
- Entity extraction (persons, organizations, places, dates, values)
- Domain classification (medical, resume, invoice, tax, legal, etc.)
- Noun and keyword extraction for search optimization
"""

from __future__ import annotations

import hashlib
import json
from src.utils.logging_utils import get_logger
import mimetypes
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = get_logger(__name__)

class DocumentDomain(Enum):
    """Supported document domains for specialized processing."""
    RESUME = "resume"
    INVOICE = "invoice"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    TAX = "tax"
    TECHNICAL = "technical"
    SCANNED = "scanned"
    GENERIC = "generic"

@dataclass
class DocumentMetadata:
    """
    Comprehensive metadata extracted from a document.
    
    Captures all file-level information during upload/extraction.
    """
    document_id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    mime_type: Optional[str] = None
    page_count: int = 1
    language: str = "en"
    languages_detected: List[str] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    has_images: bool = False
    has_tables: bool = False
    is_scanned: bool = False
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    content_hash: Optional[str] = None
    extraction_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_extension": self.file_extension,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "language": self.language,
            "languages_detected": self.languages_detected,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "has_images": self.has_images,
            "has_tables": self.has_tables,
            "is_scanned": self.is_scanned,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "author": self.author,
            "title": self.title,
            "content_hash": self.content_hash,
            "extraction_timestamp": self.extraction_timestamp,
        }

@dataclass
class ExtractedEntity:
    """A single extracted entity with metadata."""
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, etc.
    value: str
    normalized_value: str
    confidence: float
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    context: Optional[str] = None  # Surrounding text for evidence

@dataclass
class ExtractedEntities:
    """
    All entities extracted from a document.
    
    Organized by entity type for easy access.
    """
    persons: List[ExtractedEntity] = field(default_factory=list)
    organizations: List[ExtractedEntity] = field(default_factory=list)
    locations: List[ExtractedEntity] = field(default_factory=list)
    dates: List[ExtractedEntity] = field(default_factory=list)
    monetary_values: List[ExtractedEntity] = field(default_factory=list)
    products: List[ExtractedEntity] = field(default_factory=list)
    skills: List[ExtractedEntity] = field(default_factory=list)
    nouns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "persons": [{"value": e.value, "confidence": e.confidence} for e in self.persons],
            "organizations": [{"value": e.value, "confidence": e.confidence} for e in self.organizations],
            "locations": [{"value": e.value, "confidence": e.confidence} for e in self.locations],
            "dates": [{"value": e.value, "confidence": e.confidence} for e in self.dates],
            "monetary_values": [{"value": e.value, "confidence": e.confidence} for e in self.monetary_values],
            "products": [{"value": e.value, "confidence": e.confidence} for e in self.products],
            "skills": [{"value": e.value, "confidence": e.confidence} for e in self.skills],
            "nouns": self.nouns[:50],  # Limit for storage
            "keywords": self.keywords[:30],
            "emails": self.emails,
            "phones": self.phones,
            "urls": self.urls,
        }
    
    def get_all_searchable_terms(self) -> List[str]:
        """Get all terms useful for search/retrieval."""
        terms = []
        terms.extend(e.value for e in self.persons)
        terms.extend(e.value for e in self.organizations)
        terms.extend(e.value for e in self.locations)
        terms.extend(e.value for e in self.products)
        terms.extend(e.value for e in self.skills)
        terms.extend(self.keywords)
        return list(set(terms))

@dataclass
class DocumentSection:
    """A structured section within a document."""
    section_id: str
    section_type: str  # heading, paragraph, table, list, image_caption, etc.
    heading: Optional[str] = None
    content: str = ""
    level: int = 0  # Hierarchy level (0=root, 1=h1, 2=h2, etc.)
    page_number: Optional[int] = None
    parent_section_id: Optional[str] = None
    child_section_ids: List[str] = field(default_factory=list)
    entities: Optional[ExtractedEntities] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_id": self.section_id,
            "section_type": self.section_type,
            "heading": self.heading,
            "content": self.content,
            "level": self.level,
            "page_number": self.page_number,
            "parent_section_id": self.parent_section_id,
            "child_section_ids": self.child_section_ids,
            "entities": self.entities.to_dict() if self.entities else None,
            "metadata": self.metadata,
        }

@dataclass
class StructuredDocument:
    """
    Fully structured document with all extracted intelligence.
    
    This is the primary output of the document intelligence pipeline.
    """
    document_id: str
    metadata: DocumentMetadata
    domain: DocumentDomain
    domain_confidence: float
    sections: List[DocumentSection]
    entities: ExtractedEntities
    table_of_contents: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    key_facts: List[str] = field(default_factory=list)
    extraction_quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "metadata": self.metadata.to_dict(),
            "domain": self.domain.value,
            "domain_confidence": self.domain_confidence,
            "sections": [s.to_dict() for s in self.sections],
            "entities": self.entities.to_dict(),
            "table_of_contents": self.table_of_contents,
            "summary": self.summary,
            "key_facts": self.key_facts,
            "extraction_quality_score": self.extraction_quality_score,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def get_section_by_heading(self, heading: str) -> Optional[DocumentSection]:
        """Find a section by its heading."""
        heading_lower = heading.lower()
        for section in self.sections:
            if section.heading and heading_lower in section.heading.lower():
                return section
        return None
    
    def get_full_text(self) -> str:
        """Get concatenated text from all sections."""
        return "\n\n".join(
            f"{s.heading}\n{s.content}" if s.heading else s.content
            for s in self.sections if s.content
        )

class DocumentIntelligence:
    """
    Main document intelligence processor.
    
    Orchestrates extraction, entity recognition, domain classification,
    and structured output generation.
    """
    
    # Domain detection keywords
    DOMAIN_KEYWORDS = {
        DocumentDomain.RESUME: {
            "resume", "cv", "curriculum vitae", "work experience", "education",
            "skills", "employment history", "professional experience", "career objective",
            "certifications", "qualifications", "references",
        },
        DocumentDomain.INVOICE: {
            "invoice", "bill", "amount due", "payment terms", "subtotal", "total",
            "tax", "vat", "purchase order", "billing address", "due date",
        },
        DocumentDomain.LEGAL: {
            "agreement", "contract", "clause", "party", "hereby", "whereas",
            "terms and conditions", "liability", "indemnification", "jurisdiction",
        },
        DocumentDomain.MEDICAL: {
            "patient", "diagnosis", "prescription", "treatment", "symptoms",
            "medical history", "dosage", "physician", "hospital", "clinic",
        },
        DocumentDomain.FINANCIAL: {
            "balance sheet", "income statement", "cash flow", "assets", "liabilities",
            "equity", "revenue", "expenses", "profit", "loss", "fiscal year",
        },
        DocumentDomain.TAX: {
            "tax return", "irs", "deduction", "taxable income", "form 1040",
            "schedule c", "w-2", "1099", "federal tax", "state tax",
        },
        DocumentDomain.TECHNICAL: {
            "api", "documentation", "specification", "architecture", "implementation",
            "endpoint", "request", "response", "parameter", "function",
        },
    }
    
    # Entity extraction patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
    URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    DATE_PATTERN = re.compile(
        r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2}[\s,]+\d{4})\b',
        re.IGNORECASE
    )
    MONEY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR)\b')
    PERSON_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')
    ORG_PATTERN = re.compile(
        r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s+(?:Inc|LLC|Ltd|Corp|Company|Co|PLC|Group|University|Institute)\b'
    )
    
    def __init__(
        self,
        enable_deep_ner: bool = True,
        extract_nouns: bool = True,
        max_keywords: int = 50,
    ):
        """
        Initialize the document intelligence processor.
        
        Args:
            enable_deep_ner: Enable deep NER using patterns/models.
            extract_nouns: Extract noun phrases for search.
            max_keywords: Maximum keywords to extract.
        """
        self.enable_deep_ner = enable_deep_ner
        self.extract_nouns = extract_nouns
        self.max_keywords = max_keywords
    
    def process_document(
        self,
        document_id: str,
        text: str,
        filename: str,
        file_size: int = 0,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> StructuredDocument:
        """
        Process a document and extract all intelligence.
        
        Args:
            document_id: Unique document identifier.
            text: Extracted text content.
            filename: Original filename.
            file_size: File size in bytes.
            raw_metadata: Additional metadata from extraction.
        
        Returns:
            StructuredDocument with all extracted information.
        """
        raw_metadata = raw_metadata or {}
        
        # 1. Extract file metadata
        metadata = self._extract_metadata(
            document_id, filename, text, file_size, raw_metadata
        )
        
        # 2. Classify document domain
        domain, domain_confidence = self._classify_domain(text)
        
        # 3. Extract structured sections
        sections = self._extract_sections(text, raw_metadata)
        
        # 4. Extract entities from full text
        entities = self._extract_entities(text)
        
        # 5. Build table of contents
        toc = self._build_table_of_contents(sections)
        
        # 6. Generate key facts
        key_facts = self._extract_key_facts(text, entities, domain)
        
        # 7. Calculate extraction quality
        quality_score = self._calculate_quality_score(text, sections, entities)
        
        return StructuredDocument(
            document_id=document_id,
            metadata=metadata,
            domain=domain,
            domain_confidence=domain_confidence,
            sections=sections,
            entities=entities,
            table_of_contents=toc,
            key_facts=key_facts,
            extraction_quality_score=quality_score,
        )
    
    def _extract_metadata(
        self,
        document_id: str,
        filename: str,
        text: str,
        file_size: int,
        raw_metadata: Dict[str, Any],
    ) -> DocumentMetadata:
        """Extract comprehensive file metadata."""
        file_ext = Path(filename).suffix.lower() if filename else ""
        mime_type, _ = mimetypes.guess_type(filename) if filename else (None, None)
        
        # Count characters and words
        char_count = len(text)
        word_count = len(text.split())
        
        # Detect page count from metadata or estimate
        page_count = raw_metadata.get("page_count") or raw_metadata.get("pages") or 1
        if page_count == 1 and char_count > 3000:
            page_count = max(1, char_count // 3000)  # Rough estimate
        
        # Detect language (simple heuristic)
        language = self._detect_language(text)
        
        # Check for images/tables
        has_images = bool(raw_metadata.get("has_images")) or "[IMAGE]" in text.upper()
        has_tables = bool(raw_metadata.get("has_tables")) or self._detect_tables(text)
        
        # Check if scanned (OCR indicators)
        is_scanned = bool(raw_metadata.get("is_scanned")) or self._detect_scanned(text)
        
        # Content hash for deduplication
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        return DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_extension=file_ext,
            file_size_bytes=file_size,
            mime_type=mime_type,
            page_count=page_count,
            language=language,
            languages_detected=[language],
            char_count=char_count,
            word_count=word_count,
            has_images=has_images,
            has_tables=has_tables,
            is_scanned=is_scanned,
            creation_date=raw_metadata.get("creation_date"),
            modification_date=raw_metadata.get("modification_date"),
            author=raw_metadata.get("author"),
            title=raw_metadata.get("title"),
            content_hash=content_hash,
        )
    
    def _classify_domain(self, text: str) -> Tuple[DocumentDomain, float]:
        """Classify document domain based on content."""
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                scores[domain] = matches / len(keywords)
        
        if not scores:
            return DocumentDomain.GENERIC, 0.5
        
        best_domain = max(scores, key=scores.get)
        confidence = min(0.95, scores[best_domain] + 0.3)
        
        return best_domain, confidence
    
    def _extract_sections(
        self,
        text: str,
        raw_metadata: Dict[str, Any],
    ) -> List[DocumentSection]:
        """Extract structured sections from text."""
        sections = []
        
        # Check for pre-extracted sections in metadata
        if raw_metadata.get("sections"):
            for idx, sec in enumerate(raw_metadata["sections"]):
                sections.append(DocumentSection(
                    section_id=f"section_{idx}",
                    section_type=sec.get("type", "paragraph"),
                    heading=sec.get("heading") or sec.get("title"),
                    content=sec.get("text") or sec.get("content", ""),
                    level=sec.get("level", 0),
                    page_number=sec.get("page"),
                    metadata=sec.get("metadata", {}),
                ))
            return sections
        
        # Parse sections from text using heading detection
        lines = text.split('\n')
        current_section = None
        section_idx = 0
        content_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Detect headings (ALL CAPS or Title Case followed by colon/newline)
            is_heading = (
                (stripped.isupper() and len(stripped) < 80 and len(stripped.split()) <= 6) or
                (stripped.endswith(':') and len(stripped) < 60) or
                re.match(r'^[A-Z][A-Za-z\s]+$', stripped) and len(stripped) < 50
            )
            
            if is_heading:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(content_lines).strip()
                    if current_section.content or current_section.heading:
                        sections.append(current_section)
                
                # Start new section
                section_idx += 1
                current_section = DocumentSection(
                    section_id=f"section_{section_idx}",
                    section_type="heading",
                    heading=stripped.rstrip(':'),
                    level=1 if stripped.isupper() else 2,
                )
                content_lines = []
            else:
                content_lines.append(stripped)
        
        # Add final section
        if current_section:
            current_section.content = '\n'.join(content_lines).strip()
            if current_section.content or current_section.heading:
                sections.append(current_section)
        elif content_lines:
            sections.append(DocumentSection(
                section_id="section_0",
                section_type="paragraph",
                content='\n'.join(content_lines).strip(),
            ))
        
        return sections
    
    def _extract_entities(self, text: str) -> ExtractedEntities:
        """Extract all entities from text."""
        entities = ExtractedEntities()
        
        # Extract emails
        entities.emails = list(set(self.EMAIL_PATTERN.findall(text)))
        
        # Extract phones
        phones = self.PHONE_PATTERN.findall(text)
        entities.phones = list(set(re.sub(r'\D', '', p) for p in phones if len(re.sub(r'\D', '', p)) >= 10))
        
        # Extract URLs
        entities.urls = list(set(self.URL_PATTERN.findall(text)))
        
        # Extract dates
        for match in self.DATE_PATTERN.finditer(text):
            entities.dates.append(ExtractedEntity(
                entity_type="DATE",
                value=match.group(),
                normalized_value=match.group().lower(),
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end(),
            ))
        
        # Extract monetary values
        for match in self.MONEY_PATTERN.finditer(text):
            entities.monetary_values.append(ExtractedEntity(
                entity_type="MONEY",
                value=match.group(),
                normalized_value=re.sub(r'[^\d.]', '', match.group()),
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end(),
            ))
        
        # Extract persons
        seen_persons = set()
        for match in self.PERSON_PATTERN.finditer(text):
            name = match.group()
            if name.lower() not in seen_persons and len(name.split()) >= 2:
                seen_persons.add(name.lower())
                entities.persons.append(ExtractedEntity(
                    entity_type="PERSON",
                    value=name,
                    normalized_value=name.lower(),
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
        
        # Extract organizations
        for match in self.ORG_PATTERN.finditer(text):
            entities.organizations.append(ExtractedEntity(
                entity_type="ORGANIZATION",
                value=match.group(),
                normalized_value=match.group().lower(),
                confidence=0.75,
                start_pos=match.start(),
                end_pos=match.end(),
            ))
        
        # Extract keywords and nouns
        if self.extract_nouns:
            entities.keywords = self._extract_keywords(text)
            entities.nouns = self._extract_nouns(text)
        
        # Extract skills (common technical and soft skills)
        entities.skills = self._extract_skills(text)
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple TF-based keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will',
            'would', 'could', 'should', 'their', 'there', 'which', 'about',
            'other', 'into', 'more', 'some', 'such', 'than', 'these', 'then',
            'when', 'also', 'been', 'over', 'only', 'very', 'just', 'after',
            'before', 'being', 'both', 'each', 'under', 'between',
        }
        
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:self.max_keywords]]
    
    def _extract_nouns(self, text: str) -> List[str]:
        """Extract noun phrases from text."""
        # Simple capitalized phrase extraction
        nouns = set()
        
        # Title case words that aren't at start of sentences
        for match in re.finditer(r'(?<=[.!?]\s)[a-z].*?([A-Z][a-z]+)', text):
            noun = match.group(1)
            if len(noun) > 2:
                nouns.add(noun.lower())
        
        # Capitalized sequences
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
            phrase = match.group()
            if len(phrase.split()) <= 3:
                nouns.add(phrase.lower())
        
        return list(nouns)[:50]
    
    def _extract_skills(self, text: str) -> List[ExtractedEntity]:
        """Extract skills from text."""
        skills = []
        text_lower = text.lower()
        
        # Common skills to detect
        skill_patterns = [
            # Programming languages
            "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust",
            "ruby", "php", "swift", "kotlin", "scala",
            # Frameworks
            "react", "angular", "vue", "django", "flask", "spring", "node.js", "express",
            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            # Cloud
            "aws", "azure", "gcp", "docker", "kubernetes",
            # Skills
            "machine learning", "deep learning", "data analysis", "project management",
            "agile", "scrum", "leadership", "communication",
        ]
        
        seen = set()
        for skill in skill_patterns:
            if skill in text_lower and skill not in seen:
                seen.add(skill)
                skills.append(ExtractedEntity(
                    entity_type="SKILL",
                    value=skill,
                    normalized_value=skill,
                    confidence=0.85,
                ))
        
        return skills
    
    def _build_table_of_contents(self, sections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """Build table of contents from sections."""
        toc = []
        for section in sections:
            if section.heading:
                toc.append({
                    "section_id": section.section_id,
                    "heading": section.heading,
                    "level": section.level,
                    "page": section.page_number,
                })
        return toc
    
    def _extract_key_facts(
        self,
        text: str,
        entities: ExtractedEntities,
        domain: DocumentDomain,
    ) -> List[str]:
        """Extract key facts based on domain."""
        facts = []
        
        if entities.persons:
            facts.append(f"Person(s) mentioned: {', '.join(e.value for e in entities.persons[:3])}")
        
        if entities.organizations:
            facts.append(f"Organization(s): {', '.join(e.value for e in entities.organizations[:3])}")
        
        if entities.dates:
            facts.append(f"Date(s) found: {', '.join(e.value for e in entities.dates[:3])}")
        
        if entities.monetary_values:
            facts.append(f"Amount(s): {', '.join(e.value for e in entities.monetary_values[:3])}")
        
        if entities.skills:
            facts.append(f"Skills: {', '.join(e.value for e in entities.skills[:5])}")
        
        return facts
    
    def _calculate_quality_score(
        self,
        text: str,
        sections: List[DocumentSection],
        entities: ExtractedEntities,
    ) -> float:
        """Calculate extraction quality score (0-1)."""
        score = 0.0
        
        # Text quality (not too short, not gibberish)
        if len(text) > 100:
            score += 0.2
        if len(text) > 500:
            score += 0.1
        
        # Section structure
        if sections:
            score += min(0.2, len(sections) * 0.02)
        
        # Entity extraction success
        entity_count = (
            len(entities.persons) + len(entities.organizations) +
            len(entities.emails) + len(entities.dates)
        )
        score += min(0.3, entity_count * 0.02)
        
        # Keywords extracted
        if entities.keywords:
            score += min(0.2, len(entities.keywords) * 0.01)
        
        return min(1.0, score)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Very basic - just check for common words
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['the', 'and', 'of', 'to', 'in']):
            return "en"
        if any(w in text_lower for w in ['der', 'die', 'und', 'ist']):
            return "de"
        if any(w in text_lower for w in ['le', 'la', 'de', 'et', 'est']):
            return "fr"
        if any(w in text_lower for w in ['el', 'la', 'de', 'en', 'es']):
            return "es"
        
        return "en"  # Default
    
    def _detect_tables(self, text: str) -> bool:
        """Detect if text contains tables."""
        # Check for table patterns
        table_indicators = [
            r'\|.*\|.*\|',  # Pipe-separated
            r'\t.*\t.*\t',  # Tab-separated
            r'^\s*\d+\s+\S+\s+\d+',  # Numeric columns
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _detect_scanned(self, text: str) -> bool:
        """Detect if document was likely scanned (OCR)."""
        # OCR typically has certain artifacts
        if not text:
            return False
        
        # Check for common OCR issues
        ocr_indicators = [
            len(re.findall(r'[Il1|]', text)) / max(1, len(text)) > 0.05,  # Confusion chars
            len(re.findall(r'\s{3,}', text)) > 10,  # Excessive whitespace
            bool(re.search(r'[^\x00-\x7F]{3,}', text)),  # Non-ASCII sequences
        ]

        return sum(bool(x) for x in ocr_indicators) >= 2

__all__ = [
    "DocumentIntelligence",
    "DocumentMetadata",
    "StructuredDocument",
    "DocumentSection",
    "ExtractedEntities",
    "ExtractedEntity",
    "DocumentDomain",
]
