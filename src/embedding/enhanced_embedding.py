"""
Enhanced Embedding Module - Improves embedding quality for better retrieval accuracy.

Key improvements:
1. Context-enriched embedding text (section context prepended)
2. Domain-aware embedding templates
3. Chunk quality scoring
4. Deduplication before upsert
5. Enhanced semantic keyword extraction
"""

from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = get_logger(__name__)

@dataclass
class EnhancedChunk:
    """A chunk optimized for embedding and retrieval."""
    chunk_id: str
    original_text: str  # Raw chunk text
    embedding_text: str  # Context-enriched text for embedding
    canonical_text: str  # Normalized text for storage

    # Metadata
    section_title: str
    section_path: str
    section_type: str
    document_id: str
    document_type: str
    document_domain: str

    # Position
    page_number: int
    chunk_index: int
    total_chunks: int

    # Quality
    quality_score: float
    content_type: str  # "narrative", "entity_list", "tabular", "structural"

    # Semantic
    semantic_keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)

    # Deduplication
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.canonical_text.encode("utf-8")
            ).hexdigest()[:16]

class EnhancedEmbeddingBuilder:
    """
    Builds optimized embeddings for better retrieval accuracy.

    Key features:
    - Prepends section context to embedding text
    - Extracts semantic keywords and entities
    - Scores chunk quality
    - Deduplicates similar content
    """

    # Domain-specific embedding templates
    EMBEDDING_TEMPLATES = {
        "resume": {
            "skills": "Skills and Technologies: {content}",
            "experience": "Work Experience: {section_title}. {content}",
            "education": "Education and Qualifications: {content}",
            "certifications": "Certifications and Credentials: {content}",
            "contact": "Contact Information: {content}",
            "summary": "Professional Summary: {content}",
            "default": "{section_path}. {content}",
        },
        "invoice": {
            "line_items": "Invoice Items: {content}",
            "totals": "Invoice Totals and Amounts: {content}",
            "parties": "Billing Parties: {content}",
            "terms": "Payment Terms: {content}",
            "default": "Invoice {section_title}: {content}",
        },
        "legal": {
            "clause": "Legal Clause - {section_title}: {content}",
            "terms": "Terms and Conditions: {content}",
            "definitions": "Legal Definitions: {content}",
            "default": "Legal Document {section_path}: {content}",
        },
        "medical": {
            "diagnosis": "Medical Diagnosis: {content}",
            "medications": "Prescribed Medications: {content}",
            "history": "Patient History: {content}",
            "findings": "Clinical Findings: {content}",
            "default": "Medical Record {section_title}: {content}",
        },
        "generic": {
            "default": "{section_path}: {content}",
        },
    }

    # Content type indicators
    CONTENT_TYPE_PATTERNS = {
        "entity_list": [
            r"^(?:[-•*]|\d+\.)\s+",  # Bullet points
            r"(?:,\s+){3,}",  # Comma-separated lists
            r"(?:\n\s*[-•*]){2,}",  # Multiple bullets
        ],
        "tabular": [
            r"\|.*\|",  # Table with pipes
            r"(?:\S+\s{2,}){2,}\S+",  # Space-aligned columns
        ],
        "structural": [
            r"^(?:Article|Section|Clause|Chapter)\s+\d",
            r"^\d+\.\d+",  # Numbered sections
        ],
    }

    # Stop words for keyword extraction
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
        "they", "my", "your", "his", "her", "our", "their", "what", "which",
        "who", "whom", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no", "not",
        "only", "same", "so", "than", "too", "very", "just", "also", "into",
    }

    def __init__(self, domain: str = "generic"):
        self.domain = domain.lower()
        self.templates = self.EMBEDDING_TEMPLATES.get(
            self.domain,
            self.EMBEDDING_TEMPLATES["generic"]
        )

    def build_enhanced_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_metadata: Dict[str, Any],
    ) -> List[EnhancedChunk]:
        """
        Build enhanced chunks with optimized embedding text.

        Args:
            chunks: Raw chunks with metadata
            document_metadata: Document-level metadata

        Returns:
            List of EnhancedChunk with optimized embedding text
        """
        enhanced_chunks: List[EnhancedChunk] = []
        seen_hashes: Set[str] = set()

        document_id = document_metadata.get("document_id", "")
        document_type = document_metadata.get("document_type", "generic")
        document_domain = document_metadata.get("document_domain", self.domain)

        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            original_text = chunk.get("text", "") or chunk.get("content", "")
            if not original_text.strip():
                continue

            # Normalize text
            canonical_text = self._normalize_text(original_text)

            # Check for duplicates
            content_hash = hashlib.sha256(canonical_text.encode()).hexdigest()[:16]
            if content_hash in seen_hashes:
                logger.debug(f"Skipping duplicate chunk: {content_hash}")
                continue
            seen_hashes.add(content_hash)

            # Extract metadata
            section_title = chunk.get("section_title", "") or chunk.get("section", "")
            section_path = chunk.get("section_path", "") or section_title
            section_type = self._infer_section_type(section_title, canonical_text)

            # Build embedding text with context
            embedding_text = self._build_embedding_text(
                content=canonical_text,
                section_title=section_title,
                section_path=section_path,
                section_type=section_type,
            )

            # Detect content type
            content_type = self._detect_content_type(canonical_text)

            # Extract semantic information
            semantic_keywords = self._extract_semantic_keywords(
                canonical_text, section_type
            )
            entities = self._extract_entities(canonical_text, section_type)
            key_phrases = self._extract_key_phrases(canonical_text)

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                text=canonical_text,
                section_type=section_type,
                content_type=content_type,
                has_entities=bool(entities),
            )

            enhanced_chunk = EnhancedChunk(
                chunk_id=chunk.get("chunk_id", f"{document_id}_chunk_{idx}"),
                original_text=original_text,
                embedding_text=embedding_text,
                canonical_text=canonical_text,
                section_title=section_title,
                section_path=section_path,
                section_type=section_type,
                document_id=document_id,
                document_type=document_type,
                document_domain=document_domain,
                page_number=chunk.get("page", 1) or 1,
                chunk_index=idx,
                total_chunks=total_chunks,
                quality_score=quality_score,
                content_type=content_type,
                semantic_keywords=semantic_keywords,
                entities=entities,
                key_phrases=key_phrases,
                content_hash=content_hash,
            )

            enhanced_chunks.append(enhanced_chunk)

        # Log statistics
        if enhanced_chunks:
            avg_quality = sum(c.quality_score for c in enhanced_chunks) / len(enhanced_chunks)
            logger.info(
                f"Built {len(enhanced_chunks)} enhanced chunks "
                f"(avg quality: {avg_quality:.2f}, deduplicated: {len(chunks) - len(enhanced_chunks)})"
            )

        return enhanced_chunks

    def _build_embedding_text(
        self,
        content: str,
        section_title: str,
        section_path: str,
        section_type: str,
    ) -> str:
        """Build context-enriched embedding text."""
        template = self.templates.get(section_type, self.templates.get("default", "{content}"))

        # Prepare template variables
        vars = {
            "content": content,
            "section_title": section_title or "Content",
            "section_path": section_path or section_title or "Document",
        }

        try:
            embedding_text = template.format(**vars)
        except KeyError:
            embedding_text = f"{section_path}: {content}"

        return embedding_text.strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent embedding."""
        if not text:
            return ""

        # Fix broken hyphenation
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        return text.strip()

    def _infer_section_type(self, section_title: str, content: str) -> str:
        """Infer section type from title and content."""
        combined = f"{section_title} {content}".lower()

        section_patterns = {
            "skills": ["skill", "technolog", "proficien", "competenc", "expert"],
            "experience": ["experience", "employment", "work history", "career"],
            "education": ["education", "degree", "university", "college", "school"],
            "certifications": ["certif", "credential", "license", "accredit"],
            "contact": ["contact", "email", "phone", "address", "linkedin"],
            "summary": ["summary", "objective", "profile", "about me", "overview"],
            "line_items": ["item", "product", "service", "description", "qty"],
            "totals": ["total", "amount", "subtotal", "tax", "grand total"],
            "clause": ["clause", "article", "section", "provision"],
            "diagnosis": ["diagnos", "assessment", "impression"],
            "medications": ["medication", "prescription", "drug", "dosage"],
        }

        for section_type, patterns in section_patterns.items():
            if any(p in combined for p in patterns):
                return section_type

        return "content"

    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in the chunk."""
        for content_type, patterns in self.CONTENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.MULTILINE):
                    return content_type
        return "narrative"

    def _extract_semantic_keywords(
        self,
        text: str,
        section_type: str,
    ) -> List[str]:
        """Extract semantic keywords from text."""
        keywords: List[str] = []

        # Tokenize
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9+#.-]*\b", text)

        # Filter stop words and short words
        candidates = [
            w for w in words
            if w.lower() not in self.STOP_WORDS and len(w) > 2
        ]

        # Extract based on section type
        if section_type == "skills":
            # Capitalize words are likely technologies/skills
            keywords = [w for w in candidates if w[0].isupper() or not w[0].isalpha()]
            # Also include common tech patterns
            tech_patterns = re.findall(
                r"\b(?:[A-Z][a-z]*(?:\.[A-Z][a-z]*)*|[A-Z]+(?:\d+)?|[a-z]+(?:[A-Z][a-z]*)+)\b",
                text
            )
            keywords.extend(tech_patterns)

        elif section_type == "experience":
            # Extract job titles and company names (capitalized sequences)
            cap_sequences = re.findall(r"\b(?:[A-Z][a-z]+\s*){2,5}", text)
            keywords.extend([s.strip() for s in cap_sequences])

        elif section_type in ["education", "certifications"]:
            # Extract degrees and institutions
            degree_patterns = re.findall(
                r"\b(?:B\.?[ASE]\.?|M\.?[ASE]\.?|Ph\.?D\.?|MBA|BS|MS|BA|MA)\b",
                text, re.IGNORECASE
            )
            keywords.extend(degree_patterns)
            keywords.extend([w for w in candidates if w[0].isupper()])

        else:
            # General: extract capitalized words and technical terms
            keywords = [w for w in candidates if w[0].isupper()]

        # Deduplicate and limit
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        return unique_keywords[:20]

    def _extract_entities(
        self,
        text: str,
        section_type: str,
    ) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        entities: List[Dict[str, str]] = []

        # Email addresses
        emails = re.findall(r"[\w.-]+@[\w.-]+\.\w+", text)
        entities.extend([{"type": "EMAIL", "value": e} for e in emails])

        # Phone numbers
        phones = re.findall(r"\+?\d[\d\s()-]{8,}\d", text)
        entities.extend([{"type": "PHONE", "value": p.strip()} for p in phones])

        # URLs
        urls = re.findall(r"https?://[^\s]+", text)
        entities.extend([{"type": "URL", "value": u} for u in urls])

        # LinkedIn URLs specifically
        linkedins = re.findall(
            r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s]+",
            text, re.IGNORECASE
        )
        entities.extend([{"type": "LINKEDIN", "value": l} for l in linkedins])

        # Dates
        dates = re.findall(
            r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4})\b",
            text
        )
        entities.extend([{"type": "DATE", "value": d} for d in dates[:5]])

        # Money amounts
        amounts = re.findall(r"\$[\d,]+(?:\.\d{2})?", text)
        entities.extend([{"type": "MONEY", "value": a} for a in amounts])

        return entities[:15]

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases (2-3 word combinations)."""
        # Simple n-gram extraction for important phrases
        words = text.split()
        phrases = []

        for i in range(len(words) - 1):
            # Bigrams
            if (words[i].lower() not in self.STOP_WORDS and
                words[i + 1].lower() not in self.STOP_WORDS):
                phrase = f"{words[i]} {words[i + 1]}"
                if len(phrase) > 5:
                    phrases.append(phrase)

        # Trigrams
        for i in range(len(words) - 2):
            if (words[i][0].isupper() and
                words[i + 2][0].isupper()):
                phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                phrases.append(phrase)

        return phrases[:10]

    def _calculate_quality_score(
        self,
        text: str,
        section_type: str,
        content_type: str,
        has_entities: bool,
    ) -> float:
        """Calculate chunk quality score (0.0 - 1.0)."""
        score = 0.5  # Base score

        # Length contribution (prefer 200-800 chars)
        text_len = len(text)
        if 200 <= text_len <= 800:
            score += 0.2
        elif 100 <= text_len <= 1000:
            score += 0.1
        elif text_len < 50 or text_len > 2000:
            score -= 0.1

        # Section type contribution (known sections are better)
        if section_type != "content":
            score += 0.1

        # Content type contribution
        if content_type == "entity_list":
            score += 0.1
        elif content_type == "narrative":
            score += 0.05

        # Entity presence contribution
        if has_entities:
            score += 0.1

        # Sentence completeness (ends with punctuation)
        if text.strip() and text.strip()[-1] in ".!?":
            score += 0.05

        return min(1.0, max(0.0, score))

def build_enhanced_payload(
    chunk: EnhancedChunk,
    base_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build enhanced Qdrant payload with all semantic information.

    Args:
        chunk: EnhancedChunk with optimized data
        base_payload: Base payload with core identifiers

    Returns:
        Complete payload for Qdrant upsert
    """
    payload = dict(base_payload)

    # Override with enhanced content
    payload["embedding_text"] = chunk.embedding_text
    payload["canonical_text"] = chunk.canonical_text
    payload["content"] = chunk.original_text

    # Add enhanced metadata
    payload["section_type"] = chunk.section_type
    payload["content_type"] = chunk.content_type
    payload["quality_score"] = chunk.quality_score
    payload["content_hash"] = chunk.content_hash

    # Add semantic information
    payload["semantic_keywords"] = chunk.semantic_keywords
    payload["entities"] = chunk.entities
    payload["key_phrases"] = chunk.key_phrases

    # Add enhanced section metadata
    payload["section"] = {
        "title": chunk.section_title,
        "path": chunk.section_path,
        "type": chunk.section_type,
    }

    return payload

def get_embedding_builder(domain: str = "generic") -> EnhancedEmbeddingBuilder:
    """Get an embedding builder for the specified domain."""
    return EnhancedEmbeddingBuilder(domain=domain)

__all__ = [
    "EnhancedChunk",
    "EnhancedEmbeddingBuilder",
    "build_enhanced_payload",
    "get_embedding_builder",
]
