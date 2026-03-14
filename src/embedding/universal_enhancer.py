"""
Universal Embedding Enhancer - Improves embedding quality for all document types.

This module provides document-agnostic embedding improvements that work across
resumes, invoices, legal documents, medical records, and generic documents.

Key Features:
1. Smart content type detection
2. Context-enriched embedding text
3. Semantic field extraction (keywords, entities, phrases)
4. Quality scoring with multiple dimensions
5. Query enrichment for better retrieval matching
"""

from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = get_logger(__name__)

# =============================================================================
# CONTENT TYPE DETECTION
# =============================================================================

class ContentTypeDetector:
    """Detects the semantic content type of text chunks.

    Supports two detection modes:
    - ``detect()`` (classmethod): regex-based detection, no dependencies
    - ``detect_ml()`` (instance method): zero-shot embedding similarity
      using prototype descriptions — requires an embedder (e.g. BAAI/bge-large-en-v1.5)
    """

    # ── Prototype descriptions for ML zero-shot classification ────────
    CONTENT_TYPE_DESCRIPTIONS: Dict[str, List[str]] = {
        "contact_info": [
            "Personal contact information with email address, phone number, and social media profiles",
            "Name, email, phone, LinkedIn URL, GitHub profile, mailing address",
            "Contact details section: email, mobile number, LinkedIn, location",
        ],
        "skills_list": [
            "Technical skills list: Python, Java, JavaScript, React, AWS, Docker, Kubernetes, SQL",
            "Programming languages, frameworks, tools, and technologies the candidate is proficient in",
            "Skills and competencies: software development, cloud computing, databases, DevOps tools",
        ],
        "education": [
            "Educational qualifications: Bachelor of Technology, Master of Science, university name, GPA, graduation year",
            "Academic background with degree, institution, major, grade point average",
            "Education section: college degree, university, coursework, specialization, academic achievements",
        ],
        "experience": [
            "Professional work experience: job title, company name, employment dates, responsibilities and achievements",
            "Career history showing roles, organizations, date ranges from 2019 to present, projects delivered",
            "Work experience with accomplishments: managed team, developed software, deployed services, led projects",
        ],
        "financial": [
            "Invoice with line items, unit prices, quantities, subtotals, tax amounts, and grand total in dollars",
            "Financial document: payment terms, amount due, billing summary, account balance, transaction details",
            "Purchase order with item descriptions, costs, shipping charges, and total amount payable",
        ],
        "legal": [
            "Legal contract clause: governing law, indemnification, confidentiality, liability limitations",
            "Agreement terms and conditions: whereas clauses, arbitration, force majeure, termination provisions",
            "Legal document with definitions, obligations, representations, warranties, and signature blocks",
        ],
        "medical": [
            "Medical record: patient diagnosis, symptoms, treatment plan, prescribed medications and dosages",
            "Clinical findings: chief complaint, history of present illness, physical examination, lab results",
            "Healthcare document with patient demographics, medical history, prognosis, and follow-up plan",
        ],
    }

    # Content type patterns (regex fallback)
    PATTERNS = {
        "contact_info": [
            r'(?:^|(?:email|e-mail)\s*[:\-]?\s*)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',  # Email with label context or start-of-line
            r'(?:phone|tel|mobile|cell|fax)\s*[:\-]?\s*[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',  # Phone with label context
            r'linkedin\.com/in/[\w-]+',  # LinkedIn
            r'github\.com/[\w-]+',  # GitHub
        ],
        "skills_list": [
            r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|sql|aws|azure|gcp|docker|kubernetes)\b',
            r'\b(excel|powerpoint|word|salesforce|sap|oracle|tableau|power\s*bi)\b',
        ],
        "education": [
            r'\b(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|mba)\b',
            r'\b(university|college|institute|school)\s+of\b',
            r'\b(degree|diploma|certificate)\b',
            r'\b(gpa|cgpa|grade|score)\s*[:\-]?\s*[\d\.]+',
        ],
        "experience": [
            r'\b(years?\s+of\s+experience|years?\s+exp\.?)\b',
            r'\b(worked|work|working)\s+(at|for|with)\b',
            r'\b(responsible\s+for|managed|led|developed|implemented)\b',
            r'\b(20\d{2})\s*[-–]\s*(20\d{2}|present|current)\b',
        ],
        "financial": [
            r'[\$€£¥]\s*[\d,]+\.?\d*',  # Currency amounts (anchor pattern)
            r'\b(invoice\s+(?:number|no|date|#)|payment\s+(?:due|terms?)|amount\s+due|subtotal|grand\s+total)\b',
            r'\b(qty|quantity|unit\s*price|line\s*item)\b',
        ],
        "legal": [
            r'\b(clause|section|article|paragraph)\s+\d+',
            r'\b(hereby|thereof|therein|whereas|notwithstanding)\b',
            r'\b(agreement|contract|terms|conditions|liability)\b',
        ],
        "medical": [
            r'\b(diagnosis|prognosis|treatment|medication|prescription)\b',
            r'\b(patient|symptoms|condition|disease|disorder)\b',
            r'\b(mg|ml|dosage|administered)\b',
        ],
        "tabular": [
            r'\|[^|]+\|[^|]+\|',  # Pipe-separated tables
            r'\t[^\t]+\t[^\t]+',  # Tab-separated data
        ],
        "bullet_list": [
            r'^[\s]*[•\-\*▪▸◦]\s+',
            r'^[\s]*\d+[\.\)]\s+',
        ],
    }

    def __init__(self, embedder=None):
        self._embedder = embedder
        self._prototypes: Dict[str, Any] = {}  # type -> centroid vector (np.ndarray)
        self._proto_lock = threading.Lock()

    def _ensure_prototypes(self) -> None:
        """Lazily compute prototype centroids (once, thread-safe)."""
        if self._prototypes or self._embedder is None or not _HAS_NUMPY:
            return
        with self._proto_lock:
            if self._prototypes:
                return  # double-check after lock
            try:
                for ctype, descriptions in self.CONTENT_TYPE_DESCRIPTIONS.items():
                    vecs = self._embedder.encode(
                        descriptions,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    centroid = np.asarray(vecs, dtype=np.float32).mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-8:
                        centroid /= norm
                    self._prototypes[ctype] = centroid
            except Exception:
                logger.warning("Failed to compute content type prototypes", exc_info=True)
                self._prototypes = {}

    def detect_ml(self, text: str, section_title: str = "") -> Dict[str, Any]:
        """ML-based content type detection via cosine similarity to prototypes.

        Falls back to regex ``detect()`` when embedder is unavailable.
        """
        self._ensure_prototypes()
        if not self._prototypes or self._embedder is None or not _HAS_NUMPY:
            return self.detect(text, section_title)  # fallback

        # Encode input text (first 500 chars for efficiency)
        input_text = f"{section_title}: {text[:500]}" if section_title else text[:500]
        try:
            vec = np.asarray(
                self._embedder.encode(
                    input_text,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            ).ravel()
        except Exception:
            logger.debug("Embedder encode failed in detect_ml, falling back to regex", exc_info=True)
            return self.detect(text, section_title)

        return self._classify_vector(vec, text)

    def detect_ml_batch(
        self, texts: List[str], section_titles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch ML content type detection — encodes all texts in one call.

        Falls back to per-item regex ``detect()`` when embedder is unavailable.
        """
        if not texts:
            return []
        self._ensure_prototypes()
        if not self._prototypes or self._embedder is None or not _HAS_NUMPY:
            titles = section_titles or [""] * len(texts)
            return [self.detect(t, s) for t, s in zip(texts, titles)]

        titles = section_titles or [""] * len(texts)
        input_texts = [
            f"{title}: {text[:500]}" if title else text[:500]
            for text, title in zip(texts, titles)
        ]
        try:
            vecs = np.asarray(
                self._embedder.encode(
                    input_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=32,
                ),
                dtype=np.float32,
            )
        except Exception:
            logger.debug("Batch encode failed in detect_ml_batch, falling back to regex", exc_info=True)
            return [self.detect(t, s) for t, s in zip(texts, titles)]

        return [self._classify_vector(vecs[i], texts[i]) for i in range(len(texts))]

    def _classify_vector(self, vec: Any, text: str) -> Dict[str, Any]:
        """Classify a single pre-encoded vector against prototypes."""
        # Cosine similarity to each prototype
        scores: Dict[str, float] = {}
        for ctype, centroid in self._prototypes.items():
            scores[ctype] = float(np.dot(vec, centroid))

        # Structural detection stays regex-based (layout feature)
        structure = self._detect_structure(text)

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        confidence = max(0.0, min(1.0, best_score))

        # Low confidence gate — gibberish / unrelated text → narrative
        if best_score < 0.3:
            best_type = "narrative"
            confidence = 0.5

        return {
            "content_type": best_type,
            "confidence": confidence,
            "structure": structure,
            "features": {},
            "scores": scores,
        }

    @classmethod
    def detect(cls, text: str, section_title: str = "") -> Dict[str, Any]:
        """
        Detect content type and characteristics (regex fallback).

        Returns:
            Dict with content_type, confidence, and detected features
        """
        text_lower = text.lower()
        combined = f"{section_title} {text}".lower()

        # Score each content type
        scores = {}
        features = {}

        for content_type, patterns in cls.PATTERNS.items():
            matches = 0
            matched_items = []
            for pattern in patterns:
                found = re.findall(pattern, combined, re.IGNORECASE | re.MULTILINE)
                if found:
                    matches += len(found)
                    matched_items.extend(found[:5])  # Limit stored matches
            scores[content_type] = matches
            if matched_items:
                features[content_type] = matched_items[:5]

        # ── Score normalization gates ──
        # contact_info: only count if >=2 distinct signals (email, phone,
        # linkedin/github) AND text is short or has "contact" in title.
        if scores.get("contact_info", 0) > 0:
            distinct = sum(
                1 for p_idx, p in enumerate(cls.PATTERNS["contact_info"])
                if re.search(p, combined, re.IGNORECASE | re.MULTILINE)
            )
            title_has_contact = "contact" in section_title.lower() if section_title else False
            if distinct < 2 and not title_has_contact:
                scores["contact_info"] = 0
            elif len(text) > 300 and not title_has_contact:
                scores["contact_info"] = max(0, scores["contact_info"] - 2)

        # financial: only count if a currency symbol was actually found,
        # not just keyword matches alone.
        if scores.get("financial", 0) > 0:
            has_currency = bool(re.search(r'[\$€£¥]\s*[\d,]+\.?\d*', combined))
            if not has_currency:
                # Keyword-only: require >=2 financial keywords to qualify
                keyword_hits = sum(
                    1 for p in cls.PATTERNS["financial"][1:]
                    if re.search(p, combined, re.IGNORECASE)
                )
                if keyword_hits < 2:
                    scores["financial"] = 0

        # Determine primary content type
        if not scores or max(scores.values()) == 0:
            primary_type = "narrative"
            confidence = 0.5
        else:
            primary_type = max(scores, key=scores.get)
            max_score = scores[primary_type]
            confidence = min(1.0, max_score / 5.0)  # Normalize to 0-1

        # Detect structure type
        structure = cls._detect_structure(text)

        return {
            "content_type": primary_type,
            "confidence": confidence,
            "structure": structure,
            "features": features,
            "scores": scores,
        }

    @classmethod
    def _detect_structure(cls, text: str) -> str:
        """Detect the structural format of the text."""
        lines = text.strip().split('\n')

        # Check for bullet lists
        bullet_lines = sum(1 for l in lines if re.match(r'^\s*[•\-\*▪▸◦]\s+', l) or re.match(r'^\s*\d+[\.\)]\s+', l))
        if bullet_lines > len(lines) * 0.5:
            return "bullet_list"

        # Check for table structure
        pipe_lines = sum(1 for l in lines if l.count('|') >= 2)
        if pipe_lines > len(lines) * 0.3:
            return "table"

        # Check for key-value pairs
        kv_lines = sum(1 for l in lines if re.match(r'^[^:]+:\s*.+', l))
        if kv_lines > len(lines) * 0.3:
            return "key_value"

        # Check for comma-separated list
        if len(lines) <= 3 and text.count(',') > 3:
            return "comma_list"

        return "prose"

# =============================================================================
# SEMANTIC FIELD EXTRACTION
# =============================================================================

class SemanticFieldExtractor:
    """Extracts semantic fields from text for enhanced retrieval."""

    # Common stopwords to filter
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    }

    # Entity patterns
    ENTITY_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
        "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "linkedin": r'linkedin\.com/in/[\w-]+',
        "date": r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        "money": r'[\$€£¥]\s*[\d,]+\.?\d*',
        "percentage": r'\d+\.?\d*\s*%',
        "year_range": r'\b(19|20)\d{2}\s*[-–]\s*(?:(19|20)\d{2}|present|current)\b',
    }

    # Technology and skill keywords
    TECH_KEYWORDS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql',
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'spark',
        'linux', 'windows', 'macos', 'bash', 'powershell',
        'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api', 'rest', 'graphql',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai',
        'excel', 'powerpoint', 'word', 'salesforce', 'sap', 'oracle', 'tableau',
    }

    @classmethod
    def extract(cls, text: str, content_type: str = "generic") -> Dict[str, Any]:
        """
        Extract semantic fields from text.

        Returns:
            Dict with keywords, entities, phrases, and domain_terms
        """
        # Extract entities
        entities = cls._extract_entities(text)

        # Extract keywords
        keywords = cls._extract_keywords(text)

        # Extract key phrases (bigrams/trigrams)
        phrases = cls._extract_phrases(text)

        # Extract domain-specific terms
        domain_terms = cls._extract_domain_terms(text, content_type)

        return {
            "entities": entities,
            "keywords": keywords,
            "phrases": phrases,
            "domain_terms": domain_terms,
        }

    @classmethod
    def _extract_entities(cls, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        entities = []

        for entity_type, pattern in cls.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:5]:  # Limit per type
                if isinstance(match, tuple):
                    match = match[0]
                entities.append({"type": entity_type, "value": str(match)})

        return entities

    @classmethod
    def _extract_keywords(cls, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.-]*\b', text.lower())

        # Filter stopwords and short words
        keywords = [w for w in words if w not in cls.STOPWORDS and len(w) > 2]

        # Count frequency
        word_counts = Counter(keywords)

        # Get top keywords
        top_keywords = [word for word, _ in word_counts.most_common(20)]

        # Also add any tech keywords found
        text_lower = text.lower()
        for tech in cls.TECH_KEYWORDS:
            if tech in text_lower and tech not in top_keywords:
                top_keywords.append(tech)

        return top_keywords[:25]

    @classmethod
    def _extract_phrases(cls, text: str) -> List[str]:
        """Extract meaningful 2-3 word phrases."""
        # Clean text
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()

        # Generate bigrams and trigrams
        phrases = []

        for i in range(len(words) - 1):
            if words[i] not in cls.STOPWORDS and words[i+1] not in cls.STOPWORDS:
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:
                    phrases.append(phrase)

        for i in range(len(words) - 2):
            if words[i] not in cls.STOPWORDS and words[i+2] not in cls.STOPWORDS:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase) > 8:
                    phrases.append(phrase)

        # Count and return top phrases
        phrase_counts = Counter(phrases)
        return [phrase for phrase, _ in phrase_counts.most_common(10)]

    @classmethod
    def _extract_domain_terms(cls, text: str, content_type: str) -> List[str]:
        """Extract domain-specific terms based on content type."""
        terms = []
        text_lower = text.lower()

        if content_type in ("skills_list", "experience"):
            # Extract technology terms
            for tech in cls.TECH_KEYWORDS:
                if len(tech) <= 3:
                    # Use word boundary for short terms
                    if re.search(r'\b' + re.escape(tech) + r'\b', text_lower):
                        terms.append(tech)
                elif tech in text_lower:
                    terms.append(tech)

        elif content_type == "education":
            # Extract degree and institution terms
            degree_patterns = [
                r'\b(bachelor|master|phd|doctorate|diploma|certificate)\b',
                r'\b(computer science|engineering|business|management|arts|science)\b',
            ]
            for pattern in degree_patterns:
                matches = re.findall(pattern, text_lower)
                terms.extend(matches)

        elif content_type == "financial":
            # Extract financial terms
            financial_terms = ['invoice', 'payment', 'amount', 'total', 'subtotal',
                             'tax', 'due', 'credit', 'debit', 'balance']
            for term in financial_terms:
                if term in text_lower:
                    terms.append(term)

        elif content_type == "legal":
            # Extract legal terms
            legal_terms = ['agreement', 'contract', 'clause', 'liability', 'warranty',
                          'indemnity', 'termination', 'confidential', 'obligation']
            for term in legal_terms:
                if term in text_lower:
                    terms.append(term)

        return list(set(terms))[:15]

# =============================================================================
# QUALITY SCORING
# =============================================================================

@dataclass
class QualityScore:
    """Multi-dimensional quality score for a chunk."""
    overall: float
    length_score: float
    completeness_score: float
    information_density: float
    structure_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall": self.overall,
            "length": self.length_score,
            "completeness": self.completeness_score,
            "density": self.information_density,
            "structure": self.structure_score,
        }

class QualityScorer:
    """Scores chunk quality for better retrieval ranking."""

    # Optimal length range (characters)
    OPTIMAL_MIN_LENGTH = 150
    OPTIMAL_MAX_LENGTH = 800

    @classmethod
    def score(cls, text: str, content_type: str, entities: List[Dict],
              keywords: List[str], structure: str) -> QualityScore:
        """
        Calculate multi-dimensional quality score.

        Returns:
            QualityScore with overall and component scores
        """
        length_score = cls._score_length(text)
        completeness_score = cls._score_completeness(text)
        density_score = cls._score_information_density(text, entities, keywords)
        structure_score = cls._score_structure(text, structure)

        # Weighted overall score
        overall = (
            length_score * 0.20 +
            completeness_score * 0.25 +
            density_score * 0.30 +
            structure_score * 0.25
        )

        # Boost for certain content types
        if content_type in ("skills_list", "contact_info", "education"):
            overall = min(1.0, overall + 0.1)

        return QualityScore(
            overall=round(overall, 3),
            length_score=round(length_score, 3),
            completeness_score=round(completeness_score, 3),
            information_density=round(density_score, 3),
            structure_score=round(structure_score, 3),
        )

    @classmethod
    def _score_length(cls, text: str) -> float:
        """Score based on text length."""
        length = len(text.strip())

        if length < 50:
            return 0.2
        elif length < cls.OPTIMAL_MIN_LENGTH:
            return 0.3 + (length / cls.OPTIMAL_MIN_LENGTH) * 0.3
        elif length <= cls.OPTIMAL_MAX_LENGTH:
            return 1.0
        elif length <= 1500:
            return 0.8
        else:
            return 0.6

    @classmethod
    def _score_completeness(cls, text: str) -> float:
        """Score based on sentence completeness."""
        score = 0.5

        stripped = text.strip()

        # Check for complete sentences
        if stripped and stripped[-1] in '.!?':
            score += 0.2

        # Check for incomplete start
        if stripped and stripped[0].islower():
            score -= 0.1

        # Count sentences
        sentences = re.split(r'[.!?]+', stripped)
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 20)

        if complete_sentences >= 2:
            score += 0.2
        elif complete_sentences >= 1:
            score += 0.1

        return min(1.0, max(0.0, score))

    @classmethod
    def _score_information_density(cls, text: str, entities: List[Dict],
                                    keywords: List[str]) -> float:
        """Score based on information density."""
        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        # Entity density
        entity_count = len(entities)
        entity_density = entity_count / (word_count / 50)  # Per 50 words
        entity_score = min(1.0, entity_density / 3)  # Optimal: 3 entities per 50 words

        # Keyword density
        keyword_score = min(1.0, len(keywords) / 10)  # Optimal: 10 keywords

        # Unique word ratio
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        unique_score = min(1.0, unique_ratio / 0.5)  # Optimal: 50% unique

        return (entity_score * 0.4 + keyword_score * 0.3 + unique_score * 0.3)

    @classmethod
    def _score_structure(cls, text: str, structure: str) -> float:
        """Score based on structure quality."""
        scores = {
            "bullet_list": 0.9,
            "key_value": 0.85,
            "comma_list": 0.8,
            "table": 0.75,
            "prose": 0.7,
        }

        base_score = scores.get(structure, 0.6)

        # Bonus for clear formatting
        lines = text.split('\n')
        if len(lines) > 1:
            avg_line_length = sum(len(l) for l in lines) / len(lines)
            if 30 < avg_line_length < 150:
                base_score += 0.05

        return min(1.0, base_score)

# =============================================================================
# EMBEDDING TEXT BUILDER
# =============================================================================

class EmbeddingTextBuilder:
    """Builds optimized text for embedding with context enrichment."""

    # Domain-aware templates
    TEMPLATES = {
        "contact_info": "Contact Information: {content}",
        "skills_list": "Skills and Competencies: {content}",
        "education": "Education and Qualifications: {content}",
        "experience": "Professional Experience: {content}",
        "financial": "Financial Information: {content}",
        "legal": "Legal Terms: {content}",
        "medical": "Medical Information: {content}",
        "tabular": "Data: {content}",
        "narrative": "{section_context}{content}",
    }

    @classmethod
    def build(cls, text: str, content_type: str, section_title: str = "",
              section_path: str = "", document_type: str = "") -> str:
        """
        Build optimized embedding text with context.

        Args:
            text: Original chunk text
            content_type: Detected content type
            section_title: Section heading
            section_path: Full section path
            document_type: Type of document

        Returns:
            Context-enriched text optimized for embedding
        """
        # Get appropriate template
        template = cls.TEMPLATES.get(content_type, cls.TEMPLATES["narrative"])

        # Build section context
        section_context = ""
        if section_title:
            section_context = f"{section_title}: "
        elif section_path:
            section_context = f"{section_path}: "

        # Clean content
        content = cls._clean_for_embedding(text)

        # Apply template
        embedding_text = template.format(
            content=content,
            section_context=section_context,
            section_title=section_title or "",
            section_path=section_path or "",
        )

        return embedding_text.strip()

    @classmethod
    def _clean_for_embedding(cls, text: str) -> str:
        """Clean text for embedding while preserving meaning."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page markers
        text = re.sub(r'---\s*Page\s*\d+\s*---', '', text, flags=re.IGNORECASE)

        # Remove extraction artifacts
        text = re.sub(r"(Extracted Document|full_text=)['\"]?", '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")

        return text.strip()

# =============================================================================
# QUERY ENRICHMENT
# =============================================================================

class QueryEnricher:
    """Enriches queries for better embedding match with documents."""

    # Query type patterns
    QUERY_PATTERNS = {
        "ranking": r'\b(rank|top|best|compare|order by|sort)\b',
        "contact": r'\b(email|phone|contact|reach|call)\b',
        "skills": r'\b(skills?|technologies?|proficient|experience with|knows?)\b',
        "education": r'\b(education|degree|university|college|studied|graduated)\b',
        "experience": r'\b(experience|worked|years?|employment|job history)\b',
        "financial": r'\b(amount|total|invoice|payment|cost|price)\b',
        "legal": r'\b(clause|terms|agreement|contract|liability)\b',
    }

    @classmethod
    def enrich(cls, query: str) -> Dict[str, Any]:
        """
        Enrich query for better matching.

        Returns:
            Dict with enriched_query, query_type, and expansion_terms
        """
        query_lower = query.lower()

        # Detect query type
        query_type = "general"
        for qtype, pattern in cls.QUERY_PATTERNS.items():
            if re.search(pattern, query_lower):
                query_type = qtype
                break

        # Get expansion terms based on query type
        expansion_terms = cls._get_expansion_terms(query_type)

        # Build enriched query
        enriched = cls._build_enriched_query(query, query_type, expansion_terms)

        return {
            "original_query": query,
            "enriched_query": enriched,
            "query_type": query_type,
            "expansion_terms": expansion_terms,
        }

    @classmethod
    def _get_expansion_terms(cls, query_type: str) -> List[str]:
        """Get expansion terms for query type."""
        expansions = {
            "ranking": ["experience", "skills", "qualifications", "years"],
            "contact": ["email", "phone", "linkedin", "address", "reach"],
            "skills": ["technologies", "programming", "tools", "proficient", "expertise"],
            "education": ["degree", "university", "college", "graduated", "studied", "major"],
            "experience": ["worked", "employed", "position", "role", "company", "years"],
            "financial": ["amount", "total", "payment", "invoice", "due", "balance"],
            "legal": ["clause", "terms", "agreement", "liability", "warranty", "obligations"],
        }
        return expansions.get(query_type, [])

    @classmethod
    def _build_enriched_query(cls, query: str, query_type: str,
                               expansion_terms: List[str]) -> str:
        """Build enriched query with context."""
        # Add query type context
        prefixes = {
            "ranking": "Rank and compare candidates based on: ",
            "contact": "Contact information including: ",
            "skills": "Technical skills and competencies: ",
            "education": "Educational background and qualifications: ",
            "experience": "Professional experience and work history: ",
            "financial": "Financial details including: ",
            "legal": "Legal terms and clauses regarding: ",
        }

        prefix = prefixes.get(query_type, "")

        return f"{prefix}{query}".strip()

# =============================================================================
# MAIN ENHANCER
# =============================================================================

@dataclass
class EnhancedEmbeddingResult:
    """Result of embedding enhancement."""
    embedding_text: str
    canonical_text: str
    content_type: str
    structure: str
    quality_score: QualityScore
    semantic_fields: Dict[str, Any]
    content_hash: str
    metadata: Dict[str, Any]

class UniversalEmbeddingEnhancer:
    """
    Universal enhancer for all document types.

    Provides comprehensive embedding improvements including:
    - Content type detection
    - Semantic field extraction
    - Quality scoring
    - Context-enriched embedding text
    - Query enrichment
    """

    def __init__(self, embedder=None):
        self.content_detector = ContentTypeDetector(embedder=embedder)
        self.field_extractor = SemanticFieldExtractor()
        self.quality_scorer = QualityScorer()
        self.text_builder = EmbeddingTextBuilder()
        self.query_enricher = QueryEnricher()

    def enhance_chunk(
        self,
        text: str,
        section_title: str = "",
        section_path: str = "",
        document_type: str = "",
        document_domain: str = "",
        _content_info: Optional[Dict[str, Any]] = None,
    ) -> EnhancedEmbeddingResult:
        """
        Enhance a single chunk for embedding.

        Args:
            text: Raw chunk text
            section_title: Section heading
            section_path: Full section hierarchy
            document_type: Document type (resume, invoice, etc.)
            document_domain: Document domain classification
            _content_info: Pre-computed content type info (from batch detection)

        Returns:
            EnhancedEmbeddingResult with all enhancements
        """
        # Use pre-computed content info if provided, otherwise detect
        if _content_info is not None:
            content_info = _content_info
        elif self.content_detector._embedder is not None:
            content_info = self.content_detector.detect_ml(text, section_title)
        else:
            content_info = self.content_detector.detect(text, section_title)
        content_type = content_info["content_type"]
        structure = content_info["structure"]

        # Extract semantic fields
        semantic_fields = self.field_extractor.extract(text, content_type)

        # Calculate quality score
        quality = self.quality_scorer.score(
            text=text,
            content_type=content_type,
            entities=semantic_fields["entities"],
            keywords=semantic_fields["keywords"],
            structure=structure,
        )

        # Build embedding text
        embedding_text = self.text_builder.build(
            text=text,
            content_type=content_type,
            section_title=section_title,
            section_path=section_path,
            document_type=document_type,
        )

        # Clean canonical text
        canonical_text = self.text_builder._clean_for_embedding(text)

        # Generate content hash
        content_hash = hashlib.sha256(canonical_text.encode()).hexdigest()[:16]

        return EnhancedEmbeddingResult(
            embedding_text=embedding_text,
            canonical_text=canonical_text,
            content_type=content_type,
            structure=structure,
            quality_score=quality,
            semantic_fields=semantic_fields,
            content_hash=content_hash,
            metadata={
                "content_confidence": content_info["confidence"],
                "detected_features": content_info.get("features", {}),
            },
        )

    def enhance_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        document_metadata: Dict[str, Any] = None,
    ) -> List[EnhancedEmbeddingResult]:
        """
        Enhance a batch of chunks.

        Args:
            chunks: List of chunk dicts with 'text', 'section_title', etc.
            document_metadata: Document-level metadata

        Returns:
            List of EnhancedEmbeddingResult
        """
        document_metadata = document_metadata or {}
        doc_type = document_metadata.get("document_type", "")
        doc_domain = document_metadata.get("document_domain", "")

        results = []
        for chunk in chunks:
            result = self.enhance_chunk(
                text=chunk.get("text", ""),
                section_title=chunk.get("section_title", ""),
                section_path=chunk.get("section_path", ""),
                document_type=doc_type,
                document_domain=doc_domain,
            )
            results.append(result)

        return results

    def enrich_query(self, query: str) -> Dict[str, Any]:
        """
        Enrich a query for better retrieval matching.

        Args:
            query: User query text

        Returns:
            Dict with enriched query and metadata
        """
        return self.query_enricher.enrich(query)

    def build_enhanced_payload(
        self,
        enhanced_result: EnhancedEmbeddingResult,
        base_payload: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Build enhanced Qdrant payload with semantic fields.

        Args:
            enhanced_result: Result from enhance_chunk
            base_payload: Existing payload to extend

        Returns:
            Enhanced payload dict for Qdrant
        """
        base_payload = base_payload or {}

        # Add semantic fields
        semantic = enhanced_result.semantic_fields

        enhanced_payload = {
            **base_payload,

            # Content classification
            "content_type": enhanced_result.content_type,
            "content_structure": enhanced_result.structure,
            "content_hash": enhanced_result.content_hash,

            # Quality metrics
            "quality_score": enhanced_result.quality_score.overall,
            "quality_metrics": enhanced_result.quality_score.to_dict(),

            # Semantic fields (for filtering and boosting)
            "semantic_keywords": semantic.get("keywords", [])[:15],
            "semantic_entities": semantic.get("entities", [])[:10],
            "semantic_phrases": semantic.get("phrases", [])[:10],
            "domain_terms": semantic.get("domain_terms", [])[:10],

            # Text variants
            "embedding_text": enhanced_result.embedding_text,
            "canonical_text": enhanced_result.canonical_text,
        }

        return enhanced_payload

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def enhance_for_embedding(
    text: str,
    section_title: str = "",
    section_path: str = "",
    document_type: str = "",
    embedder=None,
) -> EnhancedEmbeddingResult:
    """
    Convenience function to enhance a single chunk.

    Args:
        text: Chunk text
        section_title: Section heading
        section_path: Section hierarchy
        document_type: Document type
        embedder: Optional embedding model for ML content detection

    Returns:
        EnhancedEmbeddingResult
    """
    enhancer = UniversalEmbeddingEnhancer(embedder=embedder)
    return enhancer.enhance_chunk(
        text=text,
        section_title=section_title,
        section_path=section_path,
        document_type=document_type,
    )

def enrich_query(query: str) -> str:
    """
    Convenience function to enrich a query.

    Args:
        query: User query

    Returns:
        Enriched query text
    """
    enricher = QueryEnricher()
    result = enricher.enrich(query)
    return result["enriched_query"]

def get_enhanced_payload(
    text: str,
    section_title: str = "",
    base_payload: Dict[str, Any] = None,
    embedder=None,
) -> Dict[str, Any]:
    """
    Convenience function to get enhanced payload for Qdrant.

    Args:
        text: Chunk text
        section_title: Section heading
        base_payload: Existing payload to extend
        embedder: Optional embedding model for ML content detection

    Returns:
        Enhanced payload dict
    """
    enhancer = UniversalEmbeddingEnhancer(embedder=embedder)
    result = enhancer.enhance_chunk(text=text, section_title=section_title)
    return enhancer.build_enhanced_payload(result, base_payload)
