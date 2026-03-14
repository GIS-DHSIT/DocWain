"""
Document Classifier - Identifies document type and extracts type-specific metadata.

Enables the system to understand what kind of document is being processed and apply
appropriate extraction and embedding strategies.

Supported Types:
- RESUME / CV
- MEDICAL_RECORD
- LEGAL_DOCUMENT
- INVOICE
- PURCHASE_ORDER
- BANK_STATEMENT
- TAX_DOCUMENT
- POLICY
- GENERIC
"""

import re
from src.utils.logging_utils import get_logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

class DocumentType(Enum):
    """Canonical document types."""
    RESUME = "RESUME"
    CV = "CV"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    LEGAL_DOCUMENT = "LEGAL_DOCUMENT"
    INVOICE = "INVOICE"
    PURCHASE_ORDER = "PURCHASE_ORDER"
    BANK_STATEMENT = "BANK_STATEMENT"
    TAX_DOCUMENT = "TAX_DOCUMENT"
    POLICY = "POLICY"
    GENERIC = "GENERIC"

@dataclass
class DocumentClassification:
    """Result of document classification."""
    primary_type: DocumentType
    confidence: float  # 0.0-1.0
    secondary_types: List[Tuple[DocumentType, float]] = None
    domain: Optional[str] = None
    key_indicators: List[str] = None
    structured_fields: Dict[str, any] = None

    def __post_init__(self):
        if self.secondary_types is None:
            self.secondary_types = []
        if self.key_indicators is None:
            self.key_indicators = []
        if self.structured_fields is None:
            self.structured_fields = {}

class DocumentClassifier:
    """Classify documents by type and extract type-specific metadata."""

    # Resume/CV indicators
    RESUME_KEYWORDS = {
        "resume", "cv", "curriculum vitae", "c.v.", "c.v",
        "professional experience", "education", "skills", "technical skills",
        "qualifications", "objective", "summary", "career", "employment",
        "work experience", "certifications", "achievements", "references"
    }

    # Medical record indicators
    MEDICAL_KEYWORDS = {
        "patient", "diagnosis", "treatment", "medication", "prescription",
        "medical history", "vital signs", "clinical", "hospital", "physician",
        "doctor", "appointment", "disease", "symptom", "laboratory", "lab results",
        "radiology", "surgery", "procedure", "discharge summary", "admission"
    }

    # Legal document indicators
    LEGAL_KEYWORDS = {
        "agreement", "contract", "clause", "hereby", "whereas", "party",
        "parties", "terms and conditions", "legal", "jurisdiction", "liability",
        "indemnification", "termination", "section", "article", "hereto",
        "defendant", "plaintiff", "court", "law", "statute", "regulation",
        "legal document", "terms", "conditions", "covenant", "warranty"
    }

    # Invoice indicators
    INVOICE_KEYWORDS = {
        "invoice", "bill", "receipt", "order", "line item", "qty", "quantity",
        "unit price", "total", "subtotal", "tax", "amount due", "payment",
        "invoice number", "invoice date", "due date", "vendor", "seller",
        "buyer", "customer", "invoice total", "net total", "item", "description"
    }

    def __init__(self):
        """Initialize classifier."""
        self.type_patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[DocumentType, Dict]:
        """Build pattern matching rules for each document type."""
        return {
            DocumentType.RESUME: {
                "keywords": self.RESUME_KEYWORDS,
                "min_keywords": 3,
                "patterns": [
                    r"(Resume|CV|Curriculum Vitae)",
                    r"(Professional\s+Summary|Career\s+Objective|Objective Statement)",
                    r"(Work\s+Experience|Employment\s+History|Professional\s+Experience)",
                    r"(Education|Academic\s+Background|Qualifications)",
                    r"(Technical\s+Skills|Core\s+Competencies|Skills\s+&\s+Expertise)",
                ]
            },
            DocumentType.MEDICAL_RECORD: {
                "keywords": self.MEDICAL_KEYWORDS,
                "min_keywords": 4,
                "patterns": [
                    r"(Medical\s+History|Patient\s+History|Clinical\s+History)",
                    r"(Diagnosis|Chief\s+Complaint|Presenting\s+Problem)",
                    r"(Vital\s+Signs|Temperature|Blood\s+Pressure|Heart\s+Rate)",
                    r"(Medication|Prescription|Drug|Treatment)",
                    r"(Discharge\s+Summary|Hospital\s+Course)",
                ]
            },
            DocumentType.LEGAL_DOCUMENT: {
                "keywords": self.LEGAL_KEYWORDS,
                "min_keywords": 5,
                "patterns": [
                    r"(Agreement|Contract|Terms\s+and\s+Conditions)",
                    r"(Whereas|Party|Parties|Hereby)",
                    r"(Section|Article|Clause|Subsection)",
                    r"(Jurisdiction|Governing\s+Law|Applicable\s+Law)",
                    r"(Liability|Indemnification|Warranty|Covenant)",
                ]
            },
            DocumentType.INVOICE: {
                "keywords": self.INVOICE_KEYWORDS,
                "min_keywords": 5,
                "patterns": [
                    r"(Invoice|Bill|Receipt|Order)",
                    r"(Invoice\s+Number|Invoice\s+Date|Invoice\s+Total)",
                    r"(Line\s+Item|Item\s+Description|Qty|Quantity)",
                    r"(Unit\s+Price|Total\s+Amount|Subtotal|Tax|Amount\s+Due)",
                    r"(Due\s+Date|Payment\s+Terms|Terms)",
                ]
            }
        }

    def classify(self, text: str, metadata: Optional[Dict] = None) -> DocumentClassification:
        """
        Classify a document and extract type-specific information.

        Args:
            text: Document text content
            metadata: Optional metadata from document (filename, content_type, etc.)

        Returns:
            DocumentClassification with type, confidence, and indicators
        """

        if not text:
            return DocumentClassification(
                primary_type=DocumentType.GENERIC,
                confidence=0.0,
                key_indicators=["empty_document"]
            )

        # Score each document type
        scores = self._score_types(text)

        # Add metadata-based signals if available
        if metadata:
            scores = self._enhance_scores_with_metadata(scores, metadata)

        # Find primary and secondary types
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_scores:
            return DocumentClassification(
                primary_type=DocumentType.GENERIC,
                confidence=0.0
            )

        primary_type, primary_score = sorted_scores[0]
        secondary_types = [(t, s) for t, s in sorted_scores[1:3] if s > 0.3]

        # Get indicators and structured fields
        indicators = self._extract_indicators(text, primary_type)
        structured = self._extract_structured_fields(text, primary_type)

        return DocumentClassification(
            primary_type=primary_type,
            confidence=min(1.0, max(0.0, primary_score)),
            secondary_types=secondary_types,
            domain=self._infer_domain(primary_type),
            key_indicators=indicators,
            structured_fields=structured
        )

    def _score_types(self, text: str) -> Dict[DocumentType, float]:
        """Score each document type based on text content."""
        text_lower = text.lower()
        scores = {}

        for doc_type, patterns in self.type_patterns.items():
            score = 0.0

            # Count keyword matches
            keyword_matches = sum(1 for kw in patterns["keywords"] if kw in text_lower)
            if keyword_matches > 0:
                keyword_score = min(1.0, keyword_matches / patterns["min_keywords"])
                score += keyword_score * 0.4  # 40% weight

            # Count pattern matches
            pattern_matches = sum(1 for p in patterns["patterns"] if re.search(p, text, re.IGNORECASE))
            if pattern_matches > 0:
                pattern_score = min(1.0, pattern_matches / len(patterns["patterns"]))
                score += pattern_score * 0.6  # 60% weight

            scores[doc_type] = score

        return scores

    def _enhance_scores_with_metadata(self, scores: Dict, metadata: Dict) -> Dict:
        """Enhance classification scores using metadata signals."""
        filename = str(metadata.get("filename", "")).lower()
        content_type = str(metadata.get("content_type", "")).lower()

        # Boost scores based on filename
        if "resume" in filename or "cv" in filename:
            scores[DocumentType.RESUME] = scores.get(DocumentType.RESUME, 0) + 0.3
        elif "invoice" in filename or "bill" in filename:
            scores[DocumentType.INVOICE] = scores.get(DocumentType.INVOICE, 0) + 0.3
        elif "medical" in filename or "hospital" in filename:
            scores[DocumentType.MEDICAL_RECORD] = scores.get(DocumentType.MEDICAL_RECORD, 0) + 0.3
        elif "contract" in filename or "agreement" in filename:
            scores[DocumentType.LEGAL_DOCUMENT] = scores.get(DocumentType.LEGAL_DOCUMENT, 0) + 0.3

        return {k: min(1.0, v) for k, v in scores.items()}

    def _extract_indicators(self, text: str, doc_type: DocumentType) -> List[str]:
        """Extract key indicators for a document type."""
        indicators = []
        text_lower = text.lower()

        patterns = self.type_patterns.get(doc_type, {})

        # Find matching keywords
        matched_keywords = [kw for kw in patterns.get("keywords", []) if kw in text_lower]
        if matched_keywords:
            indicators.extend(matched_keywords[:5])  # Top 5 indicators

        # Add structural indicators
        if "\n" in text and text.count("\n") > 20:
            indicators.append("structured_layout")

        if re.search(r"\$\d+", text):
            indicators.append("currency_values")

        if re.search(r"\b\d{4}-\d{2}-\d{2}\b|\d{1,2}/\d{1,2}/\d{2,4}", text):
            indicators.append("date_fields")

        return indicators

    def _extract_structured_fields(self, text: str, doc_type: DocumentType) -> Dict:
        """Extract type-specific structured fields."""
        fields = {
            "type": doc_type.value,
            "layout_type": self._infer_layout(text),
            "estimated_pages": max(1, text.count("\n") // 50),
            "has_tables": "table" in text.lower() or "\t" in text,
            "has_lists": any(text.startswith(c * 2) for c in ["-", "*", "•"]),
        }

        # Type-specific fields
        if doc_type in [DocumentType.RESUME, DocumentType.CV]:
            fields["has_contact_info"] = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
            fields["has_skills_section"] = "skill" in text.lower()
            fields["has_education_section"] = "education" in text.lower()

        elif doc_type == DocumentType.INVOICE:
            fields["has_line_items"] = bool(re.search(r"\$\d+", text))
            fields["has_totals"] = "total" in text.lower()
            fields["has_dates"] = bool(re.search(r"\d{1,2}/\d{1,2}/\d{4}", text))

        elif doc_type == DocumentType.MEDICAL_RECORD:
            fields["has_vital_signs"] = any(x in text.lower() for x in ["bp", "heart rate", "temperature"])
            fields["has_medications"] = "medication" in text.lower() or "rx" in text.lower()

        elif doc_type == DocumentType.LEGAL_DOCUMENT:
            fields["has_clauses"] = any(x in text.lower() for x in ["section", "article", "clause"])
            fields["has_signatures"] = "_" * 10 in text or "signature" in text.lower()

        return fields

    def _infer_layout(self, text: str) -> str:
        """Infer document layout type."""
        lines = text.split("\n")

        if len(lines) < 20:
            return "simple_text"

        indentation_ratio = sum(1 for line in lines if line.startswith((" ", "\t"))) / len(lines)
        if indentation_ratio > 0.5:
            return "structured_indented"

        if any("\t" in line for line in lines):
            return "tabular"

        if sum(1 for line in lines if len(line) > 100) > len(lines) * 0.7:
            return "prose_heavy"

        return "mixed_layout"

    def _infer_domain(self, doc_type: DocumentType) -> Optional[str]:
        """Infer business domain from document type."""
        domain_map = {
            DocumentType.RESUME: "hr",
            DocumentType.CV: "hr",
            DocumentType.MEDICAL_RECORD: "healthcare",
            DocumentType.LEGAL_DOCUMENT: "legal",
            DocumentType.INVOICE: "finance",
            DocumentType.PURCHASE_ORDER: "procurement",
            DocumentType.BANK_STATEMENT: "finance",
            DocumentType.TAX_DOCUMENT: "finance",
            DocumentType.POLICY: "insurance",
        }
        return domain_map.get(doc_type)

# Singleton instance
_classifier_instance = None

def get_document_classifier() -> DocumentClassifier:
    """Get or create singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DocumentClassifier()
    return _classifier_instance

