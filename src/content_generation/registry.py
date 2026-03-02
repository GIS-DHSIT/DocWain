"""Content type registry — maps content type IDs to metadata and configuration.

Each content type defines what kind of document-grounded content can be generated,
which domain it belongs to, and what fields/facts are needed from source documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ContentType:
    """Metadata for a single content type."""

    id: str
    domain: str
    name: str
    description: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    min_chunks: int = 1
    supports_multi_doc: bool = False


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAINS = {
    "hr": "Human Resources / Resume",
    "invoice": "Invoice / Finance",
    "legal": "Legal / Contract",
    "medical": "Medical / Healthcare",
    "report": "Report / Analysis",
    "general": "General Purpose",
    "cross_document": "Cross-Document",
}


# ---------------------------------------------------------------------------
# Content type registry
# ---------------------------------------------------------------------------

CONTENT_TYPE_REGISTRY: Dict[str, ContentType] = {}


def _register(ct: ContentType) -> ContentType:
    CONTENT_TYPE_REGISTRY[ct.id] = ct
    return ct


# -- HR / Resume domain ----------------------------------------------------

_register(ContentType(
    id="cover_letter",
    domain="hr",
    name="Cover Letter",
    description="Professional cover letter grounded in resume evidence.",
    required_fields=["person_name", "skills"],
    optional_fields=["organizations", "experience_years", "education"],
))

_register(ContentType(
    id="professional_summary",
    domain="hr",
    name="Professional Summary",
    description="Concise career summary highlighting key qualifications.",
    required_fields=["person_name"],
    optional_fields=["skills", "experience_years", "organizations", "certifications"],
))

_register(ContentType(
    id="skills_matrix",
    domain="hr",
    name="Skills Matrix",
    description="Structured skills breakdown with proficiency categories.",
    required_fields=["skills"],
    optional_fields=["certifications", "organizations"],
))

_register(ContentType(
    id="candidate_comparison",
    domain="hr",
    name="Candidate Comparison",
    description="Side-by-side comparison of multiple candidates.",
    required_fields=["person_name", "skills"],
    supports_multi_doc=True,
    min_chunks=2,
))

_register(ContentType(
    id="interview_prep",
    domain="hr",
    name="Interview Preparation Guide",
    description="Interview talking points based on candidate profile.",
    required_fields=["person_name", "skills"],
    optional_fields=["organizations", "experience_years", "education"],
))

# -- Invoice / Finance domain -----------------------------------------------

_register(ContentType(
    id="invoice_summary",
    domain="invoice",
    name="Invoice Summary",
    description="Summary of invoice details including totals and line items.",
    required_fields=["amounts"],
    optional_fields=["vendor_name", "dates", "line_items"],
))

_register(ContentType(
    id="expense_report",
    domain="invoice",
    name="Expense Report",
    description="Consolidated expense report from multiple invoices.",
    required_fields=["amounts"],
    optional_fields=["vendor_name", "dates", "categories"],
    supports_multi_doc=True,
))

_register(ContentType(
    id="payment_reminder",
    domain="invoice",
    name="Payment Reminder",
    description="Professional payment reminder letter grounded in invoice data.",
    required_fields=["amounts", "vendor_name"],
    optional_fields=["dates", "invoice_number"],
))

# -- Legal / Contract domain ------------------------------------------------

_register(ContentType(
    id="contract_summary",
    domain="legal",
    name="Contract Summary",
    description="Plain-language summary of contract terms and obligations.",
    required_fields=["parties"],
    optional_fields=["dates", "terms", "obligations"],
))

_register(ContentType(
    id="compliance_report",
    domain="legal",
    name="Compliance Report",
    description="Compliance assessment against regulatory requirements.",
    required_fields=["terms"],
    optional_fields=["regulations", "obligations", "risks"],
))

_register(ContentType(
    id="risk_assessment",
    domain="legal",
    name="Risk Assessment",
    description="Analysis of contractual or legal risks.",
    required_fields=["terms"],
    optional_fields=["obligations", "penalties", "dates"],
))

# -- Medical / Healthcare domain --------------------------------------------

_register(ContentType(
    id="patient_summary",
    domain="medical",
    name="Patient Summary",
    description="Structured patient information summary from medical records.",
    required_fields=["patient_name"],
    optional_fields=["diagnoses", "medications", "dates", "vitals"],
))

_register(ContentType(
    id="medical_report",
    domain="medical",
    name="Medical Report",
    description="Formatted medical report from clinical document evidence.",
    required_fields=["patient_name"],
    optional_fields=["diagnoses", "medications", "procedures", "dates"],
))

# -- Report / Analysis domain -----------------------------------------------

_register(ContentType(
    id="executive_summary",
    domain="report",
    name="Executive Summary",
    description="High-level executive summary of key findings.",
    required_fields=[],
    optional_fields=["key_findings", "recommendations", "metrics"],
))

_register(ContentType(
    id="key_findings",
    domain="report",
    name="Key Findings Report",
    description="Structured report of key findings from document analysis.",
    required_fields=[],
    optional_fields=["metrics", "trends", "insights"],
))

_register(ContentType(
    id="recommendations",
    domain="report",
    name="Recommendations",
    description="Actionable recommendations based on document evidence.",
    required_fields=[],
    optional_fields=["key_findings", "metrics", "constraints"],
))

# -- General domain ---------------------------------------------------------

_register(ContentType(
    id="document_summary",
    domain="general",
    name="Document Summary",
    description="Concise summary of document contents.",
    required_fields=[],
    min_chunks=1,
))

_register(ContentType(
    id="key_points",
    domain="general",
    name="Key Points",
    description="Bullet-point extraction of key information.",
    required_fields=[],
))

_register(ContentType(
    id="faq_generation",
    domain="general",
    name="FAQ Generation",
    description="Generate frequently asked questions and answers from document.",
    required_fields=[],
))

_register(ContentType(
    id="action_items",
    domain="general",
    name="Action Items",
    description="Extract and format actionable items from documents.",
    required_fields=[],
))

_register(ContentType(
    id="talking_points",
    domain="general",
    name="Talking Points",
    description="Key talking points extracted from document evidence.",
    required_fields=[],
))

# -- Cross-document domain --------------------------------------------------

_register(ContentType(
    id="comparison_report",
    domain="cross_document",
    name="Comparison Report",
    description="Detailed comparison across multiple documents.",
    required_fields=[],
    supports_multi_doc=True,
    min_chunks=2,
))

_register(ContentType(
    id="consolidated_summary",
    domain="cross_document",
    name="Consolidated Summary",
    description="Unified summary across multiple documents.",
    required_fields=[],
    supports_multi_doc=True,
    min_chunks=2,
))

_register(ContentType(
    id="trend_analysis",
    domain="cross_document",
    name="Trend Analysis",
    description="Identify trends and patterns across multiple documents.",
    required_fields=[],
    supports_multi_doc=True,
    min_chunks=2,
))

_register(ContentType(
    id="meeting_notes",
    domain="general",
    name="Meeting Notes",
    description="Structured meeting notes with key decisions and action items.",
    required_fields=[],
    optional_fields=["dates", "participants", "decisions"],
))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_content_type(type_id: str) -> Optional[ContentType]:
    """Retrieve a content type by its ID."""
    return CONTENT_TYPE_REGISTRY.get(type_id)


def list_content_types(domain: Optional[str] = None) -> List[ContentType]:
    """List all content types, optionally filtered by domain."""
    types = list(CONTENT_TYPE_REGISTRY.values())
    if domain:
        types = [ct for ct in types if ct.domain == domain]
    return types


def list_domains() -> List[Dict[str, str]]:
    """Return all domains with their descriptions."""
    return [{"id": k, "name": v} for k, v in DOMAINS.items()]


# ---------------------------------------------------------------------------
# Natural language → content type detection
# ---------------------------------------------------------------------------

_TYPE_PATTERNS: List[tuple[str, str]] = [
    # HR domain
    (r"\b(?:cover\s+letter|application\s+letter)\b", "cover_letter"),
    (r"\b(?:professional\s+summary|career\s+summary|profile\s+summary)\b", "professional_summary"),
    (r"\b(?:skills?\s+matrix|skills?\s+breakdown|competenc(?:y|ies)\s+matrix)\b", "skills_matrix"),
    (r"\b(?:candidate\s+comparison|compare\s+candidates?|candidate\s+vs)\b", "candidate_comparison"),
    (r"\b(?:interview\s+(?:prep|preparation|guide|questions))\b", "interview_prep"),
    # Invoice
    (r"\b(?:invoice\s+summary|summarize?\s+(?:the\s+)?invoices?)\b", "invoice_summary"),
    (r"\b(?:expense\s+report|spending\s+report|expenditure\s+report)\b", "expense_report"),
    (r"\b(?:payment\s+reminder|overdue\s+notice|payment\s+notice)\b", "payment_reminder"),
    # Legal
    (r"\b(?:contract\s+summary|summarize?\s+(?:the\s+)?contract)\b", "contract_summary"),
    (r"\b(?:compliance\s+report|regulatory\s+assessment)\b", "compliance_report"),
    (r"\b(?:risk\s+assessment|risk\s+analysis|risk\s+report)\b", "risk_assessment"),
    # Medical
    (r"\b(?:patient\s+summary|clinical\s+summary)\b", "patient_summary"),
    (r"\b(?:medical\s+report|clinical\s+report)\b", "medical_report"),
    # Report
    (r"\b(?:executive\s+summary|exec\s+summary)\b", "executive_summary"),
    (r"\b(?:key\s+findings?|main\s+findings?)\b", "key_findings"),
    (r"\b(?:recommend(?:ation)?s?\s+(?:report|list)?)\b", "recommendations"),
    # General
    (r"\b(?:document\s+summary|summarize?\s+(?:the\s+)?document)\b", "document_summary"),
    (r"\b(?:key\s+points?|main\s+points?|bullet\s+points?)\b", "key_points"),
    (r"\b(?:faq|frequently\s+asked|generate\s+questions?)\b", "faq_generation"),
    (r"\b(?:action\s+items?|to-?do\s+list|next\s+steps?)\b", "action_items"),
    (r"\b(?:talking\s+points?|discussion\s+points?)\b", "talking_points"),
    (r"\b(?:meeting\s+notes?|meeting\s+minutes?|meeting\s+summary)\b", "meeting_notes"),
    # Cross-document
    (r"\b(?:comparison\s+report|comparative\s+analysis)\b", "comparison_report"),
    (r"\b(?:consolidated\s+summary|combined\s+summary)\b", "consolidated_summary"),
    (r"\b(?:trend\s+analysis|trend\s+report)\b", "trend_analysis"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), tid) for p, tid in _TYPE_PATTERNS]


def detect_content_type(query: str) -> Optional[str]:
    """Detect content type ID from a natural language query.

    Returns the content type ID or None if no match.
    """
    if not query:
        return None
    for pattern, type_id in _COMPILED_PATTERNS:
        if pattern.search(query):
            return type_id
    return None


def detect_content_type_with_domain(
    query: str,
    chunk_domain: Optional[str] = None,
) -> Optional[ContentType]:
    """Detect content type and cross-reference with chunk domain.

    If chunks indicate a specific domain, prefer content types from that domain.
    Falls back to general types if no domain-specific match.
    """
    type_id = detect_content_type(query)
    if type_id:
        return CONTENT_TYPE_REGISTRY.get(type_id)

    # Fallback: if query mentions "generate" or "create" with a domain hint
    generate_match = re.search(
        r"\b(?:generate|create|write|draft|produce|build)\b",
        query,
        re.IGNORECASE,
    )
    if not generate_match:
        return None

    # Use domain hint to pick a default type
    domain_defaults = {
        "hr": "professional_summary",
        "resume": "professional_summary",
        "invoice": "invoice_summary",
        "legal": "contract_summary",
        "medical": "patient_summary",
        "report": "executive_summary",
    }
    if chunk_domain and chunk_domain in domain_defaults:
        return CONTENT_TYPE_REGISTRY.get(domain_defaults[chunk_domain])

    return CONTENT_TYPE_REGISTRY.get("document_summary")
