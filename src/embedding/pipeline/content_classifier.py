"""Consolidated content classifier for section-kind and document-domain.

Single source of truth — called at ingestion (payload builder, chunk metadata)
and at extraction time as a fallback.  Keeps keyword lists in one place so
ingestion and retrieval always agree.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from src.intelligence.domain_classifier import (
    DOC_TYPE_HINTS,
    classify_domain,
)


# ── Section-kind title patterns (priority 1) ────────────────────────────
# Checked first; a clear title beats noisy keyword counting.

_TITLE_MAP = [
    # Domain-specific patterns FIRST (more specific beats generic)
    # Invoice section kinds
    (("line item", "item detail"), "line_items"),
    (("bill to", "ship to", "sold to"), "parties_addresses"),
    (("payment term", "terms of payment"), "terms_conditions"),
    (("invoice summary", "billing summary"), "financial_summary"),
    (("tax detail", "tax breakdown"), "financial_summary"),
    # Legal section kinds
    (("governing law", "jurisdiction"), "legal_clauses"),
    (("indemnif",), "legal_clauses"),
    (("confidential", "non-disclosure"), "legal_clauses"),
    (("recital", "preamble", "whereas"), "legal_preamble"),
    (("definition",), "legal_definitions"),
    (("signature", "execution", "attestation"), "legal_signatures"),
    # Insurance/Policy section kinds (map to existing legal/financial kinds)
    (("coverage", "covered peril", "scope of coverage"), "legal_clauses"),
    (("exclusion", "excluded peril", "not covered"), "legal_clauses"),
    (("premium", "premium schedule", "premium calculation"), "financial_summary"),
    (("claim", "claims procedure", "how to claim"), "terms_conditions"),
    (("deductible", "excess"), "terms_conditions"),
    (("beneficiary", "insured", "policyholder"), "parties_addresses"),
    # Medical section kinds
    (("diagnosis", "assessment", "impression"), "medical_findings"),
    (("medication", "prescription", "drug"), "medical_medications"),
    (("lab result", "laboratory", "test result"), "medical_lab_results"),
    (("patient info", "patient detail", "demographics"), "medical_patient_info"),
    (("procedure", "operation", "intervention"), "medical_findings"),
    # Bank section kinds
    (("account summary", "account detail"), "financial_summary"),
    (("opening balance", "closing balance"), "financial_summary"),
    (("transaction history",), "transactions"),
    # Generic resume patterns (checked after domain-specific)
    (("summary", "objective", "profile", "overview", "about me"), "summary_objective"),
    (("contact",), "identity_contact"),
    (("education", "academic", "qualification"), "education"),
    (("certif", "credential", "license"), "certifications"),
    (("work experience", "employment", "career history", "professional experience"), "experience"),
    (("achievement", "award", "honor"), "achievements"),
    (("project",), "experience"),
    # Generic transaction (after specific "transaction history")
    (("transaction",), "transactions"),
]


def _title_match(section_title: str) -> Optional[str]:
    title_lower = section_title.lower()
    for needles, kind in _TITLE_MAP:
        if any(n in title_lower for n in needles):
            return kind
    # Special handling for "skills" — sub-classify
    if "skill" in title_lower:
        if any(w in title_lower for w in ("technical", "programming", "technology")):
            return "skills_technical"
        if any(w in title_lower for w in ("soft", "functional", "business")):
            return "skills_functional"
        return "skills_technical"
    return None


# ── Content-supports-title verification ──────────────────────────────
# When the title says "Education" but the chunk text is actually work
# experience (boundary crossing), the title classification is wrong.
# This gate verifies the content actually matches the title-derived kind.

_DATE_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)\d{2}|present|current)\b",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(
    r"\b(?:bachelor|master|phd|doctorate|b\.?tech|m\.?tech|b\.?s|m\.?s|b\.?a|m\.?a|mba|diploma)\b",
    re.IGNORECASE,
)

_CONTENT_SUPPORT_RULES: dict[str, list[set[str] | re.Pattern]] = {
    "education": [
        # At least 1 of these tokens/patterns required
        {"degree", "university", "college", "school", "gpa", "cgpa",
         "graduation", "bachelor", "master", "phd", "diploma",
         "b.tech", "m.tech", "btech", "mtech", "b.sc", "m.sc"},
    ],
    "skills_technical": [
        # At least 2 tech keywords required
        {"python", "java", "javascript", "typescript", "sql", "react",
         "angular", "vue", "node.js", "aws", "azure", "docker",
         "kubernetes", "git", "html", "css", "django", "flask",
         "spring", "mongodb", "postgresql", "tensorflow", "pytorch"},
    ],
    "identity_contact": [
        # At least 2 of these contextual signals
        {"email:", "phone:", "mobile:", "linkedin.com", "contact info",
         "contact details", "tel:", "github.com"},
    ],
}


def _content_supports_title(text: str, kind: str) -> bool:
    """Return True if chunk text has enough evidence to support *kind*.

    Only gates a few high-risk kinds where cross-boundary misclassification
    is common.  For ungated kinds the function returns True (trust the title).
    When text is empty or very short, trust the title — there's nothing to
    contradict it.
    """
    rules = _CONTENT_SUPPORT_RULES.get(kind)
    if rules is None:
        return True  # Not gated — trust title

    if not text or len(text.strip()) < 30:
        return True  # Not enough text to contradict the title

    text_lower = text.lower()

    if kind == "education":
        # Need at least 1 degree/institution keyword
        kw_set = rules[0]
        if any(kw in text_lower for kw in kw_set):
            return True
        if _DEGREE_RE.search(text):
            return True
        return False

    if kind == "skills_technical":
        kw_set = rules[0]
        hits = sum(1 for kw in kw_set if kw in text_lower)
        return hits >= 2

    if kind == "identity_contact":
        kw_set = rules[0]
        hits = sum(1 for kw in kw_set if kw in text_lower)
        return hits >= 2

    return True


# ── Section-kind content keywords (priority 2) ──────────────────────────

_CONTENT_KEYWORDS = {
    "skills_technical": {
        "python", "java", "c++", "javascript", "typescript", "golang", "rust",
        "sql", "mongodb", "postgresql", "mysql", "react", "angular", "vue",
        "node.js", "express", "django", "flask", "spring", "hibernate",
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
        "html", "css", "sass", "webpack", "npm", "yarn", "api", "rest",
        "microservices", "architecture", "design patterns", "solid", "agile",
        "scrum", "machine learning", "tensorflow", "pytorch", "nlp", "bert",
        "programming language", "software", "code", "development", "backend",
        "frontend", "fullstack", "devops", "cloud", "infrastructure",
    },
    "skills_functional": {
        "communication", "leadership", "teamwork", "project management",
        "stakeholder", "requirement", "analysis", "problem solving",
        "negotiation", "presentation", "documentation", "collaboration",
        "mentoring", "training", "process improvement", "quality assurance",
        "testing", "uat", "business", "sap", "salesforce", "crm", "erp",
        "supply chain", "operations", "finance", "accounting", "compliance",
    },
    "education": {
        "b.tech", "btech", "b.sc", "bsc", "b.a", "ba", "m.tech", "mtech",
        "m.sc", "msc", "m.a", "ma", "mba", "phd", "degree", "diploma",
        "university", "college", "school", "bachelor", "master", "graduation",
        "gpa", "cgpa", "specialization", "major", "minor", "coursework",
    },
    "certifications": {
        "certified", "certification", "credential", "certificate", "license",
        "pmp", "ccna", "rhce", "comptia", "licensed", "accredited",
        "board", "exam",
    },
    "experience": {
        "years of experience", "years exp", "experienced in", "worked on",
        "responsible for", "accomplished", "achieved", "delivered", "managed",
        "led team", "oversaw", "directed", "coordinated", "spearheaded",
        "professional experience", "work experience", "employment", "career",
        "developed", "implemented", "designed", "built", "created",
        "deployed", "automated", "optimized", "improved", "reduced",
        "increased", "intern", "engineer", "analyst", "developer",
        "consultant", "associate", "company", "organization", "remote",
    },
    "identity_contact": {
        "email:", "phone:", "mobile:", "linkedin.com", "contact info",
        "contact details", "tel:", "whatsapp:", "github.com/",
    },
    "achievements": {
        "award", "achievement", "accomplishment", "recognition", "winner",
        "best", "excellence", "honor", "distinction",
        "promoted", "commendation", "merit", "outstanding", "superior",
    },
    # Invoice section kinds
    "line_items": {
        "line item", "unit price", "quantity", "item description", "sku",
        "product code", "amount", "rate", "hours", "subtotal",
    },
    "parties_addresses": {
        "bill to", "ship to", "sold to", "vendor", "supplier", "buyer",
        "seller", "remit to", "billing address", "shipping address",
    },
    "terms_conditions": {
        "payment term", "net 30", "net 60", "due upon receipt", "warranty",
        "liability", "condition", "late fee", "penalty",
    },
    "financial_summary": {
        "subtotal", "grand total", "tax total", "amount due", "balance due",
        "total amount", "opening balance", "closing balance", "net total",
        "account summary", "statement summary",
    },
    "transactions": {
        "transaction", "debit", "credit", "withdrawal", "deposit",
        "transfer", "payment", "check number", "reference number",
    },
    # Legal section kinds
    "legal_clauses": {
        "governing law", "indemnification", "confidentiality", "arbitration",
        "limitation of liability", "force majeure", "termination",
        "non-compete", "intellectual property", "dispute resolution",
    },
    "legal_preamble": {
        "whereas", "recital", "preamble", "hereby", "hereinafter",
        "party of the first part", "entered into", "effective date",
    },
    "legal_definitions": {
        "shall mean", "defined as", "herein referred", "as used in",
        "interpretation", "definition",
    },
    "legal_signatures": {
        "signature", "witness", "notary", "executed", "authorized signatory",
        "in witness whereof", "duly authorized",
    },
    # Medical section kinds
    "medical_findings": {
        "diagnosis", "assessment", "impression", "clinical finding",
        "chief complaint", "history of present illness", "review of systems",
        "physical examination", "prognosis",
    },
    "medical_medications": {
        "medication", "prescription", "dosage", "drug", "frequency",
        "route of administration", "pharmacy", "refill",
    },
    "medical_lab_results": {
        "lab result", "blood test", "urinalysis", "hemoglobin", "glucose",
        "cholesterol", "white blood cell", "platelet", "reference range",
    },
    "medical_patient_info": {
        "patient name", "date of birth", "medical record number", "mrn",
        "insurance", "allergies", "emergency contact",
    },
}

_MIN_KEYWORD_SCORE = 2  # Require ≥2 matches for a confident content-based label


def classify_section_kind_with_source(
    text: str, section_title: str = "",
) -> Tuple[str, str]:
    """Classify chunk content into a section kind and return confidence source.

    Returns ``(kind, source)`` where *source* is ``"title"`` when the kind
    was derived from a clear section title match (high confidence),
    ``"content"`` when it was derived from keyword scoring (lower confidence).
    """
    # Priority 1 — title-based (fast, high-confidence)
    if section_title:
        title_kind = _title_match(section_title)
        if title_kind:
            if _content_supports_title(text, title_kind):
                return title_kind, "title"
            # Title says one thing but content doesn't match — fall through
            # to content-based scoring for a more accurate classification.

    # Priority 2 — content keyword scoring
    if not text:
        return "section_text", "content"

    combined = f"{section_title} {text}".lower()
    scores = {
        kind: sum(1 for kw in keywords if kw in combined)
        for kind, keywords in _CONTENT_KEYWORDS.items()
    }

    max_score = max(scores.values()) if scores else 0
    if max_score < _MIN_KEYWORD_SCORE:
        return "section_text", "content"

    top = [k for k, v in scores.items() if v == max_score]
    if len(top) == 1:
        return top[0], "content"

    # Tie-breaker — prefer the one that echoes the section title
    if section_title:
        for kind in top:
            if kind.replace("_", " ") in section_title.lower():
                return kind, "content"

    # Tie-breaker — date ranges strongly indicate experience
    if "experience" in top and _DATE_RANGE_RE.search(combined):
        return "experience", "content"

    # Tie-breaker — degree keywords strongly indicate education
    if "education" in top and _DEGREE_RE.search(combined):
        return "education", "content"

    # Tie-breaker — if >50% commas suggest a skill list
    if "skills_technical" in top:
        tokens = combined.split(",")
        if len(tokens) >= 4:
            avg_len = sum(len(t.strip()) for t in tokens) / len(tokens)
            if avg_len < 30:
                return "skills_technical", "content"

    return top[0], "content"


def classify_section_kind(text: str, section_title: str = "") -> str:
    """Classify chunk content into a section kind.

    Returns one of: summary_objective, experience, skills_technical,
    skills_functional, education, certifications, achievements,
    identity_contact, section_text.
    """
    kind, _source = classify_section_kind_with_source(text, section_title)
    return kind


# ── Document domain classification ──────────────────────────────────────

_FILENAME_HINTS = {
    "resume": "resume",
    "cv": "resume",
    "profile": "resume",
    "invoice": "invoice",
    "inv": "invoice",
    "purchase_order": "purchase_order",
    "bank_statement": "bank_statement",
    "statement": "bank_statement",
    "medical": "medical",
    "contract": "legal",
    "agreement": "legal",
    "tax": "tax",
}


def classify_doc_domain(
    text: str,
    filename: str = "",
    doc_type: str = "",
) -> str:
    """Classify document domain from content + metadata hints.

    Returns one of: resume, invoice, purchase_order, bank_statement,
    medical, legal, tax, generic.
    """
    # Quick check — filename hint (e.g., "John_Resume.pdf" → resume)
    if filename:
        fn_lower = filename.lower()
        for hint, domain in _FILENAME_HINTS.items():
            if hint in fn_lower:
                return domain

    # Quick check — doc_type hint
    if doc_type:
        dt_lower = doc_type.lower().strip()
        mapped = DOC_TYPE_HINTS.get(dt_lower)
        if mapped:
            return mapped

    # Full classification via existing domain_classifier
    result = classify_domain(
        text or "",
        metadata={"source_name": filename, "doc_type": doc_type},
    )
    # When the classifier is uncertain (low confidence / small gap between
    # top domains), cross-check with keyword fallback.  If keyword matching
    # also finds no strong indicators, fall back to "generic" instead of
    # returning a random domain that happened to score marginally highest.
    if result.uncertain:
        from src.intelligence.domain_classifier import _keyword_fallback_classify
        kw = _keyword_fallback_classify(text or "", {"source_name": filename, "doc_type": doc_type})
        if not kw.uncertain:
            return kw.domain  # keyword evidence is strong — trust it
        return "generic"
    return result.domain


__all__ = ["classify_section_kind", "classify_section_kind_with_source", "classify_doc_domain"]
