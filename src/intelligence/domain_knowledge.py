"""Domain Knowledge Engine — professional domain expertise for document intelligence.

Provides domain-specific analytical context so DocWain understands *how*
professionals analyze documents in each domain:
  - HR/Recruitment: What recruiters look for in resumes, screening criteria
  - Procurement/Invoice: Purchase-to-pay process, invoice validation rules
  - Medical: Region-aware clinical standards, diagnostic context
  - Legal: Contract review methodology, risk assessment frameworks
  - Insurance/Policy: Coverage analysis, claims evaluation criteria

The engine has three knowledge tiers:
  1. **Embedded expertise** — hardcoded professional knowledge (always available)
  2. **Web-enriched context** — supplementary domain knowledge fetched from the
     internet on demand (optional, does NOT affect document content)
  3. **Region detection** — for medical documents, detects the country of origin
     and applies region-specific clinical standards

Usage::

    from src.intelligence.domain_knowledge import get_domain_knowledge_provider

    provider = get_domain_knowledge_provider()
    context = provider.get_knowledge("hr", query="rank candidates by skills")
    # → DomainContext with analytical_perspective, evaluation_criteria, etc.
"""
from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

# ── Region / Country Detection Patterns ──────────────────────────────

_COUNTRY_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # USA
    (re.compile(r"\b(?:United\s+States|USA|U\.S\.A\.|US(?:\s+(?:citizen|resident|national)))\b", re.I), "US", "United States"),
    (re.compile(r"\b(?:FDA|CDC|CMS|NIH|HIPAA|Medicare|Medicaid)\b"), "US", "United States"),
    (re.compile(r"\b(?:mg/dL|lbs?|pounds?)\b", re.I), "US", "United States"),
    # UK
    (re.compile(r"\b(?:United\s+Kingdom|UK|U\.K\.|NHS|NICE\s+guidelines?|BNF)\b", re.I), "UK", "United Kingdom"),
    (re.compile(r"\b(?:GP\s+(?:surgery|practice)|A&E|consultant|registrar)\b", re.I), "UK", "United Kingdom"),
    (re.compile(r"\bmg/dL\b.*\bmmol/[Ll]\b|\bmmol/[Ll]\b", re.I), "UK", "United Kingdom"),
    # India
    (re.compile(r"\b(?:India|Indian|AYUSH|Ayurveda|MBBS|AIIMS|ICMR)\b", re.I), "IN", "India"),
    (re.compile(r"\b(?:INR|Rs\.?|Rupees?)\b.*(?:hospital|clinic|medical)", re.I), "IN", "India"),
    # EU / Europe
    (re.compile(r"\b(?:EU|EMA|European\s+(?:Medicines?\s+Agency|Union))\b", re.I), "EU", "European Union"),
    # Canada
    (re.compile(r"\b(?:Canada|Canadian|Health\s+Canada|OHIP)\b", re.I), "CA", "Canada"),
    # Australia
    (re.compile(r"\b(?:Australia|Australian|TGA|Medicare\s+(?:Australia|card)|PBS)\b", re.I), "AU", "Australia"),
    # General / Default
    (re.compile(r"\b(?:WHO|ICD-1[01]|SNOMED|HL7|FHIR)\b", re.I), "INT", "International"),
]

_CURRENCY_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"[$]\s*[\d,]+(?:\.\d{2})?"), "US", "United States"),
    (re.compile(r"[£]\s*[\d,]+(?:\.\d{2})?"), "UK", "United Kingdom"),
    (re.compile(r"[€]\s*[\d,]+(?:\.\d{2})?"), "EU", "European Union"),
    (re.compile(r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{2})?"), "IN", "India"),
    (re.compile(r"(?:CAD|C\$)\s*[\d,]+(?:\.\d{2})?"), "CA", "Canada"),
    (re.compile(r"(?:AUD|A\$)\s*[\d,]+(?:\.\d{2})?"), "AU", "Australia"),
]

def detect_country_of_origin(text: str) -> Tuple[str, str, float]:
    """Detect the likely country of origin from document text.

    Returns (country_code, country_name, confidence).
    """
    if not text:
        return ("INT", "International", 0.0)

    sample = text[:8000]
    votes: Dict[str, int] = {}

    for pattern, code, _name in _COUNTRY_PATTERNS:
        matches = pattern.findall(sample)
        if matches:
            votes[code] = votes.get(code, 0) + len(matches)

    for pattern, code, _name in _CURRENCY_PATTERNS:
        matches = pattern.findall(sample)
        if matches:
            votes[code] = votes.get(code, 0) + len(matches)

    if not votes:
        return ("INT", "International", 0.2)

    total = sum(votes.values())
    best_code = max(votes, key=votes.get)
    confidence = min(votes[best_code] / max(total, 1), 1.0)

    name_map = {"US": "United States", "UK": "United Kingdom", "IN": "India",
                "EU": "European Union", "CA": "Canada", "AU": "Australia",
                "INT": "International"}
    return (best_code, name_map.get(best_code, best_code), round(confidence, 2))

# ── Domain Context Dataclass ─────────────────────────────────────────

@dataclass
class DomainContext:
    """Professional domain knowledge context for a specific query."""

    domain: str
    analytical_perspective: str
    evaluation_criteria: List[str]
    key_indicators: List[str]
    professional_terminology: List[str]
    common_patterns: List[str]
    red_flags: List[str]
    region_context: Optional[str] = None
    web_supplement: Optional[str] = None
    cache_key: str = ""

    def to_prompt_section(self) -> str:
        """Render as a prompt injection section for LLM calls."""
        parts = [f"DOMAIN EXPERTISE ({self.domain.upper()})"]
        parts.append(self.analytical_perspective)

        if self.evaluation_criteria:
            parts.append("\nEvaluation Criteria:")
            for c in self.evaluation_criteria[:8]:
                parts.append(f"  - {c}")

        if self.key_indicators:
            parts.append("\nKey Indicators to Analyze:")
            for k in self.key_indicators[:8]:
                parts.append(f"  - {k}")

        if self.red_flags:
            parts.append("\nRed Flags / Watch Points:")
            for r in self.red_flags[:6]:
                parts.append(f"  - {r}")

        if self.region_context:
            parts.append(f"\nRegion-Specific Context:\n{self.region_context}")

        if self.web_supplement:
            parts.append(f"\nSupplementary Domain Context:\n{self.web_supplement}")

        return "\n".join(parts)

    def to_brief(self) -> str:
        """Render a brief context string for rendering pipeline."""
        parts = [self.analytical_perspective]
        if self.evaluation_criteria:
            parts.append("Evaluate: " + "; ".join(self.evaluation_criteria[:4]))
        if self.region_context:
            parts.append(self.region_context)
        return "\n".join(parts)

# ── Embedded Domain Knowledge Base ───────────────────────────────────

_HR_KNOWLEDGE = DomainContext(
    domain="hr",
    analytical_perspective=(
        "Analyze as a senior talent acquisition specialist and HR analyst. "
        "Evaluate candidates the way an experienced recruiter would: assessing "
        "career trajectory, skill-role fit, growth velocity, and cultural indicators. "
        "Look beyond listed skills to infer competency depth from project descriptions "
        "and achievement quantification."
    ),
    evaluation_criteria=[
        "Career progression velocity (promotions, role changes, scope expansion)",
        "Skill depth vs breadth balance (specialist vs generalist profile)",
        "Achievement quantification (metrics, outcomes, impact statements)",
        "Education-to-role alignment and continuous learning indicators",
        "Industry diversity and domain expertise depth",
        "Leadership and team management indicators",
        "Technical skill currency (how recent are certifications and technologies)",
        "Cultural fit indicators (volunteer work, cross-functional projects, communication style)",
    ],
    key_indicators=[
        "Years of progressive experience in target domain",
        "Quantified achievements (revenue, cost savings, team size, throughput)",
        "Certification recency and relevance to target role",
        "Skill overlap with job requirements (primary, secondary, adjacent)",
        "Employment gap analysis and career continuity",
        "Project complexity and scope indicators",
        "Cross-functional collaboration evidence",
        "Technical depth markers (patents, publications, open-source contributions)",
    ],
    professional_terminology=[
        "ATS optimization", "competency mapping", "skill gap analysis",
        "career trajectory", "role-fit score", "talent pipeline",
        "passive candidate", "employer branding", "succession planning",
        "behavioral indicators", "cultural alignment", "retention risk",
    ],
    common_patterns=[
        "Strong candidates quantify achievements with specific metrics",
        "Career gaps over 6 months warrant explanation in context",
        "Frequent job changes (<1 year) may indicate either high demand or instability",
        "Certifications without practical application may indicate theoretical-only knowledge",
        "Cross-industry experience often brings valuable diverse perspectives",
        "Candidates with mentoring/training experience show leadership readiness",
    ],
    red_flags=[
        "Inconsistent dates or overlapping employment periods",
        "Vague achievement descriptions without quantification",
        "Missing education details for roles that typically require degrees",
        "Skills listed without supporting experience or project context",
        "Downward career trajectory without clear career change rationale",
        "Contact information missing or incomplete",
    ],
)

_INVOICE_KNOWLEDGE = DomainContext(
    domain="invoice",
    analytical_perspective=(
        "Analyze as a procurement and accounts payable specialist. "
        "Evaluate invoices through the lens of the purchase-to-pay (P2P) cycle: "
        "purchase order matching, goods receipt verification, invoice validation, "
        "approval workflow, and payment processing. Identify discrepancies, "
        "compliance issues, and optimization opportunities."
    ),
    evaluation_criteria=[
        "Three-way match validation (PO, goods receipt, invoice)",
        "Tax calculation accuracy (VAT/GST rates, taxable vs exempt items)",
        "Payment terms compliance and early payment discount opportunities",
        "Vendor information completeness and accuracy",
        "Line item pricing consistency with contracted rates",
        "Currency and exchange rate accuracy for international invoices",
        "Duplicate invoice detection across vendor and date combinations",
        "Regulatory compliance (tax registration, invoice numbering sequence)",
    ],
    key_indicators=[
        "Invoice number, date, and due date sequence",
        "Vendor/supplier identification and registration details",
        "Line item descriptions, quantities, unit prices, and totals",
        "Tax breakdown (rate, base amount, tax amount)",
        "Discount terms and early payment incentives",
        "Purchase order reference numbers",
        "Delivery/shipping details and charges",
        "Total amount, subtotal, tax, and net payable reconciliation",
    ],
    professional_terminology=[
        "purchase-to-pay cycle", "three-way match", "goods receipt note",
        "accounts payable", "vendor master data", "payment terms",
        "early payment discount", "invoice validation", "approval workflow",
        "credit memo", "debit note", "reconciliation",
    ],
    common_patterns=[
        "Invoices should have sequential numbering within a vendor",
        "Tax rates should be consistent for similar item categories",
        "Payment terms typically range from Net 15 to Net 90",
        "Discounts (e.g., 2/10 Net 30) incentivize early payment",
        "Bulk orders may have different unit pricing than individual items",
        "International invoices require exchange rate documentation",
    ],
    red_flags=[
        "Invoice total does not match sum of line items plus tax",
        "Missing or invalid vendor tax registration number",
        "Duplicate invoice numbers from the same vendor",
        "Prices significantly above contracted or market rates",
        "Invoice date before goods receipt date",
        "Missing purchase order reference for PO-based procurement",
    ],
)

_MEDICAL_KNOWLEDGE = DomainContext(
    domain="medical",
    analytical_perspective=(
        "Analyze as a clinical documentation specialist. "
        "Review medical records with focus on clinical completeness, "
        "diagnostic accuracy, medication safety, and continuity of care. "
        "Cross-reference findings across encounters to identify trends, "
        "potential interactions, and follow-up requirements. Always consider "
        "the regional healthcare context and applicable clinical guidelines."
    ),
    evaluation_criteria=[
        "Diagnosis completeness with supporting clinical evidence",
        "Medication safety (interactions, contraindications, dosing appropriateness)",
        "Lab result interpretation against reference ranges",
        "Treatment plan alignment with diagnosis and clinical guidelines",
        "Follow-up and continuity of care documentation",
        "Vital signs trending and clinical deterioration indicators",
        "Allergy documentation and cross-referencing with prescribed medications",
        "Clinical coding accuracy (ICD-10, CPT, SNOMED-CT)",
    ],
    key_indicators=[
        "Primary and secondary diagnoses with severity indicators",
        "Active medication list with dosages, frequencies, and routes",
        "Lab values with reference ranges and abnormal flags",
        "Vital signs (BP, HR, temp, SpO2, RR) and their trends",
        "Procedure dates, types, and outcomes",
        "Allergy list and adverse reaction history",
        "Immunization status and preventive care compliance",
        "Social history relevant to clinical context",
    ],
    professional_terminology=[
        "chief complaint", "history of present illness", "review of systems",
        "differential diagnosis", "clinical impression", "treatment plan",
        "medication reconciliation", "drug-drug interaction", "adverse event",
        "clinical pathway", "evidence-based medicine", "informed consent",
    ],
    common_patterns=[
        "Multiple chronic conditions require holistic management plans",
        "Polypharmacy (5+ medications) increases interaction risk significantly",
        "Lab trends over time are more clinically meaningful than single values",
        "Antibiotic prescriptions should align with culture sensitivity results",
        "Chronic disease management requires periodic reassessment intervals",
        "Discharge summaries should include medication changes and follow-up dates",
    ],
    red_flags=[
        "Known drug-drug interactions in active medication list",
        "Lab values critically outside reference ranges without documented action",
        "Prescribed medication matching documented allergy",
        "Missing follow-up plan for newly diagnosed conditions",
        "Incomplete medication reconciliation at care transitions",
        "Significant vital sign changes without clinical response documentation",
    ],
)

_MEDICAL_REGION_CONTEXT: Dict[str, str] = {
    "US": (
        "US Healthcare Context: "
        "HIPAA-compliant documentation standards apply. "
        "Lab values typically in mg/dL (glucose), g/dL (hemoglobin). "
        "FDA-approved medications; DEA scheduling for controlled substances. "
        "ICD-10-CM for diagnosis coding, CPT for procedures. "
        "Medicare/Medicaid coverage considerations. "
        "Weight in pounds (lbs), temperature in Fahrenheit."
    ),
    "UK": (
        "UK/NHS Healthcare Context: "
        "NICE clinical guidelines are the standard of care reference. "
        "Lab values in mmol/L (glucose), g/L (hemoglobin). "
        "BNF (British National Formulary) for medication reference. "
        "NHS coding: SNOMED CT for clinical terms, OPCS-4 for procedures. "
        "GP referral pathway: GP → consultant → specialist. "
        "Weight in kg/stones, temperature in Celsius."
    ),
    "IN": (
        "India Healthcare Context: "
        "Both allopathic and AYUSH (Ayurveda, Yoga, Naturopathy, Unani, Siddha, Homeopathy) systems. "
        "NABH accreditation standards for hospitals. "
        "National List of Essential Medicines (NLEM) for drug formulary. "
        "ICMR guidelines for clinical research. "
        "Common abbreviations: MBBS (doctor), MS/MD (specialist). "
        "Lab values typically in mg/dL with SI unit equivalents."
    ),
    "EU": (
        "EU Healthcare Context: "
        "EMA (European Medicines Agency) drug approvals. "
        "Lab values in SI units (mmol/L for glucose). "
        "Cross-border healthcare directive applies. "
        "Weight in kg, temperature in Celsius. "
        "Member state variations in healthcare systems."
    ),
    "CA": (
        "Canada Healthcare Context: "
        "Health Canada drug approvals; provincial formularies. "
        "Lab values in SI units (mmol/L). "
        "Provincial health insurance plans (e.g., OHIP for Ontario). "
        "Weight in kg, temperature in Celsius. "
        "Bilingual documentation may apply (English/French)."
    ),
    "AU": (
        "Australia Healthcare Context: "
        "TGA (Therapeutic Goods Administration) drug approvals. "
        "PBS (Pharmaceutical Benefits Scheme) for medication subsidies. "
        "Lab values in SI units (mmol/L). "
        "Medicare Australia coverage. "
        "Weight in kg, temperature in Celsius."
    ),
    "INT": (
        "International Healthcare Context: "
        "WHO ICD-10/ICD-11 coding system. "
        "SI units preferred for lab values. "
        "Consider local clinical guidelines and drug formularies. "
        "FHIR/HL7 interoperability standards where applicable."
    ),
}

_LEGAL_KNOWLEDGE = DomainContext(
    domain="legal",
    analytical_perspective=(
        "Analyze as a contract review specialist and legal document analyst. "
        "Evaluate documents for obligation clarity, risk allocation, compliance "
        "requirements, and enforceability. Identify asymmetric terms, ambiguous "
        "language, and missing standard clauses. Map the obligation-rights "
        "hierarchy between parties."
    ),
    evaluation_criteria=[
        "Obligation clarity (shall/must vs may/should language precision)",
        "Risk allocation balance between parties",
        "Indemnification scope and limitations",
        "Termination rights and conditions (for cause, for convenience)",
        "Intellectual property ownership and licensing terms",
        "Confidentiality scope, duration, and exceptions",
        "Dispute resolution mechanism (arbitration, jurisdiction, governing law)",
        "Liability caps and exclusions of consequential damages",
    ],
    key_indicators=[
        "Parties and their defined roles/obligations",
        "Effective date, term, and renewal conditions",
        "Payment terms, penalties, and performance metrics",
        "Representations and warranties scope",
        "Force majeure triggers and consequences",
        "Assignment and subcontracting restrictions",
        "Notice requirements and communication protocols",
        "Amendment and waiver procedures",
    ],
    professional_terminology=[
        "indemnification", "force majeure", "governing law",
        "representations and warranties", "material breach",
        "liquidated damages", "severability", "assignment",
        "non-compete", "non-solicitation", "confidentiality",
        "dispute resolution", "limitation of liability",
    ],
    common_patterns=[
        "Standard contracts include boilerplate clauses (severability, entire agreement, waiver)",
        "Indemnification obligations should be reciprocal and capped",
        "Non-compete clauses should have reasonable geographic and temporal scope",
        "Force majeure clauses should include pandemic/epidemic coverage post-2020",
        "Governing law and jurisdiction should be clearly specified",
        "Amendment clauses typically require written agreement from both parties",
    ],
    red_flags=[
        "Unlimited liability or uncapped indemnification for one party",
        "Automatic renewal without termination notice provisions",
        "Unilateral amendment rights for one party",
        "Missing governing law or dispute resolution mechanism",
        "Broad non-compete clauses without reasonable limitations",
        "One-sided termination rights without cure period",
    ],
)

_POLICY_KNOWLEDGE = DomainContext(
    domain="policy",
    analytical_perspective=(
        "Analyze as an insurance analyst and policy specialist. "
        "Evaluate policies for coverage adequacy, exclusion scope, premium "
        "structure, claims procedures, and policyholder obligations. "
        "Compare coverage terms against industry standards and identify "
        "gaps that may leave the insured exposed."
    ),
    evaluation_criteria=[
        "Coverage scope: what is covered (perils, risks, events)",
        "Exclusion scope: what is explicitly not covered",
        "Premium structure: base premium, deductibles, co-payments",
        "Sum insured / coverage limits adequacy",
        "Claims process: notification requirements, documentation, timelines",
        "Policy period and renewal terms",
        "Policyholder obligations (duty of disclosure, loss mitigation)",
        "Endorsements, riders, and add-on coverage options",
    ],
    key_indicators=[
        "Policy number, effective date, and expiry date",
        "Named insured and additional insured parties",
        "Coverage types (comprehensive, third-party, specific perils)",
        "Sum insured / coverage limit amounts",
        "Deductible / excess amounts per claim type",
        "Premium amount and payment schedule",
        "No-claim bonus / discount provisions",
        "Waiting periods and cooling-off periods",
    ],
    professional_terminology=[
        "premium", "deductible", "sum insured", "exclusion",
        "endorsement", "rider", "co-payment", "subrogation",
        "underwriting", "actuary", "claims adjuster",
        "indemnity", "insurable interest", "utmost good faith",
    ],
    common_patterns=[
        "Comprehensive policies cost more but cover a wider range of perils",
        "Higher deductibles typically result in lower premiums",
        "No-claim bonuses reward policyholders for claim-free periods",
        "Pre-existing condition exclusions are common in health and motor insurance",
        "Coverage limits should reflect current replacement/market values",
        "Policy renewals may come with revised terms and premium adjustments",
    ],
    red_flags=[
        "Broad exclusion clauses that significantly limit practical coverage",
        "Sum insured below replacement value (underinsurance)",
        "Unreasonably short claims notification windows",
        "Missing or unclear dispute resolution process",
        "Automatic premium escalation without policyholder consent",
        "Vague or ambiguous coverage trigger definitions",
    ],
)

_GENERIC_KNOWLEDGE = DomainContext(
    domain="generic",
    analytical_perspective=(
        "Analyze as a document intelligence specialist. "
        "Evaluate the document for information completeness, structural "
        "consistency, factual accuracy indicators, and cross-reference "
        "opportunities. Identify patterns, themes, and potential gaps "
        "in the information presented."
    ),
    evaluation_criteria=[
        "Information completeness and coverage of key topics",
        "Internal consistency of facts, dates, and figures",
        "Source attribution and evidence quality",
        "Structural organization and readability",
        "Cross-document pattern identification",
        "Data quality (missing fields, inconsistencies, outliers)",
    ],
    key_indicators=[
        "Document type and purpose",
        "Key entities (people, organizations, locations, dates)",
        "Quantitative data points (numbers, amounts, percentages)",
        "Temporal information (dates, periods, deadlines)",
        "Relationships between entities and concepts",
        "Action items, requirements, or obligations",
    ],
    professional_terminology=[
        "document intelligence", "information extraction",
        "entity recognition", "cross-reference", "data quality",
        "pattern analysis", "anomaly detection", "completeness audit",
    ],
    common_patterns=[
        "Well-structured documents have clear sections and headings",
        "Quantified claims are more reliable than qualitative assertions",
        "Cross-document consistency strengthens confidence in findings",
        "Date-ordered analysis reveals trends and trajectories",
    ],
    red_flags=[
        "Contradictory information within or across documents",
        "Missing expected sections or fields",
        "Significant data gaps without explanation",
        "Inconsistent formatting suggesting multiple authors or edits",
    ],
)

# Master knowledge registry
_DOMAIN_KNOWLEDGE: Dict[str, DomainContext] = {
    "hr": _HR_KNOWLEDGE,
    "resume": _HR_KNOWLEDGE,
    "invoice": _INVOICE_KNOWLEDGE,
    "procurement": _INVOICE_KNOWLEDGE,
    "medical": _MEDICAL_KNOWLEDGE,
    "clinical": _MEDICAL_KNOWLEDGE,
    "legal": _LEGAL_KNOWLEDGE,
    "contract": _LEGAL_KNOWLEDGE,
    "policy": _POLICY_KNOWLEDGE,
    "insurance": _POLICY_KNOWLEDGE,
    "generic": _GENERIC_KNOWLEDGE,
    "report": _GENERIC_KNOWLEDGE,
}

# ── Intent-Specific Knowledge Augmentation ───────────────────────────

_INTENT_AUGMENTATION: Dict[str, Dict[str, str]] = {
    "hr": {
        "rank": "When ranking candidates: weight recent experience higher, value quantified achievements, consider role-fit alignment over raw experience years.",
        "compare": "When comparing candidates: use consistent dimensions (skills, experience, education, certifications), highlight complementary strengths, note unique differentiators.",
        "summary": "When summarizing resumes: lead with the candidate's strongest qualification, include career level assessment, note specialization depth.",
        "extraction": "When extracting resume data: verify contact completeness, calculate total experience accurately, distinguish primary skills from supplementary.",
        "contact": "When presenting contact details: ensure all available channels are listed (email, phone, LinkedIn, location), flag incomplete contact information.",
    },
    "invoice": {
        "summary": "When summarizing invoices: aggregate totals by vendor, highlight any discrepancies, report tax and discount summaries.",
        "extraction": "When extracting invoice data: verify mathematical accuracy (subtotal + tax = total), capture all line items with quantities and unit prices.",
        "comparison": "When comparing invoices: align by vendor, date, or amount; identify pricing trends and volume patterns.",
        "factual": "When answering invoice questions: be precise with amounts (include currency), report exact dates, reference invoice numbers.",
    },
    "medical": {
        "summary": "When summarizing medical records: organize by problem list, note active medications with any interactions, highlight abnormal results.",
        "extraction": "When extracting medical data: include reference ranges for lab values, flag abnormal results, cross-reference medications with diagnoses.",
        "comparison": "When comparing medical records: track changes in lab values over time, note medication adjustments, identify clinical trajectory.",
    },
    "legal": {
        "summary": "When summarizing legal documents: identify all parties, map key obligations, highlight risk areas and liability provisions.",
        "extraction": "When extracting legal clauses: classify by type (obligation/right/condition), identify the responsible party, note any conditions or exceptions.",
        "comparison": "When comparing legal documents: align by clause type, identify differences in obligation scope, note changes in risk allocation.",
    },
    "policy": {
        "summary": "When summarizing policies: clearly separate coverage from exclusions, highlight premium and deductible structures.",
        "extraction": "When extracting policy data: capture all coverage types with limits, list all exclusions completely, note claims procedure requirements.",
        "comparison": "When comparing policies: align coverage types, compare premium-to-coverage ratios, identify coverage gaps between policies.",
    },
}

# ── Web Enrichment Cache ─────────────────────────────────────────────

_web_cache: Dict[str, Tuple[str, float]] = {}
_web_cache_lock = threading.Lock()
_WEB_CACHE_TTL = 3600  # 1 hour

def _get_cached_web_knowledge(cache_key: str) -> Optional[str]:
    """Get cached web knowledge if not expired."""
    with _web_cache_lock:
        if cache_key in _web_cache:
            text, ts = _web_cache[cache_key]
            if time.time() - ts < _WEB_CACHE_TTL:
                return text
            del _web_cache[cache_key]
    return None

def _set_cached_web_knowledge(cache_key: str, text: str) -> None:
    """Cache web knowledge result."""
    with _web_cache_lock:
        # Limit cache size
        if len(_web_cache) > 50:
            oldest_key = min(_web_cache, key=lambda k: _web_cache[k][1])
            del _web_cache[oldest_key]
        _web_cache[cache_key] = (text, time.time())

def _fetch_domain_web_knowledge(domain: str, query: str) -> Optional[str]:
    """Fetch supplementary domain knowledge from the web.

    This is for domain *expertise* only — it does NOT fetch or use
    any content from the user's documents.
    """
    try:
        from src.tools.web_search import search_web
    except ImportError:
        logger.debug("Web search module not available for domain enrichment")
        return None

    # Build a domain-focused search query
    search_queries: Dict[str, str] = {
        "hr": "HR recruitment best practices resume screening criteria",
        "invoice": "invoice processing best practices accounts payable validation",
        "medical": "clinical documentation best practices medical record review",
        "legal": "contract review best practices legal document analysis",
        "policy": "insurance policy analysis coverage evaluation best practices",
    }

    base_query = search_queries.get(domain, f"{domain} document analysis best practices")
    cache_key = hashlib.md5(f"{domain}:{base_query}".encode()).hexdigest()

    cached = _get_cached_web_knowledge(cache_key)
    if cached:
        return cached

    try:
        results = search_web(base_query, max_results=3, timeout=8.0)
        if not results:
            return None

        snippets = []
        for r in results[:3]:
            snippet = r.get("snippet") or r.get("body") or ""
            if snippet:
                snippets.append(snippet[:300])

        if not snippets:
            return None

        combined = " ".join(snippets)[:800]
        _set_cached_web_knowledge(cache_key, combined)
        return combined

    except Exception as exc:
        logger.debug("Web domain knowledge fetch failed: %s", exc)
        return None

# ── DomainKnowledgeProvider ──────────────────────────────────────────

class DomainKnowledgeProvider:
    """Central provider for domain-specific analytical knowledge.

    Thread-safe singleton that provides professional domain expertise
    for document analysis. Does NOT modify or replace document content —
    only adds analytical context.
    """

    def __init__(self, *, web_enrichment: bool = False, cache_ttl: int = 3600):
        self._web_enrichment = web_enrichment
        self._cache_ttl = cache_ttl
        self._knowledge = dict(_DOMAIN_KNOWLEDGE)
        self._region_context = dict(_MEDICAL_REGION_CONTEXT)
        self._intent_augmentation = dict(_INTENT_AUGMENTATION)
        _unique_domains = {ctx.domain for ctx in self._knowledge.values()}
        logger.info(
            "DomainKnowledgeProvider initialized: %d domains, web_enrichment=%s",
            len(_unique_domains),
            web_enrichment,
        )

    @property
    def supported_domains(self) -> List[str]:
        """Return list of unique supported domain names."""
        return list({ctx.domain for ctx in self._knowledge.values()})

    def get_knowledge(
        self,
        domain: str,
        *,
        query: str = "",
        intent: str = "",
        document_text: str = "",
        include_web: bool = False,
    ) -> DomainContext:
        """Get domain knowledge context for a specific query.

        Args:
            domain: The document domain (hr, invoice, medical, etc.)
            query: The user's query (for intent-specific augmentation)
            intent: The detected intent (rank, compare, summary, etc.)
            document_text: Raw document text (for region detection in medical)
            include_web: Whether to include web-enriched knowledge

        Returns:
            DomainContext with professional analytical knowledge.
        """
        base_ctx = self._knowledge.get(domain, self._knowledge["generic"])

        # Build a copy with potential augmentations
        ctx = DomainContext(
            domain=base_ctx.domain,
            analytical_perspective=base_ctx.analytical_perspective,
            evaluation_criteria=list(base_ctx.evaluation_criteria),
            key_indicators=list(base_ctx.key_indicators),
            professional_terminology=list(base_ctx.professional_terminology),
            common_patterns=list(base_ctx.common_patterns),
            red_flags=list(base_ctx.red_flags),
        )

        # Intent-specific augmentation
        if intent and domain in self._intent_augmentation:
            aug = self._intent_augmentation[domain].get(intent)
            if aug:
                ctx.analytical_perspective = f"{ctx.analytical_perspective}\n{aug}"

        # Medical region detection
        if domain in ("medical", "clinical") and document_text:
            country_code, country_name, confidence = detect_country_of_origin(document_text)
            if confidence >= 0.3:
                region_ctx = self._region_context.get(country_code, self._region_context["INT"])
                ctx.region_context = f"Detected region: {country_name} (confidence: {confidence})\n{region_ctx}"
            else:
                ctx.region_context = self._region_context["INT"]

        # Web enrichment (optional, for domain knowledge only)
        if include_web and self._web_enrichment:
            web_ctx = _fetch_domain_web_knowledge(ctx.domain, query)
            if web_ctx:
                ctx.web_supplement = web_ctx

        return ctx

    def get_brief_context(self, domain: str, intent: str = "") -> str:
        """Get a brief domain context string suitable for prompt injection.

        This is a lightweight version for embedding in generation prompts
        without consuming too many tokens.
        """
        ctx = self.get_knowledge(domain, intent=intent)
        parts = [ctx.analytical_perspective]

        if ctx.evaluation_criteria:
            parts.append("Key criteria: " + "; ".join(ctx.evaluation_criteria[:3]))

        if ctx.red_flags:
            parts.append("Watch for: " + "; ".join(ctx.red_flags[:2]))

        return "\n".join(parts)

    def get_medical_region_context(self, document_text: str) -> Tuple[str, str, str]:
        """Get medical region detection and context.

        Returns (country_code, country_name, region_context_text).
        """
        code, name, conf = detect_country_of_origin(document_text)
        ctx = self._region_context.get(code, self._region_context["INT"])
        return (code, name, ctx)

# ── Singleton Management ─────────────────────────────────────────────

_provider: Optional[DomainKnowledgeProvider] = None
_provider_lock = threading.Lock()

def get_domain_knowledge_provider() -> DomainKnowledgeProvider:
    """Get the global DomainKnowledgeProvider singleton.

    Creates a default instance if none has been explicitly set.
    """
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = DomainKnowledgeProvider()
    return _provider

def set_domain_knowledge_provider(provider: DomainKnowledgeProvider) -> None:
    """Set the global DomainKnowledgeProvider singleton."""
    global _provider
    with _provider_lock:
        _provider = provider

def ensure_domain_knowledge_provider(*, web_enrichment: bool = False) -> DomainKnowledgeProvider:
    """Ensure the global provider is initialized with the given settings."""
    global _provider
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is None:
            _provider = DomainKnowledgeProvider(web_enrichment=web_enrichment)
    return _provider
