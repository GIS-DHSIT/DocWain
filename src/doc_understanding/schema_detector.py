"""
schema_detector.py — Document schema detection and field extraction module.

Detects the schema template for a known document type, maps section roles
from structure_inference to schema sections, extracts typed fields from
existing entities and regex patterns, and scores completeness.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SCHEMA_TEMPLATES",
    "SECTION_ROLE_MAP",
    "SchemaTemplate",
    "SchemaDetectionResult",
    "detect_and_extract_schema",
]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
_SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldDefinition:
    """Defines how a single field should be extracted."""
    name: str
    field_type: str  # "str", "list", "float", "date"
    entity_labels: List[str] = field(default_factory=list)  # e.g. ["PERSON", "ORG"]
    regex_patterns: List[str] = field(default_factory=list)
    section_hint: Optional[str] = None  # look inside this section first
    description: str = ""

@dataclass
class SchemaTemplate:
    """Schema definition for one document type."""
    doc_type: str
    required_sections: List[str]
    optional_sections: List[str]
    fields: List[FieldDefinition]

@dataclass
class SchemaDetectionResult:
    """Return value of detect_and_extract_schema()."""
    doc_type: str
    completeness_score: float            # 0.0 – 1.0
    found_sections: List[str]
    missing_sections: List[str]
    extracted_fields: Dict[str, Any]
    schema_version: str

# ---------------------------------------------------------------------------
# Regex helpers (compiled once at import time)
# ---------------------------------------------------------------------------

# Resume
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
_RE_PHONE = re.compile(
    r"(?:\+?\d[\d\s\-().]{7,}\d)"
)
_RE_TOTAL_EXP_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.I)
_RE_SKILLS_SECTION = re.compile(
    r"(?:technical\s+skills?|skills?|technologies|tech\s+stack)\s*[:\-]?\s*\n(.*?)(?:\n\n|\Z)",
    re.I | re.S,
)
_RE_SKILL_ITEM = re.compile(r"[A-Za-z][A-Za-z0-9#+.\-/_ ]{1,40}")

# Invoice
_RE_INVOICE_NUMBER = re.compile(
    r"(?:invoice|inv|bill)\s*[#no.:]*\s*([A-Z0-9\-/]{3,20})", re.I
)
_RE_TOTAL_AMOUNT = re.compile(
    r"(?:total|grand\s+total|amount\s+due)\s*[:\$]?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)", re.I
)
_RE_PAYMENT_TERMS = re.compile(
    r"(?:net\s*-?\s*\d+|due\s+within\s+\d+\s+days?|payment\s+due\s+in\s+\d+)", re.I
)

# Contract
_RE_GOVERNING_LAW = re.compile(
    r"(?:governed\s+by|governing\s+law|laws?\s+of)\s+(?:the\s+)?([A-Z][A-Za-z\s]{2,40}?)(?:\.|,|;|\n)",
    re.I,
)
_RE_CONTRACT_TYPE = re.compile(
    r"\b(non[\-\s]?disclosure\s+agreement|nda|service\s+agreement|employment\s+agreement|"
    r"license\s+agreement|master\s+services\s+agreement|msa|statement\s+of\s+work|sow|"
    r"purchase\s+order|partnership\s+agreement|consulting\s+agreement)\b",
    re.I,
)

# Medical
_RE_DIAGNOSIS_CODES = re.compile(
    r"\b(?:ICD[-\s]?(?:10|9|11)?[-\s]?[A-Z0-9.]{2,8}|[A-Z]\d{2}(?:\.\d{1,2})?)\b"
)
_RE_MEDICATIONS = re.compile(
    r"\b([A-Z][a-z]{3,30}(?:ine|ol|ide|ate|il|in|an|on|en|um|am)?)"
    r"\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|IU))",
    re.I,
)
_RE_LAB_RESULT = re.compile(
    r"([A-Za-z][A-Za-z0-9\s]{2,30})\s*[:\-=]\s*([\d.]+)\s*([\w/%]+)?",
)

# Policy
_RE_POLICY_NUMBER = re.compile(
    r"(?:policy|document|doc)\s*[#no.:]*\s*([A-Z0-9\-/]{3,20})", re.I
)
_RE_APPLICABLE_TEAMS = re.compile(
    r"(?:applies?\s+to|applicable\s+to|scope)\s*[:\-]?\s*([^\n.]{5,120})", re.I
)

# Report
_RE_REPORT_TYPE = re.compile(
    r"\b(annual\s+report|quarterly\s+report|financial\s+report|audit\s+report|"
    r"incident\s+report|progress\s+report|executive\s+summary|feasibility\s+report|"
    r"status\s+report|research\s+report)\b",
    re.I,
)

# Shared date-like pattern used across multiple types
_RE_DATE_GENERIC = re.compile(
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Section role → schema section mapping
# ---------------------------------------------------------------------------

SECTION_ROLE_MAP: Dict[str, str] = {
    # Resume
    "contact_info": "contact_info",
    "contact": "contact_info",
    "personal_info": "contact_info",
    "experience_history": "experience",
    "work_experience": "experience",
    "professional_experience": "experience",
    "employment_history": "experience",
    "education_training": "education",
    "education": "education",
    "academic_background": "education",
    "skills": "skills",
    "technical_skills": "skills",
    "core_competencies": "skills",
    # Invoice
    "vendor_information": "vendor_info",
    "vendor_info": "vendor_info",
    "supplier_info": "vendor_info",
    "billing_details": "vendor_info",
    "line_items": "line_items",
    "itemized_list": "line_items",
    "item_details": "line_items",
    "financial_data": "totals",
    "totals": "totals",
    "summary_totals": "totals",
    "payment_info": "totals",
    # Contract
    "parties": "parties",
    "contracting_parties": "parties",
    "party_information": "parties",
    "legal_terms": "terms",
    "terms_and_conditions": "terms",
    "general_terms": "terms",
    "obligations": "obligations",
    "responsibilities": "obligations",
    "duties": "obligations",
    # Medical
    "patient_information": "patient_info",
    "patient_info": "patient_info",
    "patient_demographics": "patient_info",
    "clinical_data": "diagnosis",
    "diagnosis": "diagnosis",
    "assessment": "diagnosis",
    "clinical_assessment": "diagnosis",
    "treatment": "treatment",
    "treatment_plan": "treatment",
    "management_plan": "treatment",
    "medications": "treatment",
    # Policy
    "scope": "scope",
    "applicability": "scope",
    "purpose": "scope",
    "requirements": "requirements",
    "policy_requirements": "requirements",
    "rules": "requirements",
    "compliance": "compliance",
    "enforcement": "compliance",
    "violations": "compliance",
    # Report
    "summary_like": "summary",
    "executive_summary": "summary",
    "abstract": "summary",
    "introduction": "summary",
    "findings": "findings",
    "results": "findings",
    "analysis": "findings",
    "observations": "findings",
    # Generic mappings usable across types
    "header": "contact_info",
    "footer": None,
    "appendix": None,
    "references": None,
    "glossary": None,
}

# ---------------------------------------------------------------------------
# Schema templates
# ---------------------------------------------------------------------------

SCHEMA_TEMPLATES: Dict[str, SchemaTemplate] = {
    "resume": SchemaTemplate(
        doc_type="resume",
        required_sections=["contact_info", "experience", "education", "skills"],
        optional_sections=["summary", "certifications", "projects", "awards", "publications"],
        fields=[
            FieldDefinition(
                name="candidate_name",
                field_type="str",
                entity_labels=["PERSON"],
                description="Full name of the candidate",
            ),
            FieldDefinition(
                name="email",
                field_type="str",
                entity_labels=["EMAIL"],
                regex_patterns=[_RE_EMAIL.pattern],
                description="Email address",
            ),
            FieldDefinition(
                name="phone",
                field_type="str",
                entity_labels=["PHONE"],
                regex_patterns=[_RE_PHONE.pattern],
                description="Phone number",
            ),
            FieldDefinition(
                name="total_experience_years",
                field_type="float",
                regex_patterns=[_RE_TOTAL_EXP_YEARS.pattern],
                section_hint="experience",
                description="Total years of professional experience",
            ),
            FieldDefinition(
                name="skills_list",
                field_type="list",
                entity_labels=["SKILL"],
                section_hint="skills",
                description="List of skills mentioned",
            ),
            FieldDefinition(
                name="experience_entries",
                field_type="list",
                section_hint="experience",
                description="Individual experience entries (job titles / companies)",
            ),
        ],
    ),

    "invoice": SchemaTemplate(
        doc_type="invoice",
        required_sections=["vendor_info", "line_items", "totals"],
        optional_sections=["buyer_info", "notes", "tax_details", "shipping"],
        fields=[
            FieldDefinition(
                name="invoice_number",
                field_type="str",
                regex_patterns=[_RE_INVOICE_NUMBER.pattern],
                description="Invoice identifier",
            ),
            FieldDefinition(
                name="invoice_date",
                field_type="date",
                entity_labels=["DATE"],
                regex_patterns=[_RE_DATE_GENERIC.pattern],
                description="Date the invoice was issued",
            ),
            FieldDefinition(
                name="vendor_name",
                field_type="str",
                entity_labels=["ORG"],
                section_hint="vendor_info",
                description="Vendor / supplier organisation name",
            ),
            FieldDefinition(
                name="total_amount",
                field_type="str",
                entity_labels=["AMOUNT", "MONEY"],
                regex_patterns=[_RE_TOTAL_AMOUNT.pattern],
                section_hint="totals",
                description="Total invoice amount",
            ),
            FieldDefinition(
                name="payment_terms",
                field_type="str",
                regex_patterns=[_RE_PAYMENT_TERMS.pattern],
                description="Payment terms (e.g. Net-30)",
            ),
        ],
    ),

    "contract": SchemaTemplate(
        doc_type="contract",
        required_sections=["parties", "terms", "obligations"],
        optional_sections=["recitals", "definitions", "representations", "warranties",
                           "indemnification", "termination", "signatures"],
        fields=[
            FieldDefinition(
                name="party_names",
                field_type="list",
                entity_labels=["ORG", "PERSON"],
                section_hint="parties",
                description="Names of contracting parties",
            ),
            FieldDefinition(
                name="effective_date",
                field_type="date",
                entity_labels=["DATE"],
                regex_patterns=[_RE_DATE_GENERIC.pattern],
                description="Contract effective date",
            ),
            FieldDefinition(
                name="governing_law",
                field_type="str",
                regex_patterns=[_RE_GOVERNING_LAW.pattern],
                description="Jurisdiction governing the contract",
            ),
            FieldDefinition(
                name="contract_type",
                field_type="str",
                regex_patterns=[_RE_CONTRACT_TYPE.pattern],
                description="Type / category of contract",
            ),
        ],
    ),

    "medical": SchemaTemplate(
        doc_type="medical",
        required_sections=["patient_info", "diagnosis", "treatment"],
        optional_sections=["history", "lab_results", "imaging", "follow_up",
                           "allergies", "vital_signs"],
        fields=[
            FieldDefinition(
                name="patient_name",
                field_type="str",
                entity_labels=["PERSON"],
                section_hint="patient_info",
                description="Patient full name",
            ),
            FieldDefinition(
                name="diagnosis_codes",
                field_type="list",
                regex_patterns=[_RE_DIAGNOSIS_CODES.pattern],
                section_hint="diagnosis",
                description="ICD or other diagnosis codes",
            ),
            FieldDefinition(
                name="medications",
                field_type="list",
                regex_patterns=[_RE_MEDICATIONS.pattern],
                section_hint="treatment",
                description="Medications with dosage",
            ),
            FieldDefinition(
                name="lab_results",
                field_type="list",
                regex_patterns=[_RE_LAB_RESULT.pattern],
                description="Lab test name and value pairs",
            ),
        ],
    ),

    "policy": SchemaTemplate(
        doc_type="policy",
        required_sections=["scope", "requirements", "compliance"],
        optional_sections=["purpose", "definitions", "exceptions", "review_history",
                           "approval", "references"],
        fields=[
            FieldDefinition(
                name="policy_number",
                field_type="str",
                regex_patterns=[_RE_POLICY_NUMBER.pattern],
                description="Policy document identifier",
            ),
            FieldDefinition(
                name="effective_date",
                field_type="date",
                entity_labels=["DATE"],
                regex_patterns=[_RE_DATE_GENERIC.pattern],
                description="Date the policy becomes effective",
            ),
            FieldDefinition(
                name="review_date",
                field_type="date",
                entity_labels=["DATE"],
                regex_patterns=[_RE_DATE_GENERIC.pattern],
                description="Next scheduled review date",
            ),
            FieldDefinition(
                name="applicable_teams",
                field_type="list",
                regex_patterns=[_RE_APPLICABLE_TEAMS.pattern],
                section_hint="scope",
                description="Teams or groups to which the policy applies",
            ),
        ],
    ),

    "report": SchemaTemplate(
        doc_type="report",
        required_sections=["summary", "findings"],
        optional_sections=["introduction", "methodology", "recommendations",
                           "appendix", "references"],
        fields=[
            FieldDefinition(
                name="report_date",
                field_type="date",
                entity_labels=["DATE"],
                regex_patterns=[_RE_DATE_GENERIC.pattern],
                description="Date the report was produced",
            ),
            FieldDefinition(
                name="author",
                field_type="str",
                entity_labels=["PERSON"],
                description="Report author or authors",
            ),
            FieldDefinition(
                name="report_type",
                field_type="str",
                regex_patterns=[_RE_REPORT_TYPE.pattern],
                description="Category / type of report",
            ),
        ],
    ),
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_section_roles(
    section_roles: Optional[Dict[str, str]],
    schema: SchemaTemplate,
) -> Tuple[List[str], List[str]]:
    """
    Given a dict of {section_id: inferred_role}, return (found_sections, missing_sections)
    relative to the schema's required_sections.
    """
    mapped: set[str] = set()

    if section_roles:
        for _sid, role in section_roles.items():
            canonical = SECTION_ROLE_MAP.get(role)
            if canonical:
                mapped.add(canonical)

    found = [s for s in schema.required_sections if s in mapped]
    missing = [s for s in schema.required_sections if s not in mapped]
    return found, missing

def _build_section_text_index(
    sections: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Build a mapping of {canonical_section_name: concatenated_text} from the
    sections list.  Each section dict is expected to have at minimum:
      - "title" or "role" or "section_type"
      - "text" or "content"
    """
    index: Dict[str, str] = {}

    for sec in sections:
        role = (
            sec.get("role")
            or sec.get("section_type")
            or sec.get("title", "")
        ).lower().replace(" ", "_")
        canonical = SECTION_ROLE_MAP.get(role, role)
        text = sec.get("text") or sec.get("content") or ""
        if canonical:
            index[canonical] = index.get(canonical, "") + "\n" + text

    return index

def _entities_by_label(
    entities: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Return {label: [text, ...]} from an entity list.
    Supports dicts with "label"/"type"/"entity_type" + "text"/"value"/"name".
    """
    by_label: Dict[str, List[str]] = {}
    for ent in entities:
        label = (
            ent.get("label")
            or ent.get("type")
            or ent.get("entity_type")
            or ""
        ).upper()
        text = (
            ent.get("text")
            or ent.get("value")
            or ent.get("name")
            or ""
        ).strip()
        if label and text:
            by_label.setdefault(label, []).append(text)
    return by_label

def _first_regex_match(pattern_str: str, text: str, group: int = 1) -> Optional[str]:
    """Return the first capture group from a regex search, or None."""
    try:
        m = re.search(pattern_str, text, re.I)
        if m:
            try:
                return m.group(group).strip()
            except IndexError:
                return m.group(0).strip()
    except re.error as exc:
        logger.debug("Regex error for pattern %r: %s", pattern_str, exc)
    return None

def _all_regex_matches(pattern_str: str, text: str, group: int = 1) -> List[str]:
    """Return all non-overlapping capture-group matches from a regex."""
    results: List[str] = []
    try:
        for m in re.finditer(pattern_str, text, re.I):
            try:
                results.append(m.group(group).strip())
            except IndexError:
                results.append(m.group(0).strip())
    except re.error as exc:
        logger.debug("Regex error for pattern %r: %s", pattern_str, exc)
    return results

def _extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skills from a block of text by finding comma/bullet-separated
    lists under common 'Skills' headers.
    """
    skills: List[str] = []

    for m in _RE_SKILLS_SECTION.finditer(text):
        section_body = m.group(1)
        # Split on commas, bullets, pipes, newlines
        tokens = re.split(r"[,|\n•\-–*]+", section_body)
        for tok in tokens:
            tok = tok.strip()
            if 2 <= len(tok) <= 50 and re.match(r"[A-Za-z]", tok):
                skills.append(tok)

    return list(dict.fromkeys(skills))  # deduplicate, preserve order

def _extract_experience_entries(sections: List[Dict[str, Any]]) -> List[str]:
    """
    Pull job-title / company pairs from experience sections.
    Looks for lines that look like "Job Title at Company" or "Company | Role".
    """
    entries: List[str] = []
    exp_re = re.compile(
        r"^([A-Z][A-Za-z\s,&.'\-]+?)(?:\s+at\s+|\s*[|@]\s*|\s*,\s*)([A-Z][A-Za-z\s,&.']+)",
        re.M,
    )
    for sec in sections:
        role = (sec.get("role") or sec.get("section_type") or "").lower()
        if "experience" in role or "employment" in role or "work" in role:
            text = sec.get("text") or sec.get("content") or ""
            for m in exp_re.finditer(text):
                entry = f"{m.group(1).strip()} — {m.group(2).strip()}"
                entries.append(entry)
    return entries

# ---------------------------------------------------------------------------
# Field extraction dispatcher
# ---------------------------------------------------------------------------

def _extract_field(
    fdef: FieldDefinition,
    entity_index: Dict[str, List[str]],
    section_text_index: Dict[str, str],
    full_text: str,
) -> Any:
    """
    Attempt to extract a field value using entities first, then regex.
    Returns the extracted value (str, list, float) or None.
    """
    # Determine the text corpus to search
    search_text = full_text
    if fdef.section_hint and fdef.section_hint in section_text_index:
        search_text = section_text_index[fdef.section_hint] + "\n" + full_text

    field_type = fdef.field_type

    # --- List fields ---
    if field_type == "list":
        # Special-case: skills_list
        if fdef.name == "skills_list":
            from_entities: List[str] = []
            for lbl in fdef.entity_labels:
                from_entities.extend(entity_index.get(lbl, []))
            if from_entities:
                return list(dict.fromkeys(from_entities))
            # Fall back to text extraction
            return _extract_skills_from_text(search_text) or None

        # experience_entries — custom extractor
        if fdef.name == "experience_entries":
            return None  # populated separately from raw sections

        # Generic list from entities
        collected: List[str] = []
        for lbl in fdef.entity_labels:
            collected.extend(entity_index.get(lbl, []))

        # Also collect from regex
        for pat in fdef.regex_patterns:
            collected.extend(_all_regex_matches(pat, search_text))

        collected = list(dict.fromkeys(collected))
        return collected if collected else None

    # --- Scalar fields (str / date / float) ---
    # 1. Try entities
    for lbl in fdef.entity_labels:
        values = entity_index.get(lbl, [])
        if values:
            val = values[0]
            if field_type == "float":
                try:
                    return float(re.sub(r"[^\d.]", "", val))
                except ValueError:
                    pass
            return val

    # 2. Try regex patterns
    for pat in fdef.regex_patterns:
        val = _first_regex_match(pat, search_text)
        if val:
            if field_type == "float":
                numeric = re.sub(r"[^\d.]", "", val)
                try:
                    return float(numeric)
                except ValueError:
                    return val
            return val

    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_extract_schema(
    doc_type: str,
    sections: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    full_text: str,
    section_roles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Detect and extract a document schema for the given doc_type.

    Parameters
    ----------
    doc_type : str
        One of: resume, invoice, contract, medical, policy, report.
        Case-insensitive; unknown types return a minimal fallback result.
    sections : list[dict]
        Section objects from structure_inference / content_map.
        Each dict should have at least "role"/"title" and "text"/"content".
    entities : list[dict]
        Entity objects (e.g. from spaCy NER or deep_analyzer).
        Each dict should have "label"/"type" and "text"/"value".
    full_text : str
        Full document text used as a fallback search corpus.
    section_roles : dict[str, str] | None
        Optional mapping of {section_id: inferred_role} from structure_inference.
        Keys are section identifiers; values are role strings such as
        "experience_history", "education_training", "financial_data", etc.

    Returns
    -------
    dict with keys:
        doc_type           — echoed doc_type string
        completeness_score — float 0.0–1.0 (found_required / total_required)
        found_sections     — list of required sections that were detected
        missing_sections   — list of required sections not found
        extracted_fields   — dict of {field_name: value}
        schema_version     — str version tag
    """
    normalised_type = doc_type.lower().strip() if doc_type else ""
    schema = SCHEMA_TEMPLATES.get(normalised_type)

    if schema is None:
        logger.warning(
            "schema_detector: unknown doc_type %r — returning empty schema result",
            doc_type,
        )
        return {
            "doc_type": doc_type,
            "completeness_score": 0.0,
            "found_sections": [],
            "missing_sections": [],
            "extracted_fields": {},
            "schema_version": _SCHEMA_VERSION,
        }

    logger.debug("schema_detector: detecting schema for doc_type=%r", normalised_type)

    # ------------------------------------------------------------------
    # Step 1 — Build lookup indexes
    # ------------------------------------------------------------------
    entity_index = _entities_by_label(entities)
    section_text_index = _build_section_text_index(sections)

    # ------------------------------------------------------------------
    # Step 2 — Map section roles to schema sections
    # ------------------------------------------------------------------
    # Start with roles from the explicit section_roles dict (if provided)
    combined_roles: Dict[str, str] = {}
    if section_roles:
        combined_roles.update(section_roles)

    # Also derive roles from the sections list themselves
    for sec in sections:
        sec_id = str(sec.get("id") or sec.get("section_id") or id(sec))
        role = (
            sec.get("role")
            or sec.get("section_type")
            or sec.get("title", "")
        ).lower().replace(" ", "_")
        if role:
            combined_roles.setdefault(sec_id, role)

    found_sections, missing_sections = _map_section_roles(combined_roles, schema)

    # ------------------------------------------------------------------
    # Step 3 — Extract fields
    # ------------------------------------------------------------------
    extracted_fields: Dict[str, Any] = {}

    for fdef in schema.fields:
        try:
            value = _extract_field(fdef, entity_index, section_text_index, full_text)
            if value is not None:
                extracted_fields[fdef.name] = value
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "schema_detector: error extracting field %r for doc_type %r: %s",
                fdef.name,
                normalised_type,
                exc,
            )

    # Special-case: experience_entries for resume
    if normalised_type == "resume" and "experience_entries" not in extracted_fields:
        entries = _extract_experience_entries(sections)
        if entries:
            extracted_fields["experience_entries"] = entries

    # Special-case: policy review_date — must differ from effective_date
    if normalised_type == "policy":
        eff = extracted_fields.get("effective_date")
        rev = extracted_fields.get("review_date")
        if eff and rev and eff == rev:
            # Try to find a second date match in the text
            all_dates = _all_regex_matches(_RE_DATE_GENERIC.pattern, full_text)
            unique_dates = [d for d in all_dates if d != eff]
            if unique_dates:
                extracted_fields["review_date"] = unique_dates[0]
            else:
                extracted_fields.pop("review_date", None)

    # ------------------------------------------------------------------
    # Step 4 — Compute completeness score
    # ------------------------------------------------------------------
    total_required = len(schema.required_sections)
    if total_required > 0:
        completeness_score = round(len(found_sections) / total_required, 4)
    else:
        completeness_score = 1.0

    logger.debug(
        "schema_detector: doc_type=%r score=%.2f found=%s missing=%s fields=%s",
        normalised_type,
        completeness_score,
        found_sections,
        missing_sections,
        list(extracted_fields.keys()),
    )

    return {
        "doc_type": normalised_type,
        "completeness_score": completeness_score,
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "extracted_fields": extracted_fields,
        "schema_version": _SCHEMA_VERSION,
    }
