"""
answerability_index.py

Builds a query answerability index for documents — a catalog of query types
each document can answer. Enables pre-filtering at retrieval time (skip
documents that cannot answer the query).
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any

logger = get_logger(__name__)

__all__ = [
    "_QUERY_TYPE_TAXONOMY",
    "build_answerability_index",
    "classify_query_type",
    "filter_by_answerability",
]

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

_QUERY_TYPE_TAXONOMY: dict[str, list[str]] = {
    # HR / Resume
    "candidate_experience": [
        "experience",
        "work history",
        "career",
        "positions",
        "employment",
    ],
    "skill_check": [
        "skills",
        "proficient",
        "expertise",
        "technologies",
        "tools",
        "knows",
    ],
    "education_check": [
        "education",
        "degree",
        "university",
        "qualification",
        "certified",
    ],
    "candidate_summary": [
        "summary",
        "overview",
        "profile",
        "about",
        "background",
    ],
    "role_fit": [
        "fit",
        "suitable",
        "qualified",
        "match",
        "eligible",
    ],
    "career_timeline": [
        "timeline",
        "chronological",
        "when did",
        "how long",
    ],
    "contact_info": [
        "contact",
        "email",
        "phone",
        "address",
        "reach",
    ],
    "salary_info": [
        "salary",
        "compensation",
        "pay",
        "ctc",
        "package",
    ],
    # Invoice / Financial
    "total_amount": [
        "total",
        "amount",
        "sum",
        "grand total",
        "balance",
    ],
    "vendor_info": [
        "vendor",
        "supplier",
        "bill from",
        "issued by",
    ],
    "line_item_detail": [
        "line item",
        "items",
        "products",
        "services",
        "quantity",
    ],
    "payment_due": [
        "due date",
        "payment terms",
        "when is payment",
        "net days",
    ],
    "tax_detail": [
        "tax",
        "vat",
        "gst",
        "sales tax",
    ],
    # Medical
    "diagnosis_info": [
        "diagnosis",
        "condition",
        "disease",
        "finding",
    ],
    "medication_info": [
        "medication",
        "drug",
        "prescription",
        "dosage",
    ],
    "lab_results": [
        "lab",
        "test results",
        "blood",
        "vitals",
    ],
    "treatment_plan": [
        "treatment",
        "therapy",
        "procedure",
        "surgery",
    ],
    "patient_info": [
        "patient",
        "demographics",
        "age",
        "gender",
    ],
    # Legal / Contract
    "contract_parties": [
        "parties",
        "between",
        "licensor",
        "licensee",
    ],
    "obligations": [
        "obligations",
        "responsibilities",
        "must",
        "shall",
    ],
    "termination": [
        "termination",
        "cancellation",
        "end date",
        "expiry",
    ],
    "liability": [
        "liability",
        "indemnity",
        "damages",
        "warranty",
    ],
    "governing_law": [
        "governing law",
        "jurisdiction",
        "applicable law",
    ],
    # Policy
    "policy_scope": [
        "scope",
        "applies to",
        "applicable",
        "coverage",
    ],
    "compliance_requirements": [
        "compliance",
        "requirements",
        "must comply",
        "mandatory",
    ],
    "policy_exceptions": [
        "exceptions",
        "exemptions",
        "waivers",
    ],
    # General
    "document_summary": [
        "summary",
        "overview",
        "what is this about",
    ],
    "specific_fact": [
        "what is",
        "how much",
        "when",
        "where",
        "who",
    ],
    "comparison": [
        "compare",
        "difference",
        "versus",
        "vs",
    ],
    "list_items": [
        "list",
        "enumerate",
        "all the",
        "how many",
    ],
}

# ---------------------------------------------------------------------------
# Doc-type → baseline answerable query types
# ---------------------------------------------------------------------------

_DOC_TYPE_BASELINE: dict[str, list[str]] = {
    # Resume / CV variants
    "resume": [
        "candidate_experience",
        "skill_check",
        "education_check",
        "candidate_summary",
        "career_timeline",
        "contact_info",
        "document_summary",
        "specific_fact",
    ],
    "cv": [
        "candidate_experience",
        "skill_check",
        "education_check",
        "candidate_summary",
        "career_timeline",
        "contact_info",
        "document_summary",
        "specific_fact",
    ],
    "curriculum_vitae": [
        "candidate_experience",
        "skill_check",
        "education_check",
        "candidate_summary",
        "career_timeline",
        "contact_info",
        "document_summary",
        "specific_fact",
    ],
    # Invoice / financial
    "invoice": [
        "total_amount",
        "vendor_info",
        "line_item_detail",
        "payment_due",
        "tax_detail",
        "document_summary",
        "specific_fact",
    ],
    "receipt": [
        "total_amount",
        "vendor_info",
        "line_item_detail",
        "tax_detail",
        "document_summary",
        "specific_fact",
    ],
    "purchase_order": [
        "total_amount",
        "vendor_info",
        "line_item_detail",
        "payment_due",
        "document_summary",
        "specific_fact",
    ],
    # Medical
    "medical_record": [
        "diagnosis_info",
        "medication_info",
        "lab_results",
        "treatment_plan",
        "patient_info",
        "document_summary",
        "specific_fact",
    ],
    "prescription": [
        "medication_info",
        "patient_info",
        "document_summary",
        "specific_fact",
    ],
    "lab_report": [
        "lab_results",
        "patient_info",
        "diagnosis_info",
        "document_summary",
        "specific_fact",
    ],
    # Legal / Contract
    "contract": [
        "contract_parties",
        "obligations",
        "termination",
        "liability",
        "governing_law",
        "document_summary",
        "specific_fact",
    ],
    "agreement": [
        "contract_parties",
        "obligations",
        "termination",
        "liability",
        "governing_law",
        "document_summary",
        "specific_fact",
    ],
    "nda": [
        "contract_parties",
        "obligations",
        "termination",
        "liability",
        "governing_law",
        "document_summary",
        "specific_fact",
    ],
    # Policy
    "policy": [
        "policy_scope",
        "compliance_requirements",
        "policy_exceptions",
        "document_summary",
        "specific_fact",
    ],
    "handbook": [
        "policy_scope",
        "compliance_requirements",
        "policy_exceptions",
        "document_summary",
        "specific_fact",
        "list_items",
    ],
}

# Minimal universal baseline for any document type not explicitly mapped
_UNIVERSAL_BASELINE: list[str] = [
    "document_summary",
    "specific_fact",
]

# ---------------------------------------------------------------------------
# Section-name → query types unlocked by that section existing in schema
# ---------------------------------------------------------------------------

_SECTION_SIGNALS: dict[str, list[str]] = {
    "experience": ["candidate_experience", "career_timeline"],
    "work experience": ["candidate_experience", "career_timeline"],
    "employment": ["candidate_experience", "career_timeline"],
    "skills": ["skill_check"],
    "technical skills": ["skill_check"],
    "education": ["education_check"],
    "qualifications": ["education_check"],
    "summary": ["candidate_summary", "document_summary"],
    "objective": ["candidate_summary"],
    "contact": ["contact_info"],
    "salary": ["salary_info"],
    "compensation": ["salary_info"],
    "total": ["total_amount"],
    "line items": ["line_item_detail"],
    "items": ["line_item_detail"],
    "payment": ["payment_due"],
    "tax": ["tax_detail"],
    "vendor": ["vendor_info"],
    "supplier": ["vendor_info"],
    "diagnosis": ["diagnosis_info"],
    "medications": ["medication_info"],
    "prescriptions": ["medication_info"],
    "lab results": ["lab_results"],
    "labs": ["lab_results"],
    "treatment": ["treatment_plan"],
    "procedures": ["treatment_plan"],
    "patient": ["patient_info"],
    "parties": ["contract_parties"],
    "obligations": ["obligations"],
    "termination": ["termination"],
    "liability": ["liability"],
    "governing law": ["governing_law"],
    "scope": ["policy_scope"],
    "compliance": ["compliance_requirements"],
    "exceptions": ["policy_exceptions"],
}

# ---------------------------------------------------------------------------
# Entity-type → query types that entity presence signals
# ---------------------------------------------------------------------------

_ENTITY_TYPE_SIGNALS: dict[str, list[str]] = {
    "AMOUNT": ["total_amount", "salary_info", "tax_detail"],
    "MONEY": ["total_amount", "salary_info", "tax_detail"],
    "CURRENCY": ["total_amount", "tax_detail"],
    "PERSON": ["candidate_summary", "patient_info", "contact_info"],
    "ORG": ["vendor_info", "contract_parties"],
    "ORGANIZATION": ["vendor_info", "contract_parties"],
    "DATE": ["career_timeline", "payment_due", "termination"],
    "TIME": ["career_timeline"],
    "GPE": ["governing_law", "contact_info"],
    "LOC": ["contact_info"],
    "EMAIL": ["contact_info"],
    "PHONE": ["contact_info"],
    "DRUG": ["medication_info"],
    "DISEASE": ["diagnosis_info"],
    "SYMPTOM": ["diagnosis_info"],
    "LAB": ["lab_results"],
    "PROCEDURE": ["treatment_plan"],
    "PERCENT": ["tax_detail", "total_amount"],
    "CARDINAL": ["list_items", "total_amount"],
    "PRODUCT": ["line_item_detail"],
    "LAW": ["governing_law", "compliance_requirements"],
    "SKILL": ["skill_check"],
    "DEGREE": ["education_check"],
    "UNIVERSITY": ["education_check"],
    "JOB_TITLE": ["candidate_experience", "role_fit"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIVERSAL_QUERY_TYPES: frozenset[str] = frozenset(
    {"specific_fact", "document_summary"}
)

def _normalise(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return " ".join(text.lower().split())

def _keywords_present_in_text(keywords: list[str], text_lower: str) -> bool:
    """Return True if at least one keyword phrase appears in *text_lower*."""
    return any(kw in text_lower for kw in keywords)

def _infer_from_full_text(full_text: str) -> list[str]:
    """Scan full_text for taxonomy keyword hits; return matched query types."""
    text_lower = _normalise(full_text)
    matched: list[str] = []
    for query_type, keywords in _QUERY_TYPE_TAXONOMY.items():
        normalised_kws = [_normalise(k) for k in keywords]
        if _keywords_present_in_text(normalised_kws, text_lower):
            matched.append(query_type)
    return matched

def _infer_from_sections(schema_result: dict[str, Any]) -> list[str]:
    """
    Inspect *schema_result* for section names/keys and map them to query types.

    Accepts any dict that may contain section names under common keys such as
    "sections", "section_titles", "detected_sections", "schema", or
    top-level keys directly.
    """
    section_candidates: list[str] = []

    # Common keys that hold section information
    for key in ("sections", "section_titles", "detected_sections", "schema"):
        value = schema_result.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    section_candidates.append(item)
                elif isinstance(item, dict):
                    for sub_key in ("title", "name", "label", "section"):
                        if isinstance(item.get(sub_key), str):
                            section_candidates.append(item[sub_key])
        elif isinstance(value, dict):
            section_candidates.extend(value.keys())

    # Also walk top-level keys of schema_result itself
    section_candidates.extend(str(k) for k in schema_result.keys())

    matched: list[str] = []
    for section_name in section_candidates:
        normalised = _normalise(section_name)
        for signal, query_types in _SECTION_SIGNALS.items():
            if signal in normalised:
                matched.extend(query_types)

    return matched

def _infer_from_entities(entities: list[dict[str, Any]] | list[str]) -> list[str]:
    """
    Map entity types/labels found in *entities* to answerable query types.

    *entities* can be:
    - a list of dicts with a "label", "type", or "entity_type" key
    - a list of plain strings (treated as entity type names directly)
    """
    matched: list[str] = []
    for ent in entities:
        if isinstance(ent, str):
            label = ent.upper()
        elif isinstance(ent, dict):
            label = (
                ent.get("label") or ent.get("type") or ent.get("entity_type") or ""
            ).upper()
        else:
            continue
        if label in _ENTITY_TYPE_SIGNALS:
            matched.extend(_ENTITY_TYPE_SIGNALS[label])
    return matched

def _normalise_doc_type(doc_type: str) -> str:
    """Normalise a doc_type string for lookup in _DOC_TYPE_BASELINE."""
    return _normalise(doc_type).replace(" ", "_").replace("-", "_")

def _compute_confidence(
    baseline_count: int,
    section_count: int,
    entity_count: int,
    text_count: int,
    total_answerable: int,
) -> float:
    """
    Heuristic confidence score in [0, 1] based on how many signal sources
    contributed to the final answerable set.
    """
    if total_answerable == 0:
        return 0.0

    score = 0.0
    # Doc-type baseline always contributes some base confidence
    if baseline_count > 0:
        score += 0.4
    if section_count > 0:
        score += 0.25
    if entity_count > 0:
        score += 0.20
    if text_count > 0:
        score += 0.15

    return round(min(score, 1.0), 3)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_answerability_index(
    doc_type: str,
    schema_result: dict[str, Any],
    entities: list[dict[str, Any]] | list[str],
    full_text: str,
    section_summaries: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build an answerability index for a single document.

    Parameters
    ----------
    doc_type:
        Document type string (e.g. "resume", "invoice", "contract").
    schema_result:
        Dict from the schema detector containing detected sections/structure.
    entities:
        List of entities extracted from the document.  Each element is either
        a plain entity-type string or a dict with a "label"/"type"/"entity_type"
        key.
    full_text:
        Raw text content of the document used for keyword scanning.
    section_summaries:
        Optional list of section summary strings; used as additional text for
        keyword scanning.

    Returns
    -------
    dict with keys:
        ``answerable_query_types`` - deduplicated list of query type strings
        ``doc_type``               - normalised doc type
        ``confidence``             - float in [0, 1]
    """
    normalised_doc_type = _normalise_doc_type(doc_type)

    # 1. Baseline from doc type
    baseline: list[str] = list(
        _DOC_TYPE_BASELINE.get(normalised_doc_type, _UNIVERSAL_BASELINE)
    )
    logger.debug(
        "answerability_index: doc_type=%r baseline=%d types",
        normalised_doc_type,
        len(baseline),
    )

    # 2. Section-based inference
    section_types: list[str] = []
    if schema_result:
        try:
            section_types = _infer_from_sections(schema_result)
            logger.debug(
                "answerability_index: section inference -> %d types",
                len(section_types),
            )
        except Exception:
            logger.exception("answerability_index: error in section inference")

    # 3. Entity-based inference
    entity_types: list[str] = []
    if entities:
        try:
            entity_types = _infer_from_entities(entities)
            logger.debug(
                "answerability_index: entity inference -> %d types",
                len(entity_types),
            )
        except Exception:
            logger.exception("answerability_index: error in entity inference")

    # 4. Full-text keyword scan
    text_types: list[str] = []
    combined_text = full_text
    if section_summaries:
        combined_text = full_text + " " + " ".join(section_summaries)
    if combined_text.strip():
        try:
            text_types = _infer_from_full_text(combined_text)
            logger.debug(
                "answerability_index: text keyword scan -> %d types",
                len(text_types),
            )
        except Exception:
            logger.exception("answerability_index: error in text keyword scan")

    # 5. Merge and deduplicate (preserve insertion order)
    seen: set[str] = set()
    answerable: list[str] = []
    for qt in baseline + section_types + entity_types + text_types:
        if qt not in seen:
            seen.add(qt)
            answerable.append(qt)

    # 6. Validate all types against taxonomy
    valid_types = set(_QUERY_TYPE_TAXONOMY.keys())
    answerable = [qt for qt in answerable if qt in valid_types]

    # 7. Always include universal types
    for universal_qt in _UNIVERSAL_QUERY_TYPES:
        if universal_qt not in seen:
            answerable.append(universal_qt)

    confidence = _compute_confidence(
        baseline_count=len(baseline),
        section_count=len(section_types),
        entity_count=len(entity_types),
        text_count=len(text_types),
        total_answerable=len(answerable),
    )

    result: dict[str, Any] = {
        "answerable_query_types": answerable,
        "doc_type": normalised_doc_type,
        "confidence": confidence,
    }
    logger.info(
        "answerability_index: doc_type=%r total_answerable=%d confidence=%.3f",
        normalised_doc_type,
        len(answerable),
        confidence,
    )
    return result

def classify_query_type(query: str) -> list[str]:
    """
    Classify a user query string into one or more query types from the taxonomy.

    Parameters
    ----------
    query:
        The raw user query string.

    Returns
    -------
    List of matching query type strings.  Falls back to ``["specific_fact"]``
    when no taxonomy keywords are matched.
    """
    if not query or not query.strip():
        logger.debug("classify_query_type: empty query -> specific_fact")
        return ["specific_fact"]

    query_lower = _normalise(query)
    matched: list[str] = []
    seen: set[str] = set()

    for query_type, keywords in _QUERY_TYPE_TAXONOMY.items():
        normalised_kws = [_normalise(k) for k in keywords]
        if _keywords_present_in_text(normalised_kws, query_lower):
            if query_type not in seen:
                seen.add(query_type)
                matched.append(query_type)

    if not matched:
        logger.debug(
            "classify_query_type: no taxonomy match for %r -> specific_fact",
            query,
        )
        return ["specific_fact"]

    logger.debug(
        "classify_query_type: query=%r -> %s",
        query,
        matched,
    )
    return matched

def filter_by_answerability(
    query_types: list[str],
    chunk_answerability: list[str],
) -> bool:
    """
    Determine whether a chunk/document is worth retrieving for the given query.

    Parameters
    ----------
    query_types:
        Query type(s) returned by ``classify_query_type()``.
    chunk_answerability:
        The ``answerable_query_types`` list from ``build_answerability_index()``.

    Returns
    -------
    ``True`` if the chunk can plausibly answer the query, ``False`` otherwise.

    Notes
    -----
    Universal query types (``"specific_fact"``, ``"document_summary"``) always
    return ``True`` regardless of the chunk's answerability list.
    """
    if not query_types:
        # No type information — allow through
        return True

    for qt in query_types:
        if qt in _UNIVERSAL_QUERY_TYPES:
            logger.debug(
                "filter_by_answerability: universal type %r -> True",
                qt,
            )
            return True

    chunk_set = set(chunk_answerability)
    for qt in query_types:
        if qt in chunk_set:
            logger.debug(
                "filter_by_answerability: matched %r in chunk answerability -> True",
                qt,
            )
            return True

    logger.debug(
        "filter_by_answerability: query_types=%s not matched in chunk %s -> False",
        query_types,
        list(chunk_answerability)[:10],
    )
    return False
