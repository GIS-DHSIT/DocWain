from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.intelligence.domain_indexer import infer_domain


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _detect_output_format(query: str, task_type: str) -> str:
    """Detect desired output format using spaCy token analysis."""
    try:
        from src.nlp.nlu_engine import parse_query
        sem = parse_query(query)
        all_terms = set(sem.target_nouns + sem.context_words)
        if "json" in all_terms:
            return "json"
        if "table" in all_terms or task_type == "rank":
            return "table"
    except Exception:
        # Fallback: simple word check
        q_lower = query.lower().split()
        if "json" in q_lower:
            return "json"
        if "table" in q_lower or task_type == "rank":
            return "table"
    if task_type == "generate":
        return "cover_letter"
    if task_type == "summarize":
        return "bullets"
    if task_type == "compare":
        return "sections"
    if task_type == "timeline":
        return "chronological"
    if task_type == "extract":
        return "structured"
    return "free_text"


def _detect_task_type(query: str) -> str:
    if not query:
        return "summarize"
    try:
        from src.nlp.nlu_engine import classify_intent, parse_query

        tokens = query.strip().split()

        # Short queries (1-3 tokens) are often greetings ("hi", "hello", "hey there").
        # Always run the conversational check — it's cheap for short text.
        from src.nlp.nlu_engine import classify_conversational
        conv = classify_conversational(query)
        if conv and conv[0] == "GREETING":
            return "greet"

        intent = classify_intent(query)
        # Map NLU intent names to deterministic router task types
        _INTENT_TO_TASK = {
            "comparison": "compare",
            "ranking": "rank",
            "summary": "summarize",
            "multi_field": "extract",
            "analytics": "extract",
            "factual": "qa",
            "timeline": "timeline",
            "reasoning": "qa",
            "cross_document": "compare",
            "contact": "extract",
            "detail": "extract",
            "extract": "extract",
        }
        task = _INTENT_TO_TASK.get(intent, "qa")

        # Check for generation intent via query semantics
        # spaCy often misclassifies imperative verbs (Write, Create, Draft) as nouns,
        # so check both action_verbs and target_nouns
        sem = parse_query(query)
        all_words = sem.action_verbs + sem.target_nouns
        if any(v in ("generate", "create", "write", "draft", "compose") for v in all_words):
            return "generate"

        # Check for list intent
        if any(v in ("list", "show", "filter") for v in all_words):
            return "list"

        # Temporal/timeline intent: queries about history, progression, chronology
        if task == "qa" and _is_temporal_query(query):
            return "timeline"

        # Exhaustive/detail intent: queries explicitly asking for all details
        if task == "qa" and _is_detail_query(query):
            return "extract"

        return task
    except Exception:
        return "qa"


_TEMPORAL_RE = re.compile(
    r"\b(?:timeline|chronolog(?:y|ical)|history|progression|evolution|"
    r"over\s+(?:time|the\s+years)|year\s+by\s+year|month\s+by\s+month|"
    r"(?:from|since)\s+\d{4}\s+(?:to|until|through)\s+\d{4}|"
    r"when\s+did\s+\w+\s+(?:start|begin|join|leave|end)|"
    r"how\s+long\s+(?:has|have|did|does|is|was)\b|"
    r"career\s+(?:path|progression|trajectory))\b",
    re.IGNORECASE,
)


def _is_temporal_query(query: str) -> bool:
    """Detect queries about timelines, history, or temporal progression."""
    return bool(_TEMPORAL_RE.search(query))


_DETAIL_RE = re.compile(
    r"\b(?:all\s+(?:details?|information|data|fields?)|"
    r"(?:complete|full|comprehensive|detailed|exhaustive)\s+(?:profile|summary|overview|breakdown|report)|"
    r"everything\s+(?:about|regarding|on)|"
    r"tell\s+me\s+everything|"
    r"what\s+are\s+all\s+(?:the\s+)?|"
    r"list\s+(?:all|every)\s+|"
    r"(?:entire|whole)\s+(?:profile|document|record|report))\b",
    re.IGNORECASE,
)


def _is_detail_query(query: str) -> bool:
    """Detect queries asking for exhaustive/comprehensive detail extraction."""
    return bool(_DETAIL_RE.search(query))


def _infer_domain(query: str, session_state: Dict[str, Any], catalog: Dict[str, Any]) -> str:
    inferred = infer_domain(query)
    if inferred not in {"unknown", "generic"}:
        return inferred
    if session_state.get("active_domain"):
        return session_state["active_domain"]
    dominant = catalog.get("dominant_domains") or {}
    if dominant:
        return sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return inferred or "unknown"


_ner_nlp_cache = None


def _extract_person_from_query(query: str) -> Optional[str]:
    """Extract a person name from the query using spaCy NER.

    Uses named entity recognition to find PERSON entities in the query text.
    Falls back to ORG entities if no person is found (for vendor/company queries).
    """
    global _ner_nlp_cache
    if not query:
        return None
    try:
        import spacy
        if _ner_nlp_cache is None:
            _ner_nlp_cache = spacy.load("en_core_web_sm")
        doc = _ner_nlp_cache(query)
        # Prefer PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        # Fall back to ORG entities (for vendor/company queries)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return ent.text.strip()
    except Exception:
        pass
    return None


def _match_entity_cache(query: str, entities_cache: Dict[str, Any]) -> tuple[Optional[str], List[str]]:
    if not entities_cache:
        return None, []
    query_norm = _normalize(query)
    for ent in entities_cache.get("entities") or []:
        ent_type = str(ent.get("type") or "").upper()
        if ent_type and ent_type not in {"PERSON", "ORG", "ORGANIZATION", "VENDOR", "PATIENT"}:
            continue
        value = ent.get("value") or ""
        aliases = ent.get("aliases") or []
        candidates = [value] + [a for a in aliases if a]
        for candidate in candidates:
            if not candidate:
                continue
            if _normalize(candidate) in query_norm:
                doc_ids = [str(d) for d in (ent.get("document_ids") or []) if d]
                return str(value or candidate), doc_ids
    return None, []


def _match_document_names(query: str, catalog: Dict[str, Any]) -> List[str]:
    docs = catalog.get("documents") or []
    if not docs:
        return []
    query_norm = _normalize(query)
    matched: List[str] = []
    for doc in docs:
        source_name = doc.get("source_name") or ""
        doc_id = doc.get("document_id")
        if not source_name or not doc_id:
            continue
        import os
        base = source_name.lower()
        base_no_ext = os.path.splitext(base)[0]
        base_tokens = base_no_ext.replace("_", " ").replace("-", " ")
        if base in query_norm or base_no_ext in query_norm or base_tokens in query_norm:
            matched.append(str(doc_id))
    return matched


def _detect_section_focus(query: str, domain_hint: str) -> List[str]:
    """Detect which document sections are relevant to the query using NLU.

    Uses spaCy to parse the query's target nouns and context words,
    then maps them to section identifiers based on semantic similarity
    to section descriptions.
    """
    if not query:
        return []

    # Build section descriptions per domain — each maps a description to section IDs
    _SECTION_DESCRIPTIONS: Dict[str, Dict[str, List[str]]] = {
        "": {  # domain-agnostic
            "email address, phone number, contact information, how to reach someone": ["identity_contact"],
        },
        "resume": {
            "technical skills, programming languages, tools, technologies, tech stack": [
                "skills_technical", "tools_technologies", "skills_functional"
            ],
            "education, degree, university, college, academic qualification": ["education"],
            "work experience, employment history, professional background": ["experience"],
            "projects, portfolio, side projects, open source contributions": ["projects"],
            "certifications, licenses, professional credentials": ["certifications"],
            "professional summary, career objective, profile overview": ["summary_objective"],
        },
        "medical": {
            "patient details, patient information, patient demographics": [
                "identity_contact", "diagnoses_procedures", "medications", "notes"
            ],
            "diagnosis, medical procedures, conditions, diseases": ["diagnoses_procedures"],
            "medications, prescriptions, drugs, dosage": ["medications"],
            "lab results, test results, blood work, diagnostic tests": ["lab_results"],
            "clinical notes, doctor notes, physician remarks": ["notes"],
        },
        "invoice": {
            "line items, products, quantities, goods purchased": ["line_items", "tables"],
            "invoice number, invoice date, invoice reference": ["invoice_metadata"],
            "total amount, balance due, subtotal, financial summary": ["financial_summary"],
            "payment terms, due date, payment conditions": ["terms_conditions"],
            "bill to, ship to, vendor, supplier, customer, buyer": ["parties_addresses"],
        },
        "purchase_order": {
            "line items, products, quantities, goods purchased": ["line_items", "tables"],
            "invoice number, invoice date, invoice reference": ["invoice_metadata"],
            "total amount, balance due, subtotal, financial summary": ["financial_summary"],
            "payment terms, due date, payment conditions": ["terms_conditions"],
            "bill to, ship to, vendor, supplier, customer, buyer": ["parties_addresses"],
        },
        "tax": {
            "taxpayer, assessee, taxpayer identity": ["taxpayer_identity"],
            "identification numbers, PAN, EIN, SSN, TIN": ["ids"],
            "total tax, tax liability, amount due, tax payable": ["totals"],
            "deductions, exemptions, tax deductions": ["deductions"],
            "assessment year, fiscal year, tax year, financial period": ["ay_fy"],
            "payments, refunds, tax payments": ["payments"],
        },
        "bank_statement": {
            "account number, routing number, account identity": ["account_identity"],
            "transactions, debits, credits, deposits, withdrawals": ["transactions"],
            "balance, opening balance, closing balance": ["balances"],
            "fees, charges, interest, service charges": ["fees"],
        },
    }

    try:
        from src.nlp.nlu_engine import parse_query, get_embedder
        sem = parse_query(query)
        query_terms = set(sem.target_nouns + sem.context_words + sem.action_verbs)

        # Collect relevant section descriptions
        candidate_sections = {}
        candidate_sections.update(_SECTION_DESCRIPTIONS.get("", {}))
        candidate_sections.update(_SECTION_DESCRIPTIONS.get(domain_hint, {}))

        if not candidate_sections:
            return []

        # Use embedding similarity to find matching sections
        embedder = get_embedder()
        if embedder is not None:
            import numpy as np
            try:
                query_vec = embedder.encode([query], normalize_embeddings=True)[0]
                descriptions = list(candidate_sections.keys())
                desc_vecs = embedder.encode(descriptions, normalize_embeddings=True)

                # Score all descriptions and sort by relevance
                scored = []
                for i, desc in enumerate(descriptions):
                    score = float(np.dot(query_vec, desc_vecs[i]))
                    scored.append((score, desc))
                scored.sort(reverse=True, key=lambda x: x[0])

                # Take top matches: must score above 0.60 or within 10% of best
                focus: List[str] = []
                if scored:
                    best_score = scored[0][0]
                    for score, desc in scored:
                        if score >= 0.60 or (score >= 0.50 and score >= best_score * 0.90):
                            for section_id in candidate_sections[desc]:
                                if section_id not in focus:
                                    focus.append(section_id)
                return focus
            except Exception:
                pass

        # Fallback: spaCy noun overlap with section descriptions
        focus = []
        for desc, section_ids in candidate_sections.items():
            desc_words = set(desc.lower().split(", "))
            if query_terms & desc_words:
                for sid in section_ids:
                    if sid not in focus:
                        focus.append(sid)
        return focus

    except Exception:
        return []


def _detect_scope(query: str, session_state: Dict[str, Any], target_doc_ids: List[str]) -> str:
    lowered = _normalize(query)
    if target_doc_ids:
        return "current_document"
    if "this document" in lowered or "current document" in lowered:
        return "current_document"
    return "profile_all_docs"


_CONSTRAINT_RE = re.compile(
    r"(?:top|first|last|bottom)\s+(\d+)\b"
    r"|(\d+)\s+(?:most|least|best|worst)\b"
    r"|(?:under|below|less than|fewer than|max(?:imum)?)\s+[\$£€₹]?\s*(\d[\d,.]*[KkMm]?)"
    r"|(?:over|above|more than|greater than|min(?:imum)?|at least)\s+[\$£€₹]?\s*(\d[\d,.]*[KkMm]?)"
    r"|(?:between)\s+[\$£€₹]?\s*(\d[\d,.]*[KkMm]?)\s+(?:and|to|-)\s+[\$£€₹]?\s*(\d[\d,.]*[KkMm]?)",
    re.IGNORECASE,
)

_CONJUNCTION_SPLIT_RE = re.compile(
    r"\b(?:and also|and then|and|also|plus|as well as|additionally)\b",
    re.IGNORECASE,
)


def _detect_multi_intent(query: str) -> List[str]:
    """Detect multiple intents in compound queries.

    E.g. "Compare salaries AND list benefits" → ["compare", "list"]
    Returns list of task_type strings. Single-intent queries return [].
    """
    # Only split if query has conjunction + multiple action verbs
    parts = _CONJUNCTION_SPLIT_RE.split(query)
    if len(parts) < 2:
        return []

    intents: List[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < 5:
            continue
        detected = _detect_task_type(part)
        if detected and detected not in intents:
            intents.append(detected)

    return intents if len(intents) >= 2 else []


def _parse_numeric_with_suffix(val_str: str) -> float:
    """Parse a numeric string with optional K/M suffix.

    Examples: "50" → 50.0, "50K" → 50000.0, "1.5M" → 1500000.0
    """
    cleaned = val_str.replace(",", "").strip()
    suffix = cleaned[-1].upper() if cleaned and cleaned[-1].isalpha() else ""
    num_part = cleaned[:-1] if suffix else cleaned
    try:
        base = float(num_part)
    except ValueError:
        return 0.0
    if suffix == "K":
        return base * 1000
    if suffix == "M":
        return base * 1_000_000
    return base


def _extract_constraints(query: str) -> Dict[str, Any]:
    """Extract numeric and ranking constraints from the query.

    E.g. "top 3 candidates under $80k" → {"top_n": 3, "max_value": 80000}
    Supports K/M suffixes: "$50K" → 50000, "1.5M" → 1500000
    """
    constraints: Dict[str, Any] = {}
    for m in _CONSTRAINT_RE.finditer(query):
        # top N / first N
        if m.group(1):
            constraints["top_n"] = int(m.group(1))
        # N most/best
        elif m.group(2):
            constraints["top_n"] = int(m.group(2))
        # under/below/less than X
        elif m.group(3):
            constraints["max_value"] = _parse_numeric_with_suffix(m.group(3))
        # over/above/more than X
        elif m.group(4):
            constraints["min_value"] = _parse_numeric_with_suffix(m.group(4))
        # between X and Y
        elif m.group(5) and m.group(6):
            constraints["min_value"] = _parse_numeric_with_suffix(m.group(5))
            constraints["max_value"] = _parse_numeric_with_suffix(m.group(6))

    # Also detect K/M suffix patterns missed by main regex
    _KM_RE = re.compile(r'[\$£€₹]\s*(\d[\d,.]*[KkMm])\b')
    for km_match in _KM_RE.finditer(query):
        val = _parse_numeric_with_suffix(km_match.group(1))
        if val > 0 and "max_value" not in constraints and "min_value" not in constraints:
            # Use surrounding context to determine direction
            pre = query[:km_match.start()].lower()
            if any(w in pre for w in ("under", "below", "less", "max", "up to")):
                constraints["max_value"] = val
            elif any(w in pre for w in ("over", "above", "more", "min", "at least")):
                constraints["min_value"] = val

    return constraints


@dataclass
class DeterministicRoute:
    task_type: str
    domain_hint: str
    scope: str
    output_format: str
    section_focus: List[str] = field(default_factory=list)
    target_person: Optional[str] = None
    target_document_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    secondary_intents: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task_type": self.task_type,
            "domain_hint": self.domain_hint,
            "scope": self.scope,
            "output_format": self.output_format,
            "section_focus": self.section_focus,
            "target_person": self.target_person,
            "target_document_ids": self.target_document_ids,
            "reasons": self.reasons,
        }
        if self.secondary_intents:
            d["secondary_intents"] = self.secondary_intents
        if self.constraints:
            d["constraints"] = self.constraints
        return d


def route_query(
    query: str,
    session_state: Optional[Dict[str, Any]],
    catalog: Optional[Dict[str, Any]],
    entities_cache: Optional[Dict[str, Any]] = None,
) -> DeterministicRoute:
    session_state = session_state or {}
    catalog = catalog or {}
    entities_cache = entities_cache or {}

    task_type = _detect_task_type(query)
    domain_hint = _infer_domain(query, session_state, catalog)
    section_focus = _detect_section_focus(query, domain_hint)
    output_format = _detect_output_format(query, task_type)

    target_person = _extract_person_from_query(query)
    cache_person, target_doc_ids = _match_entity_cache(query, entities_cache)
    if not target_person and cache_person:
        target_person = cache_person

    explicit_doc_ids = _match_document_names(query, catalog)
    if explicit_doc_ids:
        target_doc_ids = list(set(target_doc_ids + explicit_doc_ids))

    scope = _detect_scope(query, session_state, target_doc_ids)
    reasons = []
    if target_doc_ids:
        reasons.append("entity_cache_match")
    if section_focus:
        reasons.append("section_focus_match")

    # Detect multi-intent and constraints
    secondary_intents = _detect_multi_intent(query)
    constraints = _extract_constraints(query)
    if secondary_intents:
        reasons.append("multi_intent")
    if constraints:
        reasons.append("has_constraints")
        # If top_n constraint found and task is not already rank, hint at ranking
        if "top_n" in constraints and task_type not in ("rank", "list"):
            task_type = "rank"

    return DeterministicRoute(
        task_type=task_type,
        domain_hint=domain_hint or "unknown",
        scope=scope,
        output_format=output_format,
        section_focus=section_focus,
        target_person=target_person,
        target_document_ids=target_doc_ids,
        reasons=reasons,
        secondary_intents=secondary_intents,
        constraints=constraints,
    )


__all__ = ["DeterministicRoute", "route_query"]
