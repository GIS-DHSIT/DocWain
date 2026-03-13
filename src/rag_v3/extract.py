from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.api.config import Config
from .types import (
    Candidate,
    CandidateField,
    Clause,
    ClauseField,
    EvidenceSpan,
    EntitySummary,
    FieldValue,
    FieldValuesField,
    GenericSchema,
    HRSchema,
    InvoiceItem,
    InvoiceItemsField,
    InvoiceSchema,
    LegalSchema,
    LLMBudget,
    MISSING_REASON,
    MedicalSchema,
    MultiEntitySchema,
    PolicySchema,
)

logger = get_logger(__name__)

EXTRACT_TIMEOUT_MS = 15000  # 60s→15s: deterministic LLM fallback should be fast, not retry-length

@dataclass
class ExtractionResult:
    domain: str
    intent: str
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema | Any  # includes LLMResponseSchema

_INVOICE_HINTS = {
    "invoice",
    "amount due",
    "subtotal",
    "total",
    "bill to",
    "billing",
    "payment terms",
    "due date",
}

_HR_HINTS = {"resume", "curriculum vitae", "cv", "candidate"}

_LEGAL_HINTS = {"agreement", "contract", "clause", "terms", "warranty", "liability"}

def _nlu_subintent(query: str) -> Optional[str]:
    """Classify sub-intent using NLU engine (replaces keyword sets)."""
    try:
        from src.nlp.nlu_engine import classify_query_subintent
        return classify_query_subintent(query)
    except Exception as exc:
        logger.debug("Failed to classify query sub-intent via NLU", exc_info=True)
        return None

_CONTACT_KEYWORDS = frozenset({
    "contact", "contacts", "email", "emails", "phone", "phones",
    "reach", "linkedin", "number", "address", "mobile",
})

def _nlu_is_contact(query: str) -> bool:
    """Check if query asks for contact information using NLU + keyword fallback."""
    ql = (query or "").lower()
    # Fast keyword check first — reliable for common patterns
    if any(w in ql for w in _CONTACT_KEYWORDS):
        return True
    try:
        from src.nlp.nlu_engine import is_contact_query
        return is_contact_query(query)
    except Exception as exc:
        logger.debug("Failed to check contact query via NLU", exc_info=True)
        return False

def schema_extract(
    *,
    query: str,
    chunks: List[Any],
    llm_client: Optional[Any],
    budget: LLMBudget,
    correlation_id: Optional[str] = None,
    scope_document_id: Optional[str] = None,
    domain_hint: Optional[str] = None,
    intent_hint: Optional[str] = None,
    query_focus: Optional[Any] = None,
    embedder: Any = None,
    intent_parse: Any = None,
) -> ExtractionResult:
    domain, intent = _infer_domain_intent(query, chunks, domain_hint=domain_hint,
                                          intent_hint=intent_hint, intent_parse=intent_parse,
                                          correlation_id=correlation_id)

    # Handle domain mismatch: query asks for a domain that doesn't match the actual chunks
    if domain == "mismatch":
        query_domain = _ml_query_domain(query, intent_parse=intent_parse) or "unknown"
        chunk_domain = _majority_chunk_domain(chunks) or "unknown"
        _DOMAIN_LABEL = {"hr": "resumes", "resume": "resumes", "invoice": "invoices",
                         "legal": "legal documents", "medical": "medical records",
                         "policy": "policy documents", "insurance": "insurance documents"}
        asked = _DOMAIN_LABEL.get(query_domain, query_domain)
        available = _DOMAIN_LABEL.get(chunk_domain, chunk_domain)
        msg = (f"No {asked} found in this profile. The available documents are {available}. "
               f"To analyze {asked}, please upload the relevant {asked} documents to this profile.")
        schema = GenericSchema(facts=FieldValuesField(items=[FieldValue(label=None, value=msg, evidence_spans=[])]))
        return ExtractionResult(domain=chunk_domain, intent=intent, schema=schema)

    # LLM-first architecture: When LLM is available, use it as the primary
    # extraction path. The LLM understands user intent and produces structured,
    # query-aware responses. Deterministic extraction is the fallback when
    # LLM is unavailable or fails.
    schema = None
    _used_llm = False

    if llm_client and budget.consume():
        llm_schema = _llm_extract(domain, intent, query, chunks, llm_client, correlation_id)
        if llm_schema is not None:
            schema = llm_schema
            _used_llm = True

    # Fallback to deterministic extraction if LLM unavailable or failed
    if schema is None:
        schema = _deterministic_extract(domain, intent, query, chunks, query_focus=query_focus, embedder=embedder)

    if _detect_multi_entity_collision(domain, schema, chunks, scope_document_id, intent):
        schema = _build_multi_entity_schema(domain, schema, chunks)
        domain = "multi"

    return ExtractionResult(domain=domain, intent=intent, schema=schema)

# Content validators: multi-word phrases that MUST appear in chunk text
# for a metadata domain tag to be trusted.  Single generic words like
# "total", "policy", "experience" cause massive false positives on
# technical manuals, product docs, etc.  Only multi-word phrases here.
_INVOICE_CONTENT_VALIDATORS = (
    "invoice number", "amount due", "bill to", "total due", "payment terms",
    "purchase order", "remittance", "balance due", "unit price", "invoice total",
    "invoice date", "net amount",
)
_HR_CONTENT_VALIDATORS = (
    "work experience", "professional experience", "career objective",
    "curriculum vitae", "resume", "professional summary", "work history",
    "years of experience", "technical skills", "key skills",
    "skills:", "education:", "experience:",
)
_LEGAL_CONTENT_VALIDATORS = (
    "governing law", "indemnification", "in witness whereof",
    "terms and conditions", "hereinafter", "contract",
)
_MEDICAL_CONTENT_VALIDATORS = (
    "patient", "diagnosis", "diagnoses", "medication", "prescription",
    "clinical", "symptoms", "vitals", "blood pressure",
    "lab result", "medical history", "allerg", "dosage",
)
_POLICY_CONTENT_VALIDATORS = (
    "insurance policy", "policy number", "coverage period",
    "premium", "deductible", "policyholder", "sum insured",
)

# ALL structured domains need content validation to prevent misclassification
_DOMAIN_CONTENT_VALIDATORS: Dict[str, tuple] = {
    "medical": _MEDICAL_CONTENT_VALIDATORS,
    "invoice": _INVOICE_CONTENT_VALIDATORS,
    "hr": _HR_CONTENT_VALIDATORS,
    "legal": _LEGAL_CONTENT_VALIDATORS,
    "policy": _POLICY_CONTENT_VALIDATORS,
}

_DOMAIN_MAP = {"resume": "hr", "invoice": "invoice", "legal": "legal",
               "policy": "policy", "report": "report"}

def _ml_query_domain(query: str, intent_parse: Any = None) -> Optional[str]:
    """Detect query domain using trained ML classifier.

    Strategy:
    1. Use intent_parse.domain if available and confident
    2. Fall back to semantic domain classifier
    """
    # 1. Use intent_parse.domain if available and confident
    if intent_parse and getattr(intent_parse, "domain", None) not in ("generic", None, ""):
        mapped = _DOMAIN_MAP.get(intent_parse.domain)
        if mapped:
            return mapped

    # 2. Use semantic domain classifier for the query itself
    try:
        from src.intelligence.domain_classifier import classify_domain
        result = classify_domain(query)
        if result and not result.uncertain and result.domain not in ("generic",):
            mapped = _DOMAIN_MAP.get(result.domain, result.domain)
            return mapped if mapped in ("hr", "invoice", "legal", "policy", "medical") else None
    except Exception:  # noqa: BLE001
        pass

    # 3. Fall back to IntentDomainClassifier
    try:
        from src.intent.intent_classifier import get_intent_classifier
        clf = get_intent_classifier()
        if clf is not None and getattr(clf, "_trained", False):
            from src.intent.llm_intent import _get_embedder
            embedder = _get_embedder()
            if embedder is not None:
                vec = embedder.encode([query], normalize_embeddings=True)[0]
                result = clf.predict(vec)
                if result and result.get("domain_confidence", 0) >= 0.70:
                    return _DOMAIN_MAP.get(result.get("domain"))
    except Exception:  # noqa: BLE001
        pass
    return None

def _majority_chunk_domain(chunks: List[Any]) -> Optional[str]:
    """Determine domain from chunk metadata with mandatory content validation.

    Every structured domain tag from metadata is validated against the actual
    chunk text.  If the text lacks domain-specific phrases, the tag is
    rejected and the domain falls back to None (→ generic extraction).
    This prevents technical manuals tagged as "invoice" or "medical" from
    being rendered with the wrong schema.
    """
    domain_counts: Dict[str, int] = {}
    for chunk in chunks:
        meta = getattr(chunk, "meta", None) or {}
        d = str(meta.get("doc_domain") or meta.get("doc_type") or "").lower().strip()
        if d and d not in ("generic", ""):
            domain_counts[d] = domain_counts.get(d, 0) + 1
    if not domain_counts:
        return None

    best = max(domain_counts, key=domain_counts.get)

    # Normalize through canonical domain map for consistency across classifiers
    try:
        from src.intelligence.domain_classifier import normalize_domain
        best = normalize_domain(best)
    except ImportError as exc:
        logger.debug("normalize_domain not available for domain normalization", exc_info=True)

    domain_map = {
        "resume": "hr", "hr": "hr", "cv": "hr",
        "invoice": "invoice", "billing": "invoice", "purchase_order": "invoice",
        "financial": "invoice",
        "legal": "legal", "contract": "legal",
        "medical": "medical", "clinical": "medical", "patient": "medical",
        "policy": "policy", "insurance": "policy",
    }
    mapped = domain_map.get(best, best)

    # Content validation: EVERY structured domain must be confirmed by
    # actual content OR filename.  This is the core guard against misclassification.
    validators = _DOMAIN_CONTENT_VALIDATORS.get(mapped)
    if validators:
        sample = " ".join((getattr(c, "text", "") or "")[:500] for c in chunks[:10]).lower()
        # Also check source filenames — "Resume.pdf", "Invoice_123.pdf" are strong signals
        source_names = " ".join(
            str((getattr(c, "meta", None) or {}).get("source_name") or "")
            for c in chunks[:10]
        ).lower()
        content_match = any(phrase in sample for phrase in validators)
        filename_match = any(phrase in source_names for phrase in validators)
        if not content_match and not filename_match:
            logger.debug(
                "Domain tag '%s' rejected — no content validators matched in chunk text or filenames",
                mapped,
            )
            return None  # Domain tag not confirmed by content

    # For domains not in the validator map, don't trust metadata alone
    if mapped not in _DOMAIN_CONTENT_VALIDATORS:
        return None

    return mapped

def _infer_domain_intent(
    query: str,
    chunks: List[Any],
    domain_hint: Optional[str] = None,
    intent_hint: Optional[str] = None,
    intent_parse: Any = None,
    correlation_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Infer domain from chunk metadata and query intent from keywords.

    Domain detection uses ML classifier first, then chunk metadata majority
    vote so that resume chunks route through the structured HR extractor
    instead of the generic one.
    """
    # Detect domain: ML classifier first, then chunk metadata majority
    # Cross-check: flag mismatch when query asks for invoice/legal but chunks are HR
    # (prevents InvoiceSchema extraction from resume text)
    # Trust query override when it says HR — user may be correcting misclassified chunks
    strategy_used = "fallback"
    domain = (domain_hint or "").strip().lower()
    if domain and domain != "generic":
        strategy_used = "hint"
    else:
        query_domain = _ml_query_domain(query, intent_parse=intent_parse)
        chunk_domain = _majority_chunk_domain(chunks)
        if query_domain and chunk_domain:
            _DOMAIN_FAMILY = {
                "hr": "hr", "resume": "hr",
                "invoice": "invoice",
                "legal": "legal_policy", "policy": "legal_policy",
                "insurance": "legal_policy",
                "medical": "medical",
                "report": "generic", "generic": "generic",
            }
            q_family = _DOMAIN_FAMILY.get(query_domain, "generic")
            c_family = _DOMAIN_FAMILY.get(chunk_domain, "generic")
            # Only flag mismatch for genuinely incompatible specific domains.
            # Invoice mismatch is reliable (invoice vs non-invoice is clear).
            # Medical mismatch is reliable (medical vs non-medical is clear,
            # content validators confirm via patient/diagnosis/medication terms).
            # Legal/policy mismatch is unreliable — "terms", "treatment plan"
            # etc. trigger false positives across medical/policy domains.
            _MISMATCH_DOMAINS = {"invoice", "medical"}
            _any_matching = False
            for _c in (chunks or []):
                _cm = getattr(_c, "meta", None) or {}
                _cd = str(_cm.get("doc_domain") or _cm.get("doc_type") or "").lower().strip()
                if _DOMAIN_FAMILY.get(_cd, "generic") == q_family:
                    _any_matching = True
                    break
            if q_family != c_family and q_family in _MISMATCH_DOMAINS and not _any_matching:
                # Query asks for invoice/medical but NO chunks match that domain → mismatch
                domain = "mismatch"
                strategy_used = "ml_classifier"
            else:
                # Domains match, or query says HR/generic/report — prefer chunk domain
                # when query domain is ambiguous (report/generic), otherwise trust query
                domain = chunk_domain if q_family == "generic" else query_domain
                strategy_used = "ml_classifier" if q_family != "generic" else "chunk_metadata"
        elif query_domain:
            domain = query_domain
            strategy_used = "ml_classifier"
        elif chunk_domain:
            domain = chunk_domain
            strategy_used = "chunk_metadata"
        else:
            domain = "generic"

    intent = _normalize_intent_hint(intent_hint) or "summary"
    lowered_query = (query or "").lower()

    # Domain-agnostic intent detection using NLU
    if intent == "summary":
        # Check contact first (highest priority — LLMs often refuse contact info)
        if _nlu_is_contact(query):
            intent = "contact"
        else:
            # 1. spaCy structural analysis — action verbs are unambiguous signals
            #    Note: spaCy often misclassifies imperative verbs (Compare, List,
            #    Show) as NOUN/PROPN, placing them in target_nouns instead of
            #    action_verbs.  Check ALL extracted words against verb sets.
            _verb_intent = None
            try:
                from src.nlp.nlu_engine import parse_query
                sem = parse_query(query)
                _COMPARE_VERBS = {"compare", "contrast", "versus", "differentiate", "distinguish"}
                _RANK_VERBS = {"rank", "rate", "score", "order", "shortlist", "prioritize"}
                _LIST_VERBS = {"list", "show", "enumerate", "display", "pull", "give",
                               "present", "exhibit", "retrieve", "fetch", "extract"}
                _SUMMARIZE_VERBS = {"summarize", "summarise", "condense", "outline",
                                     "brief", "recap", "overview"}
                _ANALYZE_VERBS = {"analyze", "analyse", "assess", "evaluate", "review",
                                   "examine", "investigate", "study", "inspect"}
                _EXPLAIN_VERBS = {"explain", "describe", "understand", "reason",
                                   "cause", "because", "clarify", "elaborate",
                                   "justify", "interpret"}
                _all_words = set(sem.action_verbs) | set(sem.target_nouns) | set(sem.context_words)
                # Also check raw query for "why/how does/what caused" patterns
                _ql = query.lower()
                _has_reasoning_pattern = any(p in _ql for p in (
                    "why ", "how does", "how did", "what caused", "what led to",
                    "reason for", "explain ", "what is the cause",
                ))
                if any(v in _COMPARE_VERBS for v in _all_words):
                    _verb_intent = "compare"
                elif any(v in _RANK_VERBS for v in _all_words):
                    _verb_intent = "rank"
                elif any(v in _LIST_VERBS for v in _all_words):
                    _verb_intent = "list"
                elif any(v in _SUMMARIZE_VERBS for v in _all_words):
                    _verb_intent = "summary"
                elif any(v in _ANALYZE_VERBS for v in _all_words) or _has_reasoning_pattern:
                    _verb_intent = "analysis"
                elif any(v in _EXPLAIN_VERBS for v in _all_words):
                    _verb_intent = "analysis"
            except Exception as exc:
                logger.debug("Failed NLU-based verb intent classification", exc_info=True)

            if _verb_intent:
                intent = _verb_intent
            else:
                # 2. NLU sub-intent classification (semantic similarity)
                subintent = _nlu_subintent(query)
                _SUBINTENT_MAP = {
                    "contact": "contact",
                    "fit_rank": "rank",
                    "totals": "totals",
                    "product_item": "products_list",
                }
                # Guard: disambiguate "totals" from "terms/conditions" using
                # spaCy target nouns — "payment terms" is about conditions,
                # not amounts.  Also require at least one financial/numeric
                # noun to avoid misclassifying generic questions like
                # "what is the weather?" as totals.
                if subintent == "totals":
                    try:
                        _tsem = parse_query(query)
                        _NON_TOTAL_NOUNS = {"term", "terms", "condition", "conditions",
                                            "policy", "deadline"}
                        _TOTAL_CONFIRM_NOUNS = {"total", "amount", "sum", "cost", "price",
                                                "balance", "invoice", "payment", "bill",
                                                "charge", "fee", "dollar", "money", "budget"}
                        if any(n in _NON_TOTAL_NOUNS for n in _tsem.target_nouns):
                            subintent = None  # not a totals query
                        elif not any(n in _TOTAL_CONFIRM_NOUNS for n in
                                     set(_tsem.target_nouns) | set(_tsem.context_words)):
                            subintent = None  # no financial signal — likely misclassified
                    except Exception as exc:
                        logger.debug("Failed NLU guard for totals sub-intent", exc_info=True)
                # Guard: "product_item" requires product/item-related nouns
                if subintent == "product_item":
                    try:
                        _psem = parse_query(query)
                        _PRODUCT_NOUNS = {"product", "item", "line", "goods", "merchandise",
                                          "order", "sku", "quantity", "unit", "catalog"}
                        if not any(n in _PRODUCT_NOUNS for n in
                                   set(_psem.target_nouns) | set(_psem.context_words)):
                            subintent = None
                    except Exception as exc:
                        logger.debug("Failed NLU guard for product_item sub-intent", exc_info=True)
                if subintent and subintent in _SUBINTENT_MAP:
                    intent = _SUBINTENT_MAP[subintent]
                else:
                    intent = "facts"

    logger.info(
        "Domain/intent resolved: domain=%s intent=%s strategy=%s",
        domain, intent, strategy_used,
        extra={"stage": "infer_domain_intent", "correlation_id": correlation_id},
    )
    return domain, intent

def _looks_like_hr_total(text: str) -> bool:
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "total years",
            "years of experience",
            "total experience",
            "total years of experience",
        )
    )

def _deterministic_extract(domain: str, intent: str, query: str, chunks: List[Any], *, query_focus: Optional[Any] = None, embedder: Any = None):
    """Route to domain-specific extractor when domain is known, else generic."""
    if domain in ("hr", "resume"):
        return _extract_hr(chunks)
    if domain == "invoice":
        return _extract_invoice(chunks, embedder=embedder)
    if domain in ("legal", "contract"):
        return _extract_legal(chunks, embedder=embedder)
    if domain == "policy":
        return _extract_policy(chunks, embedder=embedder)
    if domain == "medical":
        return _extract_medical(chunks, embedder=embedder)
    return _extract_document_intelligence(query, chunks, query_focus=query_focus)

def _extract_document_intelligence(query: str, chunks: List[Any], *, query_focus: Optional[Any] = None) -> GenericSchema:
    """Universal document intelligence extraction.

    Works for ANY document type by analysing content structure
    (KV pairs, lists, sections, contact info, relevant sentences)
    rather than relying on domain classification.
    """
    keywords = _keywords(query)
    facts: List[FieldValue] = []

    # ── Phase 1: Group chunks by document ───────────────────────────────
    doc_chunks: Dict[str, List[Tuple[str, str, str, str]]] = {}  # doc_id -> [(chunk_id, text, doc_name, section_kind)]
    doc_names: Dict[str, str] = {}

    for chunk in chunks:
        raw_text = getattr(chunk, "text", "") or ""
        text = _extract_full_text_content(raw_text)
        chunk_id = getattr(chunk, "id", "")
        meta = getattr(chunk, "meta", None) or {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or chunk_id)
        source = getattr(chunk, "source", None)
        doc_name = ""
        if source is not None:
            doc_name = str(getattr(source, "document_name", "") or "")
        if not doc_name:
            doc_name = str(meta.get("source_name") or "")
        if doc_name and doc_id not in doc_names:
            doc_names[doc_id] = doc_name

        section_kind = str(meta.get("section_kind") or meta.get("chunk_kind") or "")
        section_title = str(meta.get("section_title") or "")
        if not section_kind or section_kind in ("section_text", "misc"):
            section_kind = _infer_section_kind_from_content(text, section_title)

        doc_chunks.setdefault(doc_id, []).append((chunk_id, text, doc_name, section_kind))

    # ── Phase 2: Extract structured facts per document (per-doc dedup) ──
    for doc_id, chunk_list in doc_chunks.items():
        doc_name = doc_names.get(doc_id, "")
        doc_seen: set = set()  # Per-document dedup — same label+value in different docs is kept

        for chunk_id, text, _, section_kind in chunk_list:
            # Handle full-document dumps by splitting into sections
            if _text_has_multiple_sections(text):
                split_sections = _split_document_into_sections(text)
                if split_sections:
                    for sec_kind, sec_text in split_sections:
                        _extract_structured_facts(
                            sec_text, chunk_id, doc_name, sec_kind,
                            keywords, facts, doc_seen,
                        )
                    continue

            _extract_structured_facts(
                text, chunk_id, doc_name, section_kind,
                keywords, facts, doc_seen,
            )

    # Global dedup: remove exact (label, value, doc_name) triples
    global_seen: set = set()
    deduped: List[FieldValue] = []
    for fact in facts:
        key = _normalize_key(f"{fact.document_name or ''}:{fact.label or ''}:{fact.value}")
        if key not in global_seen:
            global_seen.add(key)
            deduped.append(fact)
    facts = deduped

    # ── Phase 3: Contact extraction pass ────────────────────────────────
    contact_seen: set = set()
    for doc_id, chunk_list in doc_chunks.items():
        doc_name = doc_names.get(doc_id, "")
        combined_text = " ".join(text for _, text, _, _ in chunk_list)
        contact = _extract_contact_fields_comprehensive(combined_text)
        for contact_type, values in [
            ("Email", contact.get("emails")),
            ("Phone", contact.get("phones")),
            ("LinkedIn", contact.get("linkedins")),
        ]:
            for val in values or []:
                key = _normalize_key(f"{doc_name}:{contact_type}:{val}")
                if key not in contact_seen:
                    contact_seen.add(key)
                    facts.append(FieldValue(
                        label=contact_type, value=val,
                        document_name=doc_name or None,
                        section="Contact",
                        evidence_spans=[_span("", val)],
                    ))

    # ── Phase 4: Score and sort by query relevance ──────────────────────
    facts = _score_and_sort_facts(facts, keywords, query)

    # ── Phase 4b: Query-focus soft cap — when focus has clear field_tags,
    # limit non-matching facts to 3 per document to emphasize relevant content.
    if query_focus and not query_focus.is_exhaustive and getattr(query_focus, "field_tags", None):
        from .query_focus import score_fact_relevance as _sfr
        matching: list = []
        non_matching: list = []
        for f in facts:
            rel = _sfr(f.label or "", f.value or "", query_focus)
            if rel >= 0.15:
                matching.append(f)
            else:
                non_matching.append(f)
        facts = matching + non_matching[:3]

    # ── Phase 5: Limit and fallback ─────────────────────────────────────
    facts = facts[:60]

    if not facts and chunks:
        fallback_seen: set = set()
        for chunk in chunks[:5]:
            text = getattr(chunk, "text", "") or ""
            chunk_id = getattr(chunk, "id", "")
            for sentence in _split_sentences(text)[:3]:
                cleaned = sentence.strip()
                if cleaned and len(cleaned) >= 10 and not _is_metadata_garbage(cleaned, max_length=500):
                    key = _normalize_key(cleaned)
                    if key not in fallback_seen:
                        fallback_seen.add(key)
                        facts.append(FieldValue(
                            label=None, value=cleaned,
                            evidence_spans=[_span(chunk_id, cleaned)],
                        ))

    return GenericSchema(facts=_field_values_field(facts))

def _overlaps_existing_facts(sentence: str, facts: List[FieldValue]) -> bool:
    """Check if sentence content is already captured by extracted facts."""
    words = set(sentence.lower().split())
    if len(words) < 3:
        return False
    for fact in facts:
        fact_words = set(fact.value.lower().split())
        if not fact_words:
            continue
        overlap = len(words & fact_words) / len(words)
        if overlap > 0.6:
            return True
    return False

def _extract_structured_facts(
    text: str,
    chunk_id: str,
    doc_name: str,
    section_kind: str,
    keywords: List[str],
    facts: List[FieldValue],
    seen: set,
) -> None:
    """Extract facts from a section of text using structural patterns."""
    section_label = section_kind.replace("_", " ").title() if section_kind else None
    current_heading = section_label  # Track inline headings

    for line in _split_lines(text):
        cleaned = line.strip()
        if not cleaned or len(cleaned) < 3:
            continue
        if _is_metadata_garbage(cleaned, max_length=500):
            continue

        # Detect heading lines: "Medications:", "Lab Results:", "Key Metrics:"
        heading_match = re.match(r"^([A-Za-z][A-Za-z\s/\-]{1,40}?)\s*:\s*$", cleaned)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            continue

        # Strategy A: KV pairs — colon-only delimiter, hyphens allowed in labels
        kv_match = re.match(r"^([A-Za-z][A-Za-z\s/\-]{1,40}?)\s*:\s*(.+)$", cleaned)
        if kv_match:
            label = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            if value and len(value) > 1:
                key = _normalize_key(f"{label}:{value}")
                if key not in seen:
                    seen.add(key)
                    facts.append(FieldValue(
                        label=label,
                        value=_sanitize_field_value(value) or value,
                        document_name=doc_name or None,
                        section=current_heading or section_label,
                        evidence_spans=[_span(chunk_id, cleaned)],
                    ))
            continue

        # Strategy B: List items (bullets, numbered) — inherit current heading
        if cleaned.startswith(("-", "•", "*")) or re.match(r"^\d+\.\s", cleaned):
            item = cleaned.lstrip("-•* ").strip()
            item = re.sub(r"^\d+\.\s*", "", item).strip()
            if item and len(item) > 2:
                key = _normalize_key(item)
                if key not in seen:
                    seen.add(key)
                    facts.append(FieldValue(
                        label=current_heading or section_label,
                        value=item,
                        document_name=doc_name or None,
                        section=current_heading or section_label,
                        evidence_spans=[_span(chunk_id, cleaned)],
                    ))
            continue

        # Strategy B2: Standalone lines under a heading (education, names, etc.)
        if current_heading and len(cleaned) < 150 and not cleaned.endswith(":"):
            key = _normalize_key(cleaned)
            if key not in seen:
                seen.add(key)
                facts.append(FieldValue(
                    label=current_heading,
                    value=cleaned,
                    document_name=doc_name or None,
                    section=current_heading,
                    evidence_spans=[_span(chunk_id, cleaned)],
                ))
            continue

    # Strategy C: Keyword-matched sentences (skip full-text dumps and redundant content)
    _MAX_SENTENCE_LEN = 300
    for sentence in _split_sentences(text):
        cleaned = sentence.strip()
        if not cleaned or len(cleaned) < 10:
            continue
        if len(cleaned) > _MAX_SENTENCE_LEN:
            continue
        if _is_metadata_garbage(cleaned, max_length=500):
            continue
        if _overlaps_existing_facts(cleaned, facts):
            continue
        if keywords and any(w in cleaned.lower() for w in keywords):
            key = _normalize_key(cleaned)
            if key not in seen:
                seen.add(key)
                facts.append(FieldValue(
                    label=None, value=cleaned,
                    document_name=doc_name or None,
                    section=current_heading or section_label,
                    evidence_spans=[_span(chunk_id, cleaned)],
                ))

def _score_and_sort_facts(
    facts: List[FieldValue],
    keywords: List[str],
    query: str,
    intent: str = "facts",
) -> List[FieldValue]:
    """Score facts by query relevance and intent, sort descending."""
    _INTENT_BOOST_LABELS = {
        "contact": {"email", "phone", "linkedin", "contact", "address", "mobile"},
        "rank": {"experience", "skills", "technical skills", "certifications", "education", "years"},
        "compare": {"experience", "skills", "technical skills", "education", "summary", "name"},
        "list": {"name", "title", "summary"},
        "summary": {"name", "summary", "experience", "skills", "education"},
        "facts": set(),
    }
    boost_labels = _INTENT_BOOST_LABELS.get(intent, set())

    def _fact_score(fact: FieldValue) -> float:
        score = 0.0
        text = (fact.value or "").lower()
        label = (fact.label or "").lower()
        # Keyword overlap (0.3 per keyword match)
        if keywords:
            score += sum(0.3 for kw in keywords if kw in text or kw in label)
        # Labeled facts (structured data) get a boost
        if fact.label:
            score += 0.2
        # Intent-specific label boost
        if boost_labels and label in boost_labels:
            score += 0.5
        # Contact boost ONLY when intent is contact
        if intent == "contact" and label in ("email", "phone", "linkedin"):
            score += 1.0
        return score

    return sorted(facts, key=_fact_score, reverse=True)

# ---------------------------------------------------------------------------
# OCR text normalization — cleans up common OCR artifacts before ML classification
# ---------------------------------------------------------------------------

_OCR_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_OCR_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_OCR_BULLET_ARTIFACT_RE = re.compile(r"^[•·▪■□◦‣⁃]\s*")
_OCR_SEMICOLON_LABEL_RE = re.compile(r"^([A-Z][A-Za-z\s]{2,30});(\s)")

def _normalize_ocr_line(line: str) -> str:
    """Normalize a single OCR-extracted line: strip control chars, collapse whitespace,
    remove bullet artifacts, fix semicolons misread as colons after labels."""
    line = _OCR_CONTROL_CHARS_RE.sub("", line)
    line = _OCR_MULTI_SPACE_RE.sub(" ", line)
    line = _OCR_BULLET_ARTIFACT_RE.sub("", line)
    line = _OCR_SEMICOLON_LABEL_RE.sub(r"\1:\2", line)
    return line.strip()

def _is_table_chunk(text: str) -> bool:
    """Detect if text looks like a pipe-delimited or tab-delimited table."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return False
    pipe_count = sum(1 for l in lines if l.count("|") >= 2)
    tab_count = sum(1 for l in lines if l.count("\t") >= 2)
    return (pipe_count >= 2) or (tab_count >= 2)

def _table_to_kv_pairs(text: str) -> List[str]:
    """Convert pipe/tab-delimited table rows into 'label: value' pairs for classification."""
    pairs: List[str] = []
    lines = text.strip().splitlines()
    headers: List[str] = []
    for line in lines:
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
        elif "\t" in line:
            cells = [c.strip() for c in line.split("\t") if c.strip()]
        else:
            continue
        # Skip separator lines (e.g., "---|---|---")
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        if not headers:
            headers = cells
            continue
        for i, cell in enumerate(cells):
            if i < len(headers) and cell:
                pairs.append(f"{headers[i]}: {cell}")
    return pairs

# ---------------------------------------------------------------------------
# ML-based line classification helper (shared by all domain extractors)
# ---------------------------------------------------------------------------

def _ml_extract_lines(chunks, domain: str, embedder):
    """Collect lines from chunks, classify with ML, return structured results.

    Returns a list of (line_text, (chunk_id, meta), LineClassification) tuples.
    When embedder is None, falls back to heuristic classification.
    """
    from .line_classifier import classify_lines as _classify_lines

    all_lines: List[str] = []
    line_meta: List[Tuple[str, dict]] = []

    for chunk in chunks:
        text = _extract_full_text_content(getattr(chunk, "text", "") or "")
        chunk_id = getattr(chunk, "id", "")
        meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}

        # Table-aware extraction: convert table rows to "label: value" pairs
        if _is_table_chunk(text):
            for pair in _table_to_kv_pairs(text):
                if pair and len(pair) >= 3:
                    all_lines.append(pair)
                    line_meta.append((chunk_id, meta))

        for line in _split_lines(text):
            cleaned = _normalize_ocr_line(line)
            if cleaned and len(cleaned) >= 3:
                all_lines.append(cleaned)
                line_meta.append((chunk_id, meta))

    if not all_lines:
        return []

    classifications = _classify_lines(all_lines, domain, embedder)
    return list(zip(all_lines, line_meta, classifications))

def _extract_invoice(chunks: List[Any], embedder: Any = None) -> InvoiceSchema:
    items: List[InvoiceItem] = []
    totals: List[FieldValue] = []
    parties: List[FieldValue] = []
    terms: List[FieldValue] = []
    invoice_metadata: List[FieldValue] = []
    seen_items = set()

    _CATEGORY_LISTS = {
        "items": None,  # items uses InvoiceItem, handled separately
        "totals": totals,
        "parties": parties,
        "terms": terms,
        "invoice_metadata": invoice_metadata,
    }

    classified = _ml_extract_lines(chunks, "invoice", embedder)
    for line, (chunk_id, meta), cls in classified:
        if cls.role == "skip":
            continue

        key = _normalize_key(line)

        if cls.category == "items" or cls.role == "bullet":
            desc = cls.value or line
            if cls.label:
                desc = cls.value
            item_key = _normalize_key(desc)
            if desc and item_key not in seen_items:
                seen_items.add(item_key)
                items.append(InvoiceItem(
                    description=desc,
                    evidence_spans=[_span(chunk_id, line)],
                ))
            continue

        if cls.role in ("kv_pair", "heading", "narrative"):
            if key in seen_items:
                continue
            seen_items.add(key)
            label = cls.label or None
            value = cls.value or line
            if not value or len(value) < 2:
                continue

            target = _CATEGORY_LISTS.get(cls.category)
            if target is not None:
                target.append(FieldValue(
                    label=label, value=value,
                    evidence_spans=[_span(chunk_id, line)],
                ))
            elif cls.category == "other" and cls.role == "kv_pair":
                # Unclassified KV — put in terms as general info
                terms.append(FieldValue(
                    label=label, value=value,
                    evidence_spans=[_span(chunk_id, line)],
                ))

    return InvoiceSchema(
        items=_items_field(items),
        totals=_field_values_field(totals),
        parties=_field_values_field(parties),
        terms=_field_values_field(terms),
        invoice_metadata=_field_values_field(invoice_metadata),
    )

def _extract_hr(chunks: List[Any]) -> HRSchema:
    """Extract HR data from chunks using intelligent content and metadata analysis."""
    candidates: List[Candidate] = []
    by_doc: Dict[str, Candidate] = {}

    # Group chunks by document and inferred section kind
    chunks_by_section: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}  # doc_id -> section_kind -> [(chunk_id, text, section_title)]
    doc_names: Dict[str, str] = {}  # doc_id -> source filename

    for chunk in chunks:
        raw_text = getattr(chunk, "text", "") or ""
        # CRITICAL: Clean extraction artifacts from the text
        text = _extract_full_text_content(raw_text)
        chunk_id = getattr(chunk, "id", "")
        meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or chunk_id)

        # Safely extract doc_name from source if it exists
        source = getattr(chunk, "source", None)
        doc_name = ""
        if source is not None:
            doc_name = str(getattr(source, "document_name", "") or "")
        if not doc_name:
            doc_name = str(meta.get("source_name") or "")

        # Remember the filename for this document (used for name extraction fallback)
        if doc_name and doc_id not in doc_names:
            doc_names[doc_id] = doc_name

        section_kind = str(meta.get("section_kind") or meta.get("chunk_kind") or "")
        section_title = str(meta.get("section_title") or "")

        # CRITICAL: If text contains multiple section headers, it's a full document dump.
        # Split it into proper sections instead of treating it as a single section.
        if _text_has_multiple_sections(text):
            split_sections = _split_document_into_sections(text)
            if split_sections:
                if doc_id not in chunks_by_section:
                    chunks_by_section[doc_id] = {}
                for sec_kind, sec_text in split_sections:
                    if sec_kind not in chunks_by_section[doc_id]:
                        chunks_by_section[doc_id][sec_kind] = []
                    chunks_by_section[doc_id][sec_kind].append((chunk_id, sec_text, sec_kind))
                continue  # Skip normal single-section processing

        # If section_kind is generic, too broad (projects covers mixed content),
        # or belongs to a different domain (e.g., invoice section kinds on a
        # resume document), re-infer from actual content.
        _WRONG_DOMAIN_KINDS = {
            "financial_summary", "line_items", "invoice_metadata",
            "parties_addresses", "terms_conditions",
        }
        _TOO_BROAD_KINDS = {"projects", "section_summary"}
        inferred = False
        if not section_kind or section_kind in ("section_text", "misc") or section_kind in _WRONG_DOMAIN_KINDS or section_kind in _TOO_BROAD_KINDS:
            section_kind = _infer_section_kind_from_content(text, section_title)
            inferred = True

        # Create a unique key combining section_kind and section_title ONLY if we inferred the section kind
        # This prevents different content types from being merged when they all have generic metadata
        section_key = section_kind
        if inferred and section_title and section_title.lower() not in section_kind.lower():
            # Add section title as secondary key only when content was inferred
            section_key = f"{section_kind}:{section_title}"

        # Initialize doc structure
        if doc_id not in chunks_by_section:
            chunks_by_section[doc_id] = {}
        if section_key not in chunks_by_section[doc_id]:
            chunks_by_section[doc_id][section_key] = []

        # If text is very long (full document dump), extract only the section-specific portion
        if len(text) > 800 and section_kind not in ("section_text", "experience"):
            text = _extract_section_from_text(text, section_kind)

        chunks_by_section[doc_id][section_key].append((chunk_id, text, section_title))

    # Extract candidates from organized chunks
    for doc_id, sections in chunks_by_section.items():
        doc_name = doc_names.get(doc_id, "")

        cand = Candidate(
            name=None,
            evidence_spans=[],
            source_type=None,  # Will be set after processing
            missing_reason={},
        )

        # Priority 0: Try to extract name from source filename first
        # (most reliable when canonical_text is garbage/empty)
        if doc_name:
            filename_name = _name_from_filename(doc_name)
            if filename_name:
                cand.name = filename_name

        by_doc[doc_id] = cand
        candidates.append(cand)

        span_seen = {span.snippet for span in (cand.evidence_spans or [])}

        # Extract from summary/objective sections
        summary_sections = ["summary_objective", "section_summary", "professional_summary"]
        for sec_kind in summary_sections:
            if sec_kind in sections:
                for chunk_id, text, section_title in sections[sec_kind]:
                    if text and not cand.experience_summary:
                        cleaned = " ".join(text.split())[:500]  # Take first 500 chars
                        cand.experience_summary = cleaned
                        _append_span(cand, chunk_id, cleaned[:120], span_seen)
                    # Summary/objective often contains years of experience
                    if text and not cand.total_years_experience:
                        years = _extract_years_experience(text)
                        if years:
                            cand.total_years_experience = years
                            _append_span(cand, chunk_id, years, span_seen)
                    if cand.experience_summary:
                        break
                if cand.experience_summary:
                    break

        # Fallback: check composite keys for summary sections
        if not cand.experience_summary:
            for section_key in sections:
                if any(s in section_key.lower() for s in ["summary", "objective", "profile"]):
                    for chunk_id, text, section_title in sections[section_key]:
                        if text and not cand.experience_summary:
                            cleaned = " ".join(text.split())[:500]
                            cand.experience_summary = cleaned
                            _append_span(cand, chunk_id, cleaned[:120], span_seen)
                            break
                    if cand.experience_summary:
                        break

        # Extract from experience sections (for years of experience and summary)
        experience_sections = ["experience", "professional_experience", "employment"]
        for sec_kind in experience_sections:
            if sec_kind in sections:
                for chunk_id, text, section_title in sections[sec_kind]:
                    if not cand.total_years_experience:
                        # Look for year patterns in experience section
                        years = _extract_years_experience(text)
                        if years:
                            cand.total_years_experience = years
                            _append_span(cand, chunk_id, years, span_seen)
                    # Extract first meaningful line as summary if missing
                    if not cand.experience_summary:
                        for line in _split_lines(text)[:3]:
                            cleaned = line.strip()
                            if cleaned and len(cleaned) > 20:
                                cand.experience_summary = cleaned
                                _append_span(cand, chunk_id, cleaned[:120], span_seen)
                                break

        # Extract from skills sections using section kind metadata
        if "skills_technical" in sections:
            for chunk_id, text, section_title in sections["skills_technical"]:
                tech_items = _flatten_skill_block(_split_lines(text))
                if tech_items:
                    cand.technical_skills = _merge_list(cand.technical_skills, tech_items)
                    for item in tech_items[:3]:
                        _append_span(cand, chunk_id, item[:120], span_seen)
                    break  # Only process first tech skills section

        # Fallback: check composite keys (section_kind:title format) for tech skills
        if not cand.technical_skills:
            for section_key in sections:
                if section_key.startswith("skills_technical:") or ("skill" in section_key.lower() and "technical" in section_key.lower()):
                    for chunk_id, text, section_title in sections[section_key]:
                        tech_items = _flatten_skill_block(_split_lines(text))
                        if tech_items:
                            cand.technical_skills = _merge_list(cand.technical_skills, tech_items)
                            for item in tech_items[:3]:
                                _append_span(cand, chunk_id, item[:120], span_seen)
                            break

        # Fallback: check generic skills section
        if not cand.technical_skills and "skills" in sections:
            for chunk_id, text, section_title in sections["skills"]:
                tech_items = _flatten_skill_block(_split_lines(text))
                if tech_items:
                    cand.technical_skills = _merge_list(cand.technical_skills, tech_items)
                    for item in tech_items[:3]:
                        _append_span(cand, chunk_id, item[:120], span_seen)
                    break

        # Extract functional skills
        if "skills_functional" in sections:
            for chunk_id, text, section_title in sections["skills_functional"]:
                func_items = _flatten_skill_block(_split_lines(text))
                if func_items:
                    cand.functional_skills = _merge_list(cand.functional_skills, func_items)
                    for item in func_items[:3]:
                        _append_span(cand, chunk_id, item[:120], span_seen)
                    break

        # Fallback: check composite keys for functional skills
        if not cand.functional_skills:
            for section_key in sections:
                if section_key.startswith("skills_functional:") or ("skill" in section_key.lower() and "functional" in section_key.lower()):
                    for chunk_id, text, section_title in sections[section_key]:
                        func_items = _flatten_skill_block(_split_lines(text))
                        if func_items:
                            cand.functional_skills = _merge_list(cand.functional_skills, func_items)
                            for item in func_items[:3]:
                                _append_span(cand, chunk_id, item[:120], span_seen)
                            break

        # Extract certifications
        if "certifications" in sections:
            for chunk_id, text, section_title in sections["certifications"]:
                cert_items = _flatten_skill_block(_split_lines(text))
                if cert_items:
                    cand.certifications = _merge_list(cand.certifications, cert_items)
                    for item in cert_items[:2]:
                        _append_span(cand, chunk_id, item[:120], span_seen)
                    break

        # Fallback: check composite keys for certifications
        if not cand.certifications:
            for section_key in sections:
                if section_key.startswith("certifications:") or "certif" in section_key.lower():
                    for chunk_id, text, section_title in sections[section_key]:
                        cert_items = _flatten_skill_block(_split_lines(text))
                        if cert_items:
                            cand.certifications = _merge_list(cand.certifications, cert_items)
                            for item in cert_items[:2]:
                                _append_span(cand, chunk_id, item[:120], span_seen)
                            break

        # Extract education
        if "education" in sections:
            for chunk_id, text, section_title in sections["education"]:
                edu_items = _parse_education_block(_split_lines(text))
                if edu_items:
                    cand.education = _merge_list(cand.education, edu_items)
                    for item in edu_items[:2]:
                        _append_span(cand, chunk_id, item[:120], span_seen)
                    break

        # Fallback: check composite keys for education
        if not cand.education:
            for section_key in sections:
                if section_key.startswith("education:") or "education" in section_key.lower():
                    for chunk_id, text, section_title in sections[section_key]:
                        edu_items = _parse_education_block(_split_lines(text))
                        if edu_items:
                            cand.education = _merge_list(cand.education, edu_items)
                            for item in edu_items[:2]:
                                _append_span(cand, chunk_id, item[:120], span_seen)
                            break

        # Extract achievements
        if "achievements" in sections:
            for chunk_id, text, section_title in sections["achievements"]:
                achieve_items = _flatten_skill_block(_split_lines(text))
                if achieve_items:
                    cand.achievements = _merge_list(cand.achievements, achieve_items[:3])
                    for item in achieve_items[:2]:
                        _append_span(cand, chunk_id, item[:120], span_seen)
                    break

        # Mine section_text chunks (failed re-inference) for missed fields
        for section_key in sections:
            if not section_key.startswith("section_text"):
                continue
            for chunk_id, text, section_title in sections[section_key]:
                if not text:
                    continue
                # Try extracting skills from unclassified chunks
                if not cand.technical_skills:
                    tech_items = _flatten_skill_block(_split_lines(text))
                    if tech_items:
                        cand.technical_skills = _merge_list(cand.technical_skills, tech_items)
                # Try extracting education
                if not cand.education:
                    edu_items = _parse_education_block(_split_lines(text))
                    if edu_items:
                        cand.education = _merge_list(cand.education, edu_items)
                # Try extracting years of experience
                if not cand.total_years_experience:
                    years = _extract_years_experience(text)
                    if years:
                        cand.total_years_experience = years

        # Extract contact info from all chunks
        for section_chunks in sections.values():
            for chunk_id, text, _ in section_chunks:
                contact = _extract_contact_fields_comprehensive(text)
                if contact:
                    if contact.get("emails"):
                        cand.emails = _merge_list(cand.emails, contact["emails"])
                    if contact.get("phones"):
                        cand.phones = _merge_list(cand.phones, contact["phones"])
                    if contact.get("linkedins"):
                        cand.linkedins = _merge_list(cand.linkedins, contact["linkedins"])
                    if contact.get("names") and not cand.name:
                        cand.name = contact["names"][0]

        # Set source type based on document name first, then content analysis
        if doc_name:
            cand.source_type = _infer_source_type(doc_name)
        if not cand.source_type:
            # Check all chunk text for LinkedIn indicators
            _has_linkedin_markers = False
            for section_chunks in sections.values():
                for _, sec_text, _ in section_chunks:
                    if sec_text and ("linkedin.com" in sec_text.lower() or "linkedin profile" in sec_text.lower()):
                        _has_linkedin_markers = True
                        break
                if _has_linkedin_markers:
                    break
            if _has_linkedin_markers and not doc_name:
                cand.source_type = "LinkedIn profile"
            else:
                # HR domain with sections like skills/education/experience → Resume
                _hr_sections = {"skills_technical", "education", "experience", "certifications",
                                "skills_functional", "summary_objective", "professional_experience"}
                if any(sk.split(":")[0] in _hr_sections for sk in sections):
                    cand.source_type = "Resume"
                else:
                    cand.source_type = "Resume"  # Default for HR domain

        # CRITICAL FALLBACK: If no data extracted via section analysis,
        # scan ALL chunk content directly for relevant patterns
        all_texts = []
        for section_chunks in sections.values():
            for chunk_id, text, _ in section_chunks:
                if text:
                    all_texts.append((chunk_id, text))

        if all_texts:
            combined_text = " ".join([t for _, t in all_texts])

            # Fallback: Extract name from identity/contact sections first
            # Handle both bare keys and compound keys (e.g., "summary_objective:Career Objective")
            if not cand.name:
                for sec_kind in ("identity_contact", "contact", "summary_objective"):
                    if cand.name:
                        break
                    for skey, sval in sections.items():
                        if skey == sec_kind or skey.startswith(f"{sec_kind}:"):
                            for _, sec_text, _ in sval:
                                name = _extract_name_from_text(sec_text)
                                if name:
                                    cand.name = name
                                    break

            # Fallback: Extract name from all combined text
            if not cand.name:
                cand.name = _extract_name_from_text(combined_text)

            # Fallback: Extract years of experience if missing
            if not cand.total_years_experience:
                years = _extract_years_experience(combined_text)
                if years:
                    cand.total_years_experience = years

            # Fallback: Extract skills directly from text if missing
            if not cand.technical_skills:
                cand.technical_skills = _extract_skills_from_text(combined_text, "technical")
                if cand.technical_skills and all_texts:
                    fb_cid, fb_txt = all_texts[0]
                    for skill in cand.technical_skills[:3]:
                        _append_span(cand, fb_cid, skill[:120], span_seen)

            if not cand.functional_skills:
                cand.functional_skills = _extract_skills_from_text(combined_text, "functional")
                if cand.functional_skills and all_texts:
                    fb_cid, fb_txt = all_texts[0]
                    for skill in cand.functional_skills[:3]:
                        _append_span(cand, fb_cid, skill[:120], span_seen)

            # Fallback: Extract education if missing
            if not cand.education:
                cand.education = _extract_education_from_text(combined_text)
                if cand.education and all_texts:
                    fb_cid, _ = all_texts[0]
                    for item in cand.education[:2]:
                        _append_span(cand, fb_cid, item[:120], span_seen)

            # Fallback: Extract certifications if missing
            if not cand.certifications:
                cand.certifications = _extract_certifications_from_text(combined_text)
                if cand.certifications and all_texts:
                    fb_cid, _ = all_texts[0]
                    for item in cand.certifications[:2]:
                        _append_span(cand, fb_cid, item[:120], span_seen)

            # Fallback: Extract experience summary if missing
            if not cand.experience_summary:
                # Look for Professional Summary or Objective section content
                summary_patterns = [
                    r"(?:professional\s+)?summary\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n[A-Z][A-Z]|\Z)",
                    r"objective\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n[A-Z][A-Z]|\Z)",
                    r"profile\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n[A-Z][A-Z]|\Z)",
                ]
                for pattern in summary_patterns:
                    match = re.search(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        summary = match.group(1).strip()
                        if len(summary) > 30:
                            cand.experience_summary = summary[:500]
                            break

                # Fallback: Take first substantial paragraph as summary
                # Prefer experience/project descriptions over skill lists
                if not cand.experience_summary:
                    # Collect multi-line summary from first meaningful text
                    summary_lines = []
                    for _, text in all_texts:
                        lines = text.split('\n')
                        for line in lines:
                            cleaned = line.strip()
                            if not cleaned:
                                if summary_lines:
                                    break  # End of paragraph
                                continue
                            lower = cleaned.lower()
                            # Skip headers and metadata
                            if any(header in lower for header in
                                ['skills', 'education', 'contact', 'certification',
                                 'key skills', 'technical', 'personal details', 'extracted document',
                                 'section_id', 'chunk_type']):
                                continue
                            # Skip comma-separated skill/tool lists (e.g., "Python, SQL, Docker, React")
                            comma_count = cleaned.count(',')
                            if comma_count >= 3 and len(cleaned) < 300:
                                avg_item_len = len(cleaned) / (comma_count + 1)
                                if avg_item_len < 25:
                                    continue  # Short comma-separated items = skill list
                            # Skip bullet-point lists of tools (e.g., "• Python • SQL • Docker")
                            if cleaned.count('•') >= 3:
                                continue
                            if len(cleaned) > 20:
                                summary_lines.append(cleaned)
                                if len(' '.join(summary_lines)) > 200:
                                    break
                        if summary_lines:
                            break
                    if summary_lines:
                        cand.experience_summary = ' '.join(summary_lines)[:500]

            # Fallback: Extract contact info if missing
            if not cand.emails or not cand.phones:
                contact = _extract_contact_fields_comprehensive(combined_text)
                if not cand.emails and contact.get("emails"):
                    cand.emails = contact["emails"]
                if not cand.phones and contact.get("phones"):
                    cand.phones = contact["phones"]
                if not cand.linkedins and contact.get("linkedins"):
                    cand.linkedins = contact["linkedins"]
                if not cand.name and contact.get("names"):
                    cand.name = contact["names"][0]

    # Sanitize all candidate fields to remove any remaining artifacts
    for cand in candidates:
        _sanitize_candidate(cand)

    # Last-resort name extraction: use the canonical _name_from_filename helper
    for doc_id, cand in by_doc.items():
        if not cand.name and doc_id in doc_names:
            filename_name = _name_from_filename(doc_names[doc_id])
            if filename_name:
                cand.name = filename_name

    for cand in candidates:
        missing = cand.missing_reason or {}
        if not cand.name:
            missing["name"] = MISSING_REASON
        if not cand.total_years_experience:
            missing["total_years_experience"] = MISSING_REASON
        if not cand.experience_summary:
            missing["experience_summary"] = MISSING_REASON
        if not cand.technical_skills:
            missing["technical_skills"] = MISSING_REASON
        if not cand.functional_skills:
            missing["functional_skills"] = MISSING_REASON
        if not cand.certifications:
            missing["certifications"] = MISSING_REASON
        if not cand.education:
            missing["education"] = MISSING_REASON
        if not cand.achievements:
            missing["achievements"] = MISSING_REASON
        if not cand.emails:
            missing["emails"] = MISSING_REASON
        if not cand.phones:
            missing["phones"] = MISSING_REASON
        if not cand.linkedins:
            missing["linkedins"] = MISSING_REASON
        if not cand.source_type:
            missing["source_type"] = MISSING_REASON
        cand.missing_reason = missing

    return HRSchema(candidates=_candidates_field(candidates))

# ============================================================================
# Candidate Field Sanitization
# ============================================================================

_METADATA_GARBAGE_RE = re.compile(
    r"(?:section_id|chunk_type|page_start|page_end|start_page|end_page|"
    r"layout_confidence|ocr_confidence|doc_quality|canonical_json|"
    r"chunk_candidates|layout_spans|key_value_pairs|ExtractedDocument|"
    r"ChunkCandidate|Section\(|Table\(|'text':|'title':|'page':)",
    re.IGNORECASE,
)

def _is_metadata_garbage(value: str, max_length: int = 200) -> bool:
    """Check if a string value contains Qdrant/extraction metadata artifacts."""
    if not value:
        return True
    # Check for known metadata patterns
    if _METADATA_GARBAGE_RE.search(value):
        return True
    # Check for raw dict/list patterns
    if re.match(r"^\s*[\[{'\"](?:section_id|chunk_type|page|text|title|csv)", value):
        return True
    # Check for very long entries (likely document dumps, not individual items)
    if max_length and len(value) > max_length:
        return True
    # Check for Python repr fragments
    if "='" in value and ("section" in value.lower() or "chunk" in value.lower()):
        return True
    return False

def _sanitize_field_value(value: Optional[str], max_length: int = 500) -> Optional[str]:
    """Clean a single string field value."""
    if not value:
        return value
    # Remove metadata patterns - use max_length for garbage detection threshold
    if _is_metadata_garbage(value, max_length=max_length):
        return None
    # Convert escaped newlines and clean stray 'n' at line starts
    cleaned = value.replace("\\n", "\n")
    # Remove stray 'n' artifacts from escaped newlines
    cleaned = re.sub(r"(?:^|\n)\s*n\s+(?=[A-Z])", "\n", cleaned)
    cleaned = re.sub(r"(?<=[a-zA-Z0-9.,:;)>])\s*\bn\s+(?=[A-Z])", " ", cleaned)
    # Clean whitespace
    cleaned = " ".join(cleaned.split())
    # Remove leading/trailing commas
    cleaned = cleaned.strip().strip(",").strip()
    # Limit length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned if cleaned else None

def _sanitize_field_list(items: Optional[List[str]], max_item_length: int = 100) -> Optional[List[str]]:
    """Clean a list of string items, removing garbage entries."""
    if not items:
        return items
    # Patterns that indicate metadata leakage into skill/list items
    _artifact_patterns = re.compile(
        r"section_title|section_kind|section_id|chunk_kind|chunk_type"
        r"|\bSection\s+\d+\s*:"
        r"|^\[[\w\s]+\]"
        r"|^text\s*:"
        r"|^key\s*:"
        r"|^value\s*:"
        r"|^title\s*[:\s]"
        r"|Extracted\s+Document"
        r"|full_text\b"
        r"|Chunk\s*Candidate"
        r"|\bSUMMARY\b.*\bRESPONSIBILITY\b"
        r"|\bROLES?\s+AND\s+RESPONSIBILIT",
        re.IGNORECASE,
    )

    # Section headers and common non-skill words that leak as individual items
    _section_header_items = {
        "summary", "objective", "skills", "experience", "education",
        "details", "introduction", "contact", "personal", "profile",
        "references", "declaration", "hobbies", "interests",
        "certifications", "achievements", "awards", "languages",
        "projects", "project details",
    }

    # Pattern for project titles that leak as skills
    _project_title_re = re.compile(
        r"^\s*(?:Projects?\s+)?(?:[A-Z][a-zA-Z]+\s+){1,4}"
        r"(?:Classifier|Predictor|Tracker|Monitor|Detector|Generator|Builder|System|App|Application|Tool|Platform)\b",
        re.IGNORECASE,
    )

    # Date range patterns that pollute skill/cert entries
    _date_tail_re = re.compile(
        r"\s*\(?\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\s*[\-–—]\s*"
        r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)?\s*\d{4}\s*\)?\s*$",
        re.IGNORECASE,
    )
    # Sentence-start patterns (bullet points + action verbs)
    _sentence_re = re.compile(
        r"^[●•▪■]\s+"
        r"|^(?:Implemented|Developed|Designed|Created|Built|Led|Managed|Worked|Used|Applied|Integrated|Achieved|Configured)\b",
        re.IGNORECASE,
    )

    cleaned = []
    seen = set()
    for item in items:
        if not item or not isinstance(item, str):
            continue
        # Skip metadata garbage
        if _is_metadata_garbage(item):
            continue
        # Clean whitespace and newlines
        item = " ".join(item.replace("\\n", " ").split()).strip().strip(",").strip().rstrip(".")
        if not item:
            continue
        # Strip date range tails: "ANN (OCT 2023 – NOV 2023)" → "ANN"
        item = _date_tail_re.sub("", item).strip()
        if not item:
            continue
        # Skip items that look like sentences (bullet point + verb)
        if _sentence_re.search(item) and len(item) > 30:
            continue
        # Skip items containing metadata artifact patterns
        if _artifact_patterns.search(item):
            continue
        # Skip single-word section headers used as skill items
        if item.strip().lower().rstrip('.') in _section_header_items:
            continue
        # Skip project titles that leak as skills (e.g., "Stock Profit Classifier")
        if _project_title_re.search(item):
            continue
        # Skip items starting with conjunctions (split sentence fragments)
        if re.match(r"^(?:and|or|but|nor)\s+", item, re.IGNORECASE) and len(item) < 50:
            continue
        # Skip education/location data that leaked into skill lists
        item_lower = item.lower().strip()
        if re.search(r'\b(?:cgpa|gpa)\b', item_lower):
            continue
        if re.search(r'\bproficiency\b', item_lower) and not re.search(r'\b(?:python|java|sql|sap|aws)\b', item_lower):
            continue
        # Indian states/cities that are NOT skills
        _location_tokens = {
            'tamil nadu', 'karnataka', 'maharashtra', 'delhi', 'mumbai', 'bangalore',
            'bengaluru', 'chennai', 'hyderabad', 'pune', 'erode', 'coimbatore',
            'kolkata', 'noida', 'gurugram', 'trivandrum', 'kerala', 'india',
        }
        if item_lower in _location_tokens:
            continue
        # Skip very long entries (likely paragraphs)
        if len(item) > max_item_length:
            continue
        # Skip entries that are mostly non-alphanumeric
        alnum_count = sum(1 for c in item if c.isalnum() or c == ' ')
        if len(item) > 5 and alnum_count / len(item) < 0.5:
            continue
        # Deduplicate
        key = item.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)
    return cleaned if cleaned else None

def _sanitize_candidate(cand: Candidate) -> None:
    """Sanitize all fields of a Candidate to remove metadata artifacts."""
    # Clean name — reject names that are actually skill/keyword lists
    if cand.name:
        cleaned_name = _sanitize_field_value(cand.name, max_length=80)
        # Reject comma-delimited lists (skill lists, not names)
        if cleaned_name and ',' in cleaned_name:
            cleaned_name = None
        # Reject names containing non-name keywords
        if cleaned_name and not _looks_like_name(cleaned_name):
            cleaned_name = None
        # Reject names starting with "Document Content" or similar metadata
        if cleaned_name and re.match(r"(?i)^(?:document\s+content|section\s+\d)", cleaned_name):
            cleaned_name = None
        cand.name = cleaned_name

    # Clean experience summary
    if cand.experience_summary:
        summary = cand.experience_summary
        # Strip "Document Content" prefix (extraction artifact)
        summary = re.sub(r"^Document\s+Content\s*", "", summary, flags=re.IGNORECASE).strip()
        # Strip "Introduction" prefix
        summary = re.sub(r"^Introduction\s+", "", summary).strip()
        # Strip leading "Projects" + everything up to first bullet point
        summary = re.sub(
            r"^Projects?\b[^●•▪■]*(?=[●•▪■])",
            "", summary, count=1, flags=re.IGNORECASE,
        ).strip()
        # Strip orphaned closing parentheses/brackets at start
        summary = re.sub(r"^[)\]}\s]+", "", summary).strip()
        # Strip leading bullet/action items
        summary = re.sub(r"^[●•▪■\-*]\s*", "", summary)
        # Strip phone/email/contact prefix that leaked into summary
        summary = re.sub(r"^(?:\+?\d[\d\s\-]{6,15}\s*)", "", summary).strip()
        # Truncate at "(cid:" artifacts
        cid_idx = summary.find("(cid:")
        if cid_idx > 50:
            summary = summary[:cid_idx].rstrip()
        cand.experience_summary = _sanitize_field_value(summary, max_length=500) if summary else None

    # Clean total years experience
    if cand.total_years_experience:
        cand.total_years_experience = _sanitize_field_value(cand.total_years_experience, max_length=30)

    # Clean source type
    if cand.source_type:
        cand.source_type = _sanitize_field_value(cand.source_type, max_length=50)

    # Clean list fields
    cand.technical_skills = _sanitize_field_list(cand.technical_skills)
    cand.functional_skills = _sanitize_field_list(cand.functional_skills)
    cand.certifications = _sanitize_field_list(cand.certifications)
    cand.education = _sanitize_field_list(cand.education, max_item_length=150)
    cand.achievements = _sanitize_field_list(cand.achievements, max_item_length=150)

def _extract_legal(chunks: List[Any], embedder: Any = None) -> LegalSchema:
    """Extract structured legal/contract intelligence from chunks."""
    clauses: List[Clause] = []
    parties: List[FieldValue] = []
    obligations: List[FieldValue] = []
    seen = set()

    current_heading_category = ""

    classified = _ml_extract_lines(chunks, "legal", embedder)
    for line, (chunk_id, meta), cls in classified:
        if cls.role == "skip" or len(line) < 5:
            continue

        # Track heading category for context propagation
        if cls.role == "heading":
            if cls.category != "other":
                current_heading_category = cls.category
            continue

        key = _normalize_key(line)
        if key in seen:
            continue
        seen.add(key)

        # Use heading context when ML category is ambiguous
        category = cls.category
        if category == "other" and current_heading_category:
            category = current_heading_category

        if category == "parties":
            label = cls.label or "Party"
            value = cls.value or line
            parties.append(FieldValue(label=label, value=value, evidence_spans=[_span(chunk_id, line)]))
        elif category == "obligations":
            obligations.append(FieldValue(label="Obligation", value=cls.value or line, evidence_spans=[_span(chunk_id, line)]))
        elif category == "clauses" or cls.role in ("kv_pair", "narrative"):
            value = cls.value or line
            if cls.role == "kv_pair" and cls.label:
                title = cls.label
            else:
                # Extract title from first sentence for narrative clauses
                title_end = value.find(".")
                title = value[:title_end].strip() if 0 < title_end < 80 else None
            if value and len(value) > 2:
                clauses.append(Clause(title=title, text=value, evidence_spans=[_span(chunk_id, line)]))

    return LegalSchema(
        clauses=_clauses_field(clauses),
        parties=_field_values_field(parties),
        obligations=_field_values_field(obligations),
    )

def _extract_medical(chunks: List[Any], embedder: Any = None) -> "MedicalSchema":
    """Extract structured medical intelligence from patient records and clinical documents."""
    patient_info: List[FieldValue] = []
    diagnoses: List[FieldValue] = []
    medications: List[FieldValue] = []
    procedures: List[FieldValue] = []
    lab_results: List[FieldValue] = []
    vitals: List[FieldValue] = []
    seen = set()

    _CATEGORY_LISTS = {
        "patient_info": patient_info,
        "diagnoses": diagnoses,
        "medications": medications,
        "procedures": procedures,
        "lab_results": lab_results,
        "vitals": vitals,
    }

    current_heading_category = ""

    classified = _ml_extract_lines(chunks, "medical", embedder)
    for line, (chunk_id, meta), cls in classified:
        if cls.role == "skip":
            continue

        # Track heading category for context propagation
        if cls.role == "heading":
            if cls.category != "other":
                current_heading_category = cls.category
            continue

        key = _normalize_key(line)
        if key in seen:
            continue

        if cls.role in ("kv_pair", "bullet", "narrative"):
            label = cls.label if cls.role == "kv_pair" else None
            value = cls.value or line

            if not value or len(value) < 2:
                continue
            # Skip long values that look like paragraphs for KV pairs
            if cls.role == "kv_pair" and len(value) > 300:
                continue

            seen.add(key)

            # Route to category from ML classification
            category = cls.category
            if category == "other":
                # Fall back to heading context or section_kind
                section_kind = str(meta.get("section_kind", "")).lower()
                if current_heading_category and current_heading_category in _CATEGORY_LISTS:
                    category = current_heading_category
                elif "diagnos" in section_kind or "procedure" in section_kind:
                    category = "diagnoses"
                elif "medicat" in section_kind:
                    category = "medications"
                elif "lab" in section_kind:
                    category = "lab_results"
                elif "identity" in section_kind or "contact" in section_kind:
                    category = "patient_info"
                else:
                    category = "diagnoses"  # default for medical domain

            target = _CATEGORY_LISTS.get(category)
            if target is not None:
                target.append(FieldValue(
                    label=label, value=value,
                    evidence_spans=[_span(chunk_id, line)],
                ))

    return MedicalSchema(
        patient_info=_field_values_field(patient_info),
        diagnoses=_field_values_field(diagnoses),
        medications=_field_values_field(medications),
        procedures=_field_values_field(procedures),
        lab_results=_field_values_field(lab_results),
        vitals=_field_values_field(vitals),
    )

def _extract_policy(chunks: List[Any], embedder: Any = None) -> "PolicySchema":
    """Extract structured insurance policy intelligence from policy documents."""
    policy_info: List[FieldValue] = []
    coverage: List[FieldValue] = []
    premiums: List[FieldValue] = []
    exclusions: List[FieldValue] = []
    terms: List[FieldValue] = []
    seen = set()

    _CATEGORY_LISTS = {
        "policy_info": policy_info,
        "coverage": coverage,
        "premiums": premiums,
        "exclusions": exclusions,
        "terms": terms,
    }

    current_heading_category = ""

    classified = _ml_extract_lines(chunks, "policy", embedder)
    for line, (chunk_id, meta), cls in classified:
        if cls.role == "skip":
            continue

        # Track heading category for context propagation
        if cls.role == "heading":
            if cls.category != "other":
                current_heading_category = cls.category
            continue

        key = _normalize_key(line)
        if key in seen:
            continue

        label = cls.label if cls.role == "kv_pair" else None
        value = cls.value or line
        if not value or len(value) < 2:
            continue
        if cls.role == "kv_pair" and len(value) > 300:
            continue

        seen.add(key)

        # Route to category from ML classification
        category = cls.category
        if category == "other":
            # Propagate heading category to content lines under that heading
            if current_heading_category:
                category = current_heading_category
            else:
                # Fall back to section_kind
                section_kind = str(meta.get("section_kind", "")).lower()
                if "financial" in section_kind or "premium" in section_kind:
                    category = "premiums"
                elif "terms" in section_kind or "condition" in section_kind:
                    category = "terms"
                else:
                    category = "policy_info"  # default for policy domain

        target = _CATEGORY_LISTS.get(category)
        if target is not None:
            target.append(FieldValue(
                label=label, value=value,
                evidence_spans=[_span(chunk_id, line)],
            ))

    return PolicySchema(
        policy_info=_field_values_field(policy_info),
        coverage=_field_values_field(coverage),
        premiums=_field_values_field(premiums),
        exclusions=_field_values_field(exclusions),
        terms=_field_values_field(terms),
    )

def _extract_generic(query: str, chunks: List[Any]) -> GenericSchema:
    keywords = _keywords(query)
    facts: List[FieldValue] = []
    seen = set()

    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        chunk_id = getattr(chunk, "id", "")

        for line in _split_lines(text):
            cleaned = line.strip()
            if not cleaned or len(cleaned) < 3:
                continue
            # Strategy 1: Extract key-value pairs (e.g., "Patient Name: John Doe")
            kv_match = re.match(r"^([A-Za-z][A-Za-z\s/]{1,40}?)\s*[:\-]\s*(.+)$", cleaned)
            if kv_match:
                label = kv_match.group(1).strip()
                value = kv_match.group(2).strip()
                if value and len(value) > 1 and not _is_metadata_garbage(f"{label}: {value}", max_length=500):
                    key = _normalize_key(f"{label}:{value}")
                    if key not in seen:
                        seen.add(key)
                        facts.append(FieldValue(
                            label=label,
                            value=value,
                            evidence_spans=[_span(chunk_id, cleaned)],
                        ))
                continue

        # Strategy 2: Extract keyword-matched sentences
        for sentence in _split_sentences(text):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            if keywords and not any(word in cleaned.lower() for word in keywords):
                continue
            key = _normalize_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            facts.append(FieldValue(label=None, value=cleaned, evidence_spans=[_span(chunk_id, cleaned)]))

    # Fallback: if keyword filter yielded nothing, include top sentences unfiltered
    if not facts and chunks:
        for chunk in chunks[:5]:
            text = getattr(chunk, "text", "") or ""
            chunk_id = getattr(chunk, "id", "")
            for sentence in _split_sentences(text)[:3]:
                cleaned = sentence.strip()
                if not cleaned or len(cleaned) < 10:
                    continue
                key = _normalize_key(cleaned)
                if key in seen:
                    continue
                seen.add(key)
                facts.append(FieldValue(label=None, value=cleaned, evidence_spans=[_span(chunk_id, cleaned)]))

    return GenericSchema(facts=_field_values_field(facts))

def _schema_is_empty(schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema | MedicalSchema | PolicySchema) -> bool:
    if isinstance(schema, InvoiceSchema):
        return not (
            (schema.items.items if schema.items else None)
            or (schema.totals.items if schema.totals else None)
            or (schema.parties.items if schema.parties else None)
            or (schema.terms.items if schema.terms else None)
            or (getattr(schema, "invoice_metadata", None) and schema.invoice_metadata.items)
        )
    if isinstance(schema, HRSchema):
        cands = (schema.candidates.items if schema.candidates else None) or []
        return not any(c.name or c.technical_skills or getattr(c, "experience_summary", None) for c in cands)
    if isinstance(schema, LegalSchema):
        return not (
            (schema.clauses.items if schema.clauses else None)
            or (schema.parties.items if schema.parties else None)
            or (schema.obligations.items if schema.obligations else None)
        )
    if isinstance(schema, MedicalSchema):
        return not (
            (schema.patient_info.items if schema.patient_info else None)
            or (schema.diagnoses.items if schema.diagnoses else None)
            or (schema.medications.items if schema.medications else None)
            or (schema.procedures.items if schema.procedures else None)
            or (schema.lab_results.items if schema.lab_results else None)
            or (schema.vitals.items if schema.vitals else None)
        )
    if isinstance(schema, PolicySchema):
        return not (
            (schema.policy_info.items if schema.policy_info else None)
            or (schema.coverage.items if schema.coverage else None)
            or (schema.premiums.items if schema.premiums else None)
            or (schema.exclusions.items if schema.exclusions else None)
            or (schema.terms.items if schema.terms else None)
        )
    if isinstance(schema, GenericSchema):
        facts = (schema.facts.items if schema.facts else None) or []
        return not any(f.value and len(str(f.value)) > 5 for f in facts)
    if isinstance(schema, MultiEntitySchema):
        return not (schema.entities or [])
    return True

def _schema_completeness(schema) -> float:
    """Return weighted ratio of populated fields (0.0=empty, 1.0=full).

    Uses domain-specific field weights — critical fields (names, items, parties)
    carry higher weight than optional fields. This ensures a schema with just
    optional fields but missing core fields triggers LLM refinement.
    """
    if isinstance(schema, InvoiceSchema):
        # items and parties are critical; totals important; terms/metadata optional
        weighted = [
            (bool(schema.items and schema.items.items), 0.30),
            (bool(schema.totals and schema.totals.items), 0.25),
            (bool(schema.parties and schema.parties.items), 0.25),
            (bool(schema.terms and schema.terms.items), 0.10),
            (bool(getattr(schema, "invoice_metadata", None) and schema.invoice_metadata.items), 0.10),
        ]
    elif isinstance(schema, HRSchema):
        cands = (schema.candidates.items if schema.candidates else None) or []
        has_name = any(c.name for c in cands)
        has_skills = any(c.technical_skills for c in cands)
        has_exp = any(getattr(c, "experience_summary", None) or getattr(c, "total_years_experience", None) for c in cands)
        has_education = any(getattr(c, "education", None) for c in cands)
        has_contact = any(getattr(c, "emails", None) or getattr(c, "phones", None) for c in cands)
        # Name is critical; skills and experience are important
        weighted = [
            (has_name, 0.30),
            (has_skills, 0.25),
            (has_exp, 0.20),
            (has_education, 0.15),
            (has_contact, 0.10),
        ]
    elif isinstance(schema, LegalSchema):
        weighted = [
            (bool(schema.clauses and schema.clauses.items), 0.40),
            (bool(schema.parties and schema.parties.items), 0.35),
            (bool(schema.obligations and schema.obligations.items), 0.25),
        ]
    elif isinstance(schema, MedicalSchema):
        # Patient info and diagnoses are critical
        weighted = [
            (bool(schema.patient_info and schema.patient_info.items), 0.25),
            (bool(schema.diagnoses and schema.diagnoses.items), 0.25),
            (bool(schema.medications and schema.medications.items), 0.20),
            (bool(schema.procedures and schema.procedures.items), 0.15),
            (bool(schema.lab_results and schema.lab_results.items), 0.10),
            (bool(schema.vitals and schema.vitals.items), 0.05),
        ]
    elif isinstance(schema, PolicySchema):
        weighted = [
            (bool(schema.policy_info and schema.policy_info.items), 0.25),
            (bool(schema.coverage and schema.coverage.items), 0.25),
            (bool(schema.premiums and schema.premiums.items), 0.20),
            (bool(schema.exclusions and schema.exclusions.items), 0.15),
            (bool(schema.terms and schema.terms.items), 0.15),
        ]
    elif isinstance(schema, GenericSchema):
        facts = (schema.facts.items if schema.facts else None) or []
        populated = sum(1 for f in facts if f.value and len(str(f.value)) > 5)
        return min(1.0, populated / 3.0) if facts else 0.0
    elif isinstance(schema, MultiEntitySchema):
        return 1.0 if (schema.entities or []) else 0.0
    else:
        return 0.0

    return sum(w for present, w in weighted if present)

def _merge_schemas(deterministic, llm_result):
    """Merge LLM result into deterministic schema — deterministic values take priority.

    For GenericSchema, keep the one with more content.
    For typed schemas, preserve deterministic fields, fill gaps from LLM.
    """
    if llm_result is None:
        return deterministic
    if _schema_is_empty(deterministic):
        return llm_result

    # For GenericSchema — keep the richer one
    if isinstance(deterministic, GenericSchema) and isinstance(llm_result, GenericSchema):
        det_facts = (deterministic.facts.items if deterministic.facts else None) or []
        llm_facts = (llm_result.facts.items if llm_result.facts else None) or []
        det_total = sum(len(str(f.value or "")) for f in det_facts)
        llm_total = sum(len(str(f.value or "")) for f in llm_facts)
        return llm_result if llm_total > det_total else deterministic

    # For typed schemas, merge field lists (deterministic items preserved, LLM fills gaps)
    def _merge_field_values(det_field, llm_field):
        """Merge two FieldValuesField objects, keeping deterministic items and adding LLM-only."""
        det_items = (det_field.items if det_field else None) or []
        llm_items = (llm_field.items if llm_field else None) or []
        if not det_items:
            return llm_field
        if not llm_items:
            return det_field
        # Keep all deterministic items, add LLM items whose labels don't overlap
        det_labels = {(f.label or "").lower().strip() for f in det_items if f.label}
        for item in llm_items:
            if item.label and (item.label.lower().strip()) not in det_labels:
                det_items.append(item)
        return det_field

    if isinstance(deterministic, InvoiceSchema) and isinstance(llm_result, InvoiceSchema):
        deterministic.items = _merge_field_values(deterministic.items, llm_result.items) if hasattr(deterministic.items, 'items') else deterministic.items or llm_result.items
        deterministic.totals = _merge_field_values(deterministic.totals, llm_result.totals)
        deterministic.parties = _merge_field_values(deterministic.parties, llm_result.parties)
        deterministic.terms = _merge_field_values(deterministic.terms, llm_result.terms)
    elif isinstance(deterministic, MedicalSchema) and isinstance(llm_result, MedicalSchema):
        deterministic.patient_info = _merge_field_values(deterministic.patient_info, llm_result.patient_info)
        deterministic.diagnoses = _merge_field_values(deterministic.diagnoses, llm_result.diagnoses)
        deterministic.medications = _merge_field_values(deterministic.medications, llm_result.medications)
        deterministic.procedures = _merge_field_values(deterministic.procedures, llm_result.procedures)
        deterministic.lab_results = _merge_field_values(deterministic.lab_results, llm_result.lab_results)
        deterministic.vitals = _merge_field_values(deterministic.vitals, llm_result.vitals)
    elif isinstance(deterministic, PolicySchema) and isinstance(llm_result, PolicySchema):
        deterministic.policy_info = _merge_field_values(deterministic.policy_info, llm_result.policy_info)
        deterministic.coverage = _merge_field_values(deterministic.coverage, llm_result.coverage)
        deterministic.premiums = _merge_field_values(deterministic.premiums, llm_result.premiums)
        deterministic.exclusions = _merge_field_values(deterministic.exclusions, llm_result.exclusions)
        deterministic.terms = _merge_field_values(deterministic.terms, llm_result.terms)

    return deterministic

def _llm_extract(
    domain: str,
    intent: str,
    query: str,
    chunks: List[Any],
    llm_client: Any,
    correlation_id: Optional[str],
):
    prompt = _build_llm_prompt(domain, intent, query, chunks, require_spans=True)
    raw, meta = _generate_extract_response(llm_client, prompt, correlation_id)
    if meta.get("timeout"):
        return None
    if _is_empty_or_truncated(raw, meta):
        retry_prompt = _build_llm_prompt(domain, intent, query, chunks, require_spans=False, max_chars=2400)
        raw, meta = _generate_extract_response(llm_client, retry_prompt, correlation_id)
        if meta.get("timeout"):
            return None
        if _is_empty_or_truncated(raw, meta):
            logger.warning(
                "RAG v3 LLM extract returned empty; falling back to deterministic",
                extra={"stage": "extract", "correlation_id": correlation_id},
            )
            return None

    payload = _extract_json(raw)
    schema_payload = payload.get("schema") if isinstance(payload, dict) else None
    if not schema_payload:
        return None

    cleaned = _sanitize_schema_payload(domain, schema_payload, chunks)
    try:
        if domain == "invoice":
            return InvoiceSchema.model_validate(cleaned)
        if domain == "hr":
            return HRSchema.model_validate(cleaned)
        if domain == "legal":
            return LegalSchema.model_validate(cleaned)
        if domain == "medical":
            return MedicalSchema.model_validate(cleaned)
        if domain == "policy":
            return PolicySchema.model_validate(cleaned)
        return GenericSchema.model_validate(cleaned)
    except Exception as exc:
        logger.debug("Failed to validate extraction schema for domain %s", domain, exc_info=True)
        return None

def _build_llm_prompt(
    domain: str,
    intent: str,
    query: str,
    chunks: List[Any],
    *,
    require_spans: bool,
    max_chars: int = 6000,  # Increased from 3200 for more context
) -> str:
    # Sort chunks by score to prioritize most relevant content
    scored_chunks = sorted(
        chunks,
        key=lambda c: getattr(c, "score", 0.0) or 0.0,
        reverse=True
    )

    # Use more chunks with better context - up to 12 chunks
    context = []
    total_chars = 0
    max_chunks = 12  # Increased from 6

    for chunk in scored_chunks[:max_chunks]:
        chunk_id = getattr(chunk, "id", "")
        text = getattr(chunk, "text", "") or ""
        # Allow up to 800 chars per chunk for better context
        snippet = " ".join(text.split())[:800]

        # Stop if we'd exceed max chars
        if total_chars + len(snippet) + 20 > max_chars:
            break

        context.append(f"[{chunk_id}] {snippet}")
        total_chars += len(snippet) + 20

    evidence = "\n\n".join(context)

    evidence_rule = (
        "Every field must include evidence_spans with chunk_id and a short snippet. "
        if require_spans
        else "You may omit evidence_spans if unavailable. "
    )

    # More focused prompt with query emphasis
    return (
        f"TASK: Answer the question by extracting structured data from the evidence.\n\n"
        f"QUESTION: {query}\n"
        f"DOMAIN: {domain}\n"
        f"INTENT: {intent}\n\n"
        "RULES:\n"
        "1. Only use facts explicitly stated in the evidence.\n"
        "2. Focus on information directly relevant to the question.\n"
        f"3. {evidence_rule}\n"
        "4. Return strict JSON only.\n\n"
        "EVIDENCE:\n"
        + evidence
        + "\n\nReturn JSON with format: {\"schema\": { ... }}"
    )

def _generate_extract_response(
    llm_client: Any,
    prompt: str,
    correlation_id: Optional[str],
) -> Tuple[str, dict]:
    options = {
        "num_predict": 512,
        "max_output_tokens": 512,
        "num_ctx": 8192,  # Fixed to match llm_extract.py — avoids Ollama model reloads
        "stop": [],
    }
    def _call():
        if hasattr(llm_client, "generate_with_metadata"):
            text, meta = llm_client.generate_with_metadata(
                prompt,
                options=options,
                max_retries=1,
                backoff=0.4,
            )
            return text or "", meta or {}
        text = llm_client.generate(prompt, max_retries=1, backoff=0.4)
        return text or "", {}

    deadline = max(0.05, float(EXTRACT_TIMEOUT_MS) / 1000.0)
    start = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        text, meta = future.result(timeout=deadline)
        _ = time.monotonic() - start
        executor.shutdown(wait=False)
        return text or "", meta or {}
    except concurrent.futures.TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning(
            "RAG v3 LLM extract timed out; falling back to deterministic",
            extra={"stage": "extract", "correlation_id": correlation_id},
        )
        return "", {"timeout": True}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 LLM extract failed: %s",
            exc,
            extra={"stage": "extract", "correlation_id": correlation_id},
        )
        return "", {}

def _is_empty_or_truncated(raw: Any, meta: dict) -> bool:
    text = str(raw or "").strip()
    if not text:
        return True
    done_reason = str(meta.get("done_reason") or "").lower()
    return done_reason in {"length", "error"}

def _sanitize_schema_payload(domain: str, payload: Any, chunks: List[Any]) -> Any:
    chunk_ids = {str(getattr(c, "id", "")) for c in chunks}

    def _clean_spans(spans: Iterable[dict]) -> List[dict]:
        cleaned: List[dict] = []
        for span in spans or []:
            chunk_id = str(span.get("chunk_id", ""))
            snippet = str(span.get("snippet", ""))
            if not chunk_id or chunk_id not in chunk_ids:
                continue
            if not snippet:
                continue
            cleaned.append({"chunk_id": chunk_id, "snippet": snippet[:120]})
        return cleaned

    def _clean_list(items: Iterable[dict], fields: List[str]) -> List[dict]:
        cleaned = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            spans = _clean_spans(item.get("evidence_spans") or [])
            if not spans:
                continue
            cleaned_item = {"evidence_spans": spans}
            for field in fields:
                if field in item and item[field]:
                    cleaned_item[field] = item[field]
            cleaned.append(cleaned_item)
        return cleaned

    if domain == "invoice":
        items = _clean_list(payload.get("items") or [], ["description", "quantity", "unit_price", "amount"])
        totals = _clean_list(payload.get("totals") or [], ["label", "value"])
        parties = _clean_list(payload.get("parties") or [], ["label", "value"])
        terms = _clean_list(payload.get("terms") or [], ["label", "value"])
        return {
            "items": _items_field(items).model_dump(),
            "totals": _field_values_field(totals).model_dump(),
            "parties": _field_values_field(parties).model_dump(),
            "terms": _field_values_field(terms).model_dump(),
        }
    if domain == "hr":
        candidates = _clean_list(
            payload.get("candidates") or [],
            [
                "name",
                "role",
                "details",
                "total_years_experience",
                "experience_summary",
                "technical_skills",
                "functional_skills",
                "certifications",
                "education",
                "achievements",
                "emails",
                "phones",
                "linkedins",
                "source_type",
            ],
        )
        return {"candidates": _candidates_field(candidates).model_dump()}
    if domain == "legal":
        clauses = _clean_list(payload.get("clauses") or [], ["title", "text"])
        return {"clauses": _clauses_field(clauses).model_dump()}
    facts = _clean_list(payload.get("facts") or [], ["label", "value"])
    return {"facts": _field_values_field(facts).model_dump()}

def _extract_json(raw: Any) -> dict:
    if not raw:
        return {}
    text = str(raw).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception as exc:
            logger.debug("Failed to parse text as JSON object", exc_info=True)
            return {}
    if "{" in text and "}" in text:
        snippet = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception as exc:
            logger.debug("Failed to parse extracted JSON snippet", exc_info=True)
            return {}
    return {}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Phone pattern - require at least 7 digits total and avoid year ranges
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4,}")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[A-Za-z0-9._/?=&-]+", re.IGNORECASE)

def _clean_extraction_text(text: str) -> str:
    """
    Aggressively clean text of all extraction/metadata artifacts.

    Removes:
    - ExtractedDocument(...) wrappers
    - Section(...), ChunkCandidate(...), Table(...) Python object reprs
    - canonical_json={...} dumps
    - Metadata field assignments (section_id=, chunk_type=, page=, etc.)
    - Page markers like '--- Page 1 ---'
    - Unicode escape sequences
    - Raw metadata dict patterns
    """
    if not text:
        return ""

    # Remove everything from canonical_json= onward (huge metadata block)
    canonical_idx = text.find("canonical_json=")
    if canonical_idx != -1:
        text = text[:canonical_idx]

    # Remove everything from chunk_candidates=[ onward
    cc_idx = text.find("chunk_candidates=[")
    if cc_idx != -1:
        text = text[:cc_idx]

    # Remove everything from layout_spans=[ onward
    ls_idx = text.find("layout_spans=[")
    if ls_idx != -1:
        text = text[:ls_idx]

    # Remove everything from key_value_pairs=[ onward
    kvp_idx = text.find("key_value_pairs=[")
    if kvp_idx != -1:
        text = text[:kvp_idx]

    # Remove ExtractedDocument(...) wrapper
    text = re.sub(r'ExtractedDocument\s*\(', '', text)

    # Remove Python object representations (handle nested parens)
    text = re.sub(r'Section\s*\((?:[^()]*|\([^()]*\))*\)', '', text)
    text = re.sub(r'ChunkCandidate\s*\((?:[^()]*|\([^()]*\))*\)', '', text)
    text = re.sub(r'Table\s*\((?:[^()]*|\([^()]*\))*\)', '', text)

    # Remove metadata list/dict assignments
    metadata_field_re = re.compile(
        r'\b(?:sections|tables|figures|errors|metrics|doc_type|doc_quality'
        r'|section_id|section_title|chunk_type|page_start|page_end'
        r'|start_page|end_page|level|ocr_confidence_avg|layout_confidence'
        r'|ocr_upgrade_suggested|chunk_candidates|layout_spans'
        r'|key_value_pairs|csv)\s*='
    )
    # Remove lines that are primarily metadata assignments
    cleaned_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        # Skip lines that are metadata assignments
        if metadata_field_re.match(stripped):
            continue
        # Skip lines that are just dict/list closers
        if stripped in ('}', ']', '},', '],', '}]', '})'):
            continue
        # Skip lines that look like raw dict entries: 'key': 'value'
        if re.match(r"^['\"](?:section_id|chunk_type|page_start|page_end|start_page|end_page"
                     r"|page_number|csv|doc_quality|ocr_confidences|layout_confidence"
                     r"|ocr_confidence_avg|ocr_upgrade_suggested|page|level|text|title"
                     r"|section_title|chunk_candidates|figures|tables|sections"
                     r"|errors|metrics|key_value_pairs|layout_spans)['\"]", stripped):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Remove embedding_text section prefixes like "[Skills Technical] Section 3:"
    text = re.sub(r'\[[\w\s]+\]\s*Section\s*\d+\s*:\s*', '', text)
    # Remove stray metadata field leaks: "section_title VALUE", "section_kind VALUE"
    text = re.sub(r'\bsection_title\s+\S+', '', text)
    text = re.sub(r'\bsection_kind\s+\S+', '', text)
    # Remove "text :" or "key :" prefix from metadata key leaks
    text = re.sub(r'(?m)^\s*(?:text|key)\s*:\s*', '', text)
    # Remove "Extracted Document full_text" partial repr artifacts
    text = re.sub(r'Extracted\s+Document\s+full_text\b', '', text)
    # Remove "Document Content" prefix (extraction pipeline artifact)
    text = re.sub(r'^Document\s+Content\s*', '', text, flags=re.IGNORECASE)
    # Remove "Introduction" prefix when followed by the actual content (extraction artifact)
    text = re.sub(r'^Introduction\s+(?=[A-Z])', '', text)
    # Remove leading 'n' artifacts from escaped newlines that became "n " (e.g., "n Certified SAP...")
    text = re.sub(r'(?:^|\n)\s*n\s+(?=[A-Z])', '\n', text)

    # Remove page markers
    text = re.sub(r'---\s*Page\s*\d+\s*---', '\n', text, flags=re.IGNORECASE)

    # Remove unicode bullet characters
    text = re.sub(r'\\uf0[a-z0-9]{2}', '', text)  # Escaped unicode
    text = re.sub(r'[\uf0b7\uf0d8]', '- ', text)  # Actual unicode bullets

    # Convert escaped newlines to actual newlines
    text = text.replace('\\n', '\n')
    # Clean stray 'n' from partially-converted escaped newlines:
    # Patterns: "...text n More text", "...textnMore text", leading "n " at line start
    # Must handle both lowercase ("text n More") AND uppercase ("HANA n Sales") preceding chars
    text = re.sub(r'(?<=[a-zA-Z0-9.,:;)>])\s*\bn\s+(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[a-zA-Z0-9.,:;)>])\s*\bn(?=[A-Z][a-z])', ' ', text)
    text = re.sub(r'(?m)^\s*n\b\s+', '', text)  # Leading "n " at start of lines

    # Remove trailing metadata fragments: ", sections=[", ", doc_type='..."
    text = re.sub(r",\s*(?:sections|tables|figures|chunk_candidates|doc_type|errors|metrics"
                  r"|canonical_json|doc_quality|layout_spans|key_value_pairs)\s*=.*$",
                  '', text, flags=re.DOTALL)

    # Remove stray Python repr artifacts
    text = re.sub(r"\b(?:none|None|True|False)\b(?=\s*[,}\]])", '', text)

    # Clean leading commas on lines (artifact of bullet-point conversion)
    text = re.sub(r'^\s*,\s*', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()

def _extract_full_text_content(text: str) -> str:
    """
    Extract the actual document content from Qdrant payload.

    Handles formats like:
    - "ExtractedDocument(full_text='...')" Python repr
    - Raw text with metadata artifacts
    - JSON-like structures with full_text field
    - Raw metadata dict dumps
    """
    if not text:
        return ""

    # Quick check: if text doesn't contain known artifact markers, just clean it
    artifact_markers = ("full_text=", "ExtractedDocument", "canonical_json",
                        "chunk_candidates=", "Section(section_id=",
                        "ChunkCandidate(text=", "'section_id':", "'chunk_type':")
    if not any(marker in text for marker in artifact_markers):
        return _clean_extraction_text(text)

    # Strategy 1: Extract full_text='...' value using string scanning
    for prefix in ["full_text='", 'full_text="']:
        idx = text.find(prefix)
        if idx >= 0:
            quote_char = prefix[-1]
            content_start = idx + len(prefix)

            # Find the closing quote by looking for known end markers
            end_markers = [
                f"{quote_char}, sections=",
                f"{quote_char}, tables=",
                f"{quote_char}, figures=",
                f"{quote_char}, chunk_candidates=",
                f"{quote_char}, doc_type=",
                f"{quote_char}, errors=",
                f"{quote_char}, metrics=",
                f"{quote_char}, canonical_json=",
                f"{quote_char}, doc_quality=",
                f"{quote_char})",
            ]

            end_idx = len(text)
            for marker in end_markers:
                pos = text.find(marker, content_start)
                if pos != -1 and pos < end_idx:
                    end_idx = pos

            content = text[content_start:end_idx]
            if content and len(content) > 20:
                return _clean_extraction_text(content)

    # Strategy 2: If text looks like a raw metadata dict dump, strip it
    if text.lstrip().startswith("{") or "'chunk_type'" in text or "'section_id'" in text:
        # This is a metadata dict, not useful text content
        # Try to extract any 'text' values from it
        text_values = re.findall(r"'text'\s*:\s*'([^']{10,})'", text)
        if text_values:
            return _clean_extraction_text("\n".join(text_values))
        # Fallback: salvage non-garbage lines (mixed garbage + real content)
        salvaged = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and not _is_metadata_garbage(stripped, max_length=500):
                salvaged.append(stripped)
        if salvaged:
            return _clean_extraction_text("\n".join(salvaged))
        return ""

    return _clean_extraction_text(text)

# Section header patterns used for boundary detection in full-document chunks
_SECTION_HEADER_MAP = {
    "summary_objective": [
        "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "PROFILE", "ABOUT ME", "CAREER OBJECTIVE",
        "EXECUTIVE SUMMARY", "PERSONAL STATEMENT", "CAREER SUMMARY", "PROFESSIONAL PROFILE", "OVERVIEW",
    ],
    "experience": [
        "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT", "CAREER HISTORY", "EMPLOYMENT HISTORY",
        "RELEVANT EXPERIENCE", "PROJECT EXPERIENCE", "PROJECTS", "KEY PROJECTS", "SELECTED PROJECTS",
        "INTERNSHIPS", "TRAINING", "WORK HISTORY", "PROFESSIONAL HISTORY",
    ],
    "skills_technical": [
        "TECHNICAL SKILLS", "SKILLS", "TECHNOLOGIES", "TECH STACK", "KEY SKILLS", "CORE COMPETENCIES",
        "TOOLS", "TECHNICAL PROFICIENCY", "AREAS OF EXPERTISE", "TECHNICAL EXPERTISE",
        "PROGRAMMING LANGUAGES", "FRAMEWORKS",
    ],
    "skills_functional": [
        "FUNCTIONAL SKILLS", "SOFT SKILLS", "BUSINESS SKILLS",
        "INTERPERSONAL SKILLS", "TRANSFERABLE SKILLS", "KEY COMPETENCIES", "LEADERSHIP SKILLS",
    ],
    "certifications": [
        "CERTIFICATIONS", "CERTIFICATES", "CREDENTIALS", "LICENSES",
        "PROFESSIONAL DEVELOPMENT", "TRAINING AND CERTIFICATIONS", "LICENSES AND CERTIFICATIONS",
    ],
    "education": [
        "EDUCATION", "ACADEMIC", "QUALIFICATIONS", "DEGREES", "ACADEMIC BACKGROUND",
        "ACADEMIC QUALIFICATIONS", "EDUCATIONAL BACKGROUND", "ACADEMIC CREDENTIALS", "TRAINING AND EDUCATION",
    ],
    "achievements": [
        "ACHIEVEMENTS", "AWARDS", "HONORS", "ACCOMPLISHMENTS", "RECOGNITION",
        "KEY ACHIEVEMENTS", "NOTABLE ACHIEVEMENTS", "PUBLICATIONS", "PUBLICATIONS AND RESEARCH", "PATENTS",
    ],
    "identity_contact": [
        "CONTACT", "PERSONAL DETAILS", "CONTACT INFORMATION",
        "PERSONAL INFORMATION", "BIO DATA", "BIODATA", "CONTACT DETAILS",
    ],
    "projects": [
        "PROJECTS", "KEY PROJECTS", "SELECTED PROJECTS", "PROJECT EXPERIENCE",
        "PERSONAL PROJECTS", "ACADEMIC PROJECTS",
    ],
    "languages": ["LANGUAGES", "LANGUAGE SKILLS", "LANGUAGE PROFICIENCY"],
    "references": ["REFERENCES", "PROFESSIONAL REFERENCES"],
    "volunteer": ["VOLUNTEER", "VOLUNTEER EXPERIENCE", "COMMUNITY SERVICE", "EXTRACURRICULAR"],
}

# All headers flattened for boundary detection
_ALL_SECTION_HEADERS = []
for _headers in _SECTION_HEADER_MAP.values():
    _ALL_SECTION_HEADERS.extend(_headers)

def _extract_section_from_text(full_text: str, section_kind: str) -> str:
    """
    Extract only the text relevant to a specific section from a full document text.

    When a chunk contains the entire document (common with full-scan retrieval)
    but is classified as a specific section, this function extracts just the
    relevant portion using section header boundaries.

    Returns the section-specific text, or the original text if section not found.
    """
    if not full_text or len(full_text) < 1500:
        # Short texts are likely already section-specific
        return full_text

    # Normalize the section_kind to match our map (strip composite keys like "certifications:CERTIFICATIONS")
    base_kind = section_kind.split(":")[0] if ":" in section_kind else section_kind

    headers_for_section = _SECTION_HEADER_MAP.get(base_kind, [])
    if not headers_for_section:
        return full_text

    text_upper = full_text.upper()

    # Find the section start
    section_start = -1
    matched_header_len = 0
    for header in headers_for_section:
        idx = text_upper.find(header)
        if idx != -1:
            section_start = idx
            matched_header_len = len(header)
            break

    if section_start == -1:
        return full_text  # Section header not found, return all text

    # Find the end: the start of the next section header
    search_from = section_start + matched_header_len + 1
    section_end = len(full_text)

    for header in _ALL_SECTION_HEADERS:
        idx = text_upper.find(header, search_from)
        if idx != -1 and idx < section_end:
            # Make sure this isn't the same header we matched
            if idx != section_start:
                section_end = idx

    extracted = full_text[section_start:section_end].strip()

    # Strip the section header line itself for cleaner extraction
    lines = extracted.split('\n')
    if lines:
        first_line_upper = lines[0].strip().upper().rstrip(':')
        if first_line_upper in [h.upper() for h in headers_for_section]:
            lines = lines[1:]
    extracted = '\n'.join(lines).strip()

    return extracted if extracted else full_text

def _text_has_multiple_sections(text: str) -> bool:
    """Check if text contains multiple recognized resume section headers."""
    if not text or len(text) < 300:
        return False
    text_upper = text.upper()
    found = 0
    for headers in _SECTION_HEADER_MAP.values():
        for header in headers:
            if header in text_upper:
                found += 1
                break  # Count one per section type
    return found >= 3  # Need at least 3 different section types

def _split_document_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split a full document text into (section_kind, section_text) tuples.

    Used when a single chunk contains the entire document content.
    Finds standalone section headers and splits at their boundaries.
    Only matches headers that appear as standalone lines, not inline labels
    like "Contact: +91..." or "Skills: Python, Java".
    """
    if not text:
        return []

    text_upper = text.upper()

    def _find_standalone_header(header: str) -> int:
        """Find header that appears as a standalone line, not as an inline label."""
        h_upper = header.upper()
        search_start = 0
        while search_start < len(text_upper):
            idx = text_upper.find(h_upper, search_start)
            if idx == -1:
                return -1
            # Must be at start of line (or start of text)
            if idx > 0 and text[idx - 1] != '\n':
                search_start = idx + 1
                continue
            # Check what follows the header on the same line
            end_idx = idx + len(header)
            newline_after = text.find('\n', end_idx)
            if newline_after != -1:
                rest_of_line = text[end_idx:newline_after].strip()
            else:
                rest_of_line = text[end_idx:].strip()
            # Allow empty or just ":" after header (standalone header)
            # Reject if there's substantial content after ":"  (inline label like "Contact: +91...")
            if rest_of_line and rest_of_line != ':':
                if rest_of_line.startswith(':') and len(rest_of_line) > 2:
                    search_start = idx + 1
                    continue
            return idx
        return -1

    # Find all section header positions
    header_positions: List[Tuple[int, str, str]] = []  # (position, section_kind, header_text)
    for section_kind, headers in _SECTION_HEADER_MAP.items():
        for header in headers:
            idx = _find_standalone_header(header)
            if idx != -1:
                header_positions.append((idx, section_kind, header))
                break  # Use first match per section type

    if not header_positions:
        return []

    # Sort by position
    header_positions.sort(key=lambda x: x[0])

    # Extract text between headers
    result: List[Tuple[str, str]] = []

    # Content before first header is identity/contact
    if header_positions[0][0] > 10:
        pre_text = text[:header_positions[0][0]].strip()
        if pre_text:
            result.append(("identity_contact", pre_text))

    for i, (pos, kind, header) in enumerate(header_positions):
        # Start after the header line
        content_start = pos + len(header)
        # Skip the rest of the header line
        newline_pos = text.find('\n', content_start)
        if newline_pos != -1:
            content_start = newline_pos + 1

        # End at next header
        if i + 1 < len(header_positions):
            content_end = header_positions[i + 1][0]
        else:
            content_end = len(text)

        section_text = text[content_start:content_end].strip()
        if section_text:
            result.append((kind, section_text))

    return result

def _extract_contact_fields(line: str) -> Dict[str, List[str]]:
    if not line:
        return {}
    emails = [_clean_contact_value(val) for val in _EMAIL_RE.findall(line)]
    phones = [_clean_contact_value(val, preserve_parens=True) for val in _PHONE_RE.findall(line)]
    linkedins = [_clean_contact_value(val) for val in _LINKEDIN_RE.findall(line)]

    if "linkedin" in line.lower() and not linkedins:
        match = re.search(r"(?i)linkedin\s*[:\-]\s*([A-Za-z0-9._/+-]+)", line)
        if match:
            linkedins.append(_clean_contact_value(match.group(1)))

    return {
        "emails": [e for e in emails if e],
        "phones": [p for p in phones if p],
        "linkedins": [l for l in linkedins if l],
    }

def _clean_contact_value(value: str, preserve_parens: bool = False) -> str:
    if preserve_parens:
        cleaned = value.strip().strip(".,;:[]{}<>")
    else:
        cleaned = value.strip().strip(".,;:()[]{}<>")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

def _is_valid_phone(phone: str) -> bool:
    """Validate that a string is actually a phone number, not a year range."""
    # Count actual digits
    digits = re.sub(r"\D", "", phone)
    # Phone numbers should have at least 7 digits
    if len(digits) < 7:
        return False
    # Check for year range patterns like 2015-2019, 2020-Present
    if re.match(r"^\d{4}\s*[-–]\s*(?:\d{4}|present|current)$", phone, re.IGNORECASE):
        return False
    # Check for standalone years
    if re.match(r"^\d{4}$", phone):
        return False
    return True

def _extract_contact_fields_comprehensive(text: str) -> Dict[str, List[str]]:
    """Extract contact fields from multi-line text more comprehensively."""
    if not text:
        return {}

    emails = []
    phones = []
    linkedins = []
    names = []

    # Pre-process: rejoin emails broken across whitespace/newlines.
    # Strategy: find every email-like match, and if its local part starts with
    # digits (e.g. "2003@gmail.com"), look backwards in the raw text for the
    # alpha prefix that was broken off (e.g. "sabareesh 2003@gmail.com" →
    # "sabareesh2003@gmail.com").  Digit-only local parts are almost never
    # standalone valid emails.
    raw_emails = list(set(_EMAIL_RE.findall(text)))
    emails = []
    for email in raw_emails:
        local_part = email.split("@")[0]
        if local_part and local_part[0].isdigit():
            # Digit-leading email — look backwards for the alpha prefix
            _prefix_pattern = re.compile(
                r'([A-Za-z][A-Za-z0-9._%-]*)\s+' + re.escape(email)
            )
            _prefix_match = _prefix_pattern.search(text)
            if _prefix_match:
                emails.append(_prefix_match.group(1) + email)
                continue
            # Also check across newlines
            _nl_prefix_pattern = re.compile(
                r'([A-Za-z][A-Za-z0-9._%-]*)\s*[\n\r]+\s*' + re.escape(email)
            )
            _nl_match = _nl_prefix_pattern.search(text)
            if _nl_match:
                emails.append(_nl_match.group(1) + email)
                continue
        emails.append(email)
    emails = list(set(emails))

    # Extract phones from entire text
    for match in _PHONE_RE.findall(text):
        cleaned_phone = _clean_contact_value(match, preserve_parens=True)
        if cleaned_phone and cleaned_phone not in phones and _is_valid_phone(cleaned_phone):
            phones.append(cleaned_phone)

    # Extract LinkedIn URLs
    linkedins = list(set(_LINKEDIN_RE.findall(text)))

    # Normalize text for name extraction
    normalized_text = text.replace('\\n', '\n').replace(' n ', '\n').replace('--- Page 1 ---', '\n')
    # Strip common garbage prefixes from extraction metadata
    normalized_text = re.sub(r'^Document\s+Content\s+', '', normalized_text, flags=re.IGNORECASE)

    # Pattern 0: Look for name in "full_text='Name\n" format (Qdrant extracted docs)
    fulltext_match = re.search(r"full_text=['\"]([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]*){0,3})\n", normalized_text)
    if fulltext_match:
        potential_name = fulltext_match.group(1).strip()
        words = potential_name.split()
        if 2 <= len(words) <= 4:
            if not any(h in potential_name.lower() for h in ['summary', 'objective', 'experience', 'education', 'document', 'professional']):
                names.append(potential_name)

    # Look for names (lines with capitalized words at start)
    for line in normalized_text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 3:
            continue

        # Pattern 1: Pipe-delimited format (e.g., "John Smith | Engineer | email@example.com")
        if '|' in line_stripped:
            parts = [p.strip() for p in line_stripped.split('|')]
            if parts:
                first_part = parts[0]
                words = first_part.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() or not w[0].islower() for w in words if w):
                    if '@' not in first_part and not any(c.isdigit() for c in first_part):
                        if not any(h in first_part.lower() for h in ['skills', 'experience', 'education', 'certification']):
                            # Use _looks_like_name validation to filter out phrases
                            if _looks_like_name(first_part):
                                names.append(first_part)
                                continue

        # Pattern 2: Line with only capitalized words (traditional name format)
        if line_stripped[0].isupper() and re.match(r'^[A-Za-z\s\-\.]+$', line_stripped[:30]):
            # Check if it's not a section header
            if not any(h in line_stripped.lower() for h in ['skills', 'experience', 'education', 'certification', 'contact', 'summary', 'achievement']):
                if len(line_stripped.split()) >= 2 and len(line_stripped) < 50:
                    # Use _looks_like_name validation to filter out phrases
                    if _looks_like_name(line_stripped):
                        names.append(line_stripped)

    return {
        "emails": [e for e in emails if e],
        "phones": [p for p in phones if p],
        "linkedins": [l for l in linkedins if l],
        "names": names[:1] if names else [],  # Return only first potential name
    }

# Known invoice/document field labels — when found inline, split before them
# to create separate "label: value" lines for the ML classifier
_INVOICE_LABEL_RE = re.compile(
    r"(?<=[a-z0-9.)\]]) (?="
    r"(?:Invoice\s*(?:No|Number|Date|#|Amount)|"
    r"Due\s*Date|Bill\s*To|Ship\s*To|"
    r"Subtotal|Total|Balance\s*Due|Amount\s*Due|"
    r"Payment\s*(?:Terms|Method|Instructions)|"
    r"Purchase\s*Order|PO\s*(?:No|Number|#)|"
    r"Tax|Discount|Qty|Quantity|"
    r"Unit\s*Price|Rate|Description|"
    r"Sl\.?|Sr\.?|S\.?No\.?)"
    r"\s*[:.]?\s)",
    re.IGNORECASE,
)

def _split_lines(text: str) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    # For long single lines (OCR'd invoices), try splitting at label boundaries
    result = []
    for line in lines:
        if len(line) > 80 and _INVOICE_LABEL_RE.search(line):
            parts = _INVOICE_LABEL_RE.split(line)
            result.extend(p.strip() for p in parts if p.strip())
        else:
            result.append(line)
    return result

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Split on sentence endings AND newlines (common document boundary)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def _keywords(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "from",
            "what", "how", "who", "where", "when", "which", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "would", "should", "will",
            "this", "that", "these", "those", "me", "my", "your", "his", "her",
            "tell", "give", "show", "find", "get", "list", "all", "about"}
    singles = [tok for tok in tokens if tok not in stop and len(tok) > 2]
    # Add bigrams for phrase matching (e.g. "machine learning" as single keyword)
    bigrams = []
    for i in range(len(tokens) - 1):
        if tokens[i] not in stop and tokens[i + 1] not in stop:
            bigrams.append(f"{tokens[i]} {tokens[i + 1]}")
    return singles + bigrams

def _span(chunk_id: str, snippet: str) -> EvidenceSpan:
    cleaned = " ".join(snippet.split())
    return EvidenceSpan(chunk_id=str(chunk_id), snippet=cleaned[:120])

_FIELD_EQUIVALENCE = {
    # Invoice totals
    "total price": "total amount",
    "total cost": "total amount",
    "grand total": "total amount",
    "net total": "total amount",
    "amount due": "total amount",
    "balance due": "total amount",
    "total payable": "total amount",
    # Invoice parties
    "bill to": "billed to",
    "sold to": "billed to",
    "ship to": "shipping address",
    "deliver to": "shipping address",
    "vendor": "supplier",
    "seller": "supplier",
    # Dates
    "invoice date": "date",
    "bill date": "date",
    "due date": "payment due date",
    "pay by": "payment due date",
    # HR/Resume
    "phone number": "phone",
    "mobile": "phone",
    "mobile number": "phone",
    "cell": "phone",
    "email address": "email",
    "e mail": "email",
    "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "career history": "experience",
    "tech skills": "technical skills",
    "key skills": "technical skills",
    "core competencies": "technical skills",
    "soft skills": "functional skills",
    "interpersonal skills": "functional skills",
    # Medical
    "patient name": "patient",
    "patient id": "patient identifier",
    "mrn": "patient identifier",
    "medical record number": "patient identifier",
    "diagnosis": "diagnoses",
    "assessment": "diagnoses",
    "impression": "diagnoses",
    "medicines": "medications",
    "drugs": "medications",
    "prescriptions": "medications",
    "rx": "medications",
}

def _normalize_key(text: str) -> str:
    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    # For label:value patterns, canonicalize the label
    if ":" in text:
        colon_idx = text.index(":")
        raw_label = re.sub(r"\W+", " ", text[:colon_idx].lower()).strip()
        canonical = _FIELD_EQUIVALENCE.get(raw_label)
        if canonical:
            raw_value = re.sub(r"\W+", " ", text[colon_idx + 1:].lower()).strip()
            return f"{canonical} {raw_value}" if raw_value else canonical
    return normalized

def _is_bullet_item(line: str) -> bool:
    if not line.startswith("-") and not line.startswith("•"):
        return False
    has_price = bool(re.search(r"[$€£]\s*\d", line))
    has_qty = bool(re.search(r"\bqty\b|\bquantity\b|\b\d+\s*x\b", line, re.IGNORECASE))
    return has_price or has_qty

def _extract_years_experience(text: str) -> Optional[str]:
    """Extract years of experience — explicit mention first, then calculate from date ranges."""
    # Strategy 1: Explicit "X years" / "X+ yrs" mention
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s*(?:of\s+)?(?:experience|exp)?\b", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} years"

    # Strategy 2: Calculate from employment date ranges
    # Match patterns like "Jan 2018 - Dec 2023", "2019 - Present", "06/2020 – 08/2024"
    import datetime
    _MONTH_MAP = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
        "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    current_year = datetime.date.today().year
    current_month = datetime.date.today().month

    # Pattern: "Mon YYYY - Mon YYYY" or "Mon YYYY - Present"
    date_range_re = re.compile(
        r"(?:(?P<sm>[A-Za-z]{3,9})[\s.,]*)?(?P<sy>\d{4})\s*[-–—to]+\s*"
        r"(?:(?P<em>[A-Za-z]{3,9})[\s.,]*)?(?P<ey>\d{4}|(?:present|current|till\s*date|now|ongoing))",
        re.IGNORECASE,
    )
    # Also match MM/YYYY - MM/YYYY
    numeric_date_re = re.compile(
        r"(?P<sm>\d{1,2})/(?P<sy>\d{4})\s*[-–—to]+\s*"
        r"(?:(?P<em>\d{1,2})/(?P<ey>\d{4})|(?P<present>present|current|till\s*date|now|ongoing))",
        re.IGNORECASE,
    )

    spans = []  # List of (start_year, start_month, end_year, end_month)

    for m in date_range_re.finditer(text):
        try:
            sy = int(m.group("sy"))
            sm_str = (m.group("sm") or "").lower()[:3]
            sm = _MONTH_MAP.get(sm_str, 1)
            ey_str = (m.group("ey") or "").strip().lower()
            if ey_str in ("present", "current", "now", "ongoing") or ey_str.startswith("till"):
                ey, em = current_year, current_month
            else:
                ey = int(ey_str)
                em_str = (m.group("em") or "").lower()[:3]
                em = _MONTH_MAP.get(em_str, 12)
            if 1970 <= sy <= current_year and 1970 <= ey <= current_year + 1 and ey >= sy:
                spans.append((sy, sm, ey, em))
        except (ValueError, TypeError):
            continue

    for m in numeric_date_re.finditer(text):
        try:
            sy = int(m.group("sy"))
            sm = int(m.group("sm"))
            if m.group("present"):
                ey, em = current_year, current_month
            else:
                ey = int(m.group("ey"))
                em = int(m.group("em"))
            if 1970 <= sy <= current_year and 1970 <= ey <= current_year + 1 and ey >= sy and 1 <= sm <= 12 and 1 <= em <= 12:
                spans.append((sy, sm, ey, em))
        except (ValueError, TypeError):
            continue

    if spans:
        # Merge overlapping spans and sum total months
        spans.sort()
        merged = []
        for s in spans:
            if merged and s[0] * 12 + s[1] <= merged[-1][2] * 12 + merged[-1][3]:
                # Overlapping — extend end if this span ends later
                if s[2] * 12 + s[3] > merged[-1][2] * 12 + merged[-1][3]:
                    merged[-1] = (merged[-1][0], merged[-1][1], s[2], s[3])
            else:
                merged.append(s)
        total_months = sum((ey * 12 + em) - (sy * 12 + sm) for sy, sm, ey, em in merged)
        if total_months >= 6:  # At least 6 months to be meaningful
            years = total_months / 12
            if years == int(years):
                return f"{int(years)} years"
            return f"{years:.1f} years"

    return None

def _infer_section_kind_from_content(text: str, section_title: str) -> str:
    """Infer section kind from actual chunk content when metadata is generic.

    Delegates to the shared content classifier so ingestion and extraction
    use identical classification logic.
    """
    from src.embedding.pipeline.content_classifier import classify_section_kind

    return classify_section_kind(text, section_title)

def _normalize_intent_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"summary", "contact", "rank", "compare", "candidate_list"}:
        return normalized
    if normalized in {"list", "listing"}:
        return "candidate_list"
    if normalized in {"qa", "answer", "facts", "factual"}:
        return "factual"
    return None

def _split_list(text: str) -> List[str]:
    cleaned = re.sub(r"(?i)^(technical skills|functional skills|key skills|skills)\s*[:\-]\s*", "", text).strip()
    if not cleaned:
        return []
    # Split on common delimiters but NOT "/" between alphanumeric characters (e.g., "S/4 HANA", "GR/IR")
    # First, protect "/" in known patterns by replacing with placeholder
    protected = re.sub(r'(\w)/(\w)', r'\1__SLASH__\2', cleaned)
    parts = re.split(r"[•\u2022,;/]+", protected)
    items = []
    for part in parts:
        item = part.strip().replace('__SLASH__', '/')
        if item:
            items.append(item)
    return items

def _parse_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {
        "skills": [],
        "technical_skills": [],
        "functional_skills": [],
        "tools": [],
        "education": [],
    }
    current = None
    for line in _split_lines(text):
        cleaned = line.strip()
        if not cleaned:
            continue
        heading = _section_heading(cleaned)
        if heading:
            current = heading
            inline = cleaned.split(":", 1)[1].strip() if ":" in cleaned else ""
            if inline:
                sections[current].append(inline)
            continue
        if current:
            if _looks_like_heading_break(cleaned):
                current = None
                continue
            sections[current].append(cleaned)
    return sections

def _section_heading(line: str) -> Optional[str]:
    lower = line.lower().strip()
    if re.match(r"^(technical skills|technical skill|technologies|tech stack)\b", lower):
        return "technical_skills"
    if re.match(r"^(functional skills|key skills|core skills)\b", lower):
        return "functional_skills"
    if re.match(r"^(skills|skill set)\b", lower):
        return "skills"
    if re.match(r"^(tools|tools & technologies|tools and technologies)\b", lower):
        return "tools"
    if re.match(r"^(education|academic|academics|qualification|qualifications)\b", lower):
        return "education"
    return None

def _looks_like_heading_break(line: str) -> bool:
    lower = line.lower().strip()
    return any(
        token in lower
        for token in (
            "experience",
            "employment",
            "professional experience",
            "projects",
            "summary",
            "objective",
            "certifications",
            "awards",
            "achievements",
            "languages",
        )
    )

def _flatten_skill_block(lines: List[str]) -> List[str]:
    items: List[str] = []
    # Headers to skip
    skip_headers = {
        'skills', 'technical skills', 'certifications', 'education',
        'experience', 'achievements', 'contact', 'summary', 'objective',
        'qualifications', 'professional skills', 'core competencies',
        'work experience', 'employment history', 'professional summary',
        'key skills', 'functional skills', 'soft skills', 'hard skills',
    }
    # Header prefixes to strip from items
    header_prefixes = [
        'certifications:', 'certification:', 'skills:', 'technical skills:',
        'education:', 'experience:', 'achievements:', 'contact:', 'key skills:',
    ]
    # Garbage patterns to filter out
    garbage_patterns = [
        'ed in', 'ing in', 'tion of', 'implementation', 'and support',
        'responsible for', 'managed', 'developed', 'coordinated', 'extracted document',
        'section_id', 'chunk_type', 'page_start', 'page_end', 'layout_confidence',
        'canonical_json', 'chunk_candidates', 'layout_spans', 'key_value_pairs',
        'doc_quality', 'ocr_confidence', 'ExtractedDocument', 'ChunkCandidate',
        "Section(", "Table(", "'text':", "'title':", "'page':", "'csv':",
        'start_page', 'end_page', 'section_title', 'ensuring', 'collaborated',
        'oversaw', 'maintained', 'prepared', 'supporting',
        # Education/grading context — not skills
        'cgpa', 'gpa:', 'percentage', 'aggregate', 'marks',
        'bachelor', 'master', 'b.e', 'b.tech', 'm.tech', 'mca', 'bca',
        'degree', 'diploma', 'graduation',
        # Language proficiency — not skills
        'proficiency', 'fluency', 'native speaker', 'professional proficiency',
        'fluent in', 'mother tongue',
        # Location/personal context — not skills
        'tamil nadu', 'karnataka', 'maharashtra', 'delhi', 'mumbai',
        'bangalore', 'chennai', 'hyderabad', 'pune', 'erode', 'coimbatore',
        'india', 'state:', 'city:', 'address:',
    ]
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # Skip section headers (lines that are just a header followed by : or nothing)
        cleaned_lower = cleaned.lower().rstrip(':').strip()
        if cleaned_lower in skip_headers:
            continue
        if cleaned.startswith(("-", "•", "*")) or re.match(r"^\d+\.", cleaned):
            cleaned = cleaned.lstrip("-•* ").strip()
        # Strip header prefixes from items
        for prefix in header_prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        # Skip garbage entries
        if any(garbage in cleaned.lower() for garbage in garbage_patterns):
            continue
        # Skip very long entries (likely paragraphs, not skills)
        if len(cleaned) > 100:
            continue
        # Skip lines that mix skills with education/location data
        # e.g. "Tamil Nadu CGPA: 7.95 Languages, English: Professional Proficiency, Erode"
        if re.search(r'\b(?:CGPA|GPA)\s*:\s*\d', cleaned, re.IGNORECASE):
            continue
        if re.search(r'\b(?:Languages?)\s*,\s*\w+\s*:\s*\w+\s+Proficiency\b', cleaned, re.IGNORECASE):
            continue
        items.extend(_split_list(cleaned))
    return items

def _parse_education_block(lines: List[str]) -> List[str]:
    entries: List[str] = []
    # Headers to skip
    skip_headers = {'education', 'academic background', 'educational qualifications', 'degrees'}
    for line in lines:
        cleaned = line.strip()
        # Skip header lines
        cleaned_lower = cleaned.lower().rstrip(':').strip()
        if cleaned_lower in skip_headers:
            continue
        # Strip leading header like "Education:"
        if cleaned.lower().startswith('education:'):
            cleaned = cleaned[10:].strip()
        if not cleaned:
            continue
        if cleaned.startswith(("-", "•", "*")) or re.match(r"^\d+\.", cleaned):
            cleaned = cleaned.lstrip("-•* ").strip()
        if _is_degree_line(cleaned) or _looks_like_education(cleaned):
            entries.append(cleaned)
    return entries

def _is_degree_line(line: str) -> bool:
    lower = line.lower()
    return bool(
        re.search(
            r"\b(b\.?e\.?|b\.?tech|btech|b\.?sc|bsc|b\.?a|ba|m\.?e\.?|m\.?tech|mtech|m\.?sc|msc|m\.?a|ma|mba|ph\.?d|phd)\b",
            lower,
        )
    )

def _append_span(cand: Candidate, chunk_id: str, line: str, span_seen: set[str]) -> None:
    snippet = " ".join(line.split())[:120]
    if not snippet or snippet in span_seen:
        return
    span_seen.add(snippet)
    cand.evidence_spans.append(EvidenceSpan(chunk_id=str(chunk_id), snippet=snippet))

def _merge_list(current: Optional[List[str]], new_items: List[str]) -> List[str]:
    merged = list(current or [])
    seen = {item.lower() for item in merged}
    for item in new_items or []:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged

def _looks_like_education(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ("bachelor", "master", "phd", "university", "college", "b.tech", "btech", "mba"))

def _infer_source_type(doc_name: str) -> Optional[str]:
    lowered = (doc_name or "").lower()
    if "linkedin" in lowered:
        return "LinkedIn profile"
    if "resume" in lowered or "cv" in lowered:
        return "Resume"
    if any(kw in lowered for kw in ("medical", "patient", "health", "hospital", "clinical")):
        return "Medical record"
    if any(kw in lowered for kw in ("invoice", "bill", "receipt")):
        return "Invoice"
    if any(kw in lowered for kw in ("contract", "agreement", "legal")):
        return "Legal document"
    return None

def _extract_name_guess(text: str, doc_name: str) -> Optional[str]:
    filename_guess = _name_from_filename(doc_name)
    if filename_guess:
        return filename_guess
    lines = _split_lines(text)[:6]
    for line in lines:
        candidate = line.split("|", 1)[0].strip()
        candidate = re.sub(r"(?i)^(name|candidate)\s*[:\-]\s*", "", candidate).strip()
        candidate = re.sub(r"[^A-Za-z\s]", "", candidate).strip()
        if _looks_like_name(candidate):
            return candidate
    return None

def _looks_like_name(value: str) -> bool:
    if not value:
        return False

    # Names don't end with colons (section headers do)
    if value.strip().endswith(':'):
        return False

    # Names NEVER contain commas — comma-separated lists are skills/keywords
    if ',' in value:
        return False

    # Clean value - remove hyphens and extra spaces
    value_cleaned = re.sub(r'\s*-\s*', ' ', value).strip()
    parts = [p for p in value_cleaned.split() if p]

    if len(parts) > 4:
        return False
    # Allow single-letter initials (e.g., "K V MADHU AADITHYA", "J. Smith")
    # but require at least one part with 2+ chars
    single_letter_count = sum(1 for p in parts if len(p.rstrip('.')) <= 1)
    has_real_name_part = any(len(p) >= 2 for p in parts)
    if single_letter_count > 0 and not has_real_name_part:
        return False
    if single_letter_count > 2:
        return False  # Too many single letters = likely abbreviation, not a name

    # Filter out common phrases that are not names (normalized without hyphens)
    phrases_not_names = {
        'cross functional', 'team collaboration', 'project management',
        'technical skills', 'professional summary', 'work experience',
        'contact information', 'personal details', 'key skills',
        'education details', 'professional experience', 'career objective',
        'supply chain', 'inventory management', 'material management',
        'vendor management', 'customer service', 'data analysis',
        'business development', 'quality assurance', 'software development',
        'results oriented', 'detail oriented', 'highly motivated',
        'cross functional team', 'team lead', 'cross functional collaboration',
        'functional team', 'team collaboration',
        # Project/product names
        'stock profit', 'profit classifier', 'machine learning', 'deep learning',
        'natural language', 'neural network', 'web application', 'mobile application',
        'data pipeline', 'real time', 'open source', 'full stack',
        'front end', 'back end', 'cloud computing', 'big data',
    }
    value_lower = value_cleaned.lower()
    if any(phrase in value_lower for phrase in phrases_not_names):
        return False

    # Filter out institution names (contain College, University, etc.)
    institution_keywords = ['college', 'university', 'institute', 'polytechnic', 'academy', 'school']
    if any(kw in value_lower for kw in institution_keywords):
        return False

    # Check for common words that indicate it's not a name
    non_name_words = {
        # Business/management terms
        'management', 'development', 'analysis', 'experience', 'skills',
        'summary', 'objective', 'professional', 'technical', 'functional',
        'collaboration', 'leadership', 'communication', 'team', 'project',
        'software', 'hardware', 'system', 'data', 'business', 'client',
        'customer', 'vendor', 'supplier', 'quality', 'process', 'results',
        'cross', 'oriented', 'driven', 'based', 'focused', 'related',
        # Section header words
        'roles', 'responsibility', 'responsibilities', 'duties', 'certification',
        'certifications', 'education', 'achievement', 'achievements', 'award',
        'awards', 'details', 'personal', 'contact', 'information',
        # Job title words
        'consultant', 'engineer', 'developer', 'analyst', 'specialist',
        'manager', 'executive', 'director', 'architect', 'administrator',
        'coordinator', 'associate', 'assistant', 'senior', 'junior', 'lead',
        # Education/institution words
        'college', 'university', 'institute', 'school', 'academy', 'polytechnic',
        'centenary', 'memorial', 'national', 'international',
        # SAP and tech terms
        'sap', 'scm', 'erp', 'crm', 'aws', 'azure', 'gcp',
        # Additional section header words
        'language', 'languages', 'proficiency', 'hobbies', 'interests',
        'references', 'declaration', 'signature', 'date', 'birth',
        # Project/product name words — never part of a person's name
        'stock', 'profit', 'classifier', 'predictor', 'tracker', 'monitor',
        'builder', 'generator', 'detector', 'parser', 'compiler', 'optimizer',
        'scheduler', 'calculator', 'converter', 'processor', 'handler',
        'factory', 'wrapper', 'helper', 'logger', 'checker', 'scanner',
        'validator', 'formatter', 'pipeline', 'framework', 'module',
        'application', 'platform', 'service', 'network', 'learning',
        'computing', 'database', 'storage', 'automation', 'integration',
    }
    # Strip trailing punctuation before matching (e.g. "Management." → "management")
    if any(part.lower().rstrip('.,;:') in non_name_words for part in parts):
        return False

    if len(parts) == 1:
        token = parts[0]
        return (token[:1].isupper() or (token[:1].isalpha() and not token[:1].islower())) and len(token) >= 3
    return all(p[:1].isupper() or (p[:1].isalpha() and not p[:1].islower()) for p in parts)

def _smart_title_case(text: str) -> str:
    """Title-case long ALL CAPS words, keep short abbreviations (PK, MM) uppercase."""
    parts = text.split()
    result = []
    for p in parts:
        if p.isupper() and len(p) > 2:
            result.append(p.title())
        else:
            result.append(p)
    return " ".join(result)

def _name_from_filename(doc_name: str) -> Optional[str]:
    if not doc_name:
        return None
    cleaned = re.sub(r"\.(pdf|docx?|rtf)$", "", doc_name, flags=re.IGNORECASE)

    # Strip role/designation suffixes after " - " (e.g., "Sabareesh M B - AI Engineer" → "Sabareesh M B")
    cleaned = re.sub(
        r"\s*[-–—]\s*(?:AI|ML|Sr\.?|Jr\.?|Senior|Junior|Lead|Staff|Principal|Associate|Chief)?\s*"
        r"(?:Engineer|Developer|Analyst|Consultant|Manager|Architect|Designer|"
        r"Specialist|Coordinator|Administrator|Officer|Intern|Trainee|Executive|Director)"
        r"(?:\s+.{0,30})?$",
        "", cleaned, flags=re.IGNORECASE,
    )

    # Split CamelCase filenames — "ManavGuptaResume" → "Manav Gupta Resume"
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)

    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\b(resume|cv|profile|linkedin)\b", "", cleaned, flags=re.IGNORECASE)
    # Strip common filename noise BEFORE name check
    cleaned = re.sub(r"\b(?:update|version|rev)\d*\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:ip|prod|test|doc|file|scan|scanned)\b", "", cleaned, flags=re.IGNORECASE)
    # Strip trailing 4-digit years — "SREELEKSHMI RESUME 2025" → "SREELEKSHMI"
    cleaned = re.sub(r"\b(20\d{2}|19\d{2})\b", "", cleaned)
    # Strip month names — "Raju July 2025" → "Raju" (after year already stripped)
    cleaned = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december"
        r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
        "", cleaned, flags=re.IGNORECASE,
    )
    # Strip leading/trailing numbers and parenthesized dedup suffixes — "21 Gokul (1)" → "Gokul"
    cleaned = re.sub(r"^\d+\s*", "", cleaned)
    cleaned = re.sub(r"\s*\(\d+\)\s*", " ", cleaned)

    # Strip noise words early (role/tech terms that leak into filenames)
    _filename_noise = {
        "new", "old", "updated", "update", "final", "draft",
        "copy", "scm", "sap", "erp", "latest", "revised", "v1", "v2", "v3",
        "ip", "prod",
        # SAP module abbreviations
        "ewm", "sd", "fi", "co", "wm", "bw", "abap", "hana",
        # Role/tech words that commonly leak into filenames
        "ai", "ml", "engineer", "developer", "analyst", "consultant",
        "manager", "architect", "designer", "specialist", "intern",
    }
    parts = cleaned.split()
    # Keep single uppercase letters (initials like M, B) but drop noise words
    name_parts = [p for p in parts if (
        (len(p) == 1 and p.isupper()) or  # Single initial
        (len(p) > 1 and p.lower() not in _filename_noise)
    )]
    cleaned = " ".join(name_parts)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Capitalize before checking (filenames may be lowercase like "raju", "aloysius")
    # Use .title() for true capitalization, then _smart_title_case for ALL CAPS handling
    capitalized = " ".join(p.capitalize() if p.islower() else p for p in cleaned.split())
    capitalized = _smart_title_case(capitalized)
    if _looks_like_name(capitalized):
        return capitalized

    return None

# ============================================================================
# Fallback Extraction Functions - Used when section-based extraction fails
# ============================================================================

def _extract_name_from_text(text: str) -> Optional[str]:
    """Extract candidate name from text using various patterns."""
    if not text:
        return None

    # Clean text first to remove extraction artifacts
    cleaned_text = _clean_extraction_text(text)

    # Normalize text - handle various newline representations
    normalized = cleaned_text.replace('\\n', '\n').replace(' n ', '\n')
    # Strip "Document Content" prefix (artifact from extraction metadata)
    normalized = re.sub(r'^Document\s+Content\s+', '', normalized, flags=re.IGNORECASE)

    # Also check if the original text has full_text format (before cleaning)
    if "full_text=" in text:
        fulltext_match = re.search(r"full_text=['\"]([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]*){0,3})\n", text)
        if fulltext_match:
            potential_name = fulltext_match.group(1).strip()
            words = potential_name.split()
            if 2 <= len(words) <= 4 and _looks_like_name(potential_name):
                if not any(h in potential_name.lower() for h in ['summary', 'objective', 'experience', 'education', 'professional', 'document']):
                    return potential_name

    lines = normalized.split('\n')[:15]  # Check first 15 lines

    for line in lines:
        line = line.strip()
        if not line or len(line) > 150:  # Allow longer lines for pipe-delimited format
            continue

        # Skip obvious non-name lines
        if any(header in line.lower() for header in [
            'skills', 'education', 'experience', 'contact', 'objective',
            'summary', 'certification', 'email:', 'phone:', 'address:'
        ]):
            continue

        # Pattern 0: Pipe-delimited format (e.g., "John Smith | Engineer | email@example.com")
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if parts:
                first_part = parts[0]
                # Check if first part looks like a name (2-4 words, capitalized)
                words = first_part.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() or not w[0].islower() for w in words if w):
                    # Filter out non-names
                    if not any(w.lower() in ['resume', 'cv', 'page', 'date', 'mr', 'ms', 'mrs', 'dr'] for w in words):
                        if '@' not in first_part and not any(c.isdigit() for c in first_part):
                            return first_part

        # Pattern 1: "Name: John Doe" format (check first to avoid false match in Pattern 2)
        name_match = re.match(r"(?i)^(?:name|candidate)\s*[:\-]\s*(.+)$", line)
        if name_match:
            name = name_match.group(1).strip()
            if _looks_like_name(name):
                return name

        # Pattern 2: Line with 2-4 capitalized words — validate with _looks_like_name()
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() or (w[0].isalpha() and not w[0].islower()) for w in words if w):
                if '@' not in line and not any(c.isdigit() for c in line):
                    if _looks_like_name(line):
                        return line

    return None

def _extract_skills_from_text(text: str, skill_type: str = "technical") -> List[str]:
    """Extract skills directly from text content."""
    skills = []

    # Clean the text first
    text = _clean_extraction_text(text)

    # Common technical skills keywords (expanded for domain expertise)
    tech_keywords = {
        # Programming languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'nosql',
        # Web frameworks
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        # Databases
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'oracle',
        # Data Science/ML
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'spark',
        # Frontend
        'html', 'css', 'sass', 'webpack', 'graphql', 'rest', 'api', 'microservices',
        # Operating Systems
        'linux', 'unix', 'windows', 'macos', 'bash', 'powershell',
        # Methodologies
        'agile', 'scrum', 'devops', 'ci/cd', 'jira', 'confluence',
        # SAP and ERP (common in enterprise)
        'sap', 'sap mm', 'sap sd', 'sap fi', 'sap co', 'sap wm', 'sap ewm',
        'sap scm', 'sap hana', 'sap s/4hana', 'sap abap', 'sap bw',
        'erp', 'crm', 'salesforce',
        # MS Office & Tools
        'excel', 'powerpoint', 'word', 'ms office', 'power bi', 'tableau',
        # Supply Chain
        'procurement', 'inventory management', 'supply chain', 'logistics',
        'warehouse management', 'material management',
    }

    # Common functional skills keywords
    func_keywords = {
        'communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
        'project management', 'time management', 'presentation', 'negotiation',
        'stakeholder management', 'requirement analysis', 'documentation',
        'mentoring', 'training', 'collaboration', 'decision making',
        'strategic planning', 'risk management', 'budget management',
        'vendor management', 'cross-functional', 'client facing', 'customer service',
    }

    keywords = tech_keywords if skill_type == "technical" else func_keywords
    text_lower = text.lower()

    # Find exact keyword matches using word boundaries for short keywords
    for keyword in keywords:
        # Use word boundary regex for short keywords (<=3 chars) to avoid false positives
        # e.g., 'r' should not match 'React', 'go' should not match 'google'
        if len(keyword) <= 3:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                skills.append(keyword.upper())
        elif keyword in text_lower:
            # Capitalize properly
            skills.append(keyword.title() if ' ' in keyword else keyword.capitalize())

    # Also extract from bullet points and comma-separated lists
    # Look for skill section patterns
    skill_section = re.search(
        r"(?:technical\s*)?skills?\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n[A-Z]|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )

    if skill_section:
        skill_text = skill_section.group(1)
        # Split by common delimiters
        items = re.split(r'[,•\|;]|\n', skill_text)
        for item in items:
            item = item.strip().strip('-').strip('*').strip()
            if item and 2 < len(item) < 40:
                # Filter out obviously non-skill items
                skip_patterns = [
                    'year', 'month', 'experience', 'worked', '@', '.com', '.org',
                    'phone', 'email', 'address', 'http', 'www', '+1', '+91',
                    'linkedin', 'github', 'twitter', 'implementation', 'support',
                    'in the', 'with the', 'for the', 'of the', 'ed in', 'ing in',
                    'responsible', 'managed', 'developed', 'coordinated',
                    'projects ', 'project ', 'classifier', 'predictor',
                    'tracker', 'monitor', 'detector', 'generator',
                ]
                if not any(pat in item.lower() for pat in skip_patterns):
                    # Also skip if it looks like a phone number or email
                    if not re.match(r'^[\d\s\-\+\(\)]+$', item) and '@' not in item:
                        # Skip items that are mostly lowercase (likely sentences, not skills)
                        if not item.islower() or item.lower() in keywords:
                            skills.append(item)

    # Deduplicate while preserving order and filter garbage
    seen = set()
    unique_skills = []
    # Regex to strip date ranges like "(OCT 2023 – NOV 2023)" or "(2023-2024)"
    _date_range_re = re.compile(
        r"\s*\(?\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\s*[\-–—]\s*"
        r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)?\s*\d{4}\s*\)?\s*$",
        re.IGNORECASE,
    )
    # Regex to detect sentence starts (bullet points + verbs)
    _sentence_start_re = re.compile(
        r"^[●•▪■\-*]\s*"
        r"|^(?:Implemented|Developed|Designed|Created|Built|Led|Managed|Worked|Used|Applied|Integrated|Achieved|Configured|Supported|Conducted)\b",
        re.IGNORECASE,
    )
    for skill in skills:
        # Clean up the skill
        skill = skill.strip()

        # Strip date ranges from skill items — "ANN (OCT 2023 – NOV 2023)" → "ANN"
        skill = _date_range_re.sub("", skill).strip()

        # Skip items containing newlines (likely garbage)
        if '\\n' in skill or '\n' in skill:
            continue

        # Skip items that start with bullet points or action verbs (sentences, not skills)
        if _sentence_start_re.search(skill):
            continue

        # Skip very long items (likely sentences, not skills)
        if len(skill) > 50:
            continue

        skill_lower = skill.lower()

        # Skip very short skills that aren't in our keyword list
        if len(skill) <= 2 and skill_lower not in {'r', 'c', 'ai', 'ml'}:
            continue

        # Skip garbage entries
        garbage_indicators = [
            'ed in', 'ing in', 'tion of', ' and ', ' or ',
            'led ', 'meetings', 'accuracy', 'operations',
            'dec 20', 'jan 20', 'feb 20', 'mar 20', 'apr 20', 'may 20',
            'jun 20', 'jul 20', 'aug 20', 'sep 20', 'oct 20', 'nov 20',
            'coimbatore', 'bangalore', 'chennai', 'mumbai', 'delhi',
            'stock profit', 'profit classifier', 'classifier', 'predictor',
        ]
        if any(garbage in skill_lower for garbage in garbage_indicators):
            continue

        if skill_lower not in seen:
            seen.add(skill_lower)
            unique_skills.append(skill)

    return unique_skills[:20]  # Return top 20 skills

def _extract_education_from_text(text: str) -> List[str]:
    """Extract education entries from text using aggressive pattern matching."""
    education = []

    # Strip common headers from text first
    text = re.sub(r'^Education\s*:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\nEducation\s*:\s*', '\n', text, flags=re.IGNORECASE)

    # Degree patterns — broad coverage of international degrees
    degree_patterns = [
        r"\b(B\.?Tech|B\.?E\.?|B\.?S\.?c?|B\.?A\.?|B\.?Com|B\.?C\.?A|B\.?B\.?A|Bachelor(?:'s)?)\b[^,\n]{0,100}",
        r"\b(M\.?Tech|M\.?E\.?|M\.?S\.?c?|M\.?A\.?|M\.?C\.?A|M\.?B\.?A|Master(?:'s)?|MBA)\b[^,\n]{0,100}",
        r"\b(Ph\.?D\.?|Doctorate|Doctor)\b[^,\n]{0,100}",
        r"\b(Diploma|Certificate|Associate)\b[^,\n]{0,100}",
        r"\b(SSLC|HSC|CBSE|ICSE|Plus\s*Two|12th|10th)\b[^,\n]{0,60}",
        r"\b(Engineering|Science|Commerce|Arts)\s+(?:in|from)\b[^,\n]{0,80}",
    ]

    for pattern in degree_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            edu_entry = match.group(0).strip()
            if len(edu_entry) > 8:
                education.append(edu_entry)

    # University/college/institute mentions with surrounding context
    uni_patterns = [
        r"(?:University|College|Institute|School|Academy)\s+(?:of\s+)?[A-Z][a-zA-Z\s,&]+(?:\d{4})?",
        r"[A-Z][a-zA-Z\s]+(?:University|College|Institute|School|Academy)[^,\n]{0,40}",
        r"(?:studied|graduated|degree)\s+(?:at|from|in)\s+[^,\n]{5,80}",
    ]
    for pattern in uni_patterns:
        for match in re.finditer(pattern, text):
            entry = match.group(0).strip()
            if len(entry) > 10:
                education.append(entry)

    # Line-by-line: look for lines that mention education keywords
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        lower = line.lower()
        if any(kw in lower for kw in ("degree", "university", "college", "institute", "bachelor", "master", "diploma", "graduated", "b.tech", "m.tech", "btech", "mtech", "b.e", "m.e", "b.sc", "m.sc", "bca", "mca", "bba", "mba")):
            if len(line) < 200 and not any(skip in lower for skip in ("skills", "experience", "contact", "phone", "email")):
                education.append(line)

    # Deduplicate
    seen = set()
    unique_edu = []
    for edu in education:
        edu_lower = edu.lower()[:50]
        if edu_lower not in seen:
            seen.add(edu_lower)
            unique_edu.append(edu)

    return unique_edu[:5]

_CONTACT_INFO_RE = re.compile(
    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # phone numbers
    r'|\S+@\S+\.\S+'                        # emails
    r'|https?://\S+'                         # URLs
    r'|\b\d{5,}\b',                          # long numbers (zip, ID)
)

def _clean_extracted_item(item: str) -> str:
    """Strip phone numbers, emails, URLs from an extracted cert/skill string."""
    item = _CONTACT_INFO_RE.sub('', item)
    return re.sub(r'\s+', ' ', item).strip()

def _is_contact_line(line: str) -> bool:
    """Return True if the line is primarily contact information (phone, email, address)."""
    return bool(re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\S+@\S+\.\S+', line))

def _extract_certifications_from_text(text: str) -> List[str]:
    """Extract certifications from text with broad pattern matching."""
    certs = []

    # Strip common headers from text first
    text = re.sub(r'^Certifications?\s*:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\nCertifications?\s*:\s*', '\n', text, flags=re.IGNORECASE)

    # Common certification patterns (tightened tails to avoid capturing contact info)
    cert_patterns = [
        r"(?:AWS|Amazon)\s+(?:Certified|Solutions?\s+Architect|Developer|SysOps)[^\n,\d@]{0,35}",
        r"(?:Azure|Microsoft)\s+(?:Certified|Administrator|Developer|Expert)[^\n,\d@]{0,35}",
        r"(?:Google|GCP)\s+(?:Certified|Cloud|Professional)[^\n,\d@]{0,35}",
        r"PMP|Project Management Professional",
        r"CAPM[^\n,\d@]{0,35}",
        r"Scrum\s+(?:Master|Product Owner|Developer)",
        r"(?:CISSP|CISM|CISA|CEH|CompTIA)[^\n,\d@]{0,25}",
        r"(?:Oracle|Salesforce)\s+Certified[^\n,\d@]{0,35}",
        r"SAP\s+(?:S/4\s*HANA|ERP|BW|FICO|SD|MM|PP|HR|SRM|SCM|Analytics\s+Cloud|SAC|Certified)[^\n,\d@]{0,35}",
        r"(?:CCNA|CCNP|CCIE)[^\n,\d@]{0,25}",
        r"(?:ITIL|Six\s+Sigma|TOGAF|PRINCE2|COBIT)[^\n,\d@]{0,25}",
        r"(?:CPA|CFA|FRM|ACCA|CA\b)[^\n,\d@]{0,25}",
        r"Business\s+Analysis[^\n,\d@]{0,30}",
        r"Certified\s+(?:Associate|Professional|Expert|Engineer|Specialist|Manager|Analyst|Auditor|ScrumMaster)[^\n,\d@]{0,35}",
    ]

    for pattern in cert_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            cert = _clean_extracted_item(match.group(0))
            if cert and len(cert) >= 3:
                certs.append(cert)

    # Line-by-line: look for lines mentioning certifications (skip contact lines)
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        if _is_contact_line(line):
            continue
        lower = line.lower()
        if any(kw in lower for kw in ("certified", "certification", "certificate", "credential", "license")):
            if len(line) < 200 and not any(skip in lower for skip in ("skills", "education", "experience")):
                cleaned = _clean_extracted_item(line)
                if cleaned and len(cleaned) >= 5:
                    certs.append(cleaned)

    # Deduplicate
    seen = set()
    unique_certs = []
    for cert in certs:
        cert_lower = cert.lower()
        if cert_lower not in seen:
            seen.add(cert_lower)
            unique_certs.append(cert)

    return unique_certs[:10]

def _items_field(items: List[InvoiceItem]) -> InvoiceItemsField:
    if items:
        return InvoiceItemsField(items=items, missing_reason=None)
    return InvoiceItemsField(items=None, missing_reason=MISSING_REASON)

def _field_values_field(items: List[FieldValue]) -> FieldValuesField:
    if items:
        return FieldValuesField(items=items, missing_reason=None)
    return FieldValuesField(items=None, missing_reason=MISSING_REASON)

def _candidates_field(items: List[Candidate]) -> CandidateField:
    if items:
        return CandidateField(items=items, missing_reason=None)
    return CandidateField(items=None, missing_reason=MISSING_REASON)

def _clauses_field(items: List[Clause]) -> ClauseField:
    if items:
        return ClauseField(items=items, missing_reason=None)
    return ClauseField(items=None, missing_reason=MISSING_REASON)

def _detect_multi_entity_collision(
    domain: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema | MedicalSchema | PolicySchema,
    chunks: List[Any],
    scope_document_id: Optional[str],
    intent: Optional[str] = None,
) -> bool:
    # GenericSchema handles multi-doc natively via document_name on facts
    if isinstance(schema, GenericSchema):
        return False
    if domain == "hr" and isinstance(schema, HRSchema):
        return False
    # Domain-specific schemas aggregate from all chunks natively — they
    # should NOT be collapsed into a shallow document listing.
    if isinstance(schema, (MedicalSchema, PolicySchema, InvoiceSchema, LegalSchema)):
        return False

    doc_ids = {
        str(_chunk_document_id(chunk))
        for chunk in chunks
        if _chunk_document_id(chunk)
    }
    if scope_document_id:
        return False
    return len(doc_ids) > 1

def _build_multi_entity_schema(
    domain: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    chunks: List[Any],
) -> MultiEntitySchema:
    entities: List[EntitySummary] = []

    if domain == "hr" and isinstance(schema, HRSchema):
        candidates = schema.candidates.items if schema.candidates else None
        for cand in candidates or []:
            label = cand.name or "Candidate"
            entities.append(
                EntitySummary(
                    label=label,
                    document_name=None,
                    document_id=None,
                    evidence_spans=cand.evidence_spans or [],
                )
            )
        return MultiEntitySchema(entities=entities or None, missing_reason=None if entities else MISSING_REASON)

    doc_map = {}
    for chunk in chunks:
        doc_id = _chunk_document_id(chunk)
        if not doc_id:
            continue
        doc_key = str(doc_id)
        if doc_key in doc_map:
            continue
        doc_map[doc_key] = chunk

    for doc_id, chunk in doc_map.items():
        doc_name = getattr(getattr(chunk, "source", None), "document_name", None)
        snippet = getattr(chunk, "text", "") or ""
        entities.append(
            EntitySummary(
                label=doc_name or "Document",
                document_name=doc_name,
                document_id=str(doc_id),
                evidence_spans=[_span(getattr(chunk, "id", ""), snippet)],
            )
        )

    return MultiEntitySchema(entities=entities or None, missing_reason=None if entities else MISSING_REASON)

def _chunk_document_id(chunk: Any) -> Optional[str]:
    meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}
    for key in ("document_id", "doc_id", "docId"):
        value = meta.get(key)
        if value:
            return str(value)
    return None

def extract_schema(
    domain: Optional[str],
    *,
    query: str,
    chunks: List[Any],
    llm_client: Optional[Any],
    budget: LLMBudget,
    correlation_id: Optional[str] = None,
    scope_document_id: Optional[str] = None,
    intent_hint: Optional[str] = None,
    document_data: Optional[Dict[str, Any]] = None,
    query_focus: Optional[Any] = None,
    tool_domain: bool = False,
    embedder: Any = None,
    tool_context: Optional[str] = None,
    intent_parse: Any = None,
    redis_client: Any = None,
    use_thinking: bool = False,
    task_spec: Any = None,
    conversation_context: Optional[str] = None,
) -> ExtractionResult:
    """Extract structured data from chunks.

    Strategy:
    1. LLM-first generic extraction — works for ANY document type.
       The LLM reads the chunks and produces a direct answer.
    2. Deterministic fallback — structured extraction when LLM is unavailable.

    When ``tool_domain=True``, the ``domain`` was set by an explicit tool
    selection (e.g. tools=resume-analysis → domain="hr").  This means:
    - Skip domain mismatch checks (user chose the tool deliberately)
    - Prefer deterministic structured extraction over LLM free-text
    """
    from .llm_extract import llm_extract_and_respond, _count_unique_documents

    # When a tool explicitly sets the domain (e.g., tools=resume-analysis → domain="hr"),
    # skip the mismatch check — the user chose this tool deliberately.
    _tool_domain_active = tool_domain and bool(domain and domain not in ("generic", ""))

    # Check for domain mismatch before any extraction (skip when tool domain is set)
    query_domain = _ml_query_domain(query, intent_parse=intent_parse)
    chunk_domain = _majority_chunk_domain(chunks)
    if not _tool_domain_active and query_domain and chunk_domain:
        _DOMAIN_FAMILY = {
            "hr": "hr", "resume": "hr",
            "invoice": "invoice",
            "legal": "legal_policy", "policy": "legal_policy",
            "insurance": "legal_policy",
            "medical": "medical",
            "report": "generic", "generic": "generic",
        }
        q_family = _DOMAIN_FAMILY.get(query_domain, "generic")
        c_family = _DOMAIN_FAMILY.get(chunk_domain, "generic")
        # Only flag mismatch for genuinely incompatible specific domains.
        # Invoice mismatch is reliable (invoice vs non-invoice is clear).
        # Medical mismatch is reliable (medical vs non-medical is clear,
        # content validators confirm via patient/diagnosis/medication terms).
        # Legal/policy mismatch is unreliable — "terms", "treatment plan"
        # etc. trigger false positives across medical/policy domains.
        _MISMATCH_DOMAINS = {"invoice", "medical"}
        _any_matching = False
        for _c in (chunks or []):
            _cm = getattr(_c, "meta", None) or {}
            _cd = str(_cm.get("doc_domain") or _cm.get("doc_type") or "").lower().strip()
            if _DOMAIN_FAMILY.get(_cd, "generic") == q_family:
                _any_matching = True
                break
        if q_family != c_family and q_family in _MISMATCH_DOMAINS and not _any_matching:
            # Query asks for invoice/legal but NO chunks match that domain
            _DOMAIN_LABEL = {"hr": "resumes", "resume": "resumes", "invoice": "invoices",
                             "legal": "legal documents", "medical": "medical records",
                             "policy": "policy documents", "insurance": "insurance documents"}
            asked = _DOMAIN_LABEL.get(query_domain, query_domain)
            available = _DOMAIN_LABEL.get(chunk_domain, chunk_domain)
            msg = (f"No {asked} found in this profile. The available documents are {available}. "
                   f"To analyze {asked}, please upload the relevant {asked} documents to this profile.")
            schema = GenericSchema(facts=FieldValuesField(items=[FieldValue(label=None, value=msg, evidence_spans=[])]))
            return ExtractionResult(domain=chunk_domain, intent=intent_hint or "answer", schema=schema)

    # Contact queries use deterministic extraction directly — LLMs often refuse
    # to share personal contact details due to safety guardrails.
    _is_contact_query = _nlu_is_contact(query)

    # When a tool explicitly sets the domain (e.g., tools=resume-analysis),
    # prefer the structured deterministic extractor which produces typed schemas
    # (HRSchema with Candidate objects, InvoiceSchema, etc.) instead of LLM
    # free-text that loses structure.
    #
    # IMPORTANT: For analytical/reasoning queries (ranking, comparison, summary,
    # generate, reasoning), ALWAYS prefer LLM over deterministic — the LLM
    # understands the query intent and produces a targeted answer, while
    # deterministic just dumps raw schema data regardless of the question asked.
    # Only prefer deterministic for simple factual/extract queries on structured
    # domains, or when tool domain is explicitly set.
    _structured_domains = {"invoice", "medical", "legal", "policy", "hr"}
    _domain_confident = chunk_domain in _structured_domains
    _query_domain_confident = query_domain in _structured_domains
    # Intents that REQUIRE LLM intelligence (never use deterministic for these)
    _llm_required_intents = {
        "rank", "ranking", "compare", "comparison", "summarize", "summary",
        "generate", "reasoning", "analyze",
        "cross_document", "analytics", "multi_field", "timeline",
    }
    _intent_str = (intent_hint or "").lower().strip()
    # Also check TaskSpec intent (NLU-based, may differ from LLM intent parser)
    _taskspec_intent = ""
    if task_spec and hasattr(task_spec, "intent"):
        _taskspec_intent = (task_spec.intent or "").lower().strip()
    _needs_llm = _intent_str in _llm_required_intents or _taskspec_intent in _llm_required_intents
    # Extra check: detect content generation verbs in query itself
    import re as _re_extract
    _has_generate_verbs = _re_extract.search(
        r"\b(generate|create|write|draft|compose|prepare|produce)\b",
        (query or "").lower(),
    )
    if not _needs_llm and _has_generate_verbs:
        _needs_llm = True
    # Extra check: detect comparison/table signals in query itself
    _has_comparison_signals = _re_extract.search(
        r"\b(compare|comparison|versus|vs\.?|side.by.side|table format|each candidate|all candidates|rank all|rank the)\b",
        (query or "").lower(),
    )
    if not _needs_llm and _has_comparison_signals:
        _needs_llm = True
        _intent_str = "comparison"
        intent_hint = "comparison"
    # When TaskSpec or query verbs indicate content generation, override the
    # LLM parser intent so downstream (llm_extract, template selection) uses
    # the correct "generate" intent instead of a misclassified "rank"/"extract".
    if _taskspec_intent == "generate" or _has_generate_verbs:
        _intent_str = "generate"
        intent_hint = "generate"
    # Prefer deterministic when: (a) tool domain is explicitly set AND intent
    # is simple extraction, OR (b) domain is confidently structured AND intent is
    # simple factual/extract (not analytical).  For analytical intents, LLM always
    # wins because it understands what the user is actually asking.
    # (c) LATENCY: For structured domains with simple factual/extract intents,
    # deterministic is faster (0.5s vs 15-45s LLM) AND more reliable — the LLM
    # often returns empty for medical/policy structured queries.
    _simple_intents = {"factual", "extraction", "extract", "qa", "list", "contact", ""}

    # Detect complex queries that need LLM even for "simple" intents:
    # Multi-part queries, conditional logic, or synthesis across many fields
    _query_lower = (query or "").lower()
    _is_complex_query = (
        _intent_str in ("comparison", "ranking", "reasoning", "cross_document", "analytics")
        or _query_lower.count(" and ") >= 2  # "Name and skills and experience"
        or (" with " in _query_lower and any(w in _query_lower for w in ("who", "which", "that")))
        or any(w in _query_lower for w in ("explain", "why", "how does", "what makes", "describe",
                                               "tell me about", "give me a summary", "summarize",
                                               "overview", "brief summary", "profile"))
        or any(w in _query_lower for w in ("total", "sum", "aggregate", "average", "count", "how many"))  # Aggregation queries need LLM computation
        or any(w in _query_lower for w in ("common", "shared", "across all"))  # Cross-document synthesis
        or (len(chunks) > 8 and len(_query_lower.split()) > 10)  # Long query + many chunks
    )

    _prefer_deterministic = (
        _is_contact_query
        or (_tool_domain_active and not _needs_llm)
        or ((_domain_confident or _query_domain_confident) and not _needs_llm
            and _intent_str in _simple_intents and not _is_complex_query)
    )
    logger.info(
        "Extraction strategy: prefer_deterministic=%s (chunk_domain=%s query_domain=%s "
        "domain_confident=%s query_domain_confident=%s intent=%r needs_llm=%s simple=%s "
        "complex_query=%s contact=%s tool=%s)",
        _prefer_deterministic, chunk_domain, query_domain,
        _domain_confident, _query_domain_confident, _intent_str,
        _needs_llm, _intent_str in _simple_intents, _is_complex_query,
        _is_contact_query, _tool_domain_active,
    )

    # ── ML-based context understanding (enriches LLM prompts) ──────────
    # Skip when deterministic extraction is preferred — context understanding
    # only enriches LLM prompts, which are skipped for deterministic paths.
    # This saves ~5-15s of embedding + clustering overhead on large chunk sets.
    _context_intelligence = None
    if embedder and chunks and not _is_contact_query and not _prefer_deterministic:
        try:
            from src.intelligence.context_understanding import understand_context_for_prompt
            _context_intelligence = understand_context_for_prompt(
                query=query,
                chunks=chunks,
                embedder=embedder,
                domain_hint=chunk_domain or query_domain or domain,
                intent_hint=_intent_str or intent_hint,
            )
            if _context_intelligence:
                logger.info(
                    "Context understanding: %d chars injected into prompt",
                    len(_context_intelligence),
                    extra={"stage": "context_understanding", "correlation_id": correlation_id},
                )
        except Exception as _cu_exc:
            logger.debug("Context understanding skipped: %s", _cu_exc)

    # ── Multi-resolution context: extract doc/section-level summaries ───────
    # When chunks include doc-level or section-level resolution metadata, build
    # a structured context dict that enriches the LLM prompt with hierarchical
    # document understanding.  This is additive — if no resolution data exists,
    # _multi_res_ctx is None and behavior is unchanged.
    _multi_res_ctx = None
    if chunks and not _is_contact_query and not _prefer_deterministic:
        try:
            from .llm_extract import _build_multi_resolution_context
            _multi_res_ctx = _build_multi_resolution_context(chunks)
            if _multi_res_ctx:
                _doc_count = len(_multi_res_ctx.get("doc_context") or [])
                _sec_count = len(_multi_res_ctx.get("section_context") or [])
                logger.info(
                    "Multi-resolution context built: %d doc summaries, %d section summaries",
                    _doc_count, _sec_count,
                    extra={"stage": "multi_resolution", "correlation_id": correlation_id},
                )
        except Exception as _mr_exc:
            logger.debug("Multi-resolution context skipped: %s", _mr_exc)

    # STRATEGY 1: LLM-first generic extraction (preferred when no tool domain)
    _llm_already_tried = False
    if llm_client and budget.allow() and not _is_contact_query and not _prefer_deterministic:
        _llm_already_tried = True
        try:
            llm_result = llm_extract_and_respond(
                query=query,
                chunks=chunks,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                intent=intent_hint,
                num_documents=_count_unique_documents(chunks),
                tool_context=tool_context,
                domain=chunk_domain or query_domain or domain or "generic",
                redis_client=redis_client,
                context_intelligence=_context_intelligence,
                use_thinking=use_thinking,
                multi_resolution_context=_multi_res_ctx,
                task_spec=task_spec,
                conversation_context=conversation_context,
            )
            if llm_result is not None:
                # Use chunk-based domain (more reliable than query override)
                inferred = chunk_domain or query_domain or domain or "generic"
                logger.info(
                    "LLM-first extraction succeeded (domain=%s)",
                    inferred,
                    extra={"stage": "extract", "correlation_id": correlation_id, "method": "llm_first"},
                )
                return ExtractionResult(
                    domain=inferred,
                    intent=intent_hint or "answer",
                    schema=llm_result,
                )
        except Exception as exc:
            logger.warning(
                "LLM-first extraction failed, falling back to deterministic: %s",
                exc,
                extra={"stage": "extract", "correlation_id": correlation_id},
            )

        # GENERATE INTENT RESILIENCE: When intent=generate and first LLM attempt
        # failed (timeout/error), retry with fewer chunks and simplified context.
        # Content generation (cover letters, summaries, etc.) MUST use LLM — a
        # deterministic HRSchema ranking is never a valid response for "write me a
        # cover letter".
        if _intent_str == "generate" and _llm_already_tried and llm_client and budget.allow():
            logger.info(
                "Generate intent: retrying LLM with reduced chunks after initial failure",
                extra={"stage": "extract", "correlation_id": correlation_id, "method": "generate_retry"},
            )
            # Use top 5 chunks only (less context = faster generation)
            _retry_chunks = chunks[:5] if len(chunks) > 5 else chunks
            try:
                llm_result = llm_extract_and_respond(
                    query=query,
                    chunks=_retry_chunks,
                    llm_client=llm_client,
                    budget=budget,
                    correlation_id=correlation_id,
                    intent=intent_hint,
                    num_documents=_count_unique_documents(_retry_chunks),
                    tool_context=tool_context,
                    domain=chunk_domain or query_domain or domain or "generic",
                    redis_client=redis_client,
                    context_intelligence=None,  # Skip heavy context for retry
                    use_thinking=False,
                    multi_resolution_context=None,
                    task_spec=task_spec,
                )
                if llm_result is not None:
                    inferred = chunk_domain or query_domain or domain or "generic"
                    logger.info(
                        "Generate retry succeeded with %d chunks (domain=%s)",
                        len(_retry_chunks), inferred,
                        extra={"stage": "extract", "correlation_id": correlation_id, "method": "generate_retry"},
                    )
                    return ExtractionResult(
                        domain=inferred,
                        intent="generate",
                        schema=llm_result,
                    )
            except Exception as _retry_exc:
                logger.warning(
                    "Generate retry also failed: %s", _retry_exc,
                    extra={"stage": "extract", "correlation_id": correlation_id},
                )

    # STRATEGY 2: Deterministic extraction (domain-aware, structured schemas)
    # When tool domain is set, this is the preferred path (structured output).
    # Use query_domain as domain_hint when chunk metadata disagrees but query is clear
    _det_domain = domain or (query_domain if _query_domain_confident else None) or ""
    deterministic_result = schema_extract(
        query=query,
        chunks=chunks,
        llm_client=None,  # deterministic only — no LLM in first pass
        budget=budget,
        correlation_id=correlation_id,
        scope_document_id=scope_document_id,
        domain_hint=_det_domain,
        intent_hint=intent_hint,
        query_focus=query_focus,
        embedder=embedder,
        intent_parse=intent_parse,
    )

    # CONTACT QUERY LLM FALLBACK: When deterministic extraction for a contact
    # query produced nothing useful, fall back to LLM extraction.  Contact info
    # may be embedded in free-text paragraphs that the deterministic parser misses.
    if _is_contact_query and llm_client and budget.allow() and not _llm_already_tried:
        from src.rag_v3.pipeline import _has_valid_deterministic_extraction
        if not _has_valid_deterministic_extraction(deterministic_result.schema):
            logger.info(
                "Contact query: deterministic extraction empty, falling back to LLM",
                extra={"stage": "extract", "correlation_id": correlation_id, "method": "contact_llm_fallback"},
            )
            try:
                llm_result = llm_extract_and_respond(
                    query=query,
                    chunks=chunks,
                    llm_client=llm_client,
                    budget=budget,
                    correlation_id=correlation_id,
                    intent=intent_hint or "contact",
                    num_documents=_count_unique_documents(chunks),
                    tool_context=tool_context,
                    domain=chunk_domain or query_domain or domain or "generic",
                    redis_client=redis_client,
                    context_intelligence=None,
                    use_thinking=False,
                    multi_resolution_context=None,
                    task_spec=task_spec,
                    conversation_context=conversation_context,
                )
                if llm_result is not None:
                    inferred = chunk_domain or query_domain or domain or "generic"
                    logger.info(
                        "Contact LLM fallback succeeded (domain=%s)", inferred,
                        extra={"stage": "extract", "correlation_id": correlation_id},
                    )
                    return ExtractionResult(
                        domain=inferred,
                        intent="contact",
                        schema=llm_result,
                    )
            except Exception as _contact_exc:
                logger.warning(
                    "Contact LLM fallback failed: %s", _contact_exc,
                    extra={"stage": "extract", "correlation_id": correlation_id},
                )

    # GENERATE INTENT GUARD: When intent=generate but deterministic produced a
    # structured schema (HRSchema, InvoiceSchema, etc.), convert the schema to
    # prompt context and retry with LLM — a structured ranking table is never
    # what the user wants for "write a cover letter" or "create interview questions".
    if _intent_str == "generate":
        _det_schema = deterministic_result.schema
        _is_structured = isinstance(_det_schema, (HRSchema, InvoiceSchema, MedicalSchema, LegalSchema, PolicySchema))
        if _is_structured and llm_client and budget.allow():
            logger.info(
                "Generate intent: converting deterministic %s schema to LLM context for content generation",
                type(_det_schema).__name__,
                extra={"stage": "extract", "correlation_id": correlation_id},
            )
            # Build schema context string to enrich the LLM prompt
            _schema_facts = []
            if isinstance(_det_schema, HRSchema) and _det_schema.candidates and getattr(_det_schema.candidates, "items", None):
                for _cand in _det_schema.candidates.items:
                    if _cand.name:
                        _schema_facts.append(f"Candidate: {_cand.name}")
                    if _cand.skills and getattr(_cand.skills, "items", None):
                        _schema_facts.append(f"Skills: {', '.join(s.value or '' for s in _cand.skills.items if s.value)}")
                    if _cand.experience and getattr(_cand.experience, "items", None):
                        _schema_facts.append(f"Experience: {', '.join(e.value or '' for e in _cand.experience.items if e.value)}")
                    if _cand.education and getattr(_cand.education, "items", None):
                        _schema_facts.append(f"Education: {', '.join(ed.value or '' for ed in _cand.education.items if ed.value)}")
            _schema_context = "\n".join(_schema_facts) if _schema_facts else None
            # Inject schema context into tool_context for the LLM
            _gen_tool_ctx = dict(tool_context) if tool_context else {}
            if _schema_context:
                _gen_tool_ctx["extracted_profile"] = _schema_context
            try:
                llm_result = llm_extract_and_respond(
                    query=query,
                    chunks=chunks[:8] if len(chunks) > 8 else chunks,
                    llm_client=llm_client,
                    budget=budget,
                    correlation_id=correlation_id,
                    intent="generate",
                    num_documents=_count_unique_documents(chunks),
                    tool_context=_gen_tool_ctx,
                    domain=chunk_domain or query_domain or domain or "generic",
                    redis_client=redis_client,
                    context_intelligence=_context_intelligence,
                    use_thinking=False,
                    multi_resolution_context=None,
                    task_spec=task_spec,
                    conversation_context=conversation_context,
                )
                if llm_result is not None:
                    inferred = deterministic_result.domain or chunk_domain or "generic"
                    logger.info(
                        "Generate intent LLM retry succeeded (domain=%s)", inferred,
                        extra={"stage": "extract", "correlation_id": correlation_id, "method": "generate_schema_retry"},
                    )
                    return ExtractionResult(
                        domain=inferred,
                        intent="generate",
                        schema=llm_result,
                    )
            except Exception as _gen_exc:
                logger.warning(
                    "Generate intent LLM retry failed: %s", _gen_exc,
                    extra={"stage": "extract", "correlation_id": correlation_id},
                )
            # If LLM also failed, return a helpful placeholder
            _gen_msg = (
                "I understand you're asking me to generate content. "
                "Let me try to create that for you based on the available documents."
            )
            _gen_schema = GenericSchema(
                facts=FieldValuesField(items=[FieldValue(label=None, value=_gen_msg, evidence_spans=[])])
            )
            return ExtractionResult(
                domain=deterministic_result.domain,
                intent="generate",
                schema=_gen_schema,
            )

    # STRATEGY 3: LLM fallback when deterministic extraction produced nothing
    # Deterministic extraction relies on line-segmented text; it fails on OCR blobs
    # without line breaks.  In that case, let the LLM synthesize from raw chunks.
    # SKIP if Strategy 1 already tried LLM and failed — no point retrying the same
    # LLM call which will just add another 30-60s of latency with no result.
    from src.rag_v3.pipeline import _has_valid_deterministic_extraction
    _det_ok = _has_valid_deterministic_extraction(deterministic_result.schema)
    if _det_ok or not llm_client or not budget.allow() or _llm_already_tried or _prefer_deterministic:
        return deterministic_result

    logger.info(
        "Deterministic extraction empty for domain=%s; falling back to LLM synthesis",
        deterministic_result.domain,
        extra={"stage": "extract", "correlation_id": correlation_id},
    )
    try:
        llm_result = llm_extract_and_respond(
            query=query,
            chunks=chunks,
            llm_client=llm_client,
            budget=budget,
            correlation_id=correlation_id,
            intent=intent_hint,
            num_documents=_count_unique_documents(chunks),
            tool_context=tool_context,
            domain=deterministic_result.domain or chunk_domain or "generic",
            redis_client=redis_client,
            context_intelligence=_context_intelligence,
            use_thinking=use_thinking,
            multi_resolution_context=_multi_res_ctx,
            task_spec=task_spec,
            conversation_context=conversation_context,
        )
        if llm_result is not None:
            inferred = deterministic_result.domain or chunk_domain or "generic"
            return ExtractionResult(
                domain=inferred,
                intent=intent_hint or "answer",
                schema=llm_result,
            )
    except Exception as exc:
        logger.warning(
            "LLM fallback for structured domain also failed: %s",
            exc,
            extra={"stage": "extract", "correlation_id": correlation_id},
        )
    return deterministic_result

def _convert_document_data_to_candidate(data: Dict[str, Any]) -> Candidate:
    """Convert document extraction result to Candidate model."""
    return Candidate(
        name=data.get("name"),
        role=None,
        details=None,
        total_years_experience=data.get("total_years_experience"),
        experience_summary=data.get("experience_summary"),
        technical_skills=data.get("technical_skills") or [],
        functional_skills=data.get("functional_skills") or [],
        certifications=data.get("certifications") or [],
        education=data.get("education") or [],
        achievements=data.get("achievements") or [],
        emails=data.get("email") or [],
        phones=data.get("phone") or [],
        linkedins=data.get("linkedin") or [],
        source_type=data.get("source_type"),
        evidence_spans=[],
        missing_reason={},
    )

