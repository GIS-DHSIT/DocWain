from __future__ import annotations

import concurrent.futures
import json
import logging
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
    MultiEntitySchema,
)

logger = logging.getLogger(__name__)

EXTRACT_TIMEOUT_MS = 15000

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

_PRODUCT_INTENTS = {"item", "items", "product", "products", "service", "services", "line item", "line items"}

_TOTAL_INTENTS = {"total", "amount due", "balance", "subtotal"}
_CONTACT_INTENTS = {"contact", "contacts", "email", "emails", "phone", "phones", "reach", "linkedin"}
_FIT_INTENTS = {"fit", "fitting", "suitable", "suitability", "best", "top", "most", "match"}


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
) -> ExtractionResult:
    domain, intent = _infer_domain_intent(query, chunks, domain_hint=domain_hint, intent_hint=intent_hint)
    schema = _deterministic_extract(domain, intent, query, chunks)

    if _schema_is_empty(schema) and llm_client and budget.consume():
        llm_schema = _llm_extract(domain, intent, query, chunks, llm_client, correlation_id)
        if llm_schema is not None:
            schema = llm_schema

    if _detect_multi_entity_collision(domain, schema, chunks, scope_document_id, intent):
        schema = _build_multi_entity_schema(domain, schema, chunks)
        domain = "multi"

    return ExtractionResult(domain=domain, intent=intent, schema=schema)


_HR_CONTENT_HINTS = ("resume", "candidate", "skills", "experience", "education", "certification")
_INVOICE_CONTENT_HINTS = ("invoice", "amount due", "bill to", "total due", "payment terms")
_LEGAL_CONTENT_HINTS = ("contract", "agreement", "clause", "liability", "parties")

# Strong query signals that override chunk-based domain detection
_QUERY_HR_STRONG = ("resume", "cv", "curriculum vitae", "candidate")
_QUERY_HR_WEAK = ("skills", "education", "certification", "certifications", "experience", "achievements")
_QUERY_INVOICE_STRONG = ("invoice", "invoices", "payment", "bill", "amount due")
_QUERY_LEGAL_STRONG = ("contract", "agreement", "clause", "liability")


def _query_domain_override(query: str) -> Optional[str]:
    """Detect domain from query keywords — strong signals override chunk majority."""
    lowered = (query or "").lower()
    # Strong HR signals — a single match is sufficient
    if any(token in lowered for token in _QUERY_HR_STRONG):
        return "hr"
    # Strong invoice/legal signals
    if any(token in lowered for token in _QUERY_INVOICE_STRONG):
        return "invoice"
    if any(token in lowered for token in _QUERY_LEGAL_STRONG):
        return "legal"
    # Weak HR signals — need at least 2 to count as HR query
    if sum(1 for token in _QUERY_HR_WEAK if token in lowered) >= 2:
        return "hr"
    return None


def _majority_chunk_domain(chunks: List[Any]) -> Optional[str]:
    """Determine domain from chunk metadata majority vote, with content fallback."""
    domain_counts: Dict[str, int] = {}
    for chunk in chunks:
        meta = getattr(chunk, "meta", None) or {}
        d = str(meta.get("doc_domain") or meta.get("doc_type") or "").lower().strip()
        if d and d not in ("generic", ""):
            domain_counts[d] = domain_counts.get(d, 0) + 1
    if domain_counts:
        best = max(domain_counts, key=domain_counts.get)
        domain_map = {"resume": "hr", "hr": "hr", "invoice": "invoice", "legal": "legal"}
        return domain_map.get(best, best)

    # Content-based fallback when metadata is missing
    sample = " ".join((getattr(c, "text", "") or "")[:200] for c in chunks[:3]).lower()
    if any(h in sample for h in _HR_CONTENT_HINTS):
        return "hr"
    if any(h in sample for h in _INVOICE_CONTENT_HINTS):
        return "invoice"
    if any(h in sample for h in _LEGAL_CONTENT_HINTS):
        return "legal"
    return None


def _infer_domain_intent(
    query: str,
    chunks: List[Any],
    domain_hint: Optional[str] = None,
    intent_hint: Optional[str] = None,
) -> Tuple[str, str]:
    """Infer domain from chunk metadata and query intent from keywords.

    Domain detection uses chunk metadata majority vote so that resume chunks
    route through the structured HR extractor instead of the generic one.
    """
    # Detect domain: query signals first, then chunk metadata majority
    domain = (domain_hint or "").strip().lower()
    if not domain or domain == "generic":
        # Query-based override takes precedence (e.g., "Abinaya's resume" → hr)
        query_domain = _query_domain_override(query)
        if query_domain:
            domain = query_domain
        else:
            detected = _majority_chunk_domain(chunks)
            if detected:
                domain = detected
            else:
                domain = "generic"

    intent = _normalize_intent_hint(intent_hint) or "summary"
    lowered_query = (query or "").lower()

    # Domain-agnostic intent detection
    if intent == "summary" and any(word in lowered_query for word in _CONTACT_INTENTS):
        intent = "contact"
    elif intent == "summary" and ("rank" in lowered_query or "ranking" in lowered_query or any(word in lowered_query for word in _FIT_INTENTS)):
        intent = "rank"
    elif intent == "summary" and "compare" in lowered_query:
        intent = "compare"
    elif intent == "summary" and ("list" in lowered_query or "candidates" in lowered_query):
        intent = "list"
    elif intent == "summary":
        intent = "facts"

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


def _deterministic_extract(domain: str, intent: str, query: str, chunks: List[Any]):
    """Route to domain-specific extractor when domain is known, else generic."""
    if domain in ("hr", "resume"):
        return _extract_hr(chunks)
    return _extract_document_intelligence(query, chunks)


def _extract_document_intelligence(query: str, chunks: List[Any]) -> GenericSchema:
    """Universal document intelligence extraction.

    Works for ANY document type by analysing content structure
    (KV pairs, lists, sections, contact info, relevant sentences)
    rather than relying on domain classification.
    """
    keywords = _keywords(query)
    _, intent = _infer_domain_intent(query, chunks)
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
    facts = _score_and_sort_facts(facts, keywords, query, intent=intent)

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


def _extract_invoice(chunks: List[Any]) -> InvoiceSchema:
    items: List[InvoiceItem] = []
    totals: List[FieldValue] = []
    parties: List[FieldValue] = []
    terms: List[FieldValue] = []
    seen_items = set()

    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        chunk_id = getattr(chunk, "id", "")
        for line in _split_lines(text):
            cleaned = line.strip()
            if not cleaned:
                continue
            item_match = re.match(r"(?i)^\s*(?:item|product|service|description|line item)\s*[:\-]\s*(.+)$", cleaned)
            if item_match:
                desc = item_match.group(1).strip()
                if desc:
                    key = _normalize_key(desc)
                    if key not in seen_items:
                        seen_items.add(key)
                        items.append(
                            InvoiceItem(
                                description=desc,
                                evidence_spans=[_span(chunk_id, cleaned)],
                            )
                        )
                continue

            if _is_bullet_item(cleaned):
                desc = cleaned.lstrip("-• ").strip()
                key = _normalize_key(desc)
                if desc and key not in seen_items:
                    seen_items.add(key)
                    items.append(
                        InvoiceItem(
                            description=desc,
                            evidence_spans=[_span(chunk_id, cleaned)],
                        )
                    )

            total_match = re.match(r"(?i)^\s*(total|amount due|subtotal|balance due)\s*[:\-]\s*(.+)$", cleaned)
            if total_match:
                label = total_match.group(1).strip().title()
                value = total_match.group(2).strip()
                if value:
                    totals.append(FieldValue(label=label, value=value, evidence_spans=[_span(chunk_id, cleaned)]))

            party_match = re.match(r"(?i)^\s*(bill to|billed to|invoice to|from|vendor|customer)\s*[:\-]\s*(.+)$", cleaned)
            if party_match:
                label = party_match.group(1).strip().title()
                value = party_match.group(2).strip()
                if value:
                    parties.append(FieldValue(label=label, value=value, evidence_spans=[_span(chunk_id, cleaned)]))

            term_match = re.match(r"(?i)^\s*(payment terms|due date|terms)\s*[:\-]\s*(.+)$", cleaned)
            if term_match:
                label = term_match.group(1).strip().title()
                value = term_match.group(2).strip()
                if value:
                    terms.append(FieldValue(label=label, value=value, evidence_spans=[_span(chunk_id, cleaned)]))

    return InvoiceSchema(
        items=_items_field(items),
        totals=_field_values_field(totals),
        parties=_field_values_field(parties),
        terms=_field_values_field(terms),
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

        # If section_kind is generic OR belongs to a different domain (e.g., invoice
        # section kinds on a resume document), re-infer from actual content.
        _WRONG_DOMAIN_KINDS = {
            "financial_summary", "line_items", "invoice_metadata",
            "parties_addresses", "terms_conditions",
        }
        inferred = False
        if not section_kind or section_kind in ("section_text", "misc") or section_kind in _WRONG_DOMAIN_KINDS:
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

        # Set source type based on document name/type
        for section_chunks in sections.values():
            if section_chunks:
                _, _, section_title = section_chunks[0]
                # Try to infer from section title
                cand.source_type = _infer_source_type(section_title or "Resume")
                break

        if not cand.source_type:
            if doc_name:
                cand.source_type = _infer_source_type(doc_name) or "Document"
            else:
                cand.source_type = "Document"

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
            if not cand.name:
                for sec_kind in ("identity_contact", "contact", "summary_objective"):
                    if sec_kind in sections and not cand.name:
                        for _, sec_text, _ in sections[sec_kind]:
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

    # Last-resort name extraction: use cleaned filename when all other methods fail
    for doc_id, cand in by_doc.items():
        if not cand.name and doc_id in doc_names:
            fname = doc_names[doc_id]
            # Strip extension, underscores, and common noise words
            cleaned = re.sub(r"\.(pdf|docx?|rtf)$", "", fname, flags=re.IGNORECASE)
            cleaned = re.sub(r"[_\-]+", " ", cleaned)
            cleaned = re.sub(r"\b(?:resume|cv|profile|linkedin|update\d*|new|old|final|draft|copy|v\d+|dev|ip)\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned and len(cleaned) >= 2:
                cand.name = _smart_title_case(cleaned)

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
    }

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
        # Skip items containing metadata artifact patterns
        if _artifact_patterns.search(item):
            continue
        # Skip single-word section headers used as skill items
        if item.strip().lower().rstrip('.') in _section_header_items:
            continue
        # Skip items starting with conjunctions (split sentence fragments)
        if re.match(r"^(?:and|or|but|nor)\s+", item, re.IGNORECASE) and len(item) < 50:
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
    # Clean name
    if cand.name:
        cleaned_name = _sanitize_field_value(cand.name, max_length=80)
        cand.name = cleaned_name

    # Clean experience summary
    if cand.experience_summary:
        cand.experience_summary = _sanitize_field_value(cand.experience_summary, max_length=500)

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


def _extract_legal(chunks: List[Any]) -> LegalSchema:
    clauses: List[Clause] = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        chunk_id = getattr(chunk, "id", "")
        for line in _split_lines(text):
            cleaned = line.strip()
            if not cleaned:
                continue
            match = re.match(r"(?i)^\s*(clause|section)\s*[:\-]\s*(.+)$", cleaned)
            if match:
                title = match.group(1).strip().title()
                body = match.group(2).strip()
                clauses.append(Clause(title=title, text=body, evidence_spans=[_span(chunk_id, cleaned)]))
    return LegalSchema(clauses=_clauses_field(clauses))


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


def _schema_is_empty(schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema) -> bool:
    if isinstance(schema, InvoiceSchema):
        return not (
            (schema.items.items if schema.items else None)
            or (schema.totals.items if schema.totals else None)
            or (schema.parties.items if schema.parties else None)
            or (schema.terms.items if schema.terms else None)
        )
    if isinstance(schema, HRSchema):
        cands = (schema.candidates.items if schema.candidates else None) or []
        return not any(c.name or c.technical_skills or getattr(c, "experience_summary", None) for c in cands)
    if isinstance(schema, LegalSchema):
        return not (schema.clauses.items if schema.clauses else None)
    if isinstance(schema, GenericSchema):
        facts = (schema.facts.items if schema.facts else None) or []
        return not any(f.value and len(str(f.value)) > 5 for f in facts)
    if isinstance(schema, MultiEntitySchema):
        return not (schema.entities or [])
    return True


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
        return GenericSchema.model_validate(cleaned)
    except Exception:
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
        "num_ctx": 4096,
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
        except Exception:
            return {}
    if "{" in text and "}" in text:
        snippet = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception:
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
    "summary_objective": ["SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "PROFILE", "ABOUT ME", "CAREER OBJECTIVE"],
    "experience": ["WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT", "CAREER HISTORY", "EMPLOYMENT HISTORY"],
    "skills_technical": ["TECHNICAL SKILLS", "SKILLS", "TECHNOLOGIES", "TECH STACK", "KEY SKILLS", "CORE COMPETENCIES"],
    "skills_functional": ["FUNCTIONAL SKILLS", "SOFT SKILLS", "BUSINESS SKILLS"],
    "certifications": ["CERTIFICATIONS", "CERTIFICATES", "CREDENTIALS", "LICENSES"],
    "education": ["EDUCATION", "ACADEMIC", "QUALIFICATIONS", "DEGREES", "ACADEMIC BACKGROUND"],
    "achievements": ["ACHIEVEMENTS", "AWARDS", "HONORS", "ACCOMPLISHMENTS", "RECOGNITION"],
    "identity_contact": ["CONTACT", "PERSONAL DETAILS", "CONTACT INFORMATION"],
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

    # Extract emails from entire text
    emails = list(set(_EMAIL_RE.findall(text)))

    # Extract phones from entire text
    for match in _PHONE_RE.findall(text):
        cleaned_phone = _clean_contact_value(match, preserve_parens=True)
        if cleaned_phone and cleaned_phone not in phones and _is_valid_phone(cleaned_phone):
            phones.append(cleaned_phone)

    # Extract LinkedIn URLs
    linkedins = list(set(_LINKEDIN_RE.findall(text)))

    # Normalize text for name extraction
    normalized_text = text.replace('\\n', '\n').replace(' n ', '\n').replace('--- Page 1 ---', '\n')

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
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
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


def _split_lines(text: str) -> List[str]:
    return [line for line in text.splitlines() if line.strip()]


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


def _normalize_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _is_bullet_item(line: str) -> bool:
    if not line.startswith("-") and not line.startswith("•"):
        return False
    has_price = bool(re.search(r"[$€£]\s*\d", line))
    has_qty = bool(re.search(r"\bqty\b|\bquantity\b|\b\d+\s*x\b", line, re.IGNORECASE))
    return has_price or has_qty


def _extract_years_experience(text: str) -> Optional[str]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} years"
    return None


def _infer_section_kind_from_content(text: str, section_title: str) -> str:
    """Infer section kind from actual chunk content when metadata is generic.

    Delegates to the shared content classifier so ingestion and extraction
    use identical classification logic.
    """
    from src.embedding.pipeline.content_classifier import classify_section_kind

    return classify_section_kind(text, section_title)


def _extract_years_experience(text: str) -> Optional[str]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} years"
    return None


def _normalize_intent_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"summary", "contact", "rank", "compare", "candidate_list"}:
        return normalized
    if normalized in {"list", "listing"}:
        return "candidate_list"
    if normalized in {"qa", "answer", "facts"}:
        return "summary"
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
    }
    value_lower = value_cleaned.lower()
    if any(phrase in value_lower for phrase in phrases_not_names):
        return False

    # Filter out institution names (contain College, University, etc.)
    institution_keywords = ['college', 'university', 'institute', 'polytechnic', 'academy', 'school']
    if any(kw in value_lower for kw in institution_keywords):
        return False

    # If value contains more than 4 non-hyphen words, it's likely not a name
    if len(parts) > 4:
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
    }
    if any(part.lower() in non_name_words for part in parts):
        return False

    if len(parts) == 1:
        token = parts[0]
        return token[:1].isupper() and len(token) >= 3
    return all(p[:1].isupper() for p in parts)


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
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\b(resume|cv|profile|linkedin)\b", "", cleaned, flags=re.IGNORECASE)
    # Strip common filename noise BEFORE name check
    cleaned = re.sub(r"\b(?:update|version|rev)\d*\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:dev|ip|prod|test|doc|file|scan|scanned)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if _looks_like_name(cleaned):
        return _smart_title_case(cleaned)

    # More aggressive cleaning: strip non-name words from filename
    _filename_noise = {
        "new", "old", "updated", "update", "final", "draft",
        "copy", "scm", "sap", "erp", "latest", "revised", "v1", "v2", "v3",
        "ip", "dev", "prod", "test", "doc", "file", "scan", "scanned",
    }
    parts = cleaned.split()
    name_parts = [p.title() for p in parts if p.lower() not in _filename_noise and len(p) > 1]
    if name_parts:
        candidate_name = " ".join(name_parts)
        if _looks_like_name(candidate_name):
            return candidate_name

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
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
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

        # Pattern 2: Line with 2-4 capitalized words
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w):
                # Filter out common non-name patterns
                if not any(w.lower() in ['resume', 'cv', 'page', 'date', 'name'] for w in words):
                    # Make sure it's not an email or has numbers
                    if '@' not in line and not any(c.isdigit() for c in line):
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
    for skill in skills:
        # Clean up the skill
        skill = skill.strip()

        # Skip items containing newlines (likely garbage)
        if '\\n' in skill or '\n' in skill:
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


def _extract_certifications_from_text(text: str) -> List[str]:
    """Extract certifications from text with broad pattern matching."""
    certs = []

    # Strip common headers from text first
    text = re.sub(r'^Certifications?\s*:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\nCertifications?\s*:\s*', '\n', text, flags=re.IGNORECASE)

    # Common certification patterns
    cert_patterns = [
        r"(?:AWS|Amazon)\s+(?:Certified|Solutions?\s+Architect|Developer|SysOps)[^\n,]{0,50}",
        r"(?:Azure|Microsoft)\s+(?:Certified|Administrator|Developer|Expert)[^\n,]{0,50}",
        r"(?:Google|GCP)\s+(?:Certified|Cloud|Professional)[^\n,]{0,50}",
        r"PMP|Project Management Professional",
        r"CAPM[^\n,]{0,60}",
        r"Scrum\s+(?:Master|Product Owner|Developer)",
        r"(?:CISSP|CISM|CISA|CEH|CompTIA)[^\n,]{0,30}",
        r"(?:Oracle|Salesforce)\s+Certified[^\n,]{0,50}",
        r"SAP\s+(?:S/4\s*HANA|ERP|BW|FICO|SD|MM|PP|HR|SRM|SCM|Analytics\s+Cloud|SAC|Certified)[^\n,]{0,60}",
        r"(?:CCNA|CCNP|CCIE)[^\n,]{0,30}",
        r"(?:ITIL|Six\s+Sigma|TOGAF|PRINCE2|COBIT)[^\n,]{0,30}",
        r"(?:CPA|CFA|FRM|ACCA|CA\b)[^\n,]{0,30}",
        r"Business\s+Analysis[^\n,]{0,40}",
        r"Certified\s+(?:Associate|Professional|Expert|Engineer|Specialist|Manager|Analyst|Auditor|ScrumMaster)[^\n,]{0,50}",
    ]

    for pattern in cert_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            cert = match.group(0).strip()
            if cert:
                certs.append(cert)

    # Line-by-line: look for lines mentioning certifications
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        lower = line.lower()
        if any(kw in lower for kw in ("certified", "certification", "certificate", "credential", "license")):
            if len(line) < 200 and not any(skip in lower for skip in ("skills", "education", "experience")):
                certs.append(line)

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
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    chunks: List[Any],
    scope_document_id: Optional[str],
    intent: Optional[str] = None,
) -> bool:
    # GenericSchema handles multi-doc natively via document_name on facts
    if isinstance(schema, GenericSchema):
        return False
    if domain == "hr" and isinstance(schema, HRSchema):
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


def _query_is_hr_like(query: str) -> bool:
    """Check if a query is related to HR/resume content."""
    lowered = (query or "").lower()
    # Strong HR signals — a single match is sufficient
    strong = ("resume", "cv", "candidate")
    if any(kw in lowered for kw in strong):
        return True
    # Weak HR signals — need at least 2 to count
    weak = ("skills", "experience", "education", "certification")
    return sum(1 for kw in weak if kw in lowered) >= 2


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
) -> ExtractionResult:
    """Extract structured data from chunks.

    Strategy:
    1. LLM-first generic extraction — works for ANY document type.
       The LLM reads the chunks and produces a direct answer.
    2. Deterministic fallback — regex-based extraction when LLM is unavailable.
    """
    from .llm_extract import llm_extract_and_respond, _count_unique_documents

    # STRATEGY 1: LLM-first generic extraction (preferred for ALL document types)
    if llm_client and budget.allow():
        try:
            llm_result = llm_extract_and_respond(
                query=query,
                chunks=chunks,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                intent=intent_hint,
                num_documents=_count_unique_documents(chunks),
            )
            if llm_result is not None:
                # Query-based domain takes precedence over chunk majority
                inferred = _query_domain_override(query) or _majority_chunk_domain(chunks) or domain or "generic"
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

    # STRATEGY 2: Deterministic fallback (chunk-based, domain-aware)
    return schema_extract(
        query=query,
        chunks=chunks,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
        scope_document_id=scope_document_id,
        domain_hint=domain,
        intent_hint=intent_hint,
    )


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

