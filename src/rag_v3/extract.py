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

EXTRACT_TIMEOUT_MS = 2500

@dataclass
class ExtractionResult:
    domain: str
    intent: str
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema


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

_HR_HINTS = {"resume", "curriculum vitae", "cv", "candidate", "experience", "education"}

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


def _infer_domain_intent(
    query: str,
    chunks: List[Any],
    domain_hint: Optional[str] = None,
    intent_hint: Optional[str] = None,
) -> Tuple[str, str]:
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        return "generic", "facts"
    combined = " ".join([query] + [getattr(c, "text", "") or "" for c in chunks]).lower()
    query_lower = (query or "").lower()
    domain = (domain_hint or "").strip().lower()
    if not domain:
        domain = "generic"
        if any(hint in query_lower for hint in _HR_HINTS) or _looks_like_hr_total(query_lower):
            domain = "hr"
        elif any(hint in query_lower for hint in _INVOICE_HINTS) and not _looks_like_hr_total(query_lower):
            domain = "invoice"
        elif any(hint in query_lower for hint in _LEGAL_HINTS):
            domain = "legal"
        elif any(hint in combined for hint in _HR_HINTS) or _looks_like_hr_total(combined):
            domain = "hr"
        elif any(hint in combined for hint in _INVOICE_HINTS) and not _looks_like_hr_total(combined):
            domain = "invoice"
        elif any(hint in combined for hint in _LEGAL_HINTS):
            domain = "legal"

    intent = _normalize_intent_hint(intent_hint) or "summary"
    lowered_query = query.lower()
    if domain == "invoice":
        if intent == "summary" and any(word in lowered_query for word in _PRODUCT_INTENTS):
            intent = "products_list"
        elif intent == "summary" and any(word in lowered_query for word in _TOTAL_INTENTS):
            intent = "totals"
        elif intent == "summary":
            intent = "summary"
    elif domain == "hr":
        if intent == "summary" and any(word in lowered_query for word in _CONTACT_INTENTS):
            intent = "contact"
        elif intent == "summary" and (
            "rank" in lowered_query or "ranking" in lowered_query or any(word in lowered_query for word in _FIT_INTENTS)
        ):
            intent = "rank"
        elif intent == "summary" and "compare" in lowered_query:
            intent = "compare"
        elif intent == "summary" and ("list" in lowered_query or "candidates" in lowered_query):
            intent = "candidate_list"
        elif intent == "summary":
            intent = "summary"
    elif domain == "legal":
        if intent == "summary":
            intent = "clauses"
    else:
        if intent == "summary":
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
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        return _extract_generic(query, chunks)
    if domain == "invoice":
        return _extract_invoice(chunks)
    if domain == "hr":
        return _extract_hr(chunks)
    if domain == "legal":
        return _extract_legal(chunks)
    return _extract_generic(query, chunks)


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
    candidates: List[Candidate] = []
    by_doc: Dict[str, Candidate] = {}
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        chunk_id = getattr(chunk, "id", "")
        meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or chunk_id)
        doc_name = getattr(getattr(chunk, "source", None), "document_name", "") or str(meta.get("source_name") or "")

        cand = by_doc.get(doc_id)
        if cand is None:
            cand = Candidate(
                name=None,
                evidence_spans=[],
                source_type=_infer_source_type(doc_name),
                missing_reason={},
            )
            by_doc[doc_id] = cand
            candidates.append(cand)
        span_seen = {span.snippet for span in (cand.evidence_spans or [])}
        if not cand.name:
            cand.name = _extract_name_guess(text, doc_name)

        summary_lines: List[str] = []
        capture_summary = False
        sections = _parse_sections(text)
        for line in _split_lines(text):
            cleaned = line.strip()
            if not cleaned:
                continue
            lower = cleaned.lower()
            match = re.match(r"(?i)^\s*(name|candidate)\s*[:\-]\s*(.+)$", cleaned)
            if match:
                name = match.group(2).strip()
                if not cand.name:
                    cand.name = name or cand.name
                if name:
                    _append_span(cand, chunk_id, cleaned, span_seen)
                continue

            years = _extract_years_experience(cleaned)
            if years and not cand.total_years_experience:
                cand.total_years_experience = years
                _append_span(cand, chunk_id, cleaned, span_seen)

            if any(tag in lower for tag in ("summary", "profile", "objective", "professional summary")):
                capture_summary = True
                inline = cleaned.split(":", 1)[1].strip() if ":" in cleaned else ""
                if inline:
                    summary_lines.append(inline)
                continue
            if capture_summary:
                summary_lines.append(cleaned)
                if len(summary_lines) >= 3:
                    capture_summary = False

            if "certification" in lower or "certified" in lower:
                cand.certifications = _merge_list(cand.certifications, _split_list(cleaned))
                _append_span(cand, chunk_id, cleaned, span_seen)
                continue
            if "award" in lower or "achievement" in lower or "honor" in lower:
                cand.achievements = _merge_list(cand.achievements, [cleaned])
                _append_span(cand, chunk_id, cleaned, span_seen)
                continue

            contact = _extract_contact_fields(cleaned)
            if contact:
                if contact.get("emails"):
                    cand.emails = _merge_list(cand.emails, contact["emails"])
                if contact.get("phones"):
                    cand.phones = _merge_list(cand.phones, contact["phones"])
                if contact.get("linkedins"):
                    cand.linkedins = _merge_list(cand.linkedins, contact["linkedins"])
                _append_span(cand, chunk_id, cleaned, span_seen)
                continue

        tech_block = sections.get("technical_skills") or []
        func_block = sections.get("functional_skills") or []
        skills_block = sections.get("skills") or []
        tools_block = sections.get("tools") or []
        tech_block_all = tech_block + skills_block + tools_block
        func_block_all = func_block
        tech_items = _flatten_skill_block(tech_block_all)
        func_items = _flatten_skill_block(func_block_all)
        if tech_items:
            cand.technical_skills = _merge_list(cand.technical_skills, tech_items)
            for line in tech_block_all:
                _append_span(cand, chunk_id, line, span_seen)
        if func_items:
            cand.functional_skills = _merge_list(cand.functional_skills, func_items)
            for line in func_block_all:
                _append_span(cand, chunk_id, line, span_seen)

        education_block = sections.get("education") or []
        education_items = _parse_education_block(education_block)
        if education_items:
            cand.education = _merge_list(cand.education, education_items)
            for line in education_block:
                _append_span(cand, chunk_id, line, span_seen)

        if summary_lines and not cand.experience_summary:
            cand.experience_summary = " ".join(summary_lines).strip()

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
        return not (schema.candidates.items if schema.candidates else None)
    if isinstance(schema, LegalSchema):
        return not (schema.clauses.items if schema.clauses else None)
    if isinstance(schema, GenericSchema):
        return not (schema.facts.items if schema.facts else None)
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
    max_chars: int = 3200,
) -> str:
    context = []
    for chunk in chunks[:6]:
        chunk_id = getattr(chunk, "id", "")
        text = getattr(chunk, "text", "") or ""
        snippet = " ".join(text.split())[:400]
        context.append(f"[{chunk_id}] {snippet}")

    evidence = "\n".join(context)
    if len(evidence) > max_chars:
        evidence = evidence[:max_chars]

    evidence_rule = (
        "Every field must include evidence_spans with chunk_id and a short snippet. "
        if require_spans
        else "You may omit evidence_spans if unavailable. "
    )
    return (
        "Extract structured data from the evidence. "
        "Only use facts explicitly present. "
        + evidence_rule +
        "Return strict JSON only.\n\n"
        f"DOMAIN: {domain}\n"
        f"INTENT: {intent}\n"
        f"QUESTION: {query}\n\n"
        "EVIDENCE:\n"
        + evidence
        + "\n\nReturn JSON with fields: {"  # noqa: ISC003
        + "\"schema\": { ... }}"
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
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[A-Za-z0-9._/?=&-]+", re.IGNORECASE)


def _extract_contact_fields(line: str) -> Dict[str, List[str]]:
    if not line:
        return {}
    emails = [_clean_contact_value(val) for val in _EMAIL_RE.findall(line)]
    phones = [_clean_contact_value(val) for val in _PHONE_RE.findall(line)]
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


def _clean_contact_value(value: str) -> str:
    cleaned = value.strip().strip(".,;:()[]{}<>")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _split_lines(text: str) -> List[str]:
    return [line for line in text.splitlines() if line.strip()]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _keywords(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "from"}
    return [tok for tok in tokens if tok not in stop and len(tok) > 2]


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
    parts = re.split(r"[•|\u2022,;/]+", cleaned)
    items = []
    for part in parts:
        item = part.strip()
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
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith(("-", "•", "*")) or re.match(r"^\d+\.", cleaned):
            cleaned = cleaned.lstrip("-•* ").strip()
        items.extend(_split_list(cleaned))
    return items


def _parse_education_block(lines: List[str]) -> List[str]:
    entries: List[str] = []
    for line in lines:
        cleaned = line.strip()
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
    parts = [p for p in value.split() if p]
    if len(parts) > 4:
        return False
    if any(len(p) == 1 for p in parts):
        return False
    if len(parts) == 1:
        token = parts[0]
        return token[:1].isupper() and len(token) >= 3
    return all(p[:1].isupper() for p in parts)


def _name_from_filename(doc_name: str) -> Optional[str]:
    if not doc_name:
        return None
    cleaned = re.sub(r"\.(pdf|docx?|rtf)$", "", doc_name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\b(resume|cv|profile|linkedin)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if _looks_like_name(cleaned):
        return cleaned
    return None


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
) -> ExtractionResult:
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
