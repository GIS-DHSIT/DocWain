from __future__ import annotations

import re
from typing import Any, List, Optional

from .types import GenericSchema, HRSchema, InvoiceSchema, LegalSchema, MISSING_REASON, MultiEntitySchema


NO_ITEMS_MESSAGE = "Invoice pages retrieved don't show itemized products/services—only totals/parties."

# Patterns that indicate metadata garbage leaked into rendered fields
_RENDER_GARBAGE_RE = re.compile(
    r"(?:section_id|chunk_type|page_start|page_end|layout_confidence|"
    r"ocr_confidence|doc_quality|canonical_json|chunk_candidates|"
    r"layout_spans|key_value_pairs|ExtractedDocument|ChunkCandidate|"
    r"Section\(|Table\(|'text':|'csv':|'page':"
    r"|^title\s*[:\s]|^key\s*:|^value\s*:|Chunk\s*Candidate)",
    re.IGNORECASE,
)


def _sanitize_render_value(value: str, max_length: int = 500) -> str:
    """Final safety net: clean any remaining metadata from a rendered value."""
    if not value:
        return ""
    if value == MISSING_REASON:
        return ""
    # If the value contains metadata patterns, it's garbage
    if _RENDER_GARBAGE_RE.search(value):
        return ""
    # Clean whitespace
    cleaned = " ".join(value.replace("\\n", "\n").split())
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned.strip()


def _sanitize_render_list(items: List[str], max_items: int = 20) -> List[str]:
    """Clean a list of rendered items, removing any garbage entries."""
    cleaned = []
    for item in items:
        if not item:
            continue
        if _RENDER_GARBAGE_RE.search(item):
            continue
        item = " ".join(item.split()).strip()
        if item and len(item) > 2:
            cleaned.append(item)
    return cleaned[:max_items]


def render_enterprise(schema: Any, intent: str, *, domain: str | None = None, strict: bool = False, query: str = "") -> str:
    if isinstance(schema, MultiEntitySchema):
        return _render_multi(schema, intent, strict=strict)
    if isinstance(schema, InvoiceSchema):
        return _render_invoice(schema, intent, strict=strict)
    if isinstance(schema, HRSchema):
        return _render_hr(schema, intent, strict=strict, query=query)
    if isinstance(schema, LegalSchema):
        return _render_legal(schema, intent, strict=strict)
    if isinstance(schema, GenericSchema):
        return _render_generic(schema, intent, strict=strict)
    _ = domain
    return ""


def _render_generic(schema: GenericSchema, intent: str, strict: bool = False) -> str:
    """Adaptive renderer — formats based on content structure, not domain label."""
    facts_items = (schema.facts.items if schema.facts else None) or []
    if not facts_items:
        if strict:
            return ""
        if schema.facts and schema.facts.missing_reason:
            return schema.facts.missing_reason
        return ""
    facts = [fact for fact in facts_items if fact.value]
    if not facts:
        if strict:
            return ""
        return ""

    # Contact intent: filter to contact facts only
    if intent == "contact":
        contact_facts = [f for f in facts if f.label and f.label.lower() in ("email", "phone", "linkedin")]
        if contact_facts:
            facts = contact_facts

    # Check whether facts carry document_name (multi-doc grouping)
    has_doc_names = any(getattr(f, "document_name", None) for f in facts)

    if has_doc_names:
        return _render_grouped_by_document(facts)

    return _render_flat_facts(facts, intent=intent)


def _render_grouped_by_document(facts: List[Any]) -> str:
    """Render facts grouped by document_name, then by section."""
    from collections import OrderedDict

    # Cross-document dedup: remove duplicate (label, value) pairs
    seen: set[tuple[str, str]] = set()
    deduped: List[Any] = []
    for fact in facts:
        key = ((fact.label or "").lower().strip(), (fact.value or "").lower().strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)

    doc_groups: OrderedDict[str, List[Any]] = OrderedDict()
    for fact in deduped:
        doc_name = getattr(fact, "document_name", None) or "Document"
        doc_groups.setdefault(doc_name, []).append(fact)

    lines: list[str] = []
    multi_doc = len(doc_groups) > 1

    if multi_doc:
        lines.append(f"Information from {len(doc_groups)} documents:\n")

    for doc_name, doc_facts in doc_groups.items():
        if multi_doc:
            lines.append(f"**{doc_name}:**")

        # Sub-group by section within document
        section_groups: OrderedDict[str, List[Any]] = OrderedDict()
        for fact in doc_facts:
            section = getattr(fact, "section", None) or ""
            section_groups.setdefault(section, []).append(fact)

        for section, section_facts in section_groups.items():
            if section and len(section_groups) > 1:
                lines.append(f"  *{section}:*")
            for fact in section_facts:
                clean_value = _sanitize_render_value(fact.value, max_length=500)
                if not clean_value:
                    continue
                if fact.label:
                    clean_label = _sanitize_render_value(fact.label, max_length=80)
                    if clean_label:
                        lines.append(f"- **{clean_label}:** {clean_value}")
                    else:
                        lines.append(f"- {clean_value}")
                else:
                    lines.append(f"- {clean_value}")

        if multi_doc:
            lines.append("")  # Blank line between documents

    return "\n".join(lines).strip()


_INTENT_HEADINGS = {
    "contact": "Contact Information:",
    "rank": "Candidate Rankings:",
    "compare": "Comparison:",
    "summary": "Summary:",
    "list": "Listed Items:",
    "facts": None,
}


def _render_flat_facts(facts: List[Any], intent: str = "facts") -> str:
    """Render facts without document grouping (single-doc or no doc info)."""
    labeled = [(f.label, f.value) for f in facts if f.label]
    unlabeled = [f.value for f in facts if not f.label]

    lines: list[str] = []

    if labeled:
        heading = _INTENT_HEADINGS.get(intent)
        if heading:
            lines.append(f"**{heading}**")
        for label, value in labeled:
            clean_label = _sanitize_render_value(label, max_length=80)
            clean_value = _sanitize_render_value(value, max_length=500)
            if clean_label and clean_value:
                lines.append(f"- **{clean_label}:** {clean_value}")

    if unlabeled:
        cleaned = [" ".join(v.split()) for v in unlabeled if v]
        if len(cleaned) == 1 and not labeled:
            return cleaned[0].strip()
        if not labeled:
            heading = _INTENT_HEADINGS.get(intent, "Key details:")
            if heading:
                lines.append(heading)
        for fact in cleaned:
            lines.append(f"- {fact}")

    return "\n".join(lines).strip()


def _render_multi(schema: MultiEntitySchema, intent: str, strict: bool = False) -> str:
    _ = intent
    entities = schema.entities or []
    if not entities:
        if strict:
            return ""
        return schema.missing_reason or ""

    has_docs = any(item.document_name or item.document_id for item in entities)
    has_named = any(item.label and item.label.lower() not in {"document", "candidate"} for item in entities)

    if has_named and not has_docs:
        lead = "There are multiple candidates in the documents."
    elif has_named and has_docs:
        lead = "There are multiple candidates across documents."
    else:
        lead = "There are multiple documents that match this request."

    parts = []
    for item in entities:
        label = item.label or item.document_name or item.document_id or "Document"
        doc_ref = item.document_name or item.document_id
        if doc_ref and doc_ref not in label:
            parts.append(f"{label} ({doc_ref})")
        else:
            parts.append(label)

    if len(parts) == 1:
        return f"{lead} {parts[0]}."
    lines = [lead, "Matched entities:"]
    lines.extend(f"- {item}" for item in parts)
    return "\n".join(lines)


def _render_hr(schema: HRSchema, intent: str, strict: bool = False, query: str = "") -> str:
    candidates = (schema.candidates.items if schema.candidates else None) or []
    if not candidates:
        if strict:
            return ""
        if schema.candidates and schema.candidates.missing_reason:
            return schema.candidates.missing_reason
        return ""
    if intent == "contact":
        sections = []
        for cand in candidates:
            name = cand.name or "Candidate"
            lines = [f"**Candidate:** {name}"]
            emails = _render_contact_value(cand.emails)
            if emails:
                lines.append(f"- Email: {emails}")
            phones = _render_contact_value(cand.phones)
            if phones:
                lines.append(f"- Phone: {phones}")
            linkedins = _render_contact_value(cand.linkedins)
            if linkedins:
                lines.append(f"- LinkedIn: {linkedins}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections).strip()
    if intent in {"rank", "compare"} and len(candidates) > 1:
        # Try structured comparison first
        try:
            from .comparator import compare_candidates_from_schema, render_comparison
            comp_result = compare_candidates_from_schema(candidates, query)
            rendered = render_comparison(comp_result, intent)
            if rendered:
                return rendered
        except Exception:
            pass  # Fall back to existing ranking

        ranked = _rank_candidates(candidates, query=query)
        lines = ["Based on the skills mentioned, here is a ranking of the candidates:"]
        for idx, cand in enumerate(ranked, start=1):
            lines.append(_format_rank_line(idx, cand))
        return "\n".join(lines)
    if len(candidates) > 1:
        sections = []
        for cand in candidates:
            sections.append(_format_candidate_detail(cand))
        return "\n\n".join(sections)

    cand = candidates[0]
    parts = []
    name = _sanitize_render_value(cand.name or "", max_length=80) or "Candidate"
    parts.append(f"**Candidate:** {name}")

    # Only render fields that have actual content — omit optional fields that are absent
    years_val = _sanitize_render_value(cand.total_years_experience or '', max_length=30)
    if years_val:
        parts.append(f"- Total experience: {years_val}")
    summary = _sanitize_render_value(cand.experience_summary or '', max_length=500)
    if summary:
        parts.append(f"- Summary: {summary}")

    tech = _sanitize_render_list(cand.technical_skills or [])
    if tech:
        parts.append(f"- Technical skills: {', '.join(tech)}")
    func = _sanitize_render_list(cand.functional_skills or [])
    if func:
        parts.append(f"- Functional skills: {', '.join(func)}")
    certs = _sanitize_render_list(cand.certifications or [])
    if certs:
        parts.append(f"- Certifications: {', '.join(certs)}")
    edu = _sanitize_render_list(cand.education or [], max_items=5)
    if edu:
        parts.append(f"- Education: {', '.join(edu)}")
    awards = _sanitize_render_list(cand.achievements or [])
    if awards:
        parts.append(f"- Achievements/Awards: {', '.join(awards)}")
    source = _sanitize_render_value(cand.source_type or '', max_length=50)
    if source:
        parts.append(f"- Source type: {source}")

    return "\n".join(parts).strip()


_RANK_STOP_WORDS = frozenset({
    "the", "and", "or", "for", "a", "an", "is", "are", "was", "were",
    "of", "in", "to", "with", "on", "at", "by", "from", "as", "it",
    "what", "who", "how", "which", "tell", "me", "about", "do", "does",
    "can", "has", "have", "their", "this", "that", "all", "be",
})


def _rank_candidates(candidates: List[Any], query: str = ""):
    query_keywords = set(query.lower().split()) - _RANK_STOP_WORDS if query else set()
    scored = []
    for cand in candidates:
        score = 0.0
        score += 1.5 * len(cand.technical_skills or [])
        score += 1.0 * len(cand.functional_skills or [])
        score += 0.7 * len(cand.certifications or [])
        score += 0.4 * len(cand.achievements or [])
        if cand.total_years_experience:
            years = _parse_years(cand.total_years_experience)
            if years is not None:
                score += min(10.0, years) * 0.2
        # Query relevance boost
        if query_keywords:
            all_text = " ".join(cand.technical_skills or []) + " " + " ".join(cand.functional_skills or []) + " " + (cand.experience_summary or "")
            all_text_lower = all_text.lower()
            matches = sum(1 for kw in query_keywords if kw in all_text_lower)
            score += matches * 2.0
        scored.append((score, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cand for _, cand in scored]


def _parse_years(value: str):
    try:
        number = float(value.split()[0])
        return number
    except Exception:
        return None


def _format_rank_line(idx: int, cand: Any) -> str:
    name = _sanitize_render_value(cand.name or "", max_length=80) or "Candidate"
    tech = ", ".join(_sanitize_render_list((cand.technical_skills or [])[:4]))
    func = ", ".join(_sanitize_render_list((cand.functional_skills or [])[:4]))
    certs = ", ".join(_sanitize_render_list((cand.certifications or [])[:3]))
    summary = _sanitize_render_value(cand.experience_summary or '', max_length=300)
    label = "Top pick" if idx == 1 else "Next"
    detail_parts = []
    if tech:
        detail_parts.append(f"key technical skills: {tech}")
    if func:
        detail_parts.append(f"functional skills: {func}")
    if certs:
        detail_parts.append(f"certifications: {certs}")
    if summary:
        detail_parts.append(f"summary: {summary}")
    detail = "; ".join(detail_parts) if detail_parts else "See document for details"
    return f"- {label}: {name} — {detail}"


def _format_candidate_detail(cand: Any) -> str:
    name = _sanitize_render_value(cand.name or "", max_length=80) or "Candidate"
    lines = [f"**Candidate:** {name}"]
    years = _sanitize_render_value(cand.total_years_experience or '', max_length=30)
    if years:
        lines.append(f"- Total experience: {years}")
    summary = _sanitize_render_value(cand.experience_summary or '', max_length=500)
    if summary:
        lines.append(f"- Summary: {summary}")
    tech = ", ".join(_sanitize_render_list((cand.technical_skills or [])[:6]))
    if tech:
        lines.append(f"- Technical skills: {tech}")
    func = ", ".join(_sanitize_render_list((cand.functional_skills or [])[:6]))
    if func:
        lines.append(f"- Functional skills: {func}")
    certs = ", ".join(_sanitize_render_list((cand.certifications or [])[:4]))
    if certs:
        lines.append(f"- Certifications: {certs}")
    edu = ", ".join(_sanitize_render_list((cand.education or [])[:3]))
    if edu:
        lines.append(f"- Education: {edu}")
    awards = ", ".join(_sanitize_render_list((cand.achievements or [])[:3]))
    if awards:
        lines.append(f"- Achievements/Awards: {awards}")
    source = _sanitize_render_value(cand.source_type or '', max_length=50)
    if source:
        lines.append(f"- Source type: {source}")
    return "\n".join(lines)


def _render_contact_value(value: Any) -> str:
    items = [item for item in (value or []) if item]
    return ", ".join(items) if items else ""


def _render_invoice(schema: InvoiceSchema, intent: str, strict: bool = False) -> str:
    if intent == "products_list":
        items = (schema.items.items if schema.items else None) or []
        if items:
            rendered = [_format_item(item.description) for item in items if item.description]
            if rendered:
                lines = ["Items listed on the invoice:"]
                lines.extend(f"- {item}" for item in rendered)
                return "\n".join(lines)
        if schema.items and schema.items.missing_reason:
            return schema.items.missing_reason
        return NO_ITEMS_MESSAGE

    if intent == "totals":
        totals_items = (schema.totals.items if schema.totals else None) or []
        if totals_items:
            totals = [_format_field(item.label, item.value) for item in totals_items if item.value]
            if totals:
                lines = ["Totals shown:"]
                lines.extend(f"- {item}" for item in totals)
                return "\n".join(lines)
        if schema.totals and schema.totals.missing_reason:
            return schema.totals.missing_reason
        return ""

    if strict:
        return _render_totals_parties(schema, prefer_totals=True) or ""

    summary = _render_totals_parties(schema, prefer_totals=True)
    if summary:
        return summary
    items = (schema.items.items if schema.items else None) or []
    if items:
        rendered = [_format_item(item.description) for item in items if item.description]
        if rendered:
            lines = ["Items listed on the invoice:"]
            lines.extend(f"- {item}" for item in rendered)
            return "\n".join(lines)
    if schema.items and schema.items.missing_reason:
        return schema.items.missing_reason
    return ""


def _render_totals_parties(schema: InvoiceSchema, prefer_totals: bool = True) -> str:
    parts: List[str] = []
    totals_items = (schema.totals.items if schema.totals else None) or []
    parties_items = (schema.parties.items if schema.parties else None) or []
    terms_items = (schema.terms.items if schema.terms else None) or []
    if prefer_totals and totals_items:
        totals = [_format_field(item.label, item.value) for item in totals_items if item.value]
        if totals:
            parts.append("Totals shown:")
            parts.extend(f"- {item}" for item in totals)
    if parties_items:
        parties = [_format_field(item.label, item.value) for item in parties_items if item.value]
        if parties:
            parts.append("Parties listed:")
            parts.extend(f"- {item}" for item in parties)
    if terms_items:
        terms = [_format_field(item.label, item.value) for item in terms_items if item.value]
        if terms:
            parts.append("Terms noted:")
            parts.extend(f"- {item}" for item in terms)
    return "\n".join(parts).strip()


def _format_item(text: str) -> str:
    return " ".join(text.split())


def _format_field(label: str | None, value: str) -> str:
    if label:
        return f"{label}: {value}"
    return value


def _render_legal(schema: LegalSchema, intent: str, strict: bool = False) -> str:
    _ = intent
    clauses_items = (schema.clauses.items if schema.clauses else None) or []
    if not clauses_items:
        if strict:
            return ""
        if schema.clauses and schema.clauses.missing_reason:
            return schema.clauses.missing_reason
        return ""
    clauses = []
    for clause in clauses_items:
        text = clause.text.strip()
        if clause.title:
            clauses.append(f"{clause.title}: {text}")
        elif text:
            clauses.append(text)
    if not clauses:
        if strict:
            return ""
        return ""
    cleaned = [" ".join(c.split()) for c in clauses]
    if len(cleaned) == 1:
        return cleaned[0]
    lines = ["Relevant clauses:"]
    for clause in cleaned:
        lines.append(f"- {clause}")
    return "\n".join(lines)


__all__ = ["render_enterprise"]
