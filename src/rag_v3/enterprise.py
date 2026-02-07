from __future__ import annotations

from typing import Any, List

from .types import GenericSchema, HRSchema, InvoiceSchema, LegalSchema, MISSING_REASON, MultiEntitySchema


NO_ITEMS_MESSAGE = "Invoice pages retrieved don’t show itemized products/services—only totals/parties."


def render_enterprise(schema: Any, intent: str, *, domain: str | None = None, strict: bool = False) -> str:
    if isinstance(schema, MultiEntitySchema):
        return _render_multi(schema, intent, strict=strict)
    if isinstance(schema, InvoiceSchema):
        return _render_invoice(schema, intent, strict=strict)
    if isinstance(schema, HRSchema):
        return _render_hr(schema, intent, strict=strict)
    if isinstance(schema, LegalSchema):
        return _render_legal(schema, intent, strict=strict)
    if isinstance(schema, GenericSchema):
        return _render_generic(schema, intent, strict=strict)
    _ = domain
    return ""


def _render_generic(schema: GenericSchema, intent: str, strict: bool = False) -> str:
    _ = intent
    facts_items = (schema.facts.items if schema.facts else None) or []
    if not facts_items:
        if strict:
            return ""
        if schema.facts and schema.facts.missing_reason:
            return schema.facts.missing_reason
        return MISSING_REASON
    facts = [fact.value for fact in facts_items if fact.value]
    if not facts:
        if strict:
            return ""
        return MISSING_REASON
    cleaned = [" ".join(fact.split()) for fact in facts]
    if len(cleaned) == 1:
        return cleaned[0].strip()
    lines = ["Key details:"]
    for fact in cleaned:
        lines.append(f"- {fact}")
    return "\n".join(lines).strip()


def _render_multi(schema: MultiEntitySchema, intent: str, strict: bool = False) -> str:
    _ = intent
    entities = schema.entities or []
    if not entities:
        if strict:
            return ""
        return schema.missing_reason or MISSING_REASON

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


def _render_hr(schema: HRSchema, intent: str, strict: bool = False) -> str:
    candidates = (schema.candidates.items if schema.candidates else None) or []
    if not candidates:
        if strict:
            return ""
        if schema.candidates and schema.candidates.missing_reason:
            return schema.candidates.missing_reason
        return MISSING_REASON
    if intent == "contact":
        sections = []
        for cand in candidates:
            name = cand.name or "Candidate"
            emails = _render_contact_value(cand.emails)
            phones = _render_contact_value(cand.phones)
            linkedins = _render_contact_value(cand.linkedins)
            sections.append(
                "\n".join(
                    [
                        f"**Candidate:** {name}",
                        f"- Email: {emails}",
                        f"- Phone: {phones}",
                        f"- LinkedIn: {linkedins}",
                    ]
                )
            )
        return "\n\n".join(sections).strip()
    if intent in {"rank", "compare"} and len(candidates) > 1:
        ranked = _rank_candidates(candidates)
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
    name = cand.name or "Candidate"
    parts.append(f"**Candidate:** {name}")

    parts.append(f"- Total experience: {cand.total_years_experience or MISSING_REASON}")
    summary = cand.experience_summary or MISSING_REASON
    parts.append(f"- Summary: {summary}")

    parts.append(f"- Technical skills: {', '.join(cand.technical_skills or []) or MISSING_REASON}")
    parts.append(f"- Functional skills: {', '.join(cand.functional_skills or []) or MISSING_REASON}")
    parts.append(f"- Certifications: {', '.join(cand.certifications or []) or MISSING_REASON}")
    parts.append(f"- Education: {', '.join(cand.education or []) or MISSING_REASON}")
    parts.append(f"- Achievements/Awards: {', '.join(cand.achievements or []) or MISSING_REASON}")
    parts.append(f"- Source type: {cand.source_type or MISSING_REASON}")

    return "\n".join(parts).strip()


def _rank_candidates(candidates: List[Any]):
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
    name = cand.name or "Candidate"
    tech = ", ".join((cand.technical_skills or [])[:4]) or MISSING_REASON
    func = ", ".join((cand.functional_skills or [])[:4]) or MISSING_REASON
    certs = ", ".join((cand.certifications or [])[:3]) or MISSING_REASON
    summary = cand.experience_summary or MISSING_REASON
    label = "Top pick" if idx == 1 else "Next"
    return (
        f"- {label}: {name} — key technical skills: {tech}; "
        f"functional skills: {func}; certifications: {certs}; summary: {summary}"
    )


def _format_candidate_detail(cand: Any) -> str:
    name = cand.name or "Candidate"
    summary = cand.experience_summary or MISSING_REASON
    tech = ", ".join((cand.technical_skills or [])[:6]) or MISSING_REASON
    func = ", ".join((cand.functional_skills or [])[:6]) or MISSING_REASON
    certs = ", ".join((cand.certifications or [])[:4]) or MISSING_REASON
    edu = ", ".join((cand.education or [])[:3]) or MISSING_REASON
    awards = ", ".join((cand.achievements or [])[:3]) or MISSING_REASON
    years = cand.total_years_experience or MISSING_REASON
    source = cand.source_type or MISSING_REASON
    lines = [
        f"**Candidate:** {name}",
        f"- Total experience: {years}",
        f"- Summary: {summary}",
        f"- Technical skills: {tech}",
        f"- Functional skills: {func}",
        f"- Certifications: {certs}",
        f"- Education: {edu}",
        f"- Achievements/Awards: {awards}",
        f"- Source type: {source}",
    ]
    return "\n".join(lines)


def _render_contact_value(value: Any) -> str:
    items = [item for item in (value or []) if item]
    return ", ".join(items) if items else MISSING_REASON


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
        return MISSING_REASON

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
    return MISSING_REASON


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
        return MISSING_REASON
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
        return MISSING_REASON
    cleaned = [" ".join(c.split()) for c in clauses]
    if len(cleaned) == 1:
        return cleaned[0]
    lines = ["Relevant clauses:"]
    for clause in cleaned:
        lines.append(f"- {clause}")
    return "\n".join(lines)


__all__ = ["render_enterprise"]
