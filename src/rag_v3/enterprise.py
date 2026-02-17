from __future__ import annotations

import re
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .query_focus import QueryFocus

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


# Contact info pattern — items that are phone numbers, emails, or addresses should be stripped
_CONTACT_ITEM_RE = re.compile(
    r'^\s*\+?\d[\d\s\-().]{7,}\s*$'     # pure phone number
    r'|^\s*\S+@\S+\.\S+\s*$'             # pure email
    r'|^\s*https?://\S+\s*$'             # pure URL
)


def _sanitize_render_list(items: List[str], max_items: int = 20) -> List[str]:
    """Clean a list of rendered items, removing any garbage entries."""
    cleaned = []
    for item in items:
        if not item:
            continue
        if _RENDER_GARBAGE_RE.search(item):
            continue
        if _CONTACT_ITEM_RE.search(item):
            continue
        item = " ".join(item.split()).strip()
        if item and len(item) > 2:
            cleaned.append(item)
    return cleaned[:max_items]


def render_enterprise(schema: Any, intent: str, *, domain: str | None = None, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    if isinstance(schema, MultiEntitySchema):
        return _render_multi(schema, intent, strict=strict, query=query, query_focus=query_focus)
    if isinstance(schema, InvoiceSchema):
        return _render_invoice(schema, intent, strict=strict, query=query, query_focus=query_focus)
    if isinstance(schema, HRSchema):
        return _render_hr(schema, intent, strict=strict, query=query, query_focus=query_focus)
    if isinstance(schema, LegalSchema):
        return _render_legal(schema, intent, strict=strict, query=query, query_focus=query_focus)
    if isinstance(schema, GenericSchema):
        return _render_generic(schema, intent, strict=strict, query=query, query_focus=query_focus)
    _ = domain
    return ""


def _render_generic(schema: GenericSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
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

    # Query-focus filtering: score and sort facts by relevance
    if query_focus and not query_focus.is_exhaustive and (query_focus.keywords or query_focus.field_tags):
        from .query_focus import score_fact_relevance
        scored = [(score_fact_relevance(f.label or "", f.value or "", query_focus), f) for f in facts]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Keep top relevant facts (min 3, max 20)
        relevant = [(s, f) for s, f in scored if s >= 0.15]
        if len(relevant) < 3:
            facts = [f for _, f in scored[:max(3, len(facts))]]
        else:
            facts = [f for _, f in relevant[:20]]

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


def _render_multi(schema: MultiEntitySchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    entities = schema.entities or []
    if not entities:
        if strict:
            return ""
        return schema.missing_reason or ""

    has_docs = any(item.document_name or item.document_id for item in entities)
    has_named = any(item.label and item.label.lower() not in {"document", "candidate"} for item in entities)

    count = len(entities)
    if has_named:
        lead = f"Found {count} candidates in the documents."
    else:
        lead = f"Found {count} documents that match this request."

    lines = [lead, ""]
    for item in entities:
        label = item.label or item.document_name or item.document_id or "Document"
        doc_ref = item.document_name or item.document_id

        # Build a brief profile summary from available fields
        details = []
        fields = getattr(item, "fields", {}) or {}
        if fields:
            for k, v in list(fields.items())[:3]:
                if v and isinstance(v, str) and len(v) < 100:
                    details.append(f"{k}: {v}")
                elif v and isinstance(v, list) and v:
                    details.append(f"{k}: {', '.join(str(x) for x in v[:3])}")

        header = f"**{label}**"
        if doc_ref and doc_ref not in label:
            header += f" ({doc_ref})"

        lines.append(header)
        if details:
            for d in details:
                lines.append(f"  - {d}")
        lines.append("")

    # Add a helpful suggestion
    if has_named and count > 1:
        names = [item.label for item in entities if item.label][:3]
        if names:
            lines.append(f"You can ask about a specific person, e.g., \"Tell me about {names[0]}\"")

    return "\n".join(lines).strip()


def _render_hr(schema: HRSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
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
        # --- Ranking: use neural schema for intelligent intent-aware ranking ---
        if intent == "rank":
            try:
                from .neural_schema import (
                    parse_query_intent,
                    build_profile_intelligence,
                    rank_by_criteria,
                    format_ranking_response,
                )
                q_intent = parse_query_intent(query, candidate_names=[c.name for c in candidates if c.name])
                profile = build_profile_intelligence(candidates)
                ranked_results = rank_by_criteria(profile, q_intent)
                if ranked_results:
                    return format_ranking_response(ranked_results, q_intent, profile)
            except Exception:
                pass  # Fall through to legacy ranking

            # Legacy ranking fallback
            ranked = _rank_candidates(candidates, query=query)
            query_lower = (query or "").lower()
            query_kws = set(query_lower.split()) - _RANK_STOP_WORDS
            relevant_kws = [kw for kw in query_kws if len(kw) > 2]
            if relevant_kws:
                criteria = ", ".join(relevant_kws[:3])
                header = f"Here is the ranking of {len(ranked)} candidates based on {criteria}:"
            else:
                header = f"Here is the ranking of {len(ranked)} candidates based on their overall profile:"
            lines = [header]
            for idx, cand in enumerate(ranked, start=1):
                lines.append(_format_rank_line(idx, cand))
            return "\n".join(lines)

        # --- Comparison: use comparator for side-by-side analysis ---
        if intent == "compare":
            try:
                from .neural_schema import (
                    parse_query_intent as _pqi,
                    build_profile_intelligence as _bpi,
                    build_candidate_digest,
                    format_comparison_response,
                )
                _cmp_intent = _pqi(query, candidate_names=[c.name for c in candidates if c.name])
                _cmp_profile = _bpi(candidates)
                # If entities specified, filter to those
                if _cmp_intent.entities:
                    entity_lower = {e.lower() for e in _cmp_intent.entities}
                    filtered = [d for d in _cmp_profile.candidates if d.name.lower() in entity_lower]
                    if len(filtered) >= 2:
                        rendered = format_comparison_response(filtered, _cmp_intent, _cmp_profile)
                        if rendered:
                            return rendered
                # Full comparison
                rendered = format_comparison_response(_cmp_profile.candidates, _cmp_intent, _cmp_profile)
                if rendered:
                    return rendered
            except Exception:
                pass  # Fall through to legacy comparator

            try:
                from .comparator import compare_candidates_from_schema, render_comparison
                comp_result = compare_candidates_from_schema(candidates, query)
                rendered = render_comparison(comp_result, intent)
                if rendered:
                    return rendered
            except Exception:
                pass

            sections = []
            for cand in candidates:
                sections.append(_format_candidate_detail(cand))
            if sections:
                return "Here is a comparison of the candidates:\n\n" + "\n\n---\n\n".join(sections)
    if len(candidates) > 1:
        # Check if the query actually implies ranking even if intent wasn't "rank"
        # e.g., "who are the top 2?" or "best candidate"
        try:
            from .neural_schema import parse_query_intent as _pqi2
            _fallback_intent = _pqi2(query, candidate_names=[c.name for c in candidates if c.name])
            if _fallback_intent.action == "rank" or _fallback_intent.top_n is not None:
                from .neural_schema import (
                    build_profile_intelligence as _bpi2,
                    rank_by_criteria as _rbc2,
                    format_ranking_response as _frr2,
                )
                _fb_profile = _bpi2(candidates)
                _fb_ranked = _rbc2(_fb_profile, _fallback_intent)
                if _fb_ranked:
                    return _frr2(_fb_ranked, _fallback_intent, _fb_profile)
        except Exception:
            pass
        sections = []
        for cand in candidates:
            sections.append(_format_candidate_detail(cand))
        return "\n\n".join(sections)

    cand = candidates[0]
    # Determine which fields to prioritize based on intent/query
    query_lower = query.lower() if query else ""
    focus = _detect_field_focus(query_lower)

    # Use query_focus for relevance scoring when available
    _has_qf = query_focus is not None and not query_focus.is_exhaustive and bool(getattr(query_focus, "field_tags", None))

    parts = []
    name = _sanitize_render_value(cand.name or "", max_length=80) or "Candidate"
    parts.append(f"**Candidate:** {name}")

    # Build all available field renderings: (tag, full_line, summary_line)
    field_renderings = []
    years_val = _sanitize_render_value(cand.total_years_experience or '', max_length=30)
    if years_val:
        field_renderings.append(("experience", f"- Total experience: {years_val}", f"- Experience: {years_val}"))
    summary = _sanitize_render_value(cand.experience_summary or '', max_length=500)
    if summary:
        # Summary line: truncate to ~80 chars
        short_summary = (summary[:80] + "...") if len(summary) > 80 else summary
        field_renderings.append(("summary", f"- Summary: {summary}", f"- Summary: {short_summary}"))
    tech = _sanitize_render_list(cand.technical_skills or [])
    if tech:
        field_renderings.append(("skills", f"- Technical skills: {', '.join(tech)}", f"- Technical skills: {', '.join(tech[:5])}{'...' if len(tech) > 5 else ''}"))
    func = _sanitize_render_list(cand.functional_skills or [])
    if func:
        field_renderings.append(("skills", f"- Functional skills: {', '.join(func)}", f"- Functional skills: {', '.join(func[:3])}{'...' if len(func) > 3 else ''}"))
    certs = _sanitize_render_list(cand.certifications or [])
    if certs:
        field_renderings.append(("certifications", f"- Certifications: {', '.join(certs)}", f"- Certifications: {', '.join(certs[:3])}{'...' if len(certs) > 3 else ''}"))
    edu = _sanitize_render_list(cand.education or [], max_items=5)
    if edu:
        field_renderings.append(("education", f"- Education: {', '.join(edu)}", f"- Education: {', '.join(edu[:2])}{'...' if len(edu) > 2 else ''}"))
    awards = _sanitize_render_list(cand.achievements or [])
    if awards:
        field_renderings.append(("achievements", f"- Achievements/Awards: {', '.join(awards)}", f"- Achievements: {', '.join(awards[:2])}{'...' if len(awards) > 2 else ''}"))
    # Contact fields
    emails = _render_contact_value(cand.emails) if hasattr(cand, 'emails') else ""
    if emails:
        field_renderings.append(("contact", f"- Email: {emails}", f"- Email: {emails}"))
    phones = _render_contact_value(cand.phones) if hasattr(cand, 'phones') else ""
    if phones:
        field_renderings.append(("contact", f"- Phone: {phones}", f"- Phone: {phones}"))
    linkedins = _render_contact_value(cand.linkedins) if hasattr(cand, 'linkedins') else ""
    if linkedins:
        field_renderings.append(("contact", f"- LinkedIn: {linkedins}", f"- LinkedIn: {linkedins}"))

    if _has_qf:
        # Query-focus driven: focused fields get full rendering, secondary get summary, irrelevant omitted
        from .query_focus import score_field_relevance
        for tag, full_line, summary_line in field_renderings:
            relevance = score_field_relevance(tag, query_focus)
            if relevance >= 0.5:
                parts.append(full_line)
            elif relevance >= 0.1:
                parts.append(summary_line)
            # else: omitted
    elif focus:
        # Legacy focus (from _detect_field_focus): focused fields first, then remaining
        for tag, full_line, _ in field_renderings:
            if tag in focus:
                parts.append(full_line)
        for tag, full_line, _ in field_renderings:
            if tag not in focus:
                parts.append(full_line)
    else:
        # No specific focus — show all fields in standard order
        for _, full_line, _ in field_renderings:
            parts.append(full_line)

    return "\n".join(parts).strip()


# Map query keywords to field tags for intent-aware rendering
_FIELD_FOCUS_MAP = {
    "skill": {"skills"},
    "skills": {"skills"},
    "technical": {"skills"},
    "education": {"education"},
    "degree": {"education"},
    "university": {"education"},
    "college": {"education"},
    "qualification": {"education", "certifications"},
    "certif": {"certifications"},
    "experience": {"experience", "summary"},
    "years": {"experience"},
    "achievement": {"achievements"},
    "award": {"achievements"},
    "summary": {"summary"},
    "overview": {"summary"},
    "profile": {"summary", "experience", "skills"},
    "contact": {"contact"},
    "email": {"contact"},
    "phone": {"contact"},
    "linkedin": {"contact"},
    "address": {"contact"},
    "project": {"experience", "summary"},
    "work": {"experience", "summary"},
    "worked": {"experience", "summary"},
}


def _detect_field_focus(query_lower: str) -> set:
    """Detect which fields to prioritize based on query keywords."""
    focus: set = set()
    for keyword, tags in _FIELD_FOCUS_MAP.items():
        if keyword in query_lower:
            focus.update(tags)
    return focus


_RANK_STOP_WORDS = frozenset({
    "the", "and", "or", "for", "a", "an", "is", "are", "was", "were",
    "of", "in", "to", "with", "on", "at", "by", "from", "as", "it",
    "what", "who", "how", "which", "tell", "me", "about", "do", "does",
    "can", "has", "have", "their", "this", "that", "all", "be",
})


def _rank_candidates(candidates: List[Any], query: str = ""):
    """Rank candidates by a weighted score combining profile strength and query relevance.

    Scoring dimensions:
    1. Profile completeness — breadth of skills, certifications, achievements
    2. Experience depth — years of experience
    3. Query relevance — keyword overlap with skills, certs, summary, role
    4. Role fit — bonus when role/designation matches query keywords
    """
    query_keywords = set(query.lower().split()) - _RANK_STOP_WORDS if query else set()
    # Build bigrams from query for compound skill matching (e.g. "data engineering")
    query_words_list = [w for w in query.lower().split() if w not in _RANK_STOP_WORDS] if query else []
    query_bigrams = {f"{query_words_list[i]} {query_words_list[i+1]}" for i in range(len(query_words_list) - 1)} if len(query_words_list) > 1 else set()

    scored = []
    for cand in candidates:
        score = 0.0
        # Profile completeness
        tech_skills = cand.technical_skills or []
        func_skills = cand.functional_skills or []
        certs = cand.certifications or []
        achievements = cand.achievements or []
        score += 1.5 * len(tech_skills)
        score += 1.0 * len(func_skills)
        score += 0.7 * len(certs)
        score += 0.4 * len(achievements)

        # Education bonus
        education = cand.education if hasattr(cand, "education") else []
        if education:
            score += 0.5 * len(education)

        # Experience depth
        if cand.total_years_experience:
            years = _parse_years(cand.total_years_experience)
            if years is not None:
                score += min(10.0, years) * 0.2

        # Query relevance — keyword matching across all candidate text
        if query_keywords:
            all_skills = [s.lower() for s in (tech_skills + func_skills)]
            all_certs = [c.lower() for c in certs]
            role_text = (getattr(cand, "role", "") or getattr(cand, "designation", "") or "").lower()
            summary_text = (cand.experience_summary or "").lower()

            # Direct skill match (high value — exact query keyword in skills)
            skill_matches = sum(1 for kw in query_keywords if any(kw in s for s in all_skills))
            score += skill_matches * 3.0

            # Bigram skill match (even higher — "data engineering" matches "data engineering")
            bigram_matches = sum(1 for bg in query_bigrams if any(bg in s for s in all_skills))
            score += bigram_matches * 4.0

            # Cert match
            cert_matches = sum(1 for kw in query_keywords if any(kw in c for c in all_certs))
            score += cert_matches * 2.5

            # Role/designation match
            if role_text:
                role_matches = sum(1 for kw in query_keywords if kw in role_text)
                score += role_matches * 3.0

            # Summary/experience text match (lower weight — less precise)
            if summary_text:
                summary_matches = sum(1 for kw in query_keywords if kw in summary_text)
                score += summary_matches * 1.0

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
    tech = ", ".join(_sanitize_render_list((cand.technical_skills or [])[:5]))
    func = ", ".join(_sanitize_render_list((cand.functional_skills or [])[:4]))
    certs = ", ".join(_sanitize_render_list((cand.certifications or [])[:3]))
    years = _sanitize_render_value(cand.total_years_experience or '', max_length=30)
    role = _sanitize_render_value(
        (getattr(cand, "role", "") or getattr(cand, "designation", "") or ""), max_length=60
    )
    detail_parts = []
    if role:
        detail_parts.append(role)
    if years:
        detail_parts.append(f"{years} experience")
    if tech:
        detail_parts.append(f"skills: {tech}")
    if certs:
        detail_parts.append(f"certifications: {certs}")
    detail = "; ".join(detail_parts) if detail_parts else "See document for details"
    return f"{idx}. **{name}** — {detail}"


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


def _render_invoice(schema: InvoiceSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    # Query-focus section filtering for invoices
    if query_focus and not query_focus.is_exhaustive and query_focus.field_tags:
        _qf_tags = query_focus.field_tags
        _want_totals = "totals" in _qf_tags
        _want_items = "items" in _qf_tags
        _want_parties = "parties" in _qf_tags
        if _want_totals and not _want_items and not _want_parties:
            # Render totals section only
            totals_items = (schema.totals.items if schema.totals else None) or []
            if totals_items:
                totals = [_format_field(item.label, item.value) for item in totals_items if item.value]
                if totals:
                    lines = ["Totals shown:"]
                    lines.extend(f"- {item}" for item in totals)
                    return "\n".join(lines)
        elif _want_items and not _want_totals and not _want_parties:
            # Render items section only
            items = (schema.items.items if schema.items else None) or []
            if items:
                rendered = [_format_item(item.description) for item in items if item.description]
                if rendered:
                    lines = ["Items listed on the invoice:"]
                    lines.extend(f"- {item}" for item in rendered)
                    return "\n".join(lines)
        elif _want_parties and not _want_totals and not _want_items:
            # Render parties section only
            parties_items = (schema.parties.items if schema.parties else None) or []
            if parties_items:
                parties = [_format_field(item.label, item.value) for item in parties_items if item.value]
                if parties:
                    lines = ["Parties listed:"]
                    lines.extend(f"- {item}" for item in parties)
                    return "\n".join(lines)
        # Fall through to standard rendering if focused section is empty

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

    # Default: show comprehensive invoice summary with all available sections
    parts: list[str] = []

    # Totals and parties
    summary = _render_totals_parties(schema, prefer_totals=True)
    if summary:
        parts.append(summary)

    # Line items
    items = (schema.items.items if schema.items else None) or []
    if items:
        rendered = [_format_item(item.description) for item in items if item.description]
        if rendered:
            parts.append("Line items:")
            for item in rendered:
                parts.append(f"- {item}")

    if parts:
        return "\n".join(parts)

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


def _render_legal(schema: LegalSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    clauses_items = (schema.clauses.items if schema.clauses else None) or []
    if not clauses_items:
        if strict:
            return ""
        if schema.clauses and schema.clauses.missing_reason:
            return schema.clauses.missing_reason
        return ""

    # Build structured clause entries
    entries: list[tuple[str, str]] = []  # (title_or_empty, text)
    for clause in clauses_items:
        text = clause.text.strip() if clause.text else ""
        title = (clause.title or "").strip()
        if text:
            entries.append((title, text))

    if not entries:
        if strict:
            return ""
        return ""

    # Query-focus clause relevance filtering: keep top relevant clauses when >3
    if query_focus and not query_focus.is_exhaustive and len(entries) > 3 and query_focus.keywords:
        from .query_focus import score_fact_relevance
        scored_entries = [
            (score_fact_relevance(title, text, query_focus), (title, text))
            for title, text in entries
        ]
        scored_entries.sort(key=lambda pair: pair[0], reverse=True)
        # Keep top 5 most relevant, minimum 2
        relevant = [(s, e) for s, e in scored_entries if s >= 0.15]
        if len(relevant) < 2:
            entries = [e for _, e in scored_entries[:2]]
        else:
            entries = [e for _, e in relevant[:5]]

    # Intent-aware heading
    _LEGAL_HEADINGS = {
        "summary": "Legal Document Summary:",
        "contact": "Parties and Contact Information:",
        "facts": "Key Legal Provisions:",
    }
    heading = _LEGAL_HEADINGS.get(intent, "Relevant Clauses:")

    if len(entries) == 1:
        title, text = entries[0]
        cleaned = " ".join(text.split())
        if title:
            return f"**{title}:** {cleaned}"
        return cleaned

    lines = [heading]
    for title, text in entries:
        cleaned = " ".join(text.split())
        if title:
            lines.append(f"- **{title}:** {cleaned}")
        else:
            lines.append(f"- {cleaned}")
    return "\n".join(lines)


__all__ = ["render_enterprise"]
