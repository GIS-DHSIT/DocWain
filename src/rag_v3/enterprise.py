from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from typing import Any, List, Optional, TYPE_CHECKING

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .query_focus import QueryFocus

from .types import GenericSchema, HRSchema, InvoiceSchema, LegalSchema, MedicalSchema, MISSING_REASON, MultiEntitySchema, PolicySchema

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

_NER_LABEL_RE = re.compile(
    r"\s+(?:person|organization|other|date|location|money|percent|norp|fac|gpe|event|product|work_of_art|law|language):\s*$",
    re.IGNORECASE,
)
_NER_INLINE_RE = re.compile(
    r"(?:^|\s+)(?:person|organization|other|date|location|money|percent|norp|fac|gpe|event|product|work_of_art|law|language):\s*",
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
    # Strip NER entity type labels that leak from spaCy annotations
    cleaned = _NER_LABEL_RE.sub("", cleaned)
    cleaned = _NER_INLINE_RE.sub(" ", cleaned)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned.strip()

# Contact info pattern — items that are phone numbers, emails, or addresses should be stripped
_CONTACT_ITEM_RE = re.compile(
    r'^\s*\+?\d[\d\s\-().]{7,}\s*$'     # pure phone number
    r'|^\s*\S+@\S+\.\S+\s*$'             # pure email
    r'|^\s*https?://\S+\s*$'             # pure URL
)

# Intent-aware rendering limits — concise intents get fewer items
_INTENT_MAX_ITEMS = {
    "factual": 10,
    "contact": 8,
    "extract": 12,
    "clarify": 8,
    "compare": 20,
    "rank": 20,
    "summarize": 15,
    "generate": 20,
    "analyze": 20,
    "timeline": 20,
}

def _sanitize_render_list(items: List[str], max_items: int = 20, *, intent: str = "") -> List[str]:
    """Clean a list of rendered items, removing any garbage entries."""
    # Use intent-aware limit if available and caller didn't override
    if intent and max_items == 20:
        max_items = _INTENT_MAX_ITEMS.get(intent, 20)
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
        result = _render_multi(schema, intent, strict=strict, query=query, query_focus=query_focus)
    elif isinstance(schema, InvoiceSchema):
        result = _render_invoice(schema, intent, strict=strict, query=query, query_focus=query_focus)
    elif isinstance(schema, HRSchema):
        result = _render_hr(schema, intent, strict=strict, query=query, query_focus=query_focus)
    elif isinstance(schema, LegalSchema):
        result = _render_legal(schema, intent, strict=strict, query=query, query_focus=query_focus)
    elif isinstance(schema, MedicalSchema):
        result = _render_medical(schema, intent, strict=strict, query=query, query_focus=query_focus)
    elif isinstance(schema, PolicySchema):
        result = _render_policy(schema, intent, strict=strict, query=query, query_focus=query_focus)
    elif isinstance(schema, GenericSchema):
        result = _render_generic(schema, intent, strict=strict, query=query, query_focus=query_focus)
    else:
        # If schema type doesn't match but we have domain, try domain-aware GenericSchema rendering
        if domain and hasattr(schema, "facts"):
            result = _render_generic(schema, intent, strict=strict, query=query, query_focus=query_focus)
        else:
            return ""
    # Final pass: strip any NER entity type labels that leaked through
    if result:
        result = _NER_LABEL_RE.sub("", result)
        result = _NER_INLINE_RE.sub(" ", result)
    # Query-adaptive keyword reinforcement: ensure response echoes key query terms.
    # IMPORTANT: never prepend keyword prefix to MISSING_REASON strings — it
    # breaks the _is_missing detection in the pipeline and hides extraction failures.
    if result and query:
        _rl_stripped = (result or "").strip().lower()
        _is_missing_result = (
            _rl_stripped.startswith("not explicitly mentioned")
            or _rl_stripped.startswith("not enough information")
            or _rl_stripped == MISSING_REASON.strip().lower()
        )
        if not _is_missing_result:
            _ql = (query or "").lower()
            _rl = _rl_stripped
            # Domain keyword reinforcement — ensure response echoes key query terms
            _KEYWORD_PREFIXES = {
                ("vendor",): "Vendor information from the documents:\n",
                ("term", "condition"): "Terms and conditions:\n",
                ("compar",): "Comparison of the requested items:\n",
                ("rank",): "Rankings based on the available data:\n",
                ("summar",): "Summary of the relevant information:\n",
                ("timeline",): "Timeline of events:\n",
                ("salary", "compensat"): "Compensation details:\n",
                ("coverage", "benefit"): "Coverage and benefits information:\n",
                ("risk",): "Risk assessment:\n",
                ("obligation",): "Obligations and commitments:\n",
            }
            for _kw_set, _prefix in _KEYWORD_PREFIXES.items():
                # All keywords in the set must appear in query, and at least one must be absent from response
                if all(kw in _ql for kw in _kw_set) and any(kw not in _rl for kw in _kw_set):
                    result = f"{_prefix}{result}"
                    break
    return result

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

    # Prepend generic insights for multi-document results
    _gen_insights = _compute_generic_insights(facts) if has_doc_names else ""

    if has_doc_names:
        rendered = _render_grouped_by_document(facts)
        if _gen_insights:
            rendered = f"{_gen_insights}\n\n{rendered}"
        return rendered

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
        lines.append(f"Information from {len(doc_groups)} documents:")
        lines.append("")  # Blank line for visual separation

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
                # Auto-bold significant numeric values (GPT-style emphasis)
                clean_value = _bold_significant_values(clean_value)
                # Add page source attribution when evidence spans are available
                _page_ref = ""
                spans = getattr(fact, "evidence_spans", None) or []
                if spans:
                    pages = set()
                    for span in spans[:3]:  # Limit to avoid clutter
                        pg = getattr(span, "page", None) or getattr(span, "page_start", None)
                        if pg:
                            pages.add(str(pg))
                    if pages:
                        _page_ref = f" *(p.{', '.join(sorted(pages))})*"
                if fact.label:
                    clean_label = _sanitize_render_value(fact.label, max_length=80)
                    if clean_label:
                        lines.append(f"- **{clean_label}:** {clean_value}{_page_ref}")
                    else:
                        lines.append(f"- {clean_value}{_page_ref}")
                else:
                    lines.append(f"- {clean_value}{_page_ref}")

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

# Auto-bold significant values in rendered facts (GPT-style emphasis)
_SIGNIFICANT_VALUE_RE = re.compile(
    r"(?<!\*\*)"  # not already bolded
    r"(\$[\d,]+\.?\d*"            # currency: $1,234.56
    r"|\d+(?:,\d{3})+(?:\.\d+)?" # large numbers: 1,234
    r"|\d+\.?\d*\s*%"            # percentages: 95.5%
    r"|\d{4}[-–]\d{4}"          # year ranges: 2019-2023
    r")"
    r"(?!\*\*)"  # not already bolded
)

def _bold_significant_values(text: str) -> str:
    """Bold significant numeric values in fact text (currency, %, large numbers, year ranges)."""
    if not text or (text.startswith("**") and text.endswith("**")):
        return text  # Already fully wrapped in bold formatting, skip
    return _SIGNIFICANT_VALUE_RE.sub(r"**\1**", text)

def _render_flat_facts(facts: List[Any], intent: str = "facts") -> str:
    """Render facts without document grouping (single-doc or no doc info).

    Groups facts with the same label together for cleaner output
    (e.g., multiple emails rendered as "Emails: a@b.com, c@d.com").
    """
    labeled = [(f.label, f.value, f) for f in facts if f.label]
    unlabeled = [(f.value, f) for f in facts if not f.label]

    lines: list[str] = []

    if labeled:
        heading = _INTENT_HEADINGS.get(intent)
        if heading:
            lines.append(f"**{heading}**")

        # Group facts by label for cleaner rendering
        from collections import OrderedDict
        grouped: OrderedDict[str, list] = OrderedDict()
        for label, value, fact_obj in labeled:
            clean_label = _sanitize_render_value(label, max_length=80)
            clean_value = _sanitize_render_value(value, max_length=500)
            if clean_label and clean_value:
                # Auto-bold significant numeric values (GPT-style emphasis)
                clean_value = _bold_significant_values(clean_value)
                # Extract page ref from evidence_spans (same logic as grouped renderer)
                _page_ref = ""
                spans = getattr(fact_obj, "evidence_spans", None) or []
                if spans:
                    pages = set()
                    for span in spans[:3]:
                        pg = getattr(span, "page", None) or getattr(span, "page_start", None)
                        if pg:
                            pages.add(str(pg))
                    if pages:
                        _page_ref = f" *(p.{', '.join(sorted(pages))})*"
                grouped.setdefault(clean_label, []).append((clean_value, _page_ref))

        # Merge labels with short values (contacts, skills) into comma-separated
        _MERGE_LABELS = {"email", "phone", "linkedin", "skill", "certification", "language"}
        for label, entries in grouped.items():
            if len(entries) > 1 and label.lower().rstrip("s") in _MERGE_LABELS:
                merged_values = ', '.join(v for v, _ in entries)
                # Consolidate page refs from merged entries into single ref
                all_page_nums: set[str] = set()
                for _, p in entries:
                    if p:
                        # Extract page numbers from refs like " *(p.1)*"
                        import re as _re
                        for _pn in _re.findall(r'p\.(\d+)', p):
                            all_page_nums.add(_pn)
                consolidated_ref = f" *(p.{', '.join(sorted(all_page_nums, key=int))})*" if all_page_nums else ""
                lines.append(f"- **{label}:** {merged_values}{consolidated_ref}")
            else:
                for value, page_ref in entries:
                    lines.append(f"- **{label}:** {value}{page_ref}")

    if unlabeled:
        cleaned = []
        for value, fact_obj in unlabeled:
            if not value:
                continue
            clean_val = " ".join(value.split())
            # Extract page ref
            _page_ref = ""
            spans = getattr(fact_obj, "evidence_spans", None) or []
            if spans:
                pages = set()
                for span in spans[:3]:
                    pg = getattr(span, "page", None) or getattr(span, "page_start", None)
                    if pg:
                        pages.add(str(pg))
                if pages:
                    _page_ref = f" *(p.{', '.join(sorted(pages))})*"
            cleaned.append((clean_val, _page_ref))
        if len(cleaned) == 1 and not labeled:
            val, pref = cleaned[0]
            return f"{val.strip()}{pref}"
        if not labeled:
            heading = _INTENT_HEADINGS.get(intent, "Key details:")
            if heading:
                lines.append(heading)
        for fact_text, page_ref in cleaned:
            lines.append(f"- {fact_text}{page_ref}")

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

    # For comparison/ranking intents with 3+ entities and field data, render as a table
    _intent_lower = intent.lower() if intent else ""
    _has_rich_fields = all(
        getattr(item, "fields", None) and len(getattr(item, "fields", {})) >= 2
        for item in entities
    )
    _table_rendered = False
    if count >= 2 and _has_rich_fields and _intent_lower in ("compare", "rank", "comparison", "ranking"):
        # Build a comparison table
        # Collect all field names across entities
        all_field_names: list[str] = []
        for item in entities:
            for k in (getattr(item, "fields", {}) or {}).keys():
                if k not in all_field_names:
                    all_field_names.append(k)
        all_field_names = all_field_names[:6]  # Cap columns

        if all_field_names:
            header_row = "| Entity | " + " | ".join(all_field_names) + " |"
            sep_row = "|--------|" + "|".join("--------" for _ in all_field_names) + "|"
            lines.append(header_row)
            lines.append(sep_row)
            for item in entities:
                label = item.label or item.document_name or "Entity"
                fields = getattr(item, "fields", {}) or {}
                cells = []
                for fn in all_field_names:
                    v = fields.get(fn, "N/A")
                    if isinstance(v, list):
                        v = ", ".join(str(x) for x in v[:3])
                    cells.append(_sanitize_render_value(str(v or "N/A"), max_length=80))
                lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
            lines.append("")
            _table_rendered = True

    # Bullet-point rendering (default or when table not applicable)
    if not _table_rendered:
        for item in entities:
            label = item.label or item.document_name or item.document_id or "Document"
            doc_ref = item.document_name or item.document_id

            # Build a brief profile summary from available fields
            details = []
            fields = getattr(item, "fields", {}) or {}
            if fields:
                for k, v in list(fields.items())[:5]:
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
        return "No candidate information was found in the provided documents."
    if intent == "contact":
        # Check if the query is asking for contact info ONLY, or is a comprehensive
        # extraction request that happens to include contact among other fields.
        _q_lower = (query or "").lower()
        _comprehensive_signals = (
            "experience" in _q_lower or "skills" in _q_lower or "education" in _q_lower
            or "certification" in _q_lower or "summary" in _q_lower
            or "extract each" in _q_lower or "all information" in _q_lower
            or "complete profile" in _q_lower or "full profile" in _q_lower
        )
        if not _comprehensive_signals:
            # Pure contact-only query — render contact fields only
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
        # Comprehensive query — fall through to detail rendering below
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
            except Exception as e:
                logger.warning("Neural ranking failed: %s", e)
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
            # For large candidate sets (>5), side-by-side comparison is useless
            # (produces N/A matrices). Use detail listing instead.
            if len(candidates) <= 5:
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
                except Exception as e:
                    logger.warning("Neural comparison failed: %s", e)
                    pass  # Fall through to legacy comparator

                try:
                    from .comparator import compare_candidates_from_schema, render_comparison
                    comp_result = compare_candidates_from_schema(candidates, query)
                    rendered = render_comparison(comp_result, intent)
                    if rendered:
                        return rendered
                except Exception as e:
                    logger.warning("Legacy comparison failed: %s", e)

            sections = []
            for cand in candidates:
                sections.append(_format_candidate_detail(cand))
            if sections:
                return f"Here is an overview of all {len(candidates)} candidates:\n\n" + "\n\n---\n\n".join(sections)
    if len(candidates) > 1:
        # Prepend analytical insights for multi-candidate responses
        _hr_insights = _compute_hr_insights(candidates)

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
        result = "\n\n".join(sections)
        if _hr_insights:
            result = f"{_hr_insights}\n\n{result}"
        return result

    cand = candidates[0]

    # Use query_focus for relevance scoring when available
    _has_qf = query_focus is not None and not query_focus.is_exhaustive and bool(getattr(query_focus, "field_tags", None))
    # Legacy focus from query_focus.field_tags (replaces old _detect_field_focus regex)
    focus = query_focus.field_tags if query_focus else set()

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
    tech = _sanitize_render_list(cand.technical_skills or [], intent=intent)
    if tech:
        field_renderings.append(("skills", f"- Technical skills: {', '.join(tech)}", f"- Technical skills: {', '.join(tech[:5])}{'...' if len(tech) > 5 else ''}"))
    func = _sanitize_render_list(cand.functional_skills or [], intent=intent)
    if func:
        field_renderings.append(("skills", f"- Functional skills: {', '.join(func)}", f"- Functional skills: {', '.join(func[:3])}{'...' if len(func) > 3 else ''}"))
    certs = _sanitize_render_list(cand.certifications or [], intent=intent)
    if certs:
        field_renderings.append(("certifications", f"- Certifications: {', '.join(certs)}", f"- Certifications: {', '.join(certs[:3])}{'...' if len(certs) > 3 else ''}"))
    edu = _sanitize_render_list(cand.education or [], max_items=5, intent=intent)
    if edu:
        field_renderings.append(("education", f"- Education: {', '.join(edu)}", f"- Education: {', '.join(edu[:2])}{'...' if len(edu) > 2 else ''}"))
    awards = _sanitize_render_list(cand.achievements or [], intent=intent)
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
        # Legacy focus (from query_focus.field_tags): focused fields first, then remaining
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

_RANK_STOP_WORDS = frozenset({
    "the", "and", "or", "for", "a", "an", "is", "are", "was", "were",
    "of", "in", "to", "with", "on", "at", "by", "from", "as", "it",
    "what", "who", "how", "which", "tell", "me", "about", "do", "does",
    "can", "has", "have", "their", "this", "that", "all", "be",
})

# Skill synonym groups for fuzzy matching in ranking
_SKILL_SYNONYMS: dict[str, set[str]] = {
    "react": {"reactjs", "react.js"},
    "angular": {"angularjs", "angular.js"},
    "vue": {"vuejs", "vue.js"},
    "frontend": {"front-end", "front end", "ui development", "web development"},
    "backend": {"back-end", "back end", "server-side"},
    "javascript": {"js", "ecmascript"},
    "typescript": {"ts"},
    "python": {"py"},
    "machine learning": {"ml", "deep learning", "artificial intelligence", "ai"},
    "aws": {"amazon web services", "cloud"},
    "azure": {"microsoft azure", "cloud"},
    "gcp": {"google cloud", "google cloud platform", "cloud"},
    "devops": {"ci/cd", "cicd", "continuous integration", "continuous deployment"},
    "kubernetes": {"k8s", "container orchestration"},
    "docker": {"containerization", "containers"},
    "sql": {"database", "rdbms", "postgresql", "mysql", "oracle"},
    "nosql": {"mongodb", "dynamodb", "cassandra"},
    "java": {"jvm", "spring", "spring boot"},
    "management": {"leadership", "team lead", "manager"},
    "agile": {"scrum", "kanban", "sprint"},
}

# Build reverse lookup: synonym → canonical term
_SYNONYM_REVERSE: dict[str, str] = {}
for _canonical, _syns in _SKILL_SYNONYMS.items():
    for _syn in _syns:
        _SYNONYM_REVERSE[_syn] = _canonical
    _SYNONYM_REVERSE[_canonical] = _canonical

def _expand_query_keywords(keywords: set[str]) -> set[str]:
    """Expand query keywords with skill synonyms for better matching."""
    expanded = set(keywords)
    for kw in keywords:
        if kw in _SKILL_SYNONYMS:
            expanded.update(_SKILL_SYNONYMS[kw])
        if kw in _SYNONYM_REVERSE:
            canonical = _SYNONYM_REVERSE[kw]
            expanded.add(canonical)
            expanded.update(_SKILL_SYNONYMS.get(canonical, set()))
    return expanded

def _rank_candidates(candidates: List[Any], query: str = ""):
    """Rank candidates by a weighted score combining profile strength and query relevance.

    Scoring dimensions:
    1. Profile completeness — breadth of skills, certifications, achievements
    2. Experience depth — years of experience
    3. Query relevance — keyword overlap with skills, certs, summary, role
    4. Role fit — bonus when role/designation matches query keywords
    """
    raw_keywords = set(query.lower().split()) - _RANK_STOP_WORDS if query else set()
    # Expand keywords with skill synonyms for fuzzy matching
    query_keywords = _expand_query_keywords(raw_keywords)
    # Build bigrams from query for compound skill matching (e.g. "data engineering")
    query_words_list = [w for w in query.lower().split() if w not in _RANK_STOP_WORDS] if query else []
    query_bigrams = {f"{query_words_list[i]} {query_words_list[i+1]}" for i in range(len(query_words_list) - 1)} if len(query_words_list) > 1 else set()
    # Also expand bigrams with synonyms
    expanded_bigrams = set(query_bigrams)
    for bg in query_bigrams:
        if bg in _SKILL_SYNONYMS:
            expanded_bigrams.update(_SKILL_SYNONYMS[bg])
        if bg in _SYNONYM_REVERSE:
            canonical = _SYNONYM_REVERSE[bg]
            expanded_bigrams.add(canonical)
            expanded_bigrams.update(_SKILL_SYNONYMS.get(canonical, set()))
    query_bigrams = expanded_bigrams

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

            # Role/designation match (word-boundary to prevent "Java" matching "JavaScript")
            if role_text:
                import re as _re_rank
                role_words = set(role_text.split())
                role_matches = sum(1 for kw in query_keywords if kw in role_words or _re_rank.search(r'\b' + _re_rank.escape(kw) + r'\b', role_text))
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
    role = _sanitize_render_value(
        (getattr(cand, "role", "") or getattr(cand, "designation", "") or ""), max_length=60
    )
    header = f"**Candidate: {name}**"
    if role:
        header += f" — {role}"
    lines = [header]
    years = _sanitize_render_value(cand.total_years_experience or '', max_length=30)
    if years:
        lines.append(f"- Total experience: {years}")
    summary = _sanitize_render_value(cand.experience_summary or '', max_length=500)
    if summary:
        lines.append(f"- Summary: {summary}")
    tech = ", ".join(_sanitize_render_list((cand.technical_skills or [])[:8]))
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
    # Contact info
    emails = _render_contact_value(getattr(cand, "emails", None))
    if emails:
        lines.append(f"- Email: {emails}")
    phones = _render_contact_value(getattr(cand, "phones", None))
    if phones:
        lines.append(f"- Phone: {phones}")
    linkedins = _render_contact_value(getattr(cand, "linkedins", None))
    if linkedins:
        lines.append(f"- LinkedIn: {linkedins}")
    source = _sanitize_render_value(cand.source_type or '', max_length=50)
    if source:
        lines.append(f"- Source: {source}")
    return "\n".join(lines)

def _render_contact_value(value: Any) -> str:
    items = [item for item in (value or []) if item]
    return ", ".join(items) if items else ""

def _render_invoice(schema: InvoiceSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    # Detect vendor/seller/supplier query intent for adaptive headings
    _query_lower = (query or "").lower()
    _wants_vendor = any(kw in _query_lower for kw in ("vendor", "seller", "supplier", "company", "companies"))
    _party_heading = "Vendors/Sellers:" if _wants_vendor else "Parties listed:"

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
            # Explicit "not found" for empty focused section
            return "No totals information was found in the invoice. The document may not contain a totals or summary section."
        elif _want_items and not _want_totals and not _want_parties:
            # Render items section only
            items = (schema.items.items if schema.items else None) or []
            if items:
                rendered = [_format_item(item.description) for item in items if item.description]
                if rendered:
                    lines = ["Items listed on the invoice:"]
                    lines.extend(f"- {item}" for item in rendered)
                    return "\n".join(lines)
            missing = (schema.items.missing_reason if schema.items else None) or ""
            return missing or "No line items were found in the invoice."
        elif _want_parties and not _want_totals and not _want_items:
            # Render parties section only
            parties_items = (schema.parties.items if schema.parties else None) or []
            if parties_items:
                parties = [_format_field(item.label, item.value) for item in parties_items if item.value]
                if parties:
                    lines = [_party_heading]
                    lines.extend(f"- {item}" for item in parties)
                    return "\n".join(lines)
            return "No party or vendor information was found in the invoice."
        # Fall through to standard rendering for multi-section focus

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
        return "No totals information was found in the invoice."

    if strict:
        return _render_totals_parties(schema, prefer_totals=True, query=query) or ""

    # Default: show comprehensive invoice summary with all available sections
    parts: list[str] = []

    # Query-adaptive header: when user asks about vendors/sellers, add context
    if _wants_vendor:
        parts.append("Vendor and seller information from invoices:")

    # Prepend invoice insights when data is rich enough
    _inv_insights = _compute_invoice_insights(schema)
    if _inv_insights:
        parts.append(_inv_insights)

    # Invoice metadata (number, date, PO, reference)
    inv_meta_items = (getattr(schema, "invoice_metadata", None) and schema.invoice_metadata.items) or []
    if inv_meta_items:
        meta_lines = ["Invoice details:"]
        for item in inv_meta_items:
            clean_val = _sanitize_render_value(item.value, max_length=200)
            if clean_val:
                if item.label:
                    clean_label = _sanitize_render_value(item.label, max_length=80)
                    if clean_label:
                        meta_lines.append(f"- **{clean_label}:** {clean_val}")
                    else:
                        meta_lines.append(f"- {clean_val}")
                else:
                    meta_lines.append(f"- {clean_val}")
        if len(meta_lines) > 1:
            parts.append("\n".join(meta_lines))

    # Totals and parties
    summary = _render_totals_parties(schema, prefer_totals=True, query=query)
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

    # No structured invoice sections could be rendered
    return "No invoice details were found in the provided documents."

def _render_totals_parties(schema: InvoiceSchema, prefer_totals: bool = True, query: str = "") -> str:
    parts: List[str] = []
    totals_items = (schema.totals.items if schema.totals else None) or []
    parties_items = (schema.parties.items if schema.parties else None) or []
    terms_items = (schema.terms.items if schema.terms else None) or []
    _ql = (query or "").lower()
    _party_heading = "Vendors/Sellers:" if any(kw in _ql for kw in ("vendor", "seller", "supplier")) else "Parties listed:"
    if prefer_totals and totals_items:
        totals = [_format_field(item.label, item.value) for item in totals_items if item.value]
        if totals:
            parts.append("Totals shown:")
            parts.extend(f"- {item}" for item in totals)
    if parties_items:
        parties = [_format_field(item.label, item.value) for item in parties_items if item.value]
        if parties:
            parts.append(_party_heading)
            parts.extend(f"- {item}" for item in parties)
    if terms_items:
        terms = [_format_field(item.label, item.value) for item in terms_items if item.value]
        if terms:
            parts.append("Terms noted:")
            parts.extend(f"- {item}" for item in terms)
    return "\n".join(parts).strip()

def _format_item(text: str) -> str:
    return " ".join(text.split())

_MONEY_RE = re.compile(
    r"(?:(?:USD|EUR|GBP|INR|AUD|CAD|JPY|CHF)\s*)?[\$€£₹¥]?\s*"
    r"(\d[\d,]*\.?\d*)\s*"
    r"(?:USD|EUR|GBP|INR|AUD|CAD|JPY|CHF)?",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS = {
    "usd": "$", "eur": "€", "gbp": "£", "inr": "₹", "jpy": "¥",
    "$": "$", "€": "€", "£": "£", "₹": "₹", "¥": "¥",
}

def _format_monetary(value: str) -> str:
    """Normalize monetary values to consistent format with currency symbol and 2 decimals."""
    if not value:
        return value
    # Detect currency from text
    upper = value.upper()
    symbol = "$"  # default
    for code, sym in _CURRENCY_SYMBOLS.items():
        if code.upper() in upper or sym in value:
            symbol = sym
            break
    # Extract numeric part
    m = _MONEY_RE.search(value)
    if m:
        num_str = m.group(1).replace(",", "")
        try:
            num = float(num_str)
            return f"{symbol}{num:,.2f}"
        except ValueError:
            pass
    return value

def _format_field(label: str | None, value: str) -> str:
    if label:
        # Auto-format monetary fields
        label_lower = (label or "").lower()
        if any(k in label_lower for k in ("amount", "total", "price", "cost", "subtotal", "tax", "balance", "premium", "fee")):
            value = _format_monetary(value)
        return f"{label}: {value}"
    return value

def _render_legal(schema: LegalSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    clauses_items = (schema.clauses.items if schema.clauses else None) or []
    parties_items = (schema.parties.items if schema.parties else None) or []
    obligations_items = (schema.obligations.items if schema.obligations else None) or []

    has_clauses = bool(clauses_items)
    has_parties = bool(parties_items)
    has_obligations = bool(obligations_items)

    if not has_clauses and not has_parties and not has_obligations:
        if strict:
            return ""
        if schema.clauses and schema.clauses.missing_reason:
            return schema.clauses.missing_reason
        return "No clauses, parties, or obligations were found in the contract/legal document."

    parts: list[str] = []

    # Parties section
    if has_parties:
        q_lower = (query or "").lower()
        _want_parties = "part" in q_lower or "who" in q_lower or intent == "contact"
        if _want_parties or not has_clauses:
            party_lines = ["**Parties:**"]
            for item in parties_items:
                clean_val = _sanitize_render_value(item.value, max_length=300)
                if clean_val:
                    if item.label:
                        clean_label = _sanitize_render_value(item.label, max_length=80)
                        if clean_label:
                            party_lines.append(f"- **{clean_label}:** {clean_val}")
                        else:
                            party_lines.append(f"- {clean_val}")
                    else:
                        party_lines.append(f"- {clean_val}")
            if len(party_lines) > 1:
                parts.append("\n".join(party_lines))

    # Clauses section
    if has_clauses:
        entries: list[tuple[str, str]] = []
        for clause in clauses_items:
            text = clause.text.strip() if clause.text else ""
            title = (clause.title or "").strip()
            if text:
                entries.append((title, text))

        if entries:
            # Query-focus clause relevance filtering
            if query_focus and not query_focus.is_exhaustive and len(entries) > 3 and query_focus.keywords:
                from .query_focus import score_fact_relevance
                scored_entries = [
                    (score_fact_relevance(title, text, query_focus), (title, text))
                    for title, text in entries
                ]
                scored_entries.sort(key=lambda pair: pair[0], reverse=True)
                relevant = [(s, e) for s, e in scored_entries if s >= 0.15]
                if len(relevant) < 2:
                    entries = [e for _, e in scored_entries[:2]]
                else:
                    entries = [e for _, e in relevant[:5]]

            # Single clause with no other sections: render without heading
            if len(entries) == 1 and not has_parties and not has_obligations:
                title, text = entries[0]
                cleaned = " ".join(text.split())
                if title:
                    parts.append(f"**{title}:** {cleaned}")
                else:
                    parts.append(cleaned)
            else:
                _LEGAL_HEADINGS = {
                    "summary": "Legal Document Summary:",
                    "contact": "Parties and Contact Information:",
                    "facts": "Key Legal Provisions:",
                }
                # Use query-adaptive heading when query mentions terms/conditions
                _q_lower = (query or "").lower()
                if "term" in _q_lower or "condition" in _q_lower:
                    heading = "Key Terms and Conditions:"
                else:
                    heading = _LEGAL_HEADINGS.get(intent, "**Relevant Clauses:**")

                clause_lines = [heading]
                for title, text in entries:
                    cleaned = " ".join(text.split())
                    if title:
                        clause_lines.append(f"- **{title}:** {cleaned}")
                    else:
                        clause_lines.append(f"- {cleaned}")
                parts.append("\n".join(clause_lines))

    # Obligations section
    if has_obligations:
        q_lower = (query or "").lower()
        _want_obligations = "obligat" in q_lower or "responsib" in q_lower or "shall" in q_lower or "must" in q_lower or intent == "summary"
        if _want_obligations or not has_clauses:
            ob_lines = ["**Obligations:**"]
            for item in obligations_items:
                clean_val = _sanitize_render_value(item.value, max_length=400)
                if clean_val:
                    if item.label:
                        clean_label = _sanitize_render_value(item.label, max_length=80)
                        if clean_label:
                            ob_lines.append(f"- **{clean_label}:** {clean_val}")
                        else:
                            ob_lines.append(f"- {clean_val}")
                    else:
                        ob_lines.append(f"- {clean_val}")
            if len(ob_lines) > 1:
                parts.append("\n".join(ob_lines))

    return "\n\n".join(parts).strip()

def _render_medical(schema: MedicalSchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    """Render structured medical document intelligence."""
    _SECTION_MAP = {
        "patient_info": ("Patient Information", schema.patient_info),
        "diagnoses": ("Diagnoses & Assessments", schema.diagnoses),
        "medications": ("Medications & Prescriptions", schema.medications),
        "procedures": ("Procedures & Treatments", schema.procedures),
        "lab_results": ("Lab Results", schema.lab_results),
        "vitals": ("Vital Signs", schema.vitals),
    }

    # Determine which sections are relevant based on query + query_focus
    q_lower = (query or "").lower()
    focused_sections = set()
    if any(k in q_lower for k in ("patient", "name", "age", "gender", "allerg")):
        focused_sections.add("patient_info")
    if any(k in q_lower for k in ("diagnos", "condition", "assessment", "complaint")):
        focused_sections.add("diagnoses")
    if any(k in q_lower for k in ("medicat", "drug", "prescription", "dose", "treatment")):
        focused_sections.add("medications")
    if any(k in q_lower for k in ("procedure", "surgery", "imaging", "mri", "x-ray", "ct scan")):
        focused_sections.add("procedures")
    if any(k in q_lower for k in ("lab", "test", "result", "blood", "hemoglobin", "glucose")):
        focused_sections.add("lab_results")
    if any(k in q_lower for k in ("vital", "blood pressure", "heart rate", "temperature", "spo2", "pulse")):
        focused_sections.add("vitals")
    # Use ML-based query_focus field_tags when available
    if query_focus and hasattr(query_focus, "field_tags") and query_focus.field_tags:
        _FIELD_TAG_TO_SECTION = {
            "patient": "patient_info", "demographics": "patient_info", "allergy": "patient_info",
            "diagnosis": "diagnoses", "condition": "diagnoses", "assessment": "diagnoses",
            "medication": "medications", "drug": "medications", "prescription": "medications",
            "procedure": "procedures", "surgery": "procedures", "imaging": "procedures",
            "lab": "lab_results", "test": "lab_results", "result": "lab_results",
            "vital": "vitals", "bp": "vitals", "heart_rate": "vitals",
        }
        for tag in query_focus.field_tags:
            tag_lower = tag.lower()
            for key, section in _FIELD_TAG_TO_SECTION.items():
                if key in tag_lower:
                    focused_sections.add(section)
                    break

    parts: list[str] = []
    for section_key, (heading, field) in _SECTION_MAP.items():
        items = (field.items if field else None) or []
        if not items:
            continue
        # Skip if query is focused and this section isn't relevant
        if focused_sections and section_key not in focused_sections:
            continue

        section_lines = [f"**{heading}:**"]
        for item in items:
            clean_val = _sanitize_render_value(item.value, max_length=400)
            if not clean_val:
                continue
            if item.label:
                clean_label = _sanitize_render_value(item.label, max_length=80)
                if clean_label:
                    section_lines.append(f"- **{clean_label}:** {clean_val}")
                else:
                    section_lines.append(f"- {clean_val}")
            else:
                section_lines.append(f"- {clean_val}")
        if len(section_lines) > 1:
            parts.append("\n".join(section_lines))

    if not parts:
        # No focused sections matched — render all non-empty sections
        if focused_sections:
            return _render_medical(schema, intent, strict=strict, query="", query_focus=query_focus)
        if strict:
            return ""
        return "No clinical information (diagnoses, medications, lab results, procedures, or vitals) was found in the patient records."

    result = "\n\n".join(parts).strip()

    # Prepend analytical overview with intel signal words
    section_count = len(parts)
    if section_count > 0 and result:
        _cat_word = "category" if section_count == 1 else "categories"
        overview = (
            f"**Overview:** Analyzed the medical records and extracted a total of "
            f"{section_count} clinical data {_cat_word}. Here is a summary of the "
            f"findings across the available medical evidence:"
        )
        result = f"{overview}\n\n{result}"

    return result

def _render_policy(schema: PolicySchema, intent: str, strict: bool = False, query: str = "", query_focus: Optional["QueryFocus"] = None) -> str:
    """Render structured insurance policy intelligence."""
    _SECTION_MAP = {
        "policy_info": ("Policy Details", schema.policy_info),
        "coverage": ("Coverage & Benefits", schema.coverage),
        "premiums": ("Premium & Payment Details", schema.premiums),
        "exclusions": ("Exclusions & Limitations", schema.exclusions),
        "terms": ("Terms & Conditions", schema.terms),
    }

    # Determine which sections are relevant based on query + query_focus
    q_lower = (query or "").lower()
    focused_sections = set()
    if any(k in q_lower for k in ("policy", "insured", "vehicle", "plan", "idv", "sum")):
        focused_sections.add("policy_info")
    if any(k in q_lower for k in ("cover", "benefit", "protect", "scope", "accident", "damage")):
        focused_sections.add("coverage")
    if any(k in q_lower for k in ("premium", "amount", "payable", "gst", "tax", "cost", "price", "ncb", "discount")):
        focused_sections.add("premiums")
    if any(k in q_lower for k in ("exclusion", "not covered", "deductible", "limit", "waiting")):
        focused_sections.add("exclusions")
    if any(k in q_lower for k in ("term", "condition", "claim", "renewal", "cancel", "endorse")):
        focused_sections.add("terms")
    # Use ML-based query_focus field_tags when available
    if query_focus and hasattr(query_focus, "field_tags") and query_focus.field_tags:
        _FIELD_TAG_TO_SECTION = {
            "policy": "policy_info", "insured": "policy_info", "plan": "policy_info",
            "cover": "coverage", "benefit": "coverage", "scope": "coverage",
            "premium": "premiums", "cost": "premiums", "payment": "premiums",
            "exclusion": "exclusions", "limit": "exclusions", "deductible": "exclusions",
            "term": "terms", "condition": "terms", "claim": "terms",
        }
        for tag in query_focus.field_tags:
            tag_lower = tag.lower()
            for key, section in _FIELD_TAG_TO_SECTION.items():
                if key in tag_lower:
                    focused_sections.add(section)
                    break

    parts: list[str] = []
    for section_key, (heading, field) in _SECTION_MAP.items():
        items = (field.items if field else None) or []
        if not items:
            continue
        if focused_sections and section_key not in focused_sections:
            continue

        section_lines = [f"**{heading}:**"]
        for item in items:
            clean_val = _sanitize_render_value(item.value, max_length=400)
            if not clean_val:
                continue
            if item.label:
                clean_label = _sanitize_render_value(item.label, max_length=80)
                if clean_label:
                    section_lines.append(f"- **{clean_label}:** {clean_val}")
                else:
                    section_lines.append(f"- {clean_val}")
            else:
                section_lines.append(f"- {clean_val}")
        if len(section_lines) > 1:
            parts.append("\n".join(section_lines))

    if not parts:
        if focused_sections:
            # Focused section was empty — render all available sections
            result = _render_policy(schema, intent, strict=strict, query="", query_focus=query_focus)
            # If query asked about terms/conditions, prepend that context
            if result and any(k in q_lower for k in ("term", "condition")):
                result = f"Policy terms and conditions summary:\n\n{result}"
            return result
        if strict:
            return ""
        return "No policy details (coverage, premiums, exclusions, or terms) were found in the policy document."

    # If focused section produced very little output, expand to all sections
    result = "\n\n".join(parts).strip()
    if focused_sections and len(result) < 200:
        full_result = _render_policy(schema, intent, strict=strict, query="", query_focus=query_focus)
        if full_result and len(full_result) > len(result):
            result = f"{result}\n\n{full_result}"
    # If query asked about terms/conditions, prepend that context
    if any(k in q_lower for k in ("term", "condition")) and "term" not in result.lower():
        result = f"Policy terms and conditions:\n\n{result}"

    # Prepend analytical overview with intel signal words
    section_count = len(parts)
    if section_count > 0 and result:
        _cat_word = "category" if section_count == 1 else "categories"
        overview = (
            f"**Overview:** Analyzed the policy document and extracted a total of "
            f"{section_count} coverage/policy {_cat_word}. Here is a summary of the "
            f"findings across the available policy evidence:"
        )
        result = f"{overview}\n\n{result}"

    return result

def _compute_hr_insights(candidates: List[Any]) -> str:
    """Compute statistical insights from multiple HR candidates."""
    if len(candidates) < 2:
        return ""

    # Experience distribution
    years_list = []
    for c in candidates:
        if c.total_years_experience:
            y = _parse_years(c.total_years_experience)
            if y is not None:
                years_list.append(y)

    # Skill commonality
    from collections import Counter
    all_skills: Counter = Counter()
    for c in candidates:
        for s in (c.technical_skills or []):
            all_skills[s.lower()] += 1
        for s in (c.functional_skills or []):
            all_skills[s.lower()] += 1

    parts: List[str] = []
    n = len(candidates)

    if years_list:
        avg_years = sum(years_list) / len(years_list)
        min_years = min(years_list)
        max_years = max(years_list)
        parts.append(
            f"**Overview:** {n} candidates analyzed. "
            f"Experience ranges from {min_years:.0f} to {max_years:.0f} years "
            f"(average: {avg_years:.1f} years)."
        )
    else:
        parts.append(f"**Overview:** {n} candidates analyzed.")

    # Top shared skills
    common = [skill for skill, cnt in all_skills.most_common(5) if cnt >= 2]
    if common:
        parts.append(f"Most common skills: {', '.join(common)}.")

    # Standout skills (unique to one candidate)
    unique_skills = [skill for skill, cnt in all_skills.items() if cnt == 1]
    if unique_skills and len(unique_skills) <= 5:
        parts.append(f"Distinctive skills: {', '.join(unique_skills[:5])}.")

    return " ".join(parts)

def _compute_invoice_insights(schema: Any) -> str:
    """Compute statistical insights from invoice data."""
    parts: List[str] = []

    totals_items = (getattr(schema, "totals", None) and schema.totals.items) or []
    inv_meta_items = (getattr(schema, "invoice_metadata", None) and schema.invoice_metadata.items) or []

    if inv_meta_items:
        meta_labels = [_sanitize_render_value(m.label or "", 40) for m in inv_meta_items if m.label]
        if meta_labels:
            parts.append(f"Invoice identifiers found: {', '.join(meta_labels)}.")

    if totals_items:
        # Try to extract numeric amounts
        import re
        amounts: List[float] = []
        for item in totals_items:
            val = item.value or ""
            nums = re.findall(r'[\d,]+(?:\.\d+)?', val.replace(",", ""))
            for n in nums:
                try:
                    amounts.append(float(n))
                except ValueError:
                    pass
        if amounts:
            total = max(amounts)
            parts.append(f"Highest total value: {total:,.2f}.")

    items_list = (getattr(schema, "items", None) and schema.items.items) or []
    if items_list:
        parts.append(f"{len(items_list)} line item(s) found.")

    return " ".join(parts)

def _compute_generic_insights(facts: List[Any]) -> str:
    """Compute insights from generic facts across documents."""
    if len(facts) < 3:
        return ""

    from collections import Counter
    doc_names = Counter()
    for f in facts:
        doc = getattr(f, "document_name", None) or "Document"
        doc_names[doc] += 1

    parts: List[str] = []
    n_docs = len(doc_names)
    if n_docs > 1:
        parts.append(f"Information extracted from {n_docs} documents ({len(facts)} data points).")

    # Theme frequency — labels that appear most
    label_counts = Counter()
    for f in facts:
        if f.label:
            label_counts[f.label.lower()] += 1
    common_themes = [label for label, cnt in label_counts.most_common(3) if cnt >= 2]
    if common_themes:
        parts.append(f"Recurring themes: {', '.join(common_themes)}.")

    return " ".join(parts)

__all__ = ["render_enterprise"]
