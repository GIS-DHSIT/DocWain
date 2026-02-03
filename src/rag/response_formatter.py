from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from src.rag.context_reasoning import KeyFact, NumericClaim, WorkingContext
from src.rag.evidence_selector import select_evidence_lines


@dataclass
class FormattedResponse:
    text: str
    used_chunk_ids: List[str]


def _ensure_period(text: str) -> str:
    if not text:
        return text
    if text.endswith((".", "!", "?")):
        return text
    return text + "."


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text or "")
    return len([p for p in parts if p.strip()])


def _doc_list(doc_names: Sequence[str]) -> str:
    names = [d for d in doc_names if d]
    if not names:
        return "the retrieved sections for this session"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:2])}, and others"


def _dedupe(items: Iterable[Tuple[str, str, str]], limit: int) -> List[Tuple[str, str, str]]:
    seen = set()
    output = []
    for doc_name, label, value in items:
        key = (doc_name.lower(), label.lower(), value.lower())
        if key in seen:
            continue
        seen.add(key)
        output.append((doc_name, label, value))
        if len(output) >= limit:
            break
    return output


def _facts_from_context(context: WorkingContext) -> List[Tuple[str, str, str, str]]:
    facts: List[Tuple[str, str, str, str]] = []
    for fact in context.key_facts:
        facts.append((fact.doc_name, fact.label, fact.value, fact.chunk_id))
    for claim in context.numeric_claims:
        facts.append((claim.doc_name, claim.label, claim.value, claim.chunk_id))
    return facts


def _filter_total_claims(claims: Sequence[NumericClaim]) -> List[NumericClaim]:
    filtered = []
    for claim in claims:
        label = (claim.label or "").lower()
        if any(token in label for token in ("total", "subtotal", "amount due", "balance")):
            filtered.append(claim)
    return filtered


def format_structured_response(
    *,
    query: str,
    intent: str,
    context: WorkingContext,
    doc_names: Sequence[str],
    assumption_line: Optional[str] = None,
    item_evidence: Optional[Sequence[Tuple[str, str]]] = None,
    strict: bool = False,
    multi_doc: bool = False,
    only_one_doc: bool = False,
) -> FormattedResponse:
    used_chunk_ids: List[str] = []
    opening_sentences: List[str] = []

    if assumption_line:
        opening_sentences.append(_ensure_period(assumption_line))

    if intent == "PRODUCTS_SERVICES":
        if item_evidence:
            opening_sentences.append(
                "Here are the product or service items I found in the retrieved sections."
            )
        else:
            opening_sentences.append(
                "I couldn’t find explicit product or service line items in the retrieved sections."
            )
    elif intent == "TOTALS":
        opening_sentences.append("Here are the totals stated in the retrieved sections.")
    elif intent == "COMPARE":
        opening_sentences.append("Here is a comparison across the retrieved documents.")
    elif intent == "SUMMARIZE":
        opening_sentences.append("Here is a summary of the retrieved documents.")
    else:
        opening_sentences.append("Here is what the retrieved sections state about your question.")

    opening_sentences.append(f"I used {_doc_list(doc_names)} as evidence.")
    if multi_doc and only_one_doc:
        opening_sentences[-1] = f"{opening_sentences[-1].rstrip('.')} and only one document was retrieved for this request."

    opening = " ".join(_ensure_period(s) for s in opening_sentences[:3])

    evidence_lines: List[str] = []
    fact_items = _facts_from_context(context)

    if intent == "PRODUCTS_SERVICES":
        items = []
        for name, chunk_id in item_evidence or []:
            if not name:
                continue
            items.append((name, chunk_id))
        seen = set()
        for name, chunk_id in items:
            key = name.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            evidence_lines.append(f"- {name}.")
            if chunk_id:
                used_chunk_ids.append(str(chunk_id))
            if len(evidence_lines) >= 7:
                break
        if not evidence_lines:
            evidence_lines.append("- No explicit product or service line items were visible in the retrieved sections.")
    elif intent == "TOTALS":
        total_claims = _filter_total_claims(context.numeric_claims)
        for claim in total_claims[:7]:
            evidence_lines.append(f"- {claim.label}: {claim.value}.")
            if claim.chunk_id:
                used_chunk_ids.append(str(claim.chunk_id))
        if not evidence_lines:
            evidence_lines.append("- No explicit totals were visible in the retrieved sections.")
    elif intent in {"COMPARE", "SUMMARIZE"}:
        grouped: List[Tuple[str, str, str, str]] = []
        for doc_name, label, value, chunk_id in fact_items:
            grouped.append((doc_name, label, value, chunk_id))
        seen = set()
        for doc_name, label, value, chunk_id in grouped:
            key = (doc_name.lower(), label.lower(), value.lower())
            if key in seen:
                continue
            seen.add(key)
            evidence_lines.append(f"- {doc_name}: {label} is {value}.")
            if chunk_id:
                used_chunk_ids.append(str(chunk_id))
            if len(evidence_lines) >= 7:
                break
        if not evidence_lines:
            evidence_lines.append("- No labeled facts were visible in the retrieved sections.")
    else:
        grouped: List[Tuple[str, str, str, str]] = []
        for doc_name, label, value, chunk_id in fact_items:
            grouped.append((doc_name, label, value, chunk_id))
        seen = set()
        for _doc_name, label, value, chunk_id in grouped:
            key = (label.lower(), value.lower())
            if key in seen:
                continue
            seen.add(key)
            evidence_lines.append(f"- {label}: {value}.")
            if chunk_id:
                used_chunk_ids.append(str(chunk_id))
            if len(evidence_lines) >= 7:
                break
        if not evidence_lines:
            evidence_lines.append("- No labeled facts were visible in the retrieved sections.")

    if strict:
        evidence_lines = [re.sub(r"[$€£¥]?\d[\d,]*(?:\.\d+)?", "a stated value", line) for line in evidence_lines]

    if intent not in {"PRODUCTS_SERVICES", "TOTALS"}:
        selected = select_evidence_lines(query=query, context=context, max_lines_per_doc=3)
        if selected:
            evidence_lines = selected
        else:
            evidence_lines = evidence_lines[:3]
    else:
        evidence_lines = evidence_lines[:3]

    takeaways: List[str] = []
    if intent in {"COMPARE", "SUMMARIZE"} and only_one_doc:
        takeaways.append("If you want a cross-document view, specify additional documents to include.")
    if intent == "PRODUCTS_SERVICES" and not item_evidence:
        takeaways.append("I can broaden retrieval if you specify the invoice(s) or section to scan.")
    if intent == "TOTALS" and not _filter_total_claims(context.numeric_claims):
        takeaways.append("If totals exist elsewhere, point me to the relevant invoice or section.")
    if not takeaways:
        takeaways.append("Tell me if you want a deeper breakdown by document or section.")
        takeaways.append("If you need a side-by-side view, specify the fields to compare.")
    takeaways = takeaways[:3]

    evidence_block = ""
    if evidence_lines:
        evidence_block = "Evidence:\n" + "\n".join(_ensure_period(line) for line in evidence_lines)
    takeaway_block = "Takeaways:\n" + "\n".join(f"- {_ensure_period(line.lstrip('- ').strip())}" for line in takeaways)

    closing = _ensure_period(f"Documents used: {_doc_list(doc_names)}")

    blocks = [opening]
    if evidence_block:
        blocks.append(evidence_block)
    blocks.extend([takeaway_block, closing])
    assembled = "\n\n".join(blocks).strip()

    if _sentence_count(assembled) < 6:
        assembled = assembled + " " + _ensure_period("Let me know if you want me to focus on a different document or section")

    return FormattedResponse(text=assembled, used_chunk_ids=used_chunk_ids)


def _list_or_missing(items: Sequence[str]) -> str:
    if not items:
        return "Not stated in the retrieved sections."
    return ", ".join(items)


def _candidate_name(profile: dict) -> str:
    name = profile.get("candidate_name")
    return name or "Candidate"


def format_candidate_profile_response(
    *,
    profile: dict,
    assumption_line: Optional[str] = None,
) -> str:
    name = _candidate_name(profile)
    years = profile.get("total_years_experience")
    source_doc = profile.get("source_document") or "the retrieved document"
    intro = []
    if assumption_line:
        intro.append(_ensure_period(assumption_line))
    intro.append(f"I extracted a candidate profile from {source_doc}.")
    if years is not None:
        intro.append(f"{name} is associated with approximately {years:g} years of experience in the retrieved sections.")
    else:
        intro.append(f"{name}'s total years of experience were not stated in the retrieved sections.")
    summary = profile.get("experience_summary")
    if summary:
        intro.append(summary)
    else:
        intro.append("Key qualifications are summarized below based on the available evidence.")
    intro_block = " ".join(_ensure_period(s) for s in intro[:3])

    bullets = [
        f"- Technical skills: {_list_or_missing(profile.get('technical_skills') or [])}.",
        f"- Functional skills: {_list_or_missing(profile.get('functional_skills') or [])}.",
        f"- Education: {_list_or_missing(profile.get('education') or [])}.",
        f"- Certifications: {_list_or_missing(profile.get('certifications') or [])}.",
        f"- Awards: {_list_or_missing(profile.get('achievements_awards') or [])}.",
    ]

    takeaways = [
        "Missing details: ask for any gaps in role history or project dates.",
        "Follow-up: confirm preferred roles and availability for the next stage.",
    ]

    closing = _ensure_period(f"Documents used: {_doc_list([source_doc])}")
    assembled = "\n\n".join(
        [
            intro_block,
            "Details:\n" + "\n".join(bullets),
            "Takeaways:\n" + "\n".join(f"- {_ensure_period(t)}" for t in takeaways),
            closing,
        ]
    ).strip()
    if _sentence_count(assembled) < 6:
        assembled = assembled + " " + _ensure_period("Let me know if you want a deeper breakdown by section")
    return assembled


def format_multi_candidate_response(
    *,
    profiles: Sequence[dict],
    assumption_line: Optional[str] = None,
    ranking: Optional[Sequence[Tuple[dict, float, str]]] = None,
) -> str:
    intro = []
    if assumption_line:
        intro.append(_ensure_period(assumption_line))
    intro.append(f"I extracted {len(profiles)} candidate profiles across the retrieved documents.")
    intro.append("Each candidate summary below is grounded in the retrieved sections.")
    intro_block = " ".join(_ensure_period(s) for s in intro[:3])

    blocks = []
    if ranking:
        ranking_lines = []
        for idx, (profile, score, rationale) in enumerate(ranking, start=1):
            ranking_lines.append(
                f"{idx}. {_candidate_name(profile)} — score {score:.2f}. {rationale}"
            )
        blocks.append("Ranking:\n" + "\n".join(ranking_lines))

    for profile in profiles:
        name = _candidate_name(profile)
        source_doc = profile.get("source_document") or "Unknown document"
        years = profile.get("total_years_experience")
        years_text = f"{years:g} years" if years is not None else "Not stated in the retrieved sections."
        details = [
            f"- Experience: {years_text}.",
            f"- Summary: {_ensure_period(profile.get('experience_summary') or 'Not stated in the retrieved sections.')}",
            f"- Technical skills: {_list_or_missing(profile.get('technical_skills') or [])}.",
            f"- Functional skills: {_list_or_missing(profile.get('functional_skills') or [])}.",
            f"- Education: {_list_or_missing(profile.get('education') or [])}.",
        ]
        blocks.append(f"{name} (Source: {source_doc})\n" + "\n".join(details))

    takeaways = [
        "Top strengths across the pool appear in the skills lists above.",
        "Missing information should be requested for candidates with unspecified experience years.",
        "Ranking is computed from experience, skill coverage, and role consistency signals.",
    ]
    closing = _ensure_period(f"Documents used: {_doc_list([p.get('source_document') for p in profiles if p.get('source_document')])}")

    assembled = "\n\n".join(
        [
            intro_block,
            *blocks,
            "Takeaways:\n" + "\n".join(f"- {_ensure_period(t)}" for t in takeaways[:3]),
            closing,
        ]
    ).strip()
    if _sentence_count(assembled) < 6:
        assembled = assembled + " " + _ensure_period("Ask if you want the ranking adjusted to specific skills")
    return assembled


def format_conservative_response(
    *,
    intent: str,
    doc_names: Sequence[str],
    include_products_message: bool = False,
) -> FormattedResponse:
    opening = "I couldn’t verify a fully grounded answer from the retrieved sections."
    if intent == "PRODUCTS_SERVICES" and include_products_message:
        opening = (
            "I couldn’t find explicit product or service line items in the retrieved sections; "
            "I can broaden retrieval or you can specify the invoice(s)."
        )
    evidence_block = "Evidence:\n- The retrieved sections did not contain enough labeled detail for a confident answer."
    takeaway_block = "Takeaways:\n- Please specify the document or section to search.\n- I can expand the retrieval scope if needed."
    closing = _ensure_period(f"Documents used: {_doc_list(doc_names)}")
    text = "\n\n".join([_ensure_period(opening), evidence_block, takeaway_block, closing]).strip()
    if _sentence_count(text) < 6:
        text = text + " " + _ensure_period("Tell me the exact invoice or section to target next")
    return FormattedResponse(text=text, used_chunk_ids=[])


__all__ = [
    "FormattedResponse",
    "format_structured_response",
    "format_conservative_response",
    "format_candidate_profile_response",
    "format_multi_candidate_response",
]
