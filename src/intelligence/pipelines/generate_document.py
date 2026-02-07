from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from src.intelligence.pipelines.common import build_sources
from src.utils.payload_utils import get_content_text
from src.kg.entity_extractor import EntityExtractor


def _find_years_experience(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b\d{1,2}\s+years\b", text.lower())


def _supported_in_chunks(value: str, chunks: Iterable[Any]) -> bool:
    value = value.lower()
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        text = get_content_text(meta) or getattr(chunk, "text", "") or ""
        if value in text.lower():
            return True
    return False


def _find_citation(value: str, chunks: Iterable[Any]) -> str:
    value = value.lower()
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        text = get_content_text(meta) or getattr(chunk, "text", "") or ""
        if value in text.lower():
            name = meta.get("source_name") or (meta.get("source") or {}).get("name") or ""
            page = meta.get("page") or meta.get("page_start") or meta.get("page_number")
            if name and page:
                return f"({name}, p. {page})"
            if name:
                return f"({name})"
            return ""
    return ""


def _extract_facts(chunks: Iterable[Any]) -> Dict[str, List[str]]:
    extractor = EntityExtractor()
    skills: List[str] = []
    orgs: List[str] = []
    persons: List[str] = []
    years: List[str] = []
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        text = get_content_text(meta) or getattr(chunk, "text", "") or ""
        if not text:
            continue
        years.extend(_find_years_experience(text))
        for ent in extractor.extract_with_metadata(text):
            if ent.type == "SKILL":
                skills.append(ent.name)
            if ent.type == "ORGANIZATION":
                orgs.append(ent.name)
            if ent.type == "PERSON":
                persons.append(ent.name)
    return {
        "skills": _dedupe(skills),
        "orgs": _dedupe(orgs),
        "persons": _dedupe(persons),
        "years": _dedupe(years),
    }


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value.strip())
    return output


def generate_document_task(
    *,
    query: str,
    chunks: List[Any],
    target_person: Optional[str] = None,
) -> Dict[str, Any]:
    if not chunks:
        return {
            "response": "Insufficient grounded evidence to draft a document.",
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {"task": "generate_document", "query": query},
        }

    facts = _extract_facts(chunks)
    name = target_person or (facts.get("persons") or ["Candidate"])[0]

    skills = [s for s in facts.get("skills", []) if _supported_in_chunks(s, chunks)][:4]
    orgs = [o for o in facts.get("orgs", []) if _supported_in_chunks(o, chunks)][:2]
    years = [y for y in facts.get("years", []) if _supported_in_chunks(y, chunks)][:1]

    if not skills and not orgs and not years:
        return {
            "response": "I couldn't find enough evidence in the documents to draft a cover letter.",
            "sources": build_sources(chunks),
            "grounded": False,
            "context_found": True,
            "metadata": {"task": "generate_document", "query": query, "reason": "no_supported_facts"},
        }

    intro = "Dear Hiring Manager,\n\nI am writing to express interest in the role."
    if years:
        citation = _find_citation(years[0], chunks)
        intro += f" The candidate brings {years[0]} of experience relevant to this position{(' ' + citation) if citation else ''}."

    body_lines = []
    if skills:
        citation = _find_citation(skills[0], chunks)
        body_lines.append(
            f"Key skills evidenced in the documents include {', '.join(skills)}{(' ' + citation) if citation else ''}."
        )
    if orgs:
        citation = _find_citation(orgs[0], chunks)
        body_lines.append(
            f"Experience includes work with {', '.join(orgs)}{(' ' + citation) if citation else ''}."
        )

    closing = (
        "\nI would welcome the opportunity to discuss how these strengths align with your needs."
        f"\n\nSincerely,\n{name}"
    )

    response = "\n".join([intro, "\n".join(body_lines), closing]).strip()
    return {
        "response": response,
        "sources": build_sources(chunks),
        "grounded": True,
        "context_found": True,
        "metadata": {"task": "generate_document", "query": query},
    }


__all__ = ["generate_document_task"]
