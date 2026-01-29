from __future__ import annotations

import re
from typing import Iterable, List, Tuple


def extract_required_attributes(query: str, intent_type: str) -> List[str]:
    lowered = (query or "").lower()
    attributes: List[str] = []
    education_terms = [
        "education", "educational", "qualification", "degree", "college", "university",
        "school", "academic", "academics", "b.tech", "btech", "b.e", "be",
        "m.tech", "mtech", "mba", "phd", "doctorate", "diploma",
    ]
    experience_terms = [
        "experience", "work history", "employment", "career", "role", "position",
        "company", "employer", "years",
    ]
    skills_terms = ["skill", "skills", "competency", "expertise", "proficiency"]
    certification_terms = ["certification", "certificate", "license", "credential"]
    contact_terms = ["email", "phone", "contact", "address", "location", "linkedin"]

    if any(term in lowered for term in education_terms):
        attributes.extend(education_terms)
    if any(term in lowered for term in experience_terms):
        attributes.extend(experience_terms)
    if any(term in lowered for term in skills_terms):
        attributes.extend(skills_terms)
    if any(term in lowered for term in certification_terms):
        attributes.extend(certification_terms)
    if any(term in lowered for term in contact_terms):
        attributes.extend(contact_terms)

    if intent_type in {"numeric_lookup", "field_extraction"} and not attributes:
        attributes.extend(["total", "amount", "number", "count", "date", "year"])

    if not attributes:
        return []
    deduped = []
    for term in attributes:
        if term not in deduped:
            deduped.append(term)
    return deduped[:10]


def filter_chunks_by_intent(
    chunks: Iterable[object],
    required_attributes: List[str],
    entities: List[str],
    intent_type: str,
) -> List[object]:
    chunks_list = list(chunks or [])
    if not chunks_list or not required_attributes:
        return chunks_list
    filtered: List[object] = []
    attr_terms = [term.lower() for term in required_attributes if term]
    entity_terms = [ent.lower() for ent in entities if ent]
    number_re = re.compile(r"\b\d+[\d,\.]*\b")

    for chunk in chunks_list:
        text = (getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else "") or "").lower()
        meta = getattr(chunk, "metadata", None) or (chunk.get("metadata") if isinstance(chunk, dict) else {}) or {}
        section = str(meta.get("section_title") or meta.get("section_path") or meta.get("section") or "").lower()
        attr_hits = sum(1 for term in attr_terms if term in text or term in section)
        entity_hits = sum(1 for ent in entity_terms if ent and ent in text)
        numeric_hit = bool(number_re.search(text))

        valid = attr_hits > 0 or (intent_type == "numeric_lookup" and numeric_hit)
        score = attr_hits + (0.5 * entity_hits) + (0.4 if numeric_hit else 0.0)

        if isinstance(chunk, dict):
            meta.setdefault("intent_match_score", round(score, 3))
            chunk["metadata"] = meta
        else:
            meta["intent_match_score"] = round(score, 3)
            setattr(chunk, "metadata", meta)
        if valid:
            filtered.append(chunk)

    return filtered


def extract_answer_requirements(query: str, intent_type: str) -> List[Tuple[str, re.Pattern]]:
    lowered = (query or "").lower()
    requirements: List[Tuple[str, re.Pattern]] = []
    if any(term in lowered for term in ("education", "educational", "qualification", "degree", "college", "university", "school")):
        requirements.append(("degree", re.compile(r"\b(b\.?tech|b\.?e|bachelor|master|m\.?tech|m\.?sc|mba|phd|doctorate|diploma)\b", re.IGNORECASE)))
        requirements.append(("institution", re.compile(r"\b(university|college|institute|school|academy|polytechnic)\b", re.IGNORECASE)))
        if any(term in lowered for term in ("year", "when", "date", "graduat")):
            requirements.append(("year", re.compile(r"\b(19|20)\d{2}\b")))
    if intent_type == "field_extraction":
        if "email" in lowered:
            requirements.append(("email", re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)))
        if "phone" in lowered or "contact" in lowered:
            requirements.append(("phone", re.compile(r"\+?\d[\d\s().-]{6,}\d")))
    return requirements


def validate_answer_requirements(chunks: Iterable[object], requirements: List[Tuple[str, re.Pattern]]) -> List[str]:
    if not requirements:
        return []
    combined = " ".join(
        (getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else "") or "")
        for chunk in chunks
        if chunk
    )
    missing: List[str] = []
    for name, pattern in requirements:
        if not pattern.search(combined):
            missing.append(name)
    return missing


def build_intent_miss_response(entity: str, intent_label: str) -> str:
    entity = (entity or "").strip()
    if entity and intent_label:
        return f"I couldn't find {intent_label} details for {entity} in the available documents."
    if intent_label:
        return f"I couldn't find {intent_label} details in the available documents."
    return "I couldn't find that information in the available documents."
