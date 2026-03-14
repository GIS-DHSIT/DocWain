from __future__ import annotations

import re
from typing import Iterable, List, Pattern, Tuple

from src.utils.logging_utils import get_logger
from src.utils.payload_utils import get_source_name

logger = get_logger(__name__)


_ATTRIBUTE_TERMS = {
    "education": [
        "education",
        "degree",
        "university",
        "college",
        "school",
        "bachelor",
        "master",
        "phd",
        "gpa",
    ],
    "experience": [
        "experience",
        "work history",
        "employment",
        "career",
        "roles",
        "positions",
        "job",
        "responsibilities",
        "tenure",
    ],
    "skills": [
        "skill",
        "skills",
        "stack",
        "technology",
        "technologies",
        "tools",
        "languages",
        "framework",
        "expertise",
    ],
    "certification": [
        "certification",
        "certified",
        "certificate",
        "license",
        "licence",
        "credential",
    ],
    "contact": [
        "contact",
        "email",
        "phone",
        "mobile",
        "address",
        "linkedin",
        "website",
    ],
    "summary": [
        "summary",
        "overview",
        "profile",
        "bio",
    ],
    "projects": [
        "project",
        "portfolio",
        "case study",
    ],
}

_NUMERIC_HINTS = {
    "how many",
    "number of",
    "total",
    "sum",
    "amount",
    "count",
    "average",
    "median",
    "mean",
    "percent",
    "percentage",
    "ratio",
    "years",
    "months",
    "days",
    "experience",
    "salary",
    "cost",
    "price",
}

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
_NUMBER_RE = re.compile(r"\b\d+[\d,\.]*\b")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
_EDU_RE = re.compile(r"\b(bachelor|master|phd|degree|university|college|school)\b", re.IGNORECASE)
_EXP_RE = re.compile(r"\b(experience|years|worked|role|responsibilities|employment)\b", re.IGNORECASE)
_SKILL_RE = re.compile(r"\b(skills?|expertise|proficient|technolog(?:y|ies)|tools|languages?)\b", re.IGNORECASE)
_CERT_RE = re.compile(r"\b(certification|certified|certificate|license|licence|credential)\b", re.IGNORECASE)
_ADDRESS_RE = re.compile(r"\b(street|st\.|avenue|ave\.|road|rd\.|lane|ln\.|drive|dr\.|city|state|zip|postal)\b", re.IGNORECASE)


def extract_required_attributes(query: str, intent_type: str) -> List[str]:
    logger.debug("extract_required_attributes: intent_type=%s", intent_type)
    lowered = (query or "").lower()
    required: List[str] = []
    for attr, terms in _ATTRIBUTE_TERMS.items():
        if any(term in lowered for term in terms):
            required.append(attr)
    if intent_type == "field_extraction" and not required:
        for attr in ("contact", "education", "experience", "skills", "certification"):
            if attr in lowered:
                required.append(attr)
    result = sorted(set(required))
    logger.debug("extract_required_attributes: returning %d attributes", len(result))
    return result


def filter_chunks_by_intent(
    chunks: Iterable[object],
    required_attributes: List[str],
    entities: List[str],
    intent_type: str,
) -> List[object]:
    logger.debug("filter_chunks_by_intent: intent_type=%s, required_attributes=%d, entities=%d", intent_type, len(required_attributes or []), len(entities or []))
    chunk_list = list(chunks or [])
    if not chunk_list:
        return []
    if intent_type in {"summarization", "deep_analysis"}:
        return chunk_list

    required_attributes = [attr for attr in (required_attributes or []) if attr]
    entities = [entity for entity in (entities or []) if entity]
    if not required_attributes and not entities:
        return chunk_list

    entity_keys = _entity_keys(entities)

    def matches(chunk: object) -> bool:
        text = _chunk_text(chunk).lower()
        attr_match = False
        if required_attributes:
            for attr in required_attributes:
                terms = _ATTRIBUTE_TERMS.get(attr) or [attr]
                if any(term in text for term in terms):
                    attr_match = True
                    break
        entity_match = False
        if entity_keys:
            if any(key in text for key in entity_keys if key):
                entity_match = True
        if required_attributes and entities:
            return attr_match or entity_match
        if required_attributes:
            return attr_match
        if entities:
            return entity_match
        return True

    filtered = [chunk for chunk in chunk_list if matches(chunk)]
    result = filtered or chunk_list
    logger.debug("filter_chunks_by_intent: input=%d, output=%d", len(chunk_list), len(result))
    return result


def extract_answer_requirements(query: str, intent_type: str) -> List[Tuple[str, Pattern[str]]]:
    logger.debug("extract_answer_requirements: intent_type=%s", intent_type)
    lowered = (query or "").lower()
    requirements: List[Tuple[str, Pattern[str]]] = []

    if "email" in lowered or "e-mail" in lowered:
        requirements.append(("email", _EMAIL_RE))
    if "phone" in lowered or "mobile" in lowered or "contact" in lowered:
        requirements.append(("phone", _PHONE_RE))
    if "address" in lowered or "location" in lowered:
        requirements.append(("address", _ADDRESS_RE))
    if "education" in lowered or "degree" in lowered:
        requirements.append(("education", _EDU_RE))
    if "experience" in lowered or "work history" in lowered:
        requirements.append(("experience", _EXP_RE))
    if "skills" in lowered or "skill" in lowered:
        requirements.append(("skills", _SKILL_RE))
    if "certification" in lowered or "certificate" in lowered or "license" in lowered or "licence" in lowered:
        requirements.append(("certification", _CERT_RE))

    if intent_type == "numeric_lookup" or _NUMBER_RE.search(lowered) or any(hint in lowered for hint in _NUMERIC_HINTS):
        requirements.append(("number", _NUMBER_RE))
    if "date" in lowered or "when" in lowered or "during" in lowered:
        requirements.append(("date", _DATE_RE))

    logger.debug("extract_answer_requirements: returning %d requirements", len(requirements))
    return requirements


def validate_answer_requirements(
    chunks: Iterable[object],
    requirements: List[Tuple[str, Pattern[str]]],
) -> List[str]:
    logger.debug("validate_answer_requirements: requirements=%d", len(requirements or []))
    if not requirements:
        return []
    chunk_list = list(chunks or [])
    if not chunk_list:
        return [name for name, _ in requirements]
    combined = " ".join(_chunk_text(chunk) for chunk in chunk_list)
    missing: List[str] = []
    for name, pattern in requirements:
        if not pattern.search(combined):
            missing.append(name)
    if missing:
        logger.debug("validate_answer_requirements: missing=%s", missing)
    return missing


def build_intent_miss_response(entity: str, intent_label: str) -> str:
    entity = (entity or "").strip()
    intent_label = (intent_label or "").strip()
    if entity and intent_label:
        return f"I could not find {intent_label} details for {entity} in the available documents."
    if entity:
        return f"I could not find relevant details about {entity} in the available documents."
    if intent_label:
        return f"I could not find {intent_label} details in the available documents."
    return "I could not find relevant details in the available documents."


def _chunk_text(chunk: object) -> str:
    text = getattr(chunk, "text", None)
    meta = getattr(chunk, "metadata", None)
    if isinstance(chunk, dict):
        text = chunk.get("text") if text is None else text
        meta = chunk.get("metadata") if meta is None else meta
    meta = meta or {}
    section = str(meta.get("section_title") or meta.get("section_path") or meta.get("section") or "")
    source = str(get_source_name(meta) or "")
    return " ".join(part for part in (text or "", section, source) if part)


def _entity_keys(entities: List[str]) -> List[str]:
    keys: List[str] = []
    for entity in entities:
        cleaned = " ".join(entity.split()).strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        keys.append(lower)
        if " " in lower:
            last = lower.split()[-1]
            if last not in keys:
                keys.append(last)
    return keys
