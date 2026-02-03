import re
from typing import Iterable, List, Tuple

from src.docwain_intel.fact_cache import FactCache

_SKILL_HEADINGS = {"skills", "tools", "technologies", "technical skills", "functional skills"}


def _split_lines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _extract_entities(line: str) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    money = re.findall(r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?", line)
    for m in money:
        entities.append(("money", m))
    dates = re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", line, re.IGNORECASE)
    dates += re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", line)
    for d in dates:
        entities.append(("date", d))
    ids = re.findall(r"\b[A-Z0-9][A-Z0-9-]{5,}\b", line)
    for identifier in ids:
        entities.append(("id", identifier))
    people = re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", line)
    for person in people:
        entities.append(("person", person))
    orgs = re.findall(r"\b[A-Z][A-Za-z0-9&.,\- ]+\b(?:Inc|LLC|Ltd|Corp|University|Hospital|Company)\b", line)
    for org in orgs:
        entities.append(("org", org.strip()))
    return entities


def _extract_skills(line: str) -> List[str]:
    tokens = re.split(r"[,;/•|]\s*", line)
    return [tok.strip() for tok in tokens if tok.strip()]


def _chunk_doc_name(chunk) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    doc_name = (
        meta.get("source_file")
        or meta.get("file_name")
        or meta.get("filename")
        or meta.get("source")
        or getattr(chunk, "source", None)
        or "Document"
    )
    return str(doc_name)


def build_fact_cache(evidence_pack: Iterable) -> FactCache:
    cache = FactCache()
    for chunk in evidence_pack or []:
        text = getattr(chunk, "text", None) or ""
        doc_name = _chunk_doc_name(chunk)
        if doc_name not in cache.doc_names:
            cache.doc_names.append(doc_name)

        heading = None
        for line in _split_lines(text):
            if line.endswith(":") and len(line) < 60:
                heading = line.rstrip(":")
                continue
            if line.isupper() and len(line) < 60:
                heading = line.title()
                continue

            if heading:
                cache.add_section(heading, line, basis=heading, doc_name=doc_name)
                if heading.strip().lower() in _SKILL_HEADINGS:
                    for skill in _extract_skills(line):
                        cache.add_skill(skill, basis=heading, doc_name=doc_name)

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    cache.add_key_value(key, value, basis=f"{key}", doc_name=doc_name)
                    if key.lower() in _SKILL_HEADINGS:
                        for skill in _extract_skills(value):
                            cache.add_skill(skill, basis=key, doc_name=doc_name)

            for kind, value in _extract_entities(line):
                cache.add_entity(kind, value, basis=line[:80], doc_name=doc_name)

    return cache


__all__ = ["build_fact_cache"]
