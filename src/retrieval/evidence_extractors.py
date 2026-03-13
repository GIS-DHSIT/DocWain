from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://[^\s<>]+|\bwww\.[^\s<>]+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_YEAR_RE = re.compile(
    r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{4}\b",
    re.IGNORECASE,
)
YEAR_RANGE_RE = re.compile(r"\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\b")
HEADING_RE = re.compile(r"^(?P<head>[A-Z][A-Za-z0-9 /&-]{2,80}):\s*$")
IDENTIFIER_RE = re.compile(
    r"\b(?P<label>invoice|inv|po|purchase order|case|tax|vat|gst|ein|ssn)\s*[#:]*\s*(?P<value>[A-Za-z0-9-]{3,})",
    re.IGNORECASE,
)


@dataclass
class EvidenceItem:
    value: str
    snippet: str
    document_id: str
    source_name: Optional[str]
    chunk_id: str
    section_title: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    meta: Dict[str, str] = field(default_factory=dict)


def _normalize_phone(phone: str) -> str:
    cleaned = re.sub(r"[^\d+]", "", phone)
    if cleaned.startswith("+"):
        return cleaned
    return cleaned.lstrip("+")


def _make_item(value: str, chunk: Dict[str, str], snippet: Optional[str] = None, meta: Optional[Dict[str, str]] = None) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        snippet=snippet or chunk.get("text", "")[:160],
        document_id=chunk["document_id"],
        source_name=chunk.get("source_name"),
        chunk_id=chunk.get("chunk_id") or "",
        section_title=chunk.get("section_title"),
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        meta=meta or {},
    )


def extract_contacts(chunks: Iterable[Dict[str, str]]) -> Dict[str, List[EvidenceItem]]:
    logger.debug("extract_contacts: starting extraction")
    phones: Dict[str, EvidenceItem] = {}
    emails: Dict[str, EvidenceItem] = {}
    urls: Dict[str, EvidenceItem] = {}
    for chunk in chunks:
        text = chunk.get("text") or ""
        for match in EMAIL_RE.findall(text):
            key = match.lower()
            emails.setdefault(key, _make_item(match, chunk, snippet=text))
        for match in URL_RE.findall(text):
            normalized = match.rstrip(").,")
            key = normalized.lower()
            urls.setdefault(key, _make_item(normalized, chunk, snippet=text))
        for match in PHONE_RE.findall(text):
            normalized = _normalize_phone(match)
            if normalized:
                phones.setdefault(normalized, _make_item(match.strip(), chunk, snippet=text))
    result = {
        "phones": list(phones.values()),
        "emails": list(emails.values()),
        "urls": list(urls.values()),
    }
    logger.debug("extract_contacts: phones=%d, emails=%d, urls=%d", len(result["phones"]), len(result["emails"]), len(result["urls"]))
    return result


def extract_dates(chunks: Iterable[Dict[str, str]]) -> List[EvidenceItem]:
    logger.debug("extract_dates: starting extraction")
    seen: Dict[str, EvidenceItem] = {}
    for chunk in chunks:
        text = chunk.get("text") or ""
        for match in YEAR_RANGE_RE.finditer(text):
            value = match.group(0).replace(" ", "")
            seen.setdefault(value, _make_item(value, chunk, snippet=text))
        for match in MONTH_YEAR_RE.findall(text):
            value = match.strip()
            seen.setdefault(value.lower(), _make_item(value, chunk, snippet=text))
        for match in YEAR_RE.finditer(text):
            value = match.group(0)
            seen.setdefault(value, _make_item(value, chunk, snippet=text))
    logger.debug("extract_dates: returning %d items", len(seen))
    return list(seen.values())


def extract_identifiers(chunks: Iterable[Dict[str, str]]) -> List[EvidenceItem]:
    logger.debug("extract_identifiers: starting extraction")
    seen: Dict[str, EvidenceItem] = {}
    for chunk in chunks:
        text = chunk.get("text") or ""
        for match in IDENTIFIER_RE.finditer(text):
            label = match.group("label").lower()
            value = match.group("value")
            key = f"{label}:{value.lower()}"
            seen.setdefault(
                key,
                _make_item(value, chunk, snippet=text, meta={"type": label}),
            )
    logger.debug("extract_identifiers: returning %d items", len(seen))
    return list(seen.values())


def extract_entities(chunks: Iterable[Dict[str, str]]) -> List[EvidenceItem]:
    logger.debug("extract_entities: starting extraction")
    seen: Dict[str, EvidenceItem] = {}
    for chunk in chunks:
        text = chunk.get("text") or ""
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
        for cand in candidates:
            key = cand.lower()
            seen.setdefault(key, _make_item(cand, chunk, snippet=text))
    logger.debug("extract_entities: returning %d items", len(seen))
    return list(seen.values())


def extract_sections(chunks: Iterable[Dict[str, str]]) -> List[EvidenceItem]:
    logger.debug("extract_sections: starting extraction")
    seen: Dict[str, EvidenceItem] = {}
    for chunk in chunks:
        section_title = chunk.get("section_title")
        if section_title:
            key = section_title.strip().lower()
            seen.setdefault(key, _make_item(section_title, chunk, snippet=section_title))
        for line in (chunk.get("text") or "").splitlines():
            match = HEADING_RE.match(line.strip())
            if match:
                head = match.group("head").strip()
                key = head.lower()
                seen.setdefault(key, _make_item(head, chunk, snippet=head))
    logger.debug("extract_sections: returning %d items", len(seen))
    return list(seen.values())


def extract_tables(chunks: Iterable[Dict[str, str]]) -> List[EvidenceItem]:
    logger.debug("extract_tables: starting extraction")
    tables: List[EvidenceItem] = []
    for chunk in chunks:
        role = (chunk.get("chunk_role") or "").lower()
        ctype = (chunk.get("chunk_type") or "").lower()
        if role == "table_text" or ctype in {"table", "table_row", "table_header"}:
            tables.append(_make_item(chunk.get("text") or "", chunk, snippet=chunk.get("text") or ""))
    logger.debug("extract_tables: returning %d items", len(tables))
    return tables


__all__ = [
    "EvidenceItem",
    "extract_contacts",
    "extract_dates",
    "extract_identifiers",
    "extract_entities",
    "extract_sections",
    "extract_tables",
]
