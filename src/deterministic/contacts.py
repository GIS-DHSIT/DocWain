from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ContactInfo:
    phones: List[str]
    emails: List[str]
    linkedins: List[str]


_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/in/[\w%\-_.]+/?", re.IGNORECASE)
_PHONE_CANDIDATE_RE = re.compile(r"(?:\+?\d[\d\s().\-]{8,}\d)")


def _unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_phone_candidate(raw: str) -> Optional[str]:
    digits = re.sub(r"\D+", "", raw or "")
    if len(digits) < 10:
        return None

    if len(digits) == 10:
        return digits

    # India-like prefixes -> keep last 10 digits.
    india_like = digits.startswith(("91", "0091", "0"))
    if india_like and len(digits) > 10:
        return digits[-10:]

    return None


def extract_contacts(text: str) -> ContactInfo:
    """
    Deterministic contact extraction.

    - Phones: accepts 10-digit and optional +91 prefix; strips non-digits; keeps last 10 digits if India-like.
    - Emails: standard email regex.
    - LinkedIn: https?://(www.)?linkedin.com/in/...
    """
    text = text or ""

    emails = _unique_in_order(m.group(0) for m in _EMAIL_RE.finditer(text))
    linkedins = _unique_in_order(_LINKEDIN_RE.findall(text))

    phones_raw = (_normalize_phone_candidate(m.group(0)) for m in _PHONE_CANDIDATE_RE.finditer(text))
    phones = _unique_in_order(p for p in phones_raw if p)

    return ContactInfo(phones=phones, emails=emails, linkedins=linkedins)


__all__ = ["extract_contacts", "ContactInfo"]
