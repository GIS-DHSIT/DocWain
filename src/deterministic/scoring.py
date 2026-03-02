from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from src.deterministic.contacts import ContactInfo, extract_contacts


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_EXPERIENCE_RE = re.compile(
    r"(?i)\b(experience|worked|work\s+experience|employment|responsibilit(?:y|ies)|project|intern|role|years?)\b"
)
_SKILLS_LINE_RE = re.compile(r"(?im)^\s*(skills|technical skills|core skills)\s*[:\-]?\s*(.*)$")


@dataclass(frozen=True)
class ProfileSignals:
    skills_count: int
    experience_markers: int
    date_presence: bool
    total_chars: int
    contact_presence: bool
    contacts: ContactInfo


def _estimate_skills_count(text: str) -> int:
    if not text:
        return 0
    found: List[str] = []
    for match in _SKILLS_LINE_RE.finditer(text):
        remainder = (match.group(2) or "").strip()
        if remainder:
            found.append(remainder)
    if not found:
        return 0

    blob = " | ".join(found)
    # Split on common separators; keep token-like items.
    tokens = [t.strip() for t in re.split(r"[,|/;•\n\t]+", blob) if t.strip()]
    tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
    # Deduplicate case-insensitively.
    seen = set()
    uniq = []
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tok)
    return len(uniq)


def compute_profile_signals(texts: Iterable[str]) -> ProfileSignals:
    texts_list = [t or "" for t in texts or []]
    joined = "\n".join(texts_list)
    contacts = extract_contacts(joined)
    contact_presence = bool(contacts.phones or contacts.emails or contacts.linkedins)

    total_chars = sum(len(t) for t in texts_list)
    date_presence = bool(_YEAR_RE.search(joined))
    experience_markers = len(_EXPERIENCE_RE.findall(joined))
    skills_count = _estimate_skills_count(joined)

    return ProfileSignals(
        skills_count=skills_count,
        experience_markers=experience_markers,
        date_presence=date_presence,
        total_chars=total_chars,
        contact_presence=contact_presence,
        contacts=contacts,
    )


def score_profile(signals: ProfileSignals) -> float:
    """
    Deterministic scoring for ranking/compare.

    Intentionally simple and stable; ties should be broken by profile_id outside this function.
    """
    score = float(signals.total_chars)
    if signals.date_presence:
        score += 5000.0
    score += 250.0 * float(min(signals.experience_markers, 40))
    score += 25.0 * float(min(signals.skills_count, 200))
    if signals.contact_presence:
        score += 3000.0
    return score


__all__ = ["compute_profile_signals", "score_profile", "ProfileSignals"]
