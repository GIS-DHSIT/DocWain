"""Cross-document entity resolution via Jaro-Winkler similarity."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List

import jellyfish

from .models import EntitySpan

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_PERSON_TITLES_RE = re.compile(
    r"^(Dr|Mr|Mrs|Ms|Miss|Prof|Rev|Hon|Sgt|Cpl|Pvt|Capt|Lt|Col|Gen|Cmdr|Adm)\.\s*",
    re.IGNORECASE,
)

_PERSON_SUFFIXES_RE = re.compile(
    r",?\s+(Jr\.?|Sr\.?|III|IV|II|Esq\.?|Ph\.?D\.?|M\.?D\.?)$",
    re.IGNORECASE,
)

_ORG_SUFFIXES_RE = re.compile(
    r",?\s+(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|PLC|LP|LLP|GmbH|AG|SA|NV|Pty\.?|S\.?A\.?)$",
    re.IGNORECASE,
)

_JARO_WINKLER_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_person(name: str) -> str:
    """Remove titles and suffixes from a person name."""
    name = _PERSON_TITLES_RE.sub("", name).strip()
    name = _PERSON_SUFFIXES_RE.sub("", name).strip()
    return name


def _strip_org(name: str) -> str:
    """Remove common org suffixes."""
    return _ORG_SUFFIXES_RE.sub("", name).strip()


def _normalize_for_comparison(entity: EntitySpan) -> str:
    """Return a cleaned, lowercased form suitable for matching."""
    text = entity.text.strip()
    if entity.label == "PERSON":
        text = _strip_person(text)
    elif entity.label == "ORG":
        text = _strip_org(text)
    return text.lower()


def _is_initial_match(a_clean: str, b_clean: str) -> bool:
    """Check if one name is an initial form of the other.

    Handles patterns like "J. Smith" matching "John Smith" and
    "Mr. Smith" matching "John Smith" (after title stripping,
    "Mr. Smith" becomes "Smith").
    """
    parts_a = a_clean.split()
    parts_b = b_clean.split()

    if not parts_a or not parts_b:
        return False

    # Single-word vs multi-word: "smith" matches "john smith" by last name
    if len(parts_a) == 1 and len(parts_b) >= 2:
        if parts_a[0] == parts_b[-1]:
            return True
    if len(parts_b) == 1 and len(parts_a) >= 2:
        if parts_b[0] == parts_a[-1]:
            return True

    # "j. smith" vs "john smith" — initial + last name match
    if len(parts_a) >= 2 and len(parts_b) >= 2:
        # Last names must match exactly
        if parts_a[-1] != parts_b[-1]:
            return False
        first_a = parts_a[0].rstrip(".")
        first_b = parts_b[0].rstrip(".")
        # One is a single letter (initial)
        if len(first_a) == 1 and first_b.startswith(first_a):
            return True
        if len(first_b) == 1 and first_a.startswith(first_b):
            return True

    return False


def _should_merge(a: EntitySpan, b: EntitySpan) -> bool:
    """Determine whether two same-label entities refer to the same real-world entity."""
    if a.label != b.label:
        return False

    a_clean = _normalize_for_comparison(a)
    b_clean = _normalize_for_comparison(b)

    # Exact match after normalization
    if a_clean == b_clean:
        return True

    # Initial / title-only matching for persons
    if a.label == "PERSON" and _is_initial_match(a_clean, b_clean):
        return True

    # Jaro-Winkler similarity
    sim = jellyfish.jaro_winkler_similarity(a_clean, b_clean)
    if sim >= _JARO_WINKLER_THRESHOLD:
        return True

    return False


def _merge_into(primary: EntitySpan, secondary: EntitySpan) -> EntitySpan:
    """Merge *secondary* into *primary*, updating aliases and confidence."""
    aliases = set(primary.aliases)
    # Add the secondary's original text as an alias (if different from primary)
    if secondary.text != primary.text:
        aliases.add(secondary.text)
    # Carry over secondary's aliases too
    aliases.update(secondary.aliases)
    # Remove primary's own text from aliases if present
    aliases.discard(primary.text)

    return primary.model_copy(
        update={
            "confidence": max(primary.confidence, secondary.confidence),
            "aliases": sorted(aliases),
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_entities(entities: List[EntitySpan]) -> List[EntitySpan]:
    """Deduplicate entities across documents within a profile.

    Groups entities by label, then greedily merges pairs that pass the
    similarity gate (Jaro-Winkler >= 0.85, initial matching, or exact
    match after title/suffix stripping).

    Returns a new list of :class:`EntitySpan` with duplicates merged.
    The highest-confidence mention becomes the primary; other surface
    forms are stored in ``aliases``.
    """
    if not entities:
        return []

    # Group by label so different types are never compared
    by_label: dict[str, list[EntitySpan]] = defaultdict(list)
    for ent in entities:
        by_label[ent.label].append(ent)

    resolved: List[EntitySpan] = []

    for _label, group in by_label.items():
        # Sort descending by confidence so the best mention is primary
        group.sort(key=lambda e: e.confidence, reverse=True)

        clusters: List[EntitySpan] = []
        for ent in group:
            merged = False
            for idx, primary in enumerate(clusters):
                if _should_merge(primary, ent):
                    clusters[idx] = _merge_into(primary, ent)
                    merged = True
                    break
            if not merged:
                clusters.append(ent.model_copy())
        resolved.extend(clusters)

    return resolved
