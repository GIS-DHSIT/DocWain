"""
form_extractor.py — Detects and extracts key-value pairs from structured form regions in documents.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "FormField",
    "is_form_like_section",
    "extract_form_fields",
    "extract_all_form_fields",
    "form_fields_to_chunk_text",
]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALSE_POSITIVE_LABELS: frozenset[str] = frozenset(
    [
        "note",
        "notes",
        "example",
        "e.g",
        "i.e",
        "see",
        "refer",
        "page",
        "figure",
        "table",
        "section",
        "chapter",
    ]
)

# Words that suggest a label is actually a prose sentence fragment rather than a field label.
_VERB_LIKE_WORDS: frozenset[str] = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "please",
        "ensure",
        "note",
        "if",
        "when",
        "while",
        "although",
        "because",
        "however",
        "therefore",
        "furthermore",
        "additionally",
        "moreover",
        "provided",
        "required",
        "include",
        "contains",
        "describe",
        "indicate",
        "specify",
    ]
)

# Compiled regex patterns for key-value extraction.
# Each tuple: (compiled pattern, confidence score, match-mode)
# match-mode: "inline" uses group(1)/group(2) on a single line;
#             "multiline" is handled separately.
_INLINE_PATTERNS: list[tuple[re.Pattern, float]] = [
    # Colon-separated: "Name: John Smith"
    (re.compile(r"^(.{2,50}):\s+(.+)$", re.MULTILINE), 0.9),
    # Tab-separated: "Name\tJohn Smith"
    (re.compile(r"^(.{2,50})\t+(.+)$", re.MULTILINE), 0.7),
    # Space-aligned (3+ spaces): "Name   John Smith"
    (re.compile(r"^(.{2,50})\s{3,}(.+)$", re.MULTILINE), 0.7),
]

# Label on one line, value indented on next: "Name:\n    John Smith"
_MULTILINE_PATTERN: re.Pattern = re.compile(
    r"^(.{2,50})[:]\s*\n\s+(.+)$", re.MULTILINE
)
_MULTILINE_CONFIDENCE: float = 0.9

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FormField:
    """Represents a single extracted form field (key-value pair)."""

    label: str
    value: str
    confidence: float
    page: Optional[int] = None
    section_title: Optional[str] = None

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _looks_like_sentence(label: str) -> bool:
    """Return True if the label text looks like a prose sentence rather than a field name."""
    words = label.lower().split()
    if not words:
        return True
    # If any word is a verb-like word the label is probably prose.
    for word in words:
        stripped = word.strip(".,;:!?\"'()")
        if stripped in _VERB_LIKE_WORDS:
            return True
    # More than 8 words is almost certainly prose.
    if len(words) > 8:
        return True
    return False

def _normalize_label(label: str) -> str:
    """Strip whitespace and convert to title case."""
    return label.strip().title()

def _is_false_positive_label(label: str) -> bool:
    """Return True if the label matches a known false-positive pattern."""
    normalized = label.strip().lower().rstrip(":").strip()
    return normalized in _FALSE_POSITIVE_LABELS

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def is_form_like_section(text: str) -> bool:
    """
    Return True if the text section appears to be form-like.

    Criteria (all must hold):
    - At least 3 lines match any key-value pattern (colon, tab, or space-aligned).
    - Low prose density: average line length < 80 chars.
    - No lines exceed 200 chars.
    - Majority of lines (> 50 %) are shorter than 100 chars.
    """
    if not text or not text.strip():
        return False

    lines = text.splitlines()
    non_empty_lines = [ln for ln in lines if ln.strip()]
    if not non_empty_lines:
        return False

    # Prose density check.
    avg_len = sum(len(ln) for ln in non_empty_lines) / len(non_empty_lines)
    if avg_len >= 80:
        return False

    if any(len(ln) > 200 for ln in non_empty_lines):
        return False

    short_count = sum(1 for ln in non_empty_lines if len(ln) < 100)
    if short_count <= len(non_empty_lines) / 2:
        return False

    # Count key-value matches.
    kv_matches = 0
    for pattern, _ in _INLINE_PATTERNS:
        kv_matches += len(pattern.findall(text))

    # Also count multiline matches.
    kv_matches += len(_MULTILINE_PATTERN.findall(text))

    if kv_matches >= 3:
        return True

    return False

def extract_form_fields(
    text: str,
    page: Optional[int] = None,
    section_title: Optional[str] = None,
) -> List[FormField]:
    """
    Extract key-value form fields from a block of text.

    Applies four patterns (colon-separated, tab-separated, space-aligned, and
    multiline label/value).  False positives are filtered by label heuristics.

    Args:
        text: Raw section text to scan.
        page: Optional page number for provenance.
        section_title: Optional section heading for provenance.

    Returns:
        List of :class:`FormField` instances.
    """
    if not text or not text.strip():
        return []

    fields: list[FormField] = []
    seen: set[tuple[str, str]] = set()

    def _add_field(raw_label: str, raw_value: str, confidence: float) -> None:
        label = _normalize_label(raw_label)
        value = raw_value.strip()

        if not label or not value:
            return
        if _is_false_positive_label(label):
            logger.debug("Skipping false-positive label: %r", label)
            return
        if _looks_like_sentence(raw_label):
            logger.debug("Skipping sentence-like label: %r", label)
            return

        key = (label.lower(), value.lower())
        if key in seen:
            return
        seen.add(key)

        fields.append(
            FormField(
                label=label,
                value=value,
                confidence=confidence,
                page=page,
                section_title=section_title,
            )
        )

    # Process inline patterns.
    for pattern, confidence in _INLINE_PATTERNS:
        for match in pattern.finditer(text):
            raw_label, raw_value = match.group(1), match.group(2)
            _add_field(raw_label, raw_value, confidence)

    # Process multiline pattern.
    for match in _MULTILINE_PATTERN.finditer(text):
        raw_label, raw_value = match.group(1), match.group(2)
        _add_field(raw_label, raw_value, _MULTILINE_CONFIDENCE)

    logger.debug(
        "extract_form_fields: found %d fields (page=%s, section=%r)",
        len(fields),
        page,
        section_title,
    )
    return fields

def extract_all_form_fields(sections: List[dict]) -> List[FormField]:
    """
    Extract form fields from all form-like sections in a document.

    Each section dict is expected to contain:
    - ``"text"`` (str): The section body text.
    - ``"title"`` (str, optional): The section heading.
    - ``"start_page"`` (int, optional): The first page number of the section.

    Args:
        sections: List of section dicts from the extraction pipeline.

    Returns:
        Deduplicated list of :class:`FormField` across all form-like sections.
    """
    if not sections:
        return []

    all_fields: list[FormField] = []
    global_seen: set[tuple[str, str]] = set()

    for section in sections:
        text = section.get("text", "")
        title = section.get("title") or None
        start_page = section.get("start_page") or None

        if not text:
            continue

        if not is_form_like_section(text):
            logger.debug(
                "Section %r is not form-like, skipping.", title or "<untitled>"
            )
            continue

        logger.debug("Extracting form fields from section: %r", title or "<untitled>")
        fields = extract_form_fields(text, page=start_page, section_title=title)

        for field in fields:
            dedup_key = (field.label.lower(), field.value.lower())
            if dedup_key not in global_seen:
                global_seen.add(dedup_key)
                all_fields.append(field)

    logger.info(
        "extract_all_form_fields: %d unique fields extracted from %d section(s).",
        len(all_fields),
        len(sections),
    )
    return all_fields

def form_fields_to_chunk_text(fields: List[FormField]) -> str:
    """
    Convert a list of form fields into a single searchable text chunk.

    The resulting string is suitable for embedding as a vector with
    ``chunk_kind="form_fields"``.

    Args:
        fields: List of :class:`FormField` instances.

    Returns:
        A pipe-delimited string of the form
        ``"Form Fields: Label1 = Value1 | Label2 = Value2 | ..."``,
        or an empty string if *fields* is empty.
    """
    if not fields:
        return ""

    pairs = " | ".join(f"{f.label} = {f.value}" for f in fields)
    chunk_text = f"Form Fields: {pairs}"
    logger.debug("form_fields_to_chunk_text: %d fields → %d chars", len(fields), len(chunk_text))
    return chunk_text
