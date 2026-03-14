from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .sanitize import sanitize_text

# ---------------------------------------------------------------------------
# Domain-specific disclaimers appended when domain is known
# ---------------------------------------------------------------------------
_DOMAIN_DISCLAIMERS: Dict[str, str] = {
    "medical": (
        "*This information is extracted from uploaded documents and is not "
        "a substitute for professional medical advice.*"
    ),
    "legal": (
        "*This analysis is based on the uploaded documents and does not "
        "constitute legal advice.*"
    ),
    "financial": (
        "*This financial information is derived from uploaded documents "
        "and should not be used as the sole basis for financial decisions.*"
    ),
    "hr": (
        "*This HR-related information is extracted from uploaded documents "
        "and may not reflect the latest organizational policies.*"
    ),
    "policy": (
        "*This policy information is based on the uploaded documents. "
        "Please verify with the issuing authority for the most current version.*"
    ),
    "invoice": (
        "*This invoice information is extracted from uploaded documents. "
        "Please verify amounts and terms against official records.*"
    ),
}

# ---------------------------------------------------------------------------
# Domain-adaptive confidence thresholds — below this, add a caveat prefix
# ---------------------------------------------------------------------------
_DOMAIN_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "medical": 0.50,
    "legal": 0.55,
    "hr": 0.55,
    "invoice": 0.60,
    "policy": 0.55,
    "generic": 0.65,
}

# Refusal/fallback phrases that should never be prefixed with caveats
_REFUSAL_PHRASES = (
    "couldn't find", "could not find", "not enough information",
    "no relevant", "not found in the", "i don't have",
    "i couldn't find", "no information",
)

# Methodology openers to strip — the user wants the answer, not process
_METHODOLOGY_PATTERNS = re.compile(
    r"^(?:Based on (?:my|the|our) (?:analysis|review|examination|assessment|reading)"
    r"|After (?:reviewing|analyzing|examining|reading)"
    r"|(?:Having|Upon) (?:reviewed|analyzed|examined|read)"
    r"|I (?:have )?(?:reviewed|analyzed|examined|read))"
    r"[^.]*[.,:]?\s*",
    re.IGNORECASE,
)

# Patterns that indicate existing markdown structure
_HAS_STRUCTURE_RE = re.compile(
    r"(?:^#{1,4}\s|\n#{1,4}\s|^\s*[-*]\s|\n\s*[-*]\s|^\s*\d+\.\s|\n\s*\d+\.\s"
    r"|\*\*[^*]+\*\*|\|.*\|.*\|)",
    re.MULTILINE,
)

# Pattern to detect "N items" requests in queries
_COUNT_REQUEST_RE = re.compile(
    r"(?:all\s+)?(\d+)\s+(?:candidates?|items?|documents?|entries|results?|options?"
    r"|employees?|patients?|invoices?|records?|files?|people|persons?)",
    re.IGNORECASE,
)
# Word-number patterns: "both candidates", "three items", etc.
_WORD_COUNT_MAP = {
    "both": 2, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_WORD_COUNT_RE = re.compile(
    r"\b(both|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:of\s+(?:the\s+)?)?(?:candidates?|items?|documents?|entries|results?|options?"
    r"|employees?|patients?|invoices?|records?|files?|people|persons?)",
    re.IGNORECASE,
)

# Pattern to detect "list of" / "list all" requests
_LIST_REQUEST_RE = re.compile(
    r"\b(?:list\s+(?:of\s+)?(?:all\s+)?|all\s+(?:the\s+)?)",
    re.IGNORECASE,
)


# ===================================================================
# Public API
# ===================================================================

def format_rag_v3_response(
    *,
    response_text: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    domain: Optional[str] = None,
    confidence: Optional[float] = None,
    query: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Format a RAG v3 response with source attribution and domain guardrails.

    Args:
        response_text: Raw LLM response text.
        sources: List of source dicts with file_name, page, score, etc.
        domain: Detected document domain (medical, legal, hr, etc.).
        confidence: Grounding confidence score 0.0-1.0.
        query: Original user query (used for completeness checking).
    """
    cleaned = sanitize_text(response_text or "")

    # --- confidence narrative injection ---
    cleaned = _inject_confidence_narrative(cleaned, confidence, domain, intent=intent)

    # --- response structure enforcement ---
    cleaned = _ensure_response_structure(cleaned, intent=intent)

    # --- source line with quality ranking ---
    source_line = _ranked_source_line(sources or [])
    if source_line:
        # Only suppress if the LAST 3 lines already have a "Source:" attribution line
        # (not inline citations like "(Source: file.pdf)" in the body)
        _tail_lines = [ln.strip().lower() for ln in cleaned.split("\n")[-3:]]
        _has_source_line = any(
            ln.startswith("source:") or ln.startswith("sources:") or ln.startswith("*source")
            for ln in _tail_lines
        )
        if not _has_source_line:
            cleaned = f"{cleaned}\n\n{source_line}"

    # --- answer completeness indicator ---
    completeness_note = _completeness_indicator(query, cleaned)
    if completeness_note:
        cleaned = f"{cleaned}\n\n{completeness_note}"

    # --- domain disclaimer ---
    if domain:
        disclaimer = _DOMAIN_DISCLAIMERS.get(domain.lower())
        if disclaimer and disclaimer.lower() not in cleaned.lower():
            cleaned = f"{cleaned}\n\n{disclaimer}"

    return cleaned.strip()


# ===================================================================
# Private helpers
# ===================================================================

def _get_confidence_threshold(domain: Optional[str], intent: Optional[str] = None) -> float:
    """Return the confidence threshold for a given domain and intent.

    Intent-aware: factual queries need higher confidence than reasoning/generate
    because factual claims are binary (right/wrong) while reasoning allows interpretation.
    """
    base = _DOMAIN_CONFIDENCE_THRESHOLDS.get(
        (domain or "").lower(), _DOMAIN_CONFIDENCE_THRESHOLDS["generic"]
    )
    # Adjust threshold by intent — reasoning/generate get more latitude
    _INTENT_ADJUSTMENTS = {
        "factual": 0.05,      # Stricter: factual claims must be well-grounded
        "contact": 0.05,      # Stricter: contact info must be accurate
        "extract": 0.03,      # Slightly stricter
        "reasoning": -0.10,   # More lenient: reasoning allows interpretation
        "generate": -0.15,    # Most lenient: creative content based on facts
        "analytics": -0.05,   # Slightly lenient: computed values may not match exactly
        "summary": -0.05,     # Slightly lenient: summaries paraphrase
    }
    if intent:
        base += _INTENT_ADJUSTMENTS.get(intent.lower(), 0.0)
    return max(0.20, min(0.80, base))


def _inject_confidence_narrative(
    text: str, confidence: Optional[float], domain: Optional[str],
    intent: Optional[str] = None,
) -> str:
    """Return text as-is — expert SME responses don't need confidence caveats.

    An expert analyst delivers findings directly. Confidence is tracked
    in structured metadata, not injected into the prose.
    """
    return text


def _ensure_response_structure(text: str, intent: Optional[str] = None) -> str:
    """Enforce response structure for longer unstructured responses.

    - Strips methodology openers so the first sentence is the direct answer.
    - For responses >200 chars with no markdown structure, attempts to
      auto-structure with bullet points at natural sentence boundaries.
    - Adds section headers for very long responses (>600 chars, 5+ sentences).
    - Preserves existing tables, lists, and bold formatting.
    - Skips auto-bulleting for intents that benefit from prose (factual, contact, extract).
    """
    if not text:
        return text

    # Strip methodology openers
    text = _METHODOLOGY_PATTERNS.sub("", text, count=1).lstrip()

    # Intents where prose is the correct format — don't auto-bullet
    _PROSE_INTENTS = frozenset({"factual", "contact", "extract", "extraction", "qa", "detail", "reasoning"})
    if intent and intent.lower() in _PROSE_INTENTS:
        # Still apply table/structure repair even for prose intents
        if _HAS_STRUCTURE_RE.search(text):
            text = _repair_truncated_structures(text)
            text = _fix_table_column_consistency(text)
            text = _clean_trailing_empty_headers(text)
        return text

    # Only auto-structure longer, unstructured responses
    if len(text) <= 200 or _HAS_STRUCTURE_RE.search(text):
        # Repair truncated tables/lists from LLM token cutoff
        text = _repair_truncated_structures(text)
        # Fix table column consistency before returning structured responses
        text = _fix_table_column_consistency(text)
        # Even structured responses may have trailing empty bold headers
        return _clean_trailing_empty_headers(text)

    # Split into sentences and structure with bullets if 3+ sentences
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text

    # For comparison/ranking intents with prose, try entity-per-section restructure
    _comparison_intents = frozenset({"comparison", "ranking", "compare", "rank", "cross_document"})
    if intent and intent.lower() in _comparison_intents and len(text) > 300:
        restructured = _detect_prose_entity_sections(text)
        if restructured:
            return restructured

    # Overview/summary intents with 5+ sentences should get section structure
    _SECTION_INTENTS = frozenset({"overview", "summary", "summarize"})
    if intent and intent.lower() in _SECTION_INTENTS and len(sentences) >= 4 and len(text) > 400:
        return _auto_section_structure(sentences)

    # For very long responses (5+ sentences, >600 chars), group into sections
    if len(sentences) >= 5 and len(text) > 600:
        return _auto_section_structure(sentences)

    # Keep first sentence as the lead, bullet-ise the rest
    # But skip bullet-ising if rest already contains structured content
    lead = sentences[0].strip()
    rest_lines = []
    for s in sentences[1:]:
        s = s.strip()
        if not s:
            continue
        # Don't re-bullet existing structure (tables, lists, headers)
        if (s.startswith("|") or s.startswith("-") or s.startswith("*")
                or s.startswith("#") or re.match(r"^\d+[.)]\s", s)):
            rest_lines.append(s)
        else:
            rest_lines.append(f"- {s}")
    if rest_lines:
        return f"{lead}\n\n" + "\n".join(rest_lines)
    return text


def _auto_section_structure(sentences: List[str]) -> str:
    """Structure a long list of sentences into Overview + Key Details sections.

    First 1-2 sentences become the overview (no header needed).
    Remaining sentences become bulleted items under a **Key Details:** header.
    This gives GPT-like structured output for long responses.
    """
    # Overview: first 1-2 sentences as plain prose
    overview_count = 1 if len(sentences) <= 5 else 2
    overview_parts = [s.strip() for s in sentences[:overview_count] if s.strip()]
    overview = " ".join(overview_parts)

    # Key details: remaining sentences as bullets
    detail_sentences = [s.strip() for s in sentences[overview_count:] if s.strip()]
    if not detail_sentences:
        return overview

    details = [f"- {s}" for s in detail_sentences]

    return f"{overview}\n\n**Key Details:**\n" + "\n".join(details)


_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|Inc|Corp|Ltd|Co|vs|etc|e\.g|i\.e|approx|dept|div|govt|assn"
    r"|Fig|Sec|para|No|Vol|pp|Rev|Sgt|Capt|Gen|Lt|Col|Maj|Ave|Blvd|St|Rd|Apt)\."
)


def _split_sentences(text: str) -> List[str]:
    """Sentence splitter that respects abbreviations and preserves structure.

    - Protects common abbreviations (Mr., Dr., Inc., etc.) from false splits
    - Preserves table rows and list items as atomic units
    - Splits on period/question/exclamation followed by space+uppercase
    """
    lines = text.strip().splitlines()
    result: List[str] = []
    prose_buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Table rows and list items are atomic — don't split
        if stripped.startswith("|") and stripped.endswith("|"):
            if prose_buffer:
                result.extend(_split_prose(" ".join(prose_buffer)))
                prose_buffer.clear()
            result.append(stripped)
            continue
        if re.match(r"^\s*(?:[-*•]\s|\d+[.)]\s)", stripped):
            if prose_buffer:
                result.extend(_split_prose(" ".join(prose_buffer)))
                prose_buffer.clear()
            result.append(stripped)
            continue
        if stripped:
            prose_buffer.append(stripped)

    if prose_buffer:
        result.extend(_split_prose(" ".join(prose_buffer)))

    return [p for p in result if p.strip()]


def _split_prose(text: str) -> List[str]:
    """Split prose text into sentences, protecting abbreviations."""
    # Temporarily protect abbreviations
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", "\x00"), text)
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    # Restore abbreviations
    return [p.replace("\x00", ".") for p in parts if p.strip()]


def _ranked_source_line(sources: List[Dict[str, Any]]) -> Optional[str]:
    """Build a source attribution line with primary/supporting ranking."""
    if not sources:
        return None

    # Deduplicate and collect scores
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for src in sources:
        name = (
            src.get("file_name")
            or src.get("source_name")
            or src.get("document_name")
        )
        if not name:
            continue

        page = src.get("page") or src.get("page_start")
        page_end = src.get("page_end")
        if page and page_end and page_end != page:
            label = f"{name} (pp. {page}-{page_end})"
        elif page:
            label = f"{name} (p. {page})"
        else:
            label = name

        if label not in seen:
            seen.add(label)
            score = src.get("score") or src.get("relevance_score") or 0.0
            entries.append({"label": label, "score": float(score)})

    if not entries:
        return None

    # Sort by relevance score descending
    entries.sort(key=lambda e: e["score"], reverse=True)

    # Single source — simple format
    if len(entries) == 1:
        return f"Source: {entries[0]['label']}"

    # Multiple sources — clean comma-separated list (GPT-style)
    # Only include sources with meaningful relevance
    has_scores = any(e["score"] > 0 for e in entries)
    if has_scores:
        # Filter out very low-relevance sources (< 0.15) unless we'd have none
        relevant = [e for e in entries if e["score"] >= 0.15]
        if not relevant:
            relevant = entries[:3]
        labels = [e["label"] for e in relevant]
    else:
        labels = [e["label"] for e in entries]

    return "Sources: " + ", ".join(labels)


def _completeness_indicator(
    query: Optional[str], response: str
) -> Optional[str]:
    """Check if the query asked for N items and the response has fewer.

    Returns a note string or None.
    """
    if not query:
        return None

    # Look for explicit count requests like "all 3 candidates"
    m = _COUNT_REQUEST_RE.search(query)
    if m:
        requested = int(m.group(1))
    else:
        # Try word-number patterns: "both candidates", "three items"
        wm = _WORD_COUNT_RE.search(query)
        if wm:
            requested = _WORD_COUNT_MAP.get(wm.group(1).lower(), 0)
            m = wm
        else:
            return None
    if requested < 2 or requested > 20:
        return None

    # Count how many distinct items the response mentions.
    # Heuristic: count numbered items (1. 2. 3.) or bullet points,
    # or bold labels (**Name**).
    numbered = re.findall(r"(?:^|\n)\s*\d+\.\s", response)
    bullets = re.findall(r"(?:^|\n)\s*[-*]\s", response)
    bold_labels = re.findall(r"\*\*[^*]{2,60}\*\*", response)

    # Take the best signal
    found = max(len(numbered), len(bullets), len(bold_labels))

    # Only flag when we clearly have fewer
    if 0 < found < requested:
        item_word = m.group(0).strip().split()[-1]  # e.g. "candidates"
        return (
            f"*Note: Information found for {found} of {requested} "
            f"requested {item_word}.*"
        )

    # Entity-name completeness: detect named entities in query and check
    # if all are mentioned in the response (handles "Compare Alice and Bob")
    _entity_note = _check_entity_completeness(query, response)
    if _entity_note:
        return _entity_note

    return None


def _check_entity_completeness(
    query: str, response: str
) -> Optional[str]:
    """Check if all named entities from query appear in the response."""
    if not query or not response:
        return None
    # Extract proper-noun entities from query (capitalized multi-word or single proper nouns)
    _entity_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    )
    _skip_words = frozenset({
        "The", "What", "How", "Who", "Where", "When", "Which", "Can", "Could",
        "Does", "List", "Show", "Compare", "Find", "Get", "Give", "Tell",
        "Rank", "Sort", "Describe", "Explain", "Summarize", "Based", "Using",
        "Please", "All", "Each", "Both", "Between", "And", "Are", "Is",
        "Their", "From", "With", "About", "Into", "Over", "Under", "After",
        "Before", "During", "Since", "Through", "Table", "Format", "Detail",
    })
    entities = []
    for m in _entity_pattern.finditer(query):
        name = m.group(1)
        words = name.split()
        # Strip leading/trailing skip words to get clean entity names
        # "Compare Alice Smith" → "Alice Smith"
        clean_words = [w for w in words if w not in _skip_words]
        if not clean_words:
            continue
        entities.append(" ".join(clean_words))

    if len(entities) < 2:
        return None  # Only check for multi-entity queries

    # Check which entities appear in the response
    resp_lower = response.lower()
    missing = [e for e in entities if e.lower() not in resp_lower]

    if missing and len(missing) < len(entities):
        return (
            f"*Note: No information found for "
            f"{', '.join(missing)} in the available documents.*"
        )


def _fix_table_column_consistency(text: str) -> str:
    """Fix markdown tables where rows have inconsistent column counts.

    LLMs sometimes generate tables where some rows have fewer pipes than
    the header. This pads short rows with 'N/A' cells to match the header
    column count, and ensures a separator row exists after the header.
    """
    if not text or "|" not in text:
        return text

    lines = text.split("\n")
    result: list[str] = []
    table_lines: list[str] = []

    def _flush_table():
        if not table_lines:
            return
        # Parse column counts
        col_counts = []
        for tl in table_lines:
            cols = [c.strip() for c in tl.strip().strip("|").split("|")]
            col_counts.append(len(cols))
        if not col_counts:
            result.extend(table_lines)
            table_lines.clear()
            return

        max_cols = max(col_counts)
        if max_cols < 2:
            result.extend(table_lines)
            table_lines.clear()
            return

        # Fix each row to have max_cols columns
        fixed: list[str] = []
        has_separator = False
        for i, tl in enumerate(table_lines):
            cols = [c.strip() for c in tl.strip().strip("|").split("|")]
            # Detect separator row (all cells are dashes)
            if all(re.match(r"^[-:]+$", c.strip()) for c in cols if c.strip()):
                has_separator = True
                # Pad separator to match header
                while len(cols) < max_cols:
                    cols.append("---")
                fixed.append("| " + " | ".join(cols) + " |")
                continue
            # Pad data rows with N/A
            while len(cols) < max_cols:
                cols.append("N/A")
            fixed.append("| " + " | ".join(cols) + " |")

        # Insert separator after header if missing
        if len(fixed) >= 2 and not has_separator:
            header_cols = max_cols
            sep = "| " + " | ".join(["---"] * header_cols) + " |"
            fixed.insert(1, sep)

        result.extend(fixed)
        table_lines.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
        else:
            _flush_table()
            result.append(line)
    _flush_table()

    return "\n".join(result)


def _clean_trailing_empty_headers(text: str) -> str:
    """Remove bold headers at the end of the response that have no content following.

    LLMs sometimes generate a bold header as the last line with nothing after it.
    E.g., "**Additional Notes:**\\n" — this looks incomplete to the user.
    """
    if not text:
        return text
    lines = text.rstrip().split("\n")
    # Strip trailing empty lines first
    while lines and not lines[-1].strip():
        lines.pop()
    # Remove trailing bold headers with no content
    while lines:
        last = lines[-1].strip()
        if re.match(r"^\*\*[^*]+\*\*:?\s*$", last):
            lines.pop()
            # Also remove blank line before the header if any
            while lines and not lines[-1].strip():
                lines.pop()
        else:
            break
    return "\n".join(lines) if lines else text


def _repair_truncated_structures(text: str) -> str:
    """Repair LLM output that was cut mid-table or mid-list.

    Detects incomplete final rows in markdown tables (missing closing pipe)
    and incomplete numbered/bulleted list items (trailing orphan line).
    Removes the broken trailing element so the output renders cleanly.
    """
    if not text or len(text) < 30:
        return text

    lines = text.rstrip().split("\n")
    if not lines:
        return text

    # Repair truncated table row: last line starts with | but doesn't end with |
    last = lines[-1].strip()
    if last.startswith("|") and not last.endswith("|"):
        # Count pipes — if it has at least one interior pipe, it's a truncated row
        if last.count("|") >= 2:
            # Try to close the row by appending pipes to match the expected columns
            # Find a prior complete row to determine column count
            target_cols = 0
            for prev_line in reversed(lines[:-1]):
                ps = prev_line.strip()
                if ps.startswith("|") and ps.endswith("|"):
                    target_cols = ps.count("|") - 1  # -1 for leading pipe
                    break
            current_cols = last.count("|") - 1
            if target_cols > current_cols:
                # Pad with N/A cells and close
                padding = " | N/A" * (target_cols - current_cols)
                lines[-1] = last + padding + " |"
            else:
                # Just close the row
                lines[-1] = last + " |"
        else:
            # Only one pipe — too broken, remove the line
            lines.pop()

    # Repair truncated list: last line is a bare number/bullet with no content
    if lines:
        last = lines[-1].strip()
        if re.match(r"^\d+[.)]\s*$", last) or re.match(r"^[-*]\s*$", last):
            lines.pop()

    # Repair orphan bold header at end (no content after it)
    # Already handled by _clean_trailing_empty_headers, but catch mid-structure cases
    if lines:
        last = lines[-1].strip()
        if re.match(r"^#{1,4}\s+\S.*$", last) and not any(l.strip() for l in lines[-1:]):
            pass  # Keep headers that are the last substantive line

    return "\n".join(lines)


def _detect_prose_entity_sections(text: str) -> Optional[str]:
    """Detect prose responses about multiple entities and restructure.

    When a comparison/ranking response mentions entities in prose without
    structure, auto-format into entity-per-section with bold headers.
    Returns restructured text, or None if not applicable.
    """
    # Find proper-noun entities (2+ word capitalized sequences)
    _entity_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    _skip = frozenset({
        "The", "This", "However", "Therefore", "Furthermore",
        "Additionally", "In Summary", "In Conclusion", "Based On",
    })

    entities: list[str] = []
    for m in _entity_re.finditer(text):
        name = m.group(1)
        if name not in _skip and len(name) > 4:
            if name not in entities:
                entities.append(name)

    # Need 2+ distinct entities to restructure
    if len(entities) < 2:
        return None

    # Check if entities appear as distinct sections already
    for e in entities:
        if f"**{e}**" in text:
            return None  # Already structured

    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 4:
        return None

    # Group sentences by entity mention
    entity_groups: dict[str, list[str]] = {e: [] for e in entities}
    ungrouped: list[str] = []

    for sent in sentences:
        matched = False
        for e in entities:
            if e.lower() in sent.lower():
                entity_groups[e].append(sent)
                matched = True
                break
        if not matched:
            ungrouped.append(sent)

    # Only restructure if most sentences are attributable to entities
    grouped_count = sum(len(v) for v in entity_groups.values())
    if grouped_count < len(sentences) * 0.5:
        return None

    # Build structured output
    parts: list[str] = []
    if ungrouped:
        parts.append(" ".join(ungrouped))
        parts.append("")

    for e in entities:
        sents = entity_groups[e]
        if sents:
            parts.append(f"**{e}:**")
            parts.append(" ".join(sents))
            parts.append("")

    return "\n".join(parts).strip()


__all__ = ["format_rag_v3_response"]
