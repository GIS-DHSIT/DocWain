from __future__ import annotations

import re

FALLBACK_ANSWER = "Not enough information in the documents to answer that."

# Matches C0/C1 control characters EXCEPT \t (0x09), \n (0x0a), \r (0x0d).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Stray 'n' from escaped newlines: "HANAn Sales" → "HANA Sales"
_STRAY_N_RE = re.compile(r"(?<=[a-zA-Z0-9.,:;)>])\s*\bn\s+(?=[A-Z])")
_LEADING_N_RE = re.compile(r"(?m)^\s*n\b\s+")

# Metadata key fragments that leak through rendering.
# Only match when followed by = or : (assignment pattern) to avoid stripping
# legitimate content that merely contains these words.
_METADATA_LEAK_RE = re.compile(
    r"\b(?:section_id|chunk_type|chunk_kind|section_kind|section_title"
    r"|page_start|page_end|start_page|end_page"
    r"|embed_pipeline_version|canonical_json|embedding_text|canonical_text"
    r"|doc_domain|doc_type|document_type"
    r"|subscription_id|profile_id|document_id"
    r"|layout_confidence|ocr_confidence)\s*[:=]\s*\S+",
    re.IGNORECASE,
)

# NER tag leaks from spaCy/internal processing: "person: John" or "organization: Google"
_NER_TAG_LEAK_RE = re.compile(
    r"\b(?:person|organization|location|date|money|percent|ordinal|cardinal"
    r"|norp|fac|gpe|loc|product|event|work_of_art|law|language|quantity"
    r"|time)\s*:\s*",
    re.IGNORECASE,
)

# Embedded metadata ID slugs: "id: workflowsandreducingmanual..." (run-on slugs)
# Match "id:" followed by a camelCase/concatenated slug (>10 chars, no spaces).
# Also match bullet points starting with "id:" to catch list-style leaks.
_ID_SLUG_RE = re.compile(r"\bid:\s*\S{10,}")
# Catch "id: CamelCaseSlug number" patterns (e.g. "id: reducingmissingdataby 40")
_ID_CAMEL_SLUG_RE = re.compile(r"\bid:\s*[a-z]{2}[a-zA-Z]{6,}(?:\s+\d+)?")


# Re-join hyphens broken by normalize_content: "self - employed" → "self-employed"
# Only for lowercase words (not date ranges like "2019 - 2023" or "Name - Title")
_BROKEN_HYPHEN_RE = re.compile(r"([a-z]) - ([a-z])")

# LLM meta-commentary patterns — strip "Based on my analysis..." openers
# Tightened: only match explicit self-referential meta-commentary, not factual
# statements that happen to start with "The documents show..."
_LLM_PREAMBLE_RE = re.compile(
    r"^(?:Based on (?:my |the )?(?:analysis|review|examination)(?: of (?:the )?(?:provided |available |given )?(?:documents?|evidence|information))?"
    r"|After (?:careful )?(?:reviewing|analyzing|examining|review|analysis) (?:of )?(?:the )?(?:provided |available )?(?:documents?|evidence)"
    r"|I (?:have |will |can )?(?:found|reviewed|analyzed|examined) (?:the )?(?:following|that)"
    r"|According to (?:my |the )?(?:analysis|review)"
    r"|From (?:the |my )?(?:analysis|review|provided |given )?(?:information|documents?)?"
    r"|Upon (?:review(?:ing)?|examin(?:ation|ing))"
    r"|Having (?:reviewed|analyzed|examined)"
    r"|Here (?:is|are) (?:the |what I )?(?:information|results?|findings?|details?) (?:found|extracted|I found)?"
    r"|In conclusion,? based on"
    r"|Let me (?:analyze|review|examine|summarize|break down|look at)"
    r"|Looking at (?:the )?(?:provided |available )?(?:documents?|evidence|information)"
    r"|The (?:provided |available )?(?:documents?|evidence) (?:show|indicate|reveal|suggest|contain))"
    r"[^.,\n]*[,.:;—]*\s*",
    re.IGNORECASE,
)

# OCR artifacts: common misrecognitions
_OCR_ARTIFACTS = [
    (re.compile(r"\bl\b(?=[A-Z])"), "I"),     # lowercase L before uppercase → I
    (re.compile(r"(?<=\d)O(?=\d)"), "0"),       # O between digits → 0
    (re.compile(r"(?<![A-Za-z])\brn\b(?![A-Za-z.,])"), "m"),  # rn → m (OCR) but not RN/prn/urn
    (re.compile(r"(?<=\d),(?=\d{3}\b)"), ","),  # preserve comma in numbers (no-op, ensures format)
]

# Double-space artifacts from PDF extraction
_DOUBLE_SPACE_RE = re.compile(r"(?<=[a-zA-Z])  +(?=[a-zA-Z])")


def _remove_repeated_content(text: str) -> str:
    """Remove duplicate sentences/paragraphs from LLM output.

    LLMs sometimes repeat the same sentence or paragraph verbatim.
    Strips exact duplicates while preserving order and structure.
    """
    if not text or len(text) < 80:
        return text

    # Split into paragraphs (separated by blank lines)
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) <= 1:
        # Try sentence-level dedup for single-paragraph text
        return _dedup_sentences(text)

    seen: set[str] = set()
    unique: list[str] = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        # Normalize for comparison: lowercase, collapse whitespace
        key = re.sub(r"\s+", " ", stripped.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(stripped)

    return "\n\n".join(unique)


def _dedup_sentences(text: str) -> str:
    """Remove duplicate sentences within a single paragraph."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 2:
        return text
    seen: set[str] = set()
    unique: list[str] = []
    for sent in sentences:
        key = re.sub(r"\s+", " ", sent.strip().lower())
        if key in seen and len(key) > 30:  # Only dedup substantial sentences
            continue
        seen.add(key)
        unique.append(sent)
    if len(unique) < len(sentences):
        return " ".join(unique)
    return text


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub("", str(text))

    # Re-join broken compound-word hyphens (position-altering — do BEFORE fence detection)
    cleaned = _BROKEN_HYPHEN_RE.sub(r"\1-\2", cleaned)
    # Strip stray 'n' artifacts (position-altering — do BEFORE fence detection)
    cleaned = _STRAY_N_RE.sub(" ", cleaned)
    cleaned = _LEADING_N_RE.sub("", cleaned)

    # Detect code-fence regions to skip OCR/artifact fixes inside them
    # Must be done AFTER all position-altering substitutions above
    _fence_re = re.compile(r"^```", re.MULTILINE)
    _fence_positions = [m.start() for m in _fence_re.finditer(cleaned)]
    _fenced_ranges: list[tuple[int, int]] = []
    _fence_open: int | None = None
    for pos in _fence_positions:
        if _fence_open is None:
            _fence_open = pos
        else:
            _fenced_ranges.append((_fence_open, pos))
            _fence_open = None
    # Unclosed code block — protect from fence to end of text
    if _fence_open is not None:
        _fenced_ranges.append((_fence_open, len(cleaned)))

    def _in_fence(pos: int) -> bool:
        return any(s <= pos <= e for s, e in _fenced_ranges)

    # Fix common OCR artifacts — skip inside code fences
    if not _fenced_ranges:
        for pattern, replacement in _OCR_ARTIFACTS:
            cleaned = pattern.sub(replacement, cleaned)
    else:
        for pattern, replacement in _OCR_ARTIFACTS:
            def _safe_sub(m: re.Match, _rep=replacement) -> str:
                if _in_fence(m.start()):
                    return m.group(0)  # preserve original inside code fence
                return _rep
            cleaned = pattern.sub(_safe_sub, cleaned)

    # Collapse double spaces (from PDF extraction) but not inside table rows
    # Apply per-line so pipe chars in source citations don't disable the fix globally
    _ds_lines = cleaned.split("\n")
    _ds_fixed = []
    for _ds_line in _ds_lines:
        _ds_stripped = _ds_line.strip()
        if _ds_stripped.startswith("|") and _ds_stripped.endswith("|"):
            _ds_fixed.append(_ds_line)  # preserve table rows
        else:
            _ds_fixed.append(_DOUBLE_SPACE_RE.sub(" ", _ds_line))
    cleaned = "\n".join(_ds_fixed)
    # Remove metadata key leaks (skip inside code fences)
    if not _fenced_ranges:
        cleaned = _METADATA_LEAK_RE.sub("", cleaned)
        cleaned = _NER_TAG_LEAK_RE.sub("", cleaned)
        cleaned = _ID_SLUG_RE.sub("", cleaned)
        cleaned = _ID_CAMEL_SLUG_RE.sub("", cleaned)
    else:
        def _meta_sub(m: re.Match) -> str:
            if _in_fence(m.start()):
                return m.group(0)
            return ""
        cleaned = _METADATA_LEAK_RE.sub(_meta_sub, cleaned)
        cleaned = _NER_TAG_LEAK_RE.sub(_meta_sub, cleaned)
        cleaned = _ID_SLUG_RE.sub(_meta_sub, cleaned)
        cleaned = _ID_CAMEL_SLUG_RE.sub(_meta_sub, cleaned)
    # Strip LLM meta-commentary preambles — only the first occurrence
    # (MULTILINE makes ^ match interior line starts; count=1 prevents
    # stripping legitimate mid-response sentences that match the pattern)
    _before_preamble = cleaned
    cleaned = _LLM_PREAMBLE_RE.sub("", cleaned, count=1)
    # Capitalize first letter after preamble stripping (only if something was stripped)
    if cleaned != _before_preamble:
        cleaned = cleaned.lstrip()
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
    # Strip hedging openers that survive the regex (common in GPT-style output)
    _hedging_prefixes = (
        "it appears that ", "it seems that ",
        "it is worth noting that ", "it's worth noting that ",
        "it should be noted that ", "it is important to note that ",
        "it's important to note that ",
    )
    # "Let me provide/present..." strips up to the next colon/period boundary
    _let_me_re = re.compile(
        r"^let me (?:provide|present|summarize|outline)\b[^:.\n]*[:.]?\s*",
        re.IGNORECASE,
    )
    _cl_let = _let_me_re.sub("", cleaned.lstrip())
    if _cl_let != cleaned.lstrip():
        cleaned = _cl_let
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
    _cl = cleaned.lstrip()
    for _hp in _hedging_prefixes:
        if _cl.lower().startswith(_hp):
            cleaned = _cl[len(_hp):]
            # Capitalize first letter of remaining text
            if cleaned and cleaned[0].islower():
                cleaned = cleaned[0].upper() + cleaned[1:]
            break
    # Remove repeated sentences/paragraphs (common LLM failure mode)
    cleaned = _remove_repeated_content(cleaned)
    # Normalize whitespace per line, but preserve blank lines needed for markdown
    # (tables, lists, and headers need preceding blank lines for proper rendering)
    lines = cleaned.splitlines()
    result_lines: list[str] = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Allow one blank line (needed for markdown formatting)
            if not prev_blank:
                result_lines.append("")
                prev_blank = True
            continue
        result_lines.append(stripped)
        prev_blank = False
    # Remove leading/trailing blanks
    while result_lines and result_lines[0] == "":
        result_lines.pop(0)
    while result_lines and result_lines[-1] == "":
        result_lines.pop()
    return "\n".join(result_lines) or ""


def sanitize(text: str) -> str:
    return sanitize_text(text)
