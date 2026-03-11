"""Output quality engine — closed-loop validation of LLM output against rendering spec.

Runs three validation passes (structural conformance, content integrity, cleanliness)
and produces a cleaned, validated output with quality metrics.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from .evidence_organizer import OrganizedEvidence
from .models import ExtractionResult
from .rendering_spec import RenderingSpec


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class QualityResult(BaseModel):
    cleaned_text: str = ""
    original_text: str = ""
    was_modified: bool = False
    issues_found: List[str] = Field(default_factory=list)
    claims_verified: int = 0
    claims_unverified: int = 0
    structural_conformance: float = 1.0
    content_integrity: float = 1.0


# ---------------------------------------------------------------------------
# Meta-commentary patterns
# ---------------------------------------------------------------------------

_META_STARTERS = (
    "based on",
    "according to the",
    "from the provided",
    "the documents show",
    "after reviewing",
    "here are",
    "i found",
    "let me",
    "as per",
)

_META_RE = re.compile(
    r"^(?:based on|according to the|from the provided|the documents show|"
    r"after reviewing|here are|i found|let me|as per)\b",
    re.IGNORECASE,
)

# Patterns for sentences that talk about the act of answering.
_ANSWERING_RE = re.compile(
    r"(?:I (?:will|can|would|shall) (?:provide|present|summarize|list|show))|"
    r"(?:the (?:following|above|below) (?:information|data|details|results))",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_claims(text: str) -> List[str]:
    """Split text into claim-like sentences (factual statements with entities or values)."""
    if not text or not text.strip():
        return []

    # Split on sentence boundaries, preserving table rows as individual claims.
    lines = text.strip().split("\n")
    claims: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Table rows: treat each data row as a claim.
        if stripped.startswith("|") and stripped.endswith("|"):
            # Skip separator rows.
            if set(stripped.replace("|", "").strip()) <= {"-", " ", ":"}:
                continue
            claims.append(stripped)
            continue

        # Split multi-sentence lines.
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # A claim-like sentence contains a proper noun (capitalized word not at start)
            # or a numeric value.
            has_entity = bool(re.search(r"(?<!\A)[A-Z][a-z]+", sent))
            has_value = bool(re.search(r"\d", sent))
            has_bullet = sent.startswith(("- ", "* ", "• "))
            if has_entity or has_value or has_bullet:
                claims.append(sent)

    return claims


def _verify_claim(
    claim: str,
    evidence: OrganizedEvidence,
    extraction: Optional[ExtractionResult] = None,
) -> bool:
    """Check if a claim is grounded in the evidence or extraction data."""
    claim_lower = claim.strip().lower()

    # Build a corpus of evidence text fragments for matching.
    evidence_texts: List[str] = []

    for grp in evidence.entity_groups:
        if grp.entity_text:
            evidence_texts.append(grp.entity_text.lower())
        for fact in grp.facts:
            for key in ("predicate", "value", "object_value", "raw_text"):
                val = fact.get(key)
                if val:
                    evidence_texts.append(str(val).lower())
        for chunk in grp.chunks:
            text = chunk.get("text") or chunk.get("content") or ""
            if text:
                evidence_texts.append(text.lower())

    for chunk in evidence.ungrouped_chunks:
        text = chunk.get("text") or chunk.get("content") or ""
        if text:
            evidence_texts.append(text.lower())

    # Check extraction entities and facts.
    if extraction:
        for entity in extraction.entities:
            evidence_texts.append(entity.text.lower())
            for alias in entity.aliases:
                evidence_texts.append(alias.lower())
        for fact in extraction.facts:
            evidence_texts.append(fact.raw_text.lower())
            if fact.object_value:
                evidence_texts.append(fact.object_value.lower())

    if not evidence_texts:
        return False

    # Extract substantive tokens from the claim (proper nouns, numbers).
    proper_nouns = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", claim)
    numbers = re.findall(r"\d[\d,.]*", claim)

    substantive_parts = [pn.lower() for pn in proper_nouns] + numbers
    if not substantive_parts:
        # No verifiable content — treat as verified (non-factual sentence).
        return True

    evidence_blob = " ".join(evidence_texts)

    # At least one substantive part must appear in evidence.
    for part in substantive_parts:
        if part.lower() in evidence_blob:
            return True

    return False


def _strip_meta_commentary(text: str) -> str:
    """Remove LLM preambles, meta-commentary, and format artifacts."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Check if this line starts with meta-commentary.
        if stripped and _META_RE.match(stripped):
            # If it ends with ":", skip entirely (it's a pure preamble).
            if stripped.endswith(":"):
                continue
            # Otherwise strip the preamble portion up to the first comma or colon,
            # but only if there's substantive content after.
            m = re.match(
                r"^(?:based on|according to the|from the provided|the documents show|"
                r"after reviewing|here are|i found|let me|as per)[^,:]*[,:]?\s*",
                stripped,
                re.IGNORECASE,
            )
            if m:
                remainder = stripped[m.end():].strip()
                if remainder:
                    cleaned_lines.append(remainder)
                continue

        # Check for sentences referencing the act of answering.
        if stripped and _ANSWERING_RE.search(stripped):
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    # Normalize multiple blank lines to at most one.
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Strip trailing whitespace from lines.
    result = "\n".join(l.rstrip() for l in result.split("\n"))

    # Strip leading/trailing whitespace.
    result = result.strip()

    return result


def _restructure_to_spec(text: str, spec: RenderingSpec) -> str:
    """Attempt to restructure non-conformant output to match the spec layout."""
    if not text or not text.strip():
        return text

    layout = spec.layout_mode

    if layout == "table" or (layout == "comparison" and spec.use_table):
        return _convert_to_table(text)
    elif layout == "single_value":
        return _extract_single_value(text)
    elif layout == "card":
        return _convert_to_card(text)
    elif layout == "list":
        return _convert_to_list(text)

    return text


def _convert_to_table(text: str) -> str:
    """Convert bullet/prose output to a markdown table."""
    # Already a table?
    if "|" in text and "---" in text:
        return text

    # Parse lines that look like key-value or bullet items.
    rows: List[tuple] = []
    lines = text.strip().split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Bullet items: "- **Key**: Value" or "- Key: Value"
        m = re.match(r"^[-*•]\s+\*?\*?(.+?)\*?\*?\s*:\s*(.+)$", stripped)
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # Plain "Key: Value".
        m = re.match(r"^(.+?):\s+(.+)$", stripped)
        if m and len(m.group(1)) < 50:
            rows.append((m.group(1).strip(), m.group(2).strip()))
            continue

    if not rows:
        return text

    # Build markdown table.
    header = "| Field | Value |"
    separator = "| --- | --- |"
    data_rows = [f"| {k} | {v} |" for k, v in rows]

    return "\n".join([header, separator] + data_rows)


def _extract_single_value(text: str) -> str:
    """Extract just the core value from output that has preamble."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    if not lines:
        return text

    # If already short, return as-is.
    if len(lines) <= 2:
        return text.strip()

    # Try to find the substantive line (not meta-commentary).
    for line in lines:
        if not _META_RE.match(line) and not line.endswith(":"):
            return line

    return lines[-1]


def _convert_to_card(text: str) -> str:
    """Convert plain text to bold-label card format."""
    if "**" in text:
        return text

    lines = text.strip().split("\n")
    converted: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            converted.append("")
            continue

        # "Key: Value" → "**Key**: Value"
        m = re.match(r"^[-*•]?\s*(.+?):\s+(.+)$", stripped)
        if m and len(m.group(1)) < 50:
            converted.append(f"**{m.group(1).strip()}**: {m.group(2).strip()}")
        else:
            converted.append(stripped)

    return "\n".join(converted)


def _convert_to_list(text: str) -> str:
    """Convert prose to bullet list."""
    # Already has bullets?
    if any(line.strip().startswith(("- ", "* ", "• ", "1.")) for line in text.split("\n") if line.strip()):
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text

    return "\n".join(f"- {s.strip()}" for s in sentences if s.strip())


# ---------------------------------------------------------------------------
# Structural conformance checks
# ---------------------------------------------------------------------------

def _check_structural_conformance(text: str, spec: RenderingSpec) -> tuple:
    """Check how well output matches spec. Returns (score, issues)."""
    if not text.strip():
        return 0.0, ["Output is empty"]

    checks_passed = 0
    checks_total = 0
    issues: List[str] = []

    layout = spec.layout_mode

    # Check layout-specific features.
    if layout == "table" or spec.use_table:
        checks_total += 1
        if "|" in text and "---" in text:
            checks_passed += 1
        else:
            issues.append("Table layout expected but no markdown table found")

    if layout == "single_value":
        checks_total += 1
        line_count = len([l for l in text.strip().split("\n") if l.strip()])
        if line_count <= 2:
            checks_passed += 1
        else:
            issues.append(f"Single value expected but output has {line_count} lines")

    if layout == "card":
        checks_total += 1
        if "**" in text and ":" in text:
            checks_passed += 1
        else:
            issues.append("Card layout expected but no bold labels found")

    if layout == "list":
        checks_total += 1
        has_bullets = any(
            l.strip().startswith(("- ", "* ", "• ", "1.", "2.", "3."))
            for l in text.split("\n") if l.strip()
        )
        if has_bullets:
            checks_passed += 1
        else:
            issues.append("List layout expected but no bullet points found")

    if layout == "narrative":
        checks_total += 1
        # Narrative should have prose (sentences, not just bullets/tables).
        if len(text.strip()) > 10:
            checks_passed += 1
        else:
            issues.append("Narrative layout expected but output is too short")

    if layout in ("timeline", "comparison", "summary"):
        checks_total += 1
        if spec.use_headers:
            if re.search(r"^#{1,3}\s|\*\*[^*]+\*\*", text, re.MULTILINE):
                checks_passed += 1
            else:
                issues.append(f"{layout.title()} layout expected headers but none found")
        else:
            checks_passed += 1

    # Check field ordering: are expected fields present?
    if spec.field_ordering:
        checks_total += 1
        text_lower = text.lower()
        found = sum(1 for f in spec.field_ordering if f.lower() in text_lower)
        missing = [f for f in spec.field_ordering if f.lower() not in text_lower]
        if missing:
            issues.append(f"Missing expected fields: {', '.join(missing)}")
        if found == len(spec.field_ordering):
            checks_passed += 1
        elif found > 0:
            checks_passed += 0.5  # Partial credit.

    # Use bold values check.
    if spec.use_bold_values:
        checks_total += 1
        if "**" in text:
            checks_passed += 1
        else:
            issues.append("Bold values expected but none found")

    if checks_total == 0:
        return 1.0, issues

    return checks_passed / checks_total, issues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_output(
    llm_output: str,
    spec: RenderingSpec,
    evidence: OrganizedEvidence,
    extraction: Optional[ExtractionResult] = None,
) -> QualityResult:
    """Validate and clean LLM output against the rendering spec and evidence.

    Runs three passes:
      1. Structural conformance — does output shape match the spec?
      2. Content integrity — are claims grounded in evidence?
      3. Cleanliness — strip meta-commentary and format artifacts.
    """
    original = llm_output or ""

    # Handle empty output gracefully.
    if not original.strip():
        return QualityResult(
            cleaned_text="",
            original_text=original,
            was_modified=bool(original),
            issues_found=["Output is empty"],
            claims_verified=0,
            claims_unverified=0,
            structural_conformance=0.0,
            content_integrity=0.0,
        )

    issues: List[str] = []
    working_text = original

    # --- Pass 1: Structural conformance ---
    struct_score, struct_issues = _check_structural_conformance(working_text, spec)
    issues.extend(struct_issues)

    if struct_score < 1.0:
        restructured = _restructure_to_spec(working_text, spec)
        if restructured != working_text:
            working_text = restructured
            # Re-check after restructuring.
            struct_score, _ = _check_structural_conformance(working_text, spec)

    # --- Pass 2: Content integrity ---
    claims = _extract_claims(working_text)
    verified = 0
    unverified = 0

    for claim in claims:
        if _verify_claim(claim, evidence, extraction):
            verified += 1
        else:
            unverified += 1

    total_claims = verified + unverified
    if total_claims > 0:
        integrity_score = verified / total_claims
    else:
        integrity_score = 1.0  # No claims to verify — no integrity issues.

    # --- Pass 3: Cleanliness ---
    working_text = _strip_meta_commentary(working_text)

    # Fix unclosed markdown.
    open_bold = working_text.count("**")
    if open_bold % 2 != 0:
        working_text += "**"
        issues.append("Fixed unclosed bold markdown")

    open_italic = working_text.count("*") - working_text.count("**") * 2
    if open_italic % 2 != 0:
        working_text += "*"
        issues.append("Fixed unclosed italic markdown")

    open_backtick = working_text.count("`") - working_text.count("```") * 3
    if open_backtick % 2 != 0:
        working_text += "`"
        issues.append("Fixed unclosed backtick")

    # Final normalization.
    working_text = re.sub(r"\n{3,}", "\n\n", working_text)
    working_text = "\n".join(l.rstrip() for l in working_text.split("\n"))
    working_text = working_text.strip()

    was_modified = working_text != original.strip()

    return QualityResult(
        cleaned_text=working_text,
        original_text=original,
        was_modified=was_modified,
        issues_found=issues,
        claims_verified=verified,
        claims_unverified=unverified,
        structural_conformance=struct_score,
        content_integrity=integrity_score,
    )


__all__ = [
    "QualityResult",
    "validate_output",
]
