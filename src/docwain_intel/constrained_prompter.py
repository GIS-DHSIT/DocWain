"""Constrained LLM prompter — translates rendering spec + organized evidence into a
structured LLM prompt where the LLM fills content, not format."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# RenderingSpec — defined here until rendering_spec.py is created
# ---------------------------------------------------------------------------

class RenderingSpec(BaseModel):
    """Describes *how* the final answer should be laid out."""

    layout_mode: str = "narrative"  # single_value, card, table, narrative, list, timeline, comparison, summary
    field_ordering: List[str] = Field(default_factory=list)
    grouping_strategy: str = "none"
    detail_level: str = "standard"  # minimal, concise, standard, comprehensive
    use_headers: bool = True
    use_bold_values: bool = True
    use_table: bool = False
    max_items: Optional[int] = None
    include_provenance: bool = False
    include_gaps: bool = False


# ---------------------------------------------------------------------------
# Import evidence models from neighbouring module
# ---------------------------------------------------------------------------

from .evidence_organizer import (  # noqa: E402
    EvidenceGap,
    EvidenceGroup,
    OrganizedEvidence,
    ProvenanceRecord,
)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class ConstrainedPrompt(BaseModel):
    """Ready-to-send prompt payload for the LLM."""

    system_prompt: str
    user_prompt: str
    max_tokens: int
    temperature: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PROMPT_CHARS: int = 24_000

_SYSTEM_PROMPT = (
    "You are a precise document analyst. Answer ONLY using the provided evidence.\n"
    "Do NOT add preambles, greetings, or meta-commentary about your analysis.\n"
    "Do NOT invent or assume information not in the evidence.\n"
    "If information is missing, state it clearly.\n"
    "Cite source documents when relevant."
)

_DETAIL_TOKEN_MAP: Dict[str, int] = {
    "minimal": 256,
    "concise": 512,
    "standard": 1024,
    "comprehensive": 2048,
}

_TEMPERATURE_MAP: Dict[str, float] = {
    "single_value": 0.1,
    "card": 0.1,
    "table": 0.1,
    "comparison": 0.1,
    "narrative": 0.3,
    "summary": 0.3,
    "list": 0.2,
    "timeline": 0.2,
}

_DETAIL_INSTRUCTION: Dict[str, str] = {
    "minimal": "Be extremely brief — one or two sentences at most.",
    "concise": "Keep the answer concise — a short paragraph or a few bullet points.",
    "standard": "Provide a clear, moderately detailed answer.",
    "comprehensive": "Provide a thorough, comprehensive answer covering all relevant details.",
}


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

def _truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate *text* to *max_chars*, appending ellipsis if clipped."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_facts(facts: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for f in facts:
        predicate = f.get("predicate", "info")
        value = f.get("value", f.get("object_value", ""))
        lines.append(f"- **{predicate}**: {value}")
    return "\n".join(lines)


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        text = chunk.get("text", chunk.get("content", ""))
        text = _truncate_text(str(text), 500)
        source = (
            chunk.get("source_document")
            or chunk.get("metadata", {}).get("source_document")
            or chunk.get("payload", {}).get("source_document")
            or "unknown source"
        )
        lines.append(f"{idx}. [{source}] {text}")
    return "\n".join(lines)


def _format_evidence_section(evidence: OrganizedEvidence) -> str:
    """Format organized evidence into a text section for the prompt."""
    parts: List[str] = []

    for group in evidence.entity_groups:
        header = group.entity_text or group.entity_id or "Entity"
        parts.append(f"## {header}")
        if group.facts:
            parts.append(_format_facts(group.facts))
        if group.chunks:
            parts.append(_format_chunks(group.chunks))
        parts.append("")  # blank line between groups

    if evidence.ungrouped_chunks:
        parts.append("## Additional Evidence")
        parts.append(_format_chunks(evidence.ungrouped_chunks))
        parts.append("")

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Format instruction builder — algorithmically derived from spec
# ---------------------------------------------------------------------------

def _build_format_instructions(spec: RenderingSpec) -> str:
    """Build OUTPUT FORMAT instructions dynamically from the rendering spec."""
    lines: List[str] = []
    lines.append("OUTPUT FORMAT:")

    mode = spec.layout_mode

    if mode == "single_value":
        lines.append("Answer in one line only.")

    elif mode == "card":
        if spec.field_ordering:
            field_list = ", ".join(spec.field_ordering)
            lines.append(f"Present as labeled fields: {field_list}.")
        else:
            lines.append("Present as labeled fields, one per line.")
        if spec.use_bold_values:
            lines.append("Bold the field values.")

    elif mode == "table":
        if spec.field_ordering:
            cols = " | ".join(spec.field_ordering)
            lines.append(f"Present as a table with columns: {cols}.")
        else:
            lines.append("Present the data as a markdown table.")

    elif mode == "narrative":
        detail = spec.detail_level or "standard"
        lines.append(f"Write {detail} paragraphs using the evidence.")

    elif mode == "list":
        lines.append("Present the answer as a bulleted list.")
        if spec.max_items:
            lines.append(f"Include at most {spec.max_items} items.")

    elif mode == "timeline":
        lines.append("Present events in chronological order with dates.")

    elif mode == "comparison":
        if spec.field_ordering:
            cols = " | ".join(spec.field_ordering)
            lines.append(
                f"Present a side-by-side comparison table with columns: {cols}."
            )
        else:
            lines.append("Present a side-by-side comparison table.")

    elif mode == "summary":
        lines.append("Provide a concise summary of the evidence.")

    else:
        lines.append("Present the answer in a clear, readable format.")

    if spec.use_headers and mode not in ("single_value",):
        lines.append("Use section headers where appropriate.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Budget-aware evidence truncation
# ---------------------------------------------------------------------------

def _truncate_evidence_to_budget(
    evidence_text: str,
    other_text_len: int,
    max_chars: int = MAX_PROMPT_CHARS,
) -> str:
    """Trim evidence text so the total prompt stays within *max_chars*."""
    budget = max_chars - other_text_len
    if budget <= 0:
        return ""
    if len(evidence_text) <= budget:
        return evidence_text
    # Truncate to budget keeping whole lines where possible
    truncated = evidence_text[:budget]
    last_nl = truncated.rfind("\n")
    if last_nl > budget * 0.5:
        truncated = truncated[:last_nl]
    return truncated.rstrip() + "\n\n[Evidence truncated due to prompt length budget]"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_prompt(
    spec: RenderingSpec,
    evidence: OrganizedEvidence,
    query: str,
    *,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
) -> ConstrainedPrompt:
    """Build a constrained LLM prompt from a rendering spec and organized evidence."""

    # --- System prompt (always the same) ---
    system_prompt = _SYSTEM_PROMPT

    # --- User prompt parts ---
    user_parts: List[str] = []

    # 1. Query
    user_parts.append(f"QUESTION: {query}")

    # 2. Format instructions
    user_parts.append(_build_format_instructions(spec))

    # 3. Detail level instruction
    detail_instr = _DETAIL_INSTRUCTION.get(
        spec.detail_level, _DETAIL_INSTRUCTION["standard"]
    )
    user_parts.append(f"DETAIL LEVEL: {detail_instr}")

    # 4. Provenance instruction
    if spec.include_provenance:
        user_parts.append(
            "PROVENANCE: Cite the source document and page number for each claim."
        )

    # 5. Evidence section
    evidence_text = _format_evidence_section(evidence)

    # 6. Gap markers
    gap_text = ""
    if spec.include_gaps and evidence.gaps:
        gap_lines = [
            f"- {g.field_name}: {g.description}" for g in evidence.gaps
        ]
        gap_text = "NOTE: No evidence found for:\n" + "\n".join(gap_lines)

    # Assemble non-evidence parts to measure budget
    non_evidence = "\n\n".join(user_parts)
    if gap_text:
        non_evidence += "\n\n" + gap_text

    # Add evidence header
    evidence_header = "EVIDENCE:"
    overhead = len(system_prompt) + len(non_evidence) + len(evidence_header) + 10  # margins

    evidence_text = _truncate_evidence_to_budget(
        evidence_text, overhead, max_chars=max_prompt_chars
    )

    # Final user prompt assembly
    user_parts.append(f"{evidence_header}\n{evidence_text}")
    if gap_text:
        user_parts.append(gap_text)

    user_prompt = "\n\n".join(user_parts)

    # --- max_tokens ---
    max_tokens = _DETAIL_TOKEN_MAP.get(spec.detail_level, 1024)

    # --- temperature ---
    temperature = _TEMPERATURE_MAP.get(spec.layout_mode, 0.2)

    return ConstrainedPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
