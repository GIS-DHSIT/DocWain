"""Single source of truth for all LLM prompts in the DocWain Core Agent pipeline.

Every LLM call — UNDERSTAND, REASON, sub-agent — draws its prompt from
the functions and constants defined here.  No other module should contain
inline prompt text.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Core system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are DocWain, an AI document intelligence assistant.

GROUNDING RULES — follow these without exception:
1. Use EXACT values from the provided evidence. Never paraphrase numbers, dates, \
names, or identifiers — reproduce them verbatim.
2. CITE every claim using [SOURCE-N] references. If multiple sources support a \
claim, cite all of them.
3. When sources CONFLICT, explicitly report the conflict and list the differing \
values with their sources.
4. Provide COMPLETE answers that address every part of the user's question. Do \
not leave sub-questions unanswered.
5. No preamble. Do not start with "Sure", "Great question", or similar filler. \
Begin directly with the answer.
6. If the evidence is insufficient to answer, say so clearly and state what \
information is missing.
"""


def build_system_prompt() -> str:
    """Return the core system prompt used for all generation calls."""
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Task-type formatting instructions
# ---------------------------------------------------------------------------

TASK_FORMATS: Dict[str, str] = {
    "extract": (
        "Extract the requested values from the evidence. Return each value "
        "exactly as it appears in the source, with its [SOURCE-N] citation."
    ),
    "compare": (
        "Compare the items across the evidence. Highlight similarities and "
        "differences. Use a structured layout (table or side-by-side) when "
        "there are more than two attributes to compare."
    ),
    "summarize": (
        "Provide a concise summary that captures the key points from the "
        "evidence. Prioritize the most important information first."
    ),
    "investigate": (
        "Investigate the question thoroughly using all available evidence. "
        "Connect related facts, identify patterns, and draw supported "
        "conclusions. Flag any gaps."
    ),
    "lookup": (
        "Look up the specific fact or value requested. Return the exact "
        "value with its source citation. If multiple values exist, list "
        "all of them."
    ),
    "aggregate": (
        "Aggregate the relevant data points from the evidence. When "
        "performing calculations, show the values used and cite each. "
        "Clearly label any derived figures."
    ),
    "list": (
        "List the requested items from the evidence. Use a clear, "
        "consistent format. Include source citations for each item."
    ),
}

# ---------------------------------------------------------------------------
# Output format instructions
# ---------------------------------------------------------------------------

_OUTPUT_FORMATS: Dict[str, str] = {
    "table": (
        "Format the answer as a Markdown table with clear column headers."
    ),
    "bullets": (
        "Format the answer as a bulleted list. Each bullet should be a "
        "self-contained point."
    ),
    "sections": (
        "Organize the answer into clearly titled sections using Markdown "
        "headings (##)."
    ),
    "numbered": (
        "Format the answer as a numbered list."
    ),
    "prose": (
        "Write the answer as flowing prose paragraphs."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trim_history(
    history: Optional[List[Dict[str, str]]],
    max_turns: int,
) -> List[Dict[str, str]]:
    """Return the last *max_turns* items from conversation history."""
    if not history:
        return []
    return history[-max_turns:]


def _format_history(turns: List[Dict[str, str]]) -> str:
    """Render conversation turns into a prompt-friendly block."""
    if not turns:
        return ""
    lines: List[str] = []
    for turn in turns:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    """Number evidence chunks as [SOURCE-N] with metadata."""
    if not evidence:
        return "(No evidence provided.)"
    blocks: List[str] = []
    for idx, item in enumerate(evidence, start=1):
        header_parts: List[str] = [f"[SOURCE-{idx}]"]
        if item.get("source_name"):
            header_parts.append(f"source_name={item['source_name']}")
        if item.get("section"):
            header_parts.append(f"section={item['section']}")
        if item.get("page") is not None:
            header_parts.append(f"page={item['page']}")
        if item.get("relevance_score") is not None:
            header_parts.append(f"relevance={item['relevance_score']}")
        header = " | ".join(header_parts)
        text = item.get("text", "")
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _format_doc_context(doc_context: Optional[Dict[str, Any]]) -> str:
    """Render document intelligence context as key-value lines."""
    if not doc_context:
        return ""
    lines: List[str] = []
    for key, value in doc_context.items():
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_doc_intelligence(doc_intel: Optional[Dict[str, Any]]) -> str:
    """Render document intelligence metadata for UNDERSTAND prompt."""
    if not doc_intel:
        return "(No document intelligence available.)"
    lines: List[str] = []
    for key, value in doc_intel.items():
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "(No document intelligence available.)"


# ---------------------------------------------------------------------------
# UNDERSTAND prompt
# ---------------------------------------------------------------------------


def build_understand_prompt(
    query: str,
    doc_intelligence: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the UNDERSTAND prompt for intent analysis.

    Parameters
    ----------
    query:
        The user's raw question.
    doc_intelligence:
        Metadata about the document corpus (topics, doc types, profiles, etc.).
    conversation_history:
        Prior turns of conversation; last 5 are included.

    Returns
    -------
    str
        The fully assembled UNDERSTAND prompt.
    """
    history_block = _format_history(_trim_history(conversation_history, 5))
    doc_intel_block = _format_doc_intelligence(doc_intelligence)

    sections: List[str] = ["## UNDERSTAND — Intent Analysis"]

    if history_block:
        sections.append(f"### Conversation History\n{history_block}")

    sections.append(f"### Document Intelligence\n{doc_intel_block}")
    sections.append(
        f"### User Query\n{query}\n\n"
        "Analyze this query against the document intelligence above. "
        "Determine:\n"
        "1. The user's intent and what information they need.\n"
        "2. The task type (extract, compare, summarize, investigate, "
        "lookup, aggregate, list).\n"
        "3. The optimal output format (table, bullets, sections, "
        "numbered, prose).\n"
        "4. Key entities, constraints, and filters to apply during "
        "retrieval."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# REASON prompt
# ---------------------------------------------------------------------------


def build_reason_prompt(
    query: str,
    task_type: str,
    output_format: str,
    evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the REASON prompt for answer generation.

    Parameters
    ----------
    query:
        The user's question.
    task_type:
        One of the TASK_FORMATS keys.
    output_format:
        One of the _OUTPUT_FORMATS keys.
    evidence:
        Retrieved and reranked evidence chunks.
    doc_context:
        Document intelligence context (orientation).
    conversation_history:
        Prior turns; last 3 are included.

    Returns
    -------
    str
        The fully assembled REASON prompt.
    """
    history_block = _format_history(_trim_history(conversation_history, 3))
    doc_ctx_block = _format_doc_context(doc_context)
    evidence_block = _format_evidence(evidence)
    task_instruction = TASK_FORMATS.get(task_type, TASK_FORMATS["extract"])
    format_instruction = _OUTPUT_FORMATS.get(output_format, _OUTPUT_FORMATS["prose"])

    sections: List[str] = ["## REASON — Answer Generation"]

    # Task + output instructions
    sections.append(
        f"### Task Instructions\n{task_instruction}\n\n"
        f"### Output Format\n{format_instruction}"
    )

    # Conversation history (if any)
    if history_block:
        sections.append(f"### Conversation History\n{history_block}")

    # Document context BEFORE evidence (orientation first)
    if doc_ctx_block:
        sections.append(f"### Document Context\n{doc_ctx_block}")

    # Evidence
    sections.append(f"### Evidence\n{evidence_block}")

    # Query
    sections.append(f"### Query\n{query}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Sub-agent prompt
# ---------------------------------------------------------------------------


def build_subagent_prompt(
    role: str,
    evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a focused prompt for a dynamic sub-agent.

    Parameters
    ----------
    role:
        The specialist role the sub-agent should adopt (e.g.,
        "financial analyst", "legal reviewer").
    evidence:
        Evidence chunks relevant to the sub-agent's task.
    doc_context:
        Optional document intelligence context.

    Returns
    -------
    str
        The fully assembled sub-agent prompt.
    """
    evidence_block = _format_evidence(evidence)
    doc_ctx_block = _format_doc_context(doc_context)

    sections: List[str] = [
        f"## Sub-Agent: {role.title()}",
        f"You are a specialist {role.lower()}. Analyze the evidence below "
        "from your domain expertise. Provide precise, well-cited findings.",
    ]

    if doc_ctx_block:
        sections.append(f"### Document Context\n{doc_ctx_block}")

    sections.append(f"### Evidence\n{evidence_block}")

    return "\n\n".join(sections)
