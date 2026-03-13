"""Single source of truth for all LLM prompts in DocWain.

Every prompt the system sends to the LLM is constructed here.
No other module should contain prompt text or LLM instructions.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# System prompt — used as the system message for every generation call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are DocWain, an expert document intelligence analyst.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Every claim must come from the provided evidence. Use exact values — "
    "'$125,000' not 'about $125K', 'March 15, 2025' not 'mid-March'.\n"
    "2. If evidence doesn't support an answer, say exactly what's missing.\n"
    "3. If sources conflict, report both with attribution: "
    "'Contract states $50K; Invoice shows $55K.'\n"
    "4. Answer ALL parts of the question. If asked about 3 things, cover all 3.\n"
    "5. Lead with the answer. No preamble. No 'Based on my analysis...'\n"
    "6. Cite sources inline as [SOURCE-N]. Every factual claim needs a citation.\n"
    "7. Bold key values with **value**. 2-3 bold items per paragraph max.\n\n"
    "FORMATTING:\n"
    "- Use markdown headers (## or ###) to organize multi-part answers.\n"
    "- Use bullet points for lists of 3+ items.\n"
    "- Use tables when comparing items across multiple dimensions.\n"
    "- Choose the format that best fits the content — vary between "
    "sections, tables, and bullet lists as appropriate.\n"
    "- Keep responses concise and well-structured.\n"
)


def build_system_prompt() -> str:
    """Return the core system prompt used for all generation calls."""
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Task-type formatting instructions
# ---------------------------------------------------------------------------

TASK_FORMATS: Dict[str, str] = {
    "extract": (
        "TASK: Extract the requested information precisely.\n"
        "- Present exact values from the documents.\n"
        "- Use a table if multiple fields are requested, key-value pairs if few.\n"
        "- Bold the extracted values.\n"
        "- If a requested field is not found, state: 'Not found in provided documents.'\n"
    ),
    "compare": (
        "TASK: Compare the subjects systematically.\n"
        "- Start with a one-line summary naming the key difference or winner.\n"
        "- Present comparison as a markdown table (subjects as rows, criteria as columns).\n"
        "- Bold the better value in each column.\n"
        "- If sources conflict on a value, report both with source attribution.\n"
        "- End with a brief synthesis of key differences.\n"
    ),
    "summarize": (
        "TASK: Provide a structured summary.\n"
        "- Start with a one-line overview of what was analyzed.\n"
        "- Present 3-6 key highlights as bullet points with specific details.\n"
        "- Bold the most important findings.\n"
        "- Include totals, counts, and ranges where applicable.\n"
        "- End with a brief conclusion.\n"
    ),
    "investigate": (
        "TASK: Investigate and assess the question.\n"
        "- Structure as: Finding → Evidence → Assessment.\n"
        "- Flag risks, inconsistencies, or concerns explicitly.\n"
        "- Distinguish between what the evidence shows vs. what it doesn't cover.\n"
        "- Be precise about severity: critical vs. minor vs. informational.\n"
    ),
    "lookup": (
        "TASK: Provide a direct factual answer.\n"
        "- Answer in 1-3 sentences maximum.\n"
        "- Include the exact value and its source.\n"
        "- No decoration, no extended analysis.\n"
    ),
    "aggregate": (
        "TASK: Aggregate and quantify from the evidence.\n"
        "- Lead with totals, counts, or computed values.\n"
        "- Show the breakdown (table if multi-item).\n"
        "- State which documents/sources contributed to each value.\n"
        "- Flag if any expected data is missing from the aggregation.\n"
    ),
    "list": (
        "TASK: List the requested items.\n"
        "- Use a numbered or bulleted list.\n"
        "- Include relevant details for each item (not just names).\n"
        "- Order by relevance or as requested.\n"
        "- State the total count at the top.\n"
    ),
}

# ---------------------------------------------------------------------------
# Output format instructions
# ---------------------------------------------------------------------------

_OUTPUT_FORMATS: Dict[str, str] = {
    "table": "Present results as a markdown table. All rows must have the same columns. Use 'N/A' for missing cells.",
    "bullets": "Present as a bulleted list. Most important items first. Each bullet should be self-contained.",
    "sections": "Organize with ## headers. Use bullets within sections. End with a synthesis.",
    "numbered": "Use a numbered list in sequential order.",
    "prose": "Write clear paragraphs. Lead with the answer, then supporting evidence.",
}

# ---------------------------------------------------------------------------
# UNDERSTAND prompt — analyzes user intent against document intelligence
# ---------------------------------------------------------------------------

_UNDERSTAND_SYSTEM = (
    "You are a document intelligence query analyzer. "
    "Given a user query, conversation history, and document metadata, "
    "produce a JSON analysis of what the user needs.\n\n"
    "Rules:\n"
    "- Decompose multi-part queries into sub-intents.\n"
    "- Resolve pronouns using conversation history.\n"
    "- Infer output format from query semantics "
    "(table for comparisons, bullets for lists, sections for summaries, prose for factual).\n"
    "- Assess complexity: 'simple' if single document/fact, 'complex' if cross-document or multi-step.\n"
    "- Identify which documents are relevant using the document intelligence metadata.\n"
    "- If the query is conversational (greeting, thanks, meta-question), set task_type to 'conversational'.\n"
)


def build_understand_prompt(
    query: str,
    doc_intelligence: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]],
) -> str:
    """Build the UNDERSTAND prompt that analyzes user intent.

    Args:
        query: The user's question.
        doc_intelligence: List of document intelligence dicts with keys:
            document_id, profile_id, profile_name, summary, entities,
            answerable_topics.
        conversation_history: Recent turns as [{"query": ..., "response": ...}].

    Returns:
        Complete prompt string for the UNDERSTAND LLM call.
    """
    parts = [_UNDERSTAND_SYSTEM, ""]

    # Conversation context
    if conversation_history:
        parts.append("CONVERSATION HISTORY:")
        for turn in conversation_history[-5:]:  # last 5 turns max
            parts.append(f"  User: {turn.get('query', '')}")
            resp = turn.get("response", "")
            if isinstance(resp, dict):
                resp = resp.get("response", str(resp))
            parts.append(f"  DocWain: {str(resp)[:300]}")
        parts.append("")

    # Document intelligence context
    if doc_intelligence:
        parts.append("AVAILABLE DOCUMENTS:")
        for doc in doc_intelligence:
            doc_id = doc.get("document_id", "unknown")
            profile = doc.get("profile_name", doc.get("profile_id", "unknown"))
            summary = doc.get("summary", "No summary available")
            entities = doc.get("entities", [])
            topics = doc.get("answerable_topics", [])

            parts.append(f"  [{doc_id}] Profile: {profile}")
            parts.append(f"    Summary: {summary}")
            if entities:
                entity_strs = [
                    e.get("name", str(e)) if isinstance(e, dict) else str(e)
                    for e in entities[:10]
                ]
                parts.append(f"    Entities: {', '.join(entity_strs)}")
            if topics:
                parts.append(f"    Topics: {', '.join(topics[:10])}")
            parts.append("")
    else:
        parts.append("AVAILABLE DOCUMENTS: None found in this subscription.\n")

    # Query
    parts.append(f"USER QUERY: {query}")
    parts.append("")

    # Expected output — JSON schema
    parts.append(
        "Respond ONLY with JSON (no markdown fences):\n"
        "{\n"
        '  "task_type": "extract | compare | summarize | investigate | lookup | aggregate | list | conversational",\n'
        '  "complexity": "simple | complex",\n'
        '  "resolved_query": "query with pronouns resolved from conversation",\n'
        '  "output_format": "table | bullets | sections | numbered | prose",\n'
        '  "relevant_documents": [\n'
        '    {"document_id": "...", "profile_id": "...", "reason": "why this doc is relevant"}\n'
        "  ],\n"
        '  "cross_profile": true | false,\n'
        '  "sub_tasks": ["sub-task 1", "sub-task 2"] | null,\n'
        '  "entities": ["entity1", "entity2"],\n'
        '  "needs_clarification": false,\n'
        '  "clarification_question": null\n'
        "}"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# REASON prompt — generates the answer from evidence
# ---------------------------------------------------------------------------


def build_reason_prompt(
    query: str,
    task_type: str,
    output_format: str,
    evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the REASON prompt that generates the final answer.

    Args:
        query: The resolved user query.
        task_type: From UNDERSTAND step (extract, compare, etc.).
        output_format: From UNDERSTAND step (table, bullets, etc.).
        evidence: Ranked evidence chunks, each with:
            source_name, section, page, text, score, source_index.
        doc_context: Aggregated document intelligence context with:
            summary, entities, key_facts (optional fields).
        conversation_history: Recent conversation turns as
            [{"query": ..., "response": ...}].

    Returns:
        Complete prompt string for the REASON LLM call.
    """
    parts = []

    # Document intelligence context (orientation before evidence)
    if doc_context:
        parts.append("--- DOCUMENT INTELLIGENCE ---")
        if doc_context.get("summary"):
            parts.append(f"Overview: {doc_context['summary']}")

        # Structured entity context
        entities = doc_context.get("entities")
        if entities:
            parts.append("")
            parts.append("## Known Entities")
            for e in entities[:15]:
                if isinstance(e, dict):
                    etype = e.get("type", "")
                    value = e.get("value", e.get("name", ""))
                    role = e.get("role", e.get("context", ""))
                    if etype and value:
                        entry = f"- {etype}: {value}"
                    elif value:
                        entry = f"- {value}"
                    else:
                        entry = f"- {etype}"
                    if role:
                        entry += f" ({role})"
                    parts.append(entry)
                else:
                    parts.append(f"- {e}")

        # Pre-extracted facts as grounding anchors
        key_facts = doc_context.get("key_facts")
        if key_facts:
            parts.append("")
            parts.append("## Pre-Extracted Facts (use as grounding anchors)")
            for fact in key_facts[:10]:
                if isinstance(fact, dict):
                    claim = fact.get("claim", str(fact))
                    evidence_ref = fact.get("evidence", "")
                    entry = f"- {claim}"
                    if evidence_ref:
                        entry += f" [{evidence_ref}]"
                    parts.append(entry)
                else:
                    parts.append(f"- {fact}")

        parts.append("--- END DOCUMENT INTELLIGENCE ---")
        parts.append("")

    # Evidence block
    parts.append("--- EVIDENCE ---")
    if evidence:
        for item in evidence:
            idx = item.get("source_index", 0)
            name = item.get("source_name", "unknown")
            section = item.get("section", "")
            page = item.get("page", "")
            score = item.get("score", 0)
            text = item.get("text", "")

            header_parts = [f"[SOURCE-{idx}]", name]
            if section:
                header_parts.append(f"| Section: {section}")
            if page:
                header_parts.append(f"| p.{page}")
            header_parts.append(f"(relevance: {score:.2f})")

            parts.append(" ".join(header_parts))
            parts.append(text)
            parts.append("")
    else:
        parts.append("No evidence found in the uploaded documents.")
        parts.append("")
    parts.append("--- END EVIDENCE ---")
    parts.append("")

    # Conversation context
    if conversation_history:
        parts.append("CONVERSATION CONTEXT:")
        for turn in conversation_history[-3:]:
            parts.append(f"  User: {turn.get('query', '')}")
            resp = turn.get("response", "")
            if isinstance(resp, dict):
                resp = resp.get("response", str(resp))
            parts.append(f"  DocWain: {str(resp)[:200]}")
        parts.append("")

    # Task instruction
    task_instruction = TASK_FORMATS.get(task_type, TASK_FORMATS["lookup"])
    parts.append(task_instruction)

    # Output format
    format_instruction = _OUTPUT_FORMATS.get(output_format, _OUTPUT_FORMATS["prose"])
    parts.append(f"FORMAT: {format_instruction}")
    parts.append("")

    # The question
    parts.append(f"QUESTION: {query}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sub-agent prompt — focused task for dynamic sub-agents
# ---------------------------------------------------------------------------


def build_subagent_prompt(
    role: str,
    evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a focused prompt for a dynamic sub-agent.

    Args:
        role: Description of what this sub-agent should do.
        evidence: Evidence chunks scoped to this sub-agent's task.
        doc_context: Document intelligence for relevant documents.

    Returns:
        Complete prompt string for the sub-agent LLM call.
    """
    parts = [
        "You are a document analysis sub-agent. Your specific task:",
        f"  {role}",
        "",
        "Rules: Use ONLY the evidence below. Be precise. Use exact values.",
        "If the evidence doesn't contain what's needed, say so.",
        "",
    ]

    # Document context
    if doc_context:
        if doc_context.get("summary"):
            parts.append(f"Document context: {doc_context['summary']}")
            parts.append("")

    # Evidence
    parts.append("--- EVIDENCE ---")
    if evidence:
        for item in evidence:
            idx = item.get("source_index", 0)
            name = item.get("source_name", "unknown")
            score = item.get("score", 0)
            text = item.get("text", "")

            parts.append(f"[SOURCE-{idx}] {name} (relevance: {score:.2f})")
            parts.append(text)
            parts.append("")
    else:
        parts.append("No evidence provided.")
        parts.append("")
    parts.append("--- END EVIDENCE ---")

    return "\n".join(parts)
