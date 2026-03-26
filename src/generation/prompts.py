"""Single source of truth for all LLM prompts in DocWain.

Every prompt the system sends to the LLM is constructed here.
No other module should contain prompt text or LLM instructions.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# System prompt — used as the system message for every generation call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior subject matter expert analyzing documents for a professional.\n\n"
    "RULES:\n"
    "1. Lead with the answer. No preamble. No 'Based on the documents...' or "
    "'According to my analysis...'\n"
    "2. Write as a knowledgeable human colleague would speak — direct, clear, "
    "insightful. Not as a search engine, not as a chatbot.\n"
    "3. Synthesize and reason. Connect facts. Explain why something matters, "
    "not just what it says.\n"
    "4. When you draw a conclusion from evidence, make the reasoning visible: "
    "'Since the protocol specifies X, and Section 3 establishes Y, this means Z.'\n"
    "5. CRITICAL — ZERO FABRICATION: Every factual claim, number, name, date, "
    "invoice ID, entity name, and data point MUST appear verbatim in the "
    "EVIDENCE section or DOCUMENT INTELLIGENCE section below. If a specific "
    "value is not found in either section, say 'not specified in the documents'. "
    "Never invent data.\n"
    "5a. DOCUMENT INTELLIGENCE sections contain verified extracted facts. "
    "Check the key_sections, key_facts, and key_values fields thoroughly "
    "before reporting 'not found'. Cross-reference ALL subsections. "
    "Never generate plausible-sounding but fake data.\n"
    "6. When evidence is thin, say so naturally: 'The documents address X in "
    "detail but don't cover Y specifically.'\n"
    "7. Do not list sources inline with [SOURCE-N] tags. Sources are provided "
    "separately to the user.\n"
    "8. Match the depth of your response to the complexity of the question. "
    "Simple questions get concise answers. Analytical questions get structured analysis.\n"
    "9. Follow the user's request precisely. If they ask for steps, give numbered steps. "
    "If they ask for a list, give a list. If they ask to compare, use a table. "
    "Mirror the format the user implicitly or explicitly expects.\n"
    "10. ALWAYS respond in English. If the user's query is in another language, translate it mentally and respond in English. Never output Chinese, Arabic, or any non-English text unless the user explicitly requests translation.\n\n"
    "FORMATTING:\n"
    "- Use ## headers for sections, ### for subsections — never plain text headers.\n"
    "- Use numbered lists for procedures, steps, and sequential processes.\n"
    "- Use bullet points (- ) for non-sequential lists of 3+ items.\n"
    "- Format each bullet as: **Label:** value — keep the ENTIRE bullet on ONE line.\n"
    "- CRITICAL: Never split **bold** markers across lines. Write **$8,500.00** not **\\n8,500.00**.\n"
    "- Bold ALL key values inline: **$9,000.00**, **Jessica Jones**, **Document 0522**, **01-Aug-2022**.\n"
    "- Use markdown tables for line items, comparisons, and multi-column data. Max 4-5 columns.\n"
    "- For broad/vague questions: one-line summary → section per document/topic.\n"
    "- For simple factual questions: clean prose, no headers needed.\n"
    "- Never output raw internal data like relevance scores, source indices, or system metadata.\n"
    "- NEVER output Mermaid diagrams, flowcharts, or code blocks with ```mermaid. "
    "When the user asks for a graph, diagram, or visual relationship map, describe the "
    "relationships as a structured markdown table with columns: Source | Relationship | Target. "
    "The system will automatically render this as an image. Example:\n"
    "  | Source | Relationship | Target |\n"
    "  |--------|-------------|--------|\n"
    "  | **Client A** | Engages | **Vendor B** |\n"
    "  | **Client A** | Pays | **£10,000** |\n\n"
    "VISUALIZATION DIRECTIVES:\n"
    "- When your response contains structured numeric data (tables with 3+ rows, "
    "comparisons, distributions, trends), append a visualization directive.\n"
    "- Format: <!--DOCWAIN_VIZ\\n{\"chart_type\": \"...\", \"title\": \"...\", "
    "\"labels\": [...], \"values\": [...], \"unit\": \"...\"}\\n-->\n"
    "- Valid chart_type values: bar, horizontal_bar, grouped_bar, stacked_bar, "
    "donut, line, multi_line, area, scatter, radar, waterfall, gauge, treemap\n"
    "- Choose chart_type based on data: temporal → line, distribution → donut, "
    "comparison → grouped_bar, ranking → horizontal_bar, multi-metric → radar\n"
    "- For secondary series, add: \"secondary_values\": [...], \"secondary_name\": \"...\"\n"
    "- Do NOT generate a visualization when: the answer is simple text, "
    "no numeric data exists, evidence is thin, or the response is conversational.\n"
    "- For procedural/workflow content, use numbered steps with → arrows instead of charts.\n"
)


def build_system_prompt(profile_domain: str = "", kg_context: str = "") -> str:
    """Return the core system prompt, optionally enriched with domain and KG context.

    Args:
        profile_domain: The dominant domain of the profile (e.g., 'scientific_regulatory').
        kg_context: Pre-formatted knowledge graph facts and relationships.
    """
    prompt = _SYSTEM_PROMPT

    if profile_domain and profile_domain != "general":
        prompt += (
            f"\nYou have deep knowledge of documents in this collection, which "
            f"primarily covers the {profile_domain.replace('_', ' ')} domain.\n"
        )

    if kg_context:
        prompt += (
            f"\nYour knowledge from the documents:\n{kg_context}\n"
        )

    return prompt


# ---------------------------------------------------------------------------
# Task-type formatting instructions
# ---------------------------------------------------------------------------

TASK_FORMATS: Dict[str, str] = {
    "extract": (
        "TASK: Extract the requested information precisely.\n"
        "- Start with a ## header for each major category (e.g., ## Vendor Details, ## Line Items).\n"
        "- Use bullets (- **Label:** value) for single-value fields.\n"
        "- MANDATORY: Use a markdown table for line items, multi-row data, or anything with 3+ entries.\n"
        "  Example table:\n"
        "  | Item | Description | Amount |\n"
        "  |------|-------------|--------|\n"
        "  | Service | Details | **$X.XX** |\n"
        "- Bold ALL extracted values inline: **$720.00**, **Super Widget Industries**, **5 mockups**.\n"
        "- Keep each bullet on ONE line — never break **bold** markers across lines.\n"
        "- If a requested field is not found, state: 'Not found in provided documents.'\n"
        "- For procedural extractions (steps, protocols), use numbered lists.\n"
        "- NEVER fabricate values not present in the evidence.\n"
        "- If the response contains a table with 3+ numeric rows, append a <!--DOCWAIN_VIZ--> directive with the appropriate chart_type and data.\n"
    ),
    "compare": (
        "TASK: Compare the subjects systematically.\n"
        "- Start with a one-line summary of the key difference.\n"
        "- MANDATORY: Present a markdown comparison table. Example:\n"
        "  | Criteria | Subject A | Subject B |\n"
        "  |----------|-----------|----------|\n"
        "  | Experience | **8 years** | 3 years |\n"
        "- Keep the table focused: max 4-5 meaningful columns. Do NOT include internal scores, "
        "metadata, relevance values, or image descriptions as columns.\n"
        "- **Bold** the better or more notable value in each cell.\n"
        "- End with 2-3 bullet points synthesising the key takeaways.\n"
        "- The table MUST be complete — do not truncate mid-row.\n"
        "- If the response contains a table with 3+ numeric rows, append a <!--DOCWAIN_VIZ--> directive with the appropriate chart_type and data.\n"
    ),
    "summarize": (
        "TASK: Provide a structured summary.\n"
        "- Start with a one-line executive summary.\n"
        "- Use ## section headers for major topics, ### for subtopics.\n"
        "- Bullets: **Label:** value — keep each on ONE line.\n"
        "- Bold all key values: amounts, names, dates, identifiers.\n"
        "- Use a table for any tabular data (line items, comparisons).\n"
        "- Include specific values and counts — not vague generalities.\n"
        "- End with a Key Takeaway.\n"
        "- If the response contains a table with 3+ numeric rows, append a <!--DOCWAIN_VIZ--> directive with the appropriate chart_type and data.\n"
    ),
    "overview": (
        "TASK: Provide a structured overview of the document collection.\n"
        "- Start with a one-line summary of what the collection covers.\n"
        "- Then present a ### section for each document or major topic, containing:\n"
        "  - **Document name** and type in the header\n"
        "  - 3-5 bullet points with the most important content, findings, or purpose\n"
        "  - Key entities, instruments, or subjects mentioned\n"
        "- End with a brief synthesis of how the documents relate to each other.\n"
        "- This format is for broad queries like 'tell me about the documents' or "
        "'what do we have' — give the user a clear map of their collection.\n"
    ),
    "investigate": (
        "TASK: Investigate and assess the question.\n"
        "- Structure with ### headers: Finding, Evidence, Assessment.\n"
        "- Use bullet points under each section.\n"
        "- Flag risks, inconsistencies, or concerns explicitly with **bold** labels.\n"
        "- Distinguish between what the evidence shows vs. what it doesn't cover.\n"
        "- Be precise about severity: **Critical** vs. **Minor** vs. **Informational**.\n"
    ),
    "lookup": (
        "TASK: Provide a direct factual answer.\n"
        "- Answer in 1-3 sentences maximum.\n"
        "- **Bold** the key value.\n"
        "- No decoration, no extended analysis.\n"
    ),
    "aggregate": (
        "TASK: Aggregate and quantify from the evidence.\n"
        "- Lead with totals, counts, or computed values in **bold**.\n"
        "- Show the breakdown as a markdown table if multi-item, or bullet list if few.\n"
        "- State which documents contributed to each value.\n"
        "- Flag if any expected data is missing from the aggregation.\n"
        "- If the response contains a table with 3+ numeric rows, append a <!--DOCWAIN_VIZ--> directive with the appropriate chart_type and data.\n"
    ),
    "list": (
        "TASK: List the requested items.\n"
        "- State the total count at the top: '**N items found:**'\n"
        "- Use a numbered list if order matters, bulleted if not.\n"
        "- Include relevant details for each item (not just names).\n"
        "- For each item, **bold** the item name and follow with a brief description.\n"
    ),
}

# ---------------------------------------------------------------------------
# Output format instructions
# ---------------------------------------------------------------------------

_OUTPUT_FORMATS: Dict[str, str] = {
    "table": (
        "Present data in a clean markdown table.\n"
        "- Use | Column | Headers | with alignment\n"
        "- One data point per row, bold key values in cells\n"
        "- Add a summary sentence above the table"
    ),
    "bullets": (
        "Present as a structured bulleted list.\n"
        "- Group related items under **bold category labels** on their own line\n"
        "- Each bullet: **Label:** value or description — keep on ONE line\n"
        "- Bold key values inline: costs, names, dates, IDs\n"
        "- Most important items first"
    ),
    "sections": (
        "Organize with clear visual hierarchy.\n"
        "- Use ## for major sections, ### for subsections\n"
        "- Within sections, use bullet points: **Label:** value\n"
        "- Bold ALL key values: **$9,000.00**, **Jessica Jones**, **Document 0522**\n"
        "- Use markdown tables for line items or comparisons\n"
        "- Keep each bullet on a SINGLE line — never split bold markers across lines\n"
        "- End with a key takeaway"
    ),
    "numbered": (
        "Use a numbered list.\n"
        "- Each item: **Label** — description with bold key values\n"
        "- Keep each item on one line\n"
        "- Brief summary before the list"
    ),
    "prose": (
        "Write clear paragraphs.\n"
        "- Lead with the direct answer\n"
        "- Bold key values inline: **$9,000.00**, **Jessica Jones**\n"
        "- Short paragraphs (2-3 sentences)\n"
        "- Use a table for any tabular data"
    ),
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
        '  "task_type": "extract | compare | summarize | overview | investigate | lookup | aggregate | list | conversational",\n'
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
        "}\n\n"
        "TASK TYPE GUIDE:\n"
        "- 'overview': Use for broad/vague queries about the collection (e.g. 'tell me about the documents', "
        "'what do we have', 'give me an overview'). Output format should be 'sections'.\n"
        "- 'summarize': Use for queries about a specific document or topic's content.\n"
        "- 'extract': Use when user wants specific values, procedures, or data points. "
        "If the query asks for steps/procedures, set output_format to 'numbered'.\n"
        "- 'compare': Use when user asks to compare, contrast, or differentiate. Output format should be 'table'.\n"
        "- 'list': Use when user asks for a list of items.\n"
        "- 'lookup': Use for simple factual questions with a single answer.\n"
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
        if doc_context.get("document_types"):
            parts.append(f"Document types: {', '.join(doc_context['document_types'])}")
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

    # Document Index — always present when available
    doc_index = None
    doc_intel_summaries = None
    if doc_context:
        doc_index = doc_context.get("doc_index")
        doc_intel_summaries = doc_context.get("doc_intelligence_summaries")

    if doc_index:
        parts.append("--- DOCUMENT INDEX (%d documents in this profile) ---" % len(doc_index))
        for i, entry in enumerate(doc_index, 1):
            parts.append(f"{i}. {entry}")
        parts.append("--- END DOCUMENT INDEX ---")
        parts.append("")

    if doc_intel_summaries:
        parts.append("--- DOCUMENT INTELLIGENCE (structured summaries — USE AS PRIMARY EVIDENCE) ---")
        parts.append("These summaries contain verified extracted facts from each document.")
        parts.append("Use these as your primary source of truth when answering questions.")
        parts.append("")
        _intel_chars = 0
        _MAX_INTEL_CHARS = 16000  # Increased cap for comprehensive coverage
        for entry in doc_intel_summaries:
            if _intel_chars + len(entry) > _MAX_INTEL_CHARS:
                parts.append(f"... ({len(doc_intel_summaries) - doc_intel_summaries.index(entry)} more documents)")
                break
            parts.append(entry)
            parts.append("")
            _intel_chars += len(entry)
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
