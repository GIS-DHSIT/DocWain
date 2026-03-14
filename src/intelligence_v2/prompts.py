"""Prompt templates for LLM-powered document analysis."""

_MAX_TEXT_CHARS = 6_000

_ANALYSIS_TEMPLATE = """\
You are a document analysis engine. Analyze the following document and return \
a strict JSON object with exactly these keys:

{{
  "document_type": "<specific type, e.g. contract, invoice, research_paper, policy, manual>",
  "language": "<ISO 639-1 code, e.g. en, fr, de>",
  "summary": "<2-4 sentence summary of the entire document>",
  "section_summaries": {{"<section_title>": "<1-2 sentence summary>", ...}},
  "entities": [
    {{"type": "<person|organization|location|date|monetary|identifier|other>", "value": "<exact text>", "role": "<function of this entity in THIS document>"}}
  ],
  "facts": [
    {{"claim": "<factual statement>", "evidence": "<quote or location in the document>"}}
  ],
  "relationships": [
    {{"from": "<entity>", "relation": "<verb or label>", "to": "<entity>", "context": "<why this relationship matters>"}}
  ],
  "answerable_topics": ["<specific topic phrase a user might ask about>", ...]
}}

Rules:
- Only include facts explicitly stated in the text. Do not infer or speculate.
- Entity "role" must describe the function of the entity in THIS document (e.g. "buyer", "author", "subject of audit").
- Preserve exact monetary values, dates, and identifiers as they appear.
- answerable_topics must be specific enough to match real user queries (e.g. "Acme Corp Q3 2025 revenue" not "financial data").
- Return ONLY the JSON object, no extra text.

Filename: {filename}
Document type hint: {doc_type}

--- DOCUMENT TEXT ---
{text}
--- END DOCUMENT TEXT ---
"""


def build_analysis_prompt(text: str, filename: str, doc_type: str) -> str:
    """Build an LLM prompt for deep document analysis.

    Args:
        text: Raw document text (truncated to 32 000 chars).
        filename: Original filename for context.
        doc_type: Hint about the document type (e.g. "financial", "legal").

    Returns:
        Formatted prompt string.
    """
    truncated = text[:_MAX_TEXT_CHARS]
    return _ANALYSIS_TEMPLATE.format(
        text=truncated,
        filename=filename,
        doc_type=doc_type,
    )
