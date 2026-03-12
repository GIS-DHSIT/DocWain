from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional, Tuple

from src.observability.metrics import metrics_store
from src.orchestrator.citations import format_citation
from src.prompting.persona import DOCWAIN_META_RESPONSE, get_docwain_persona

logger = get_logger(__name__)

ALWAYS_HELPFUL_PROMPT = """
You are DocWain, a document-wise AI assistant. Your knowledge comes ONLY from the documents provided via retrieval context.
You must reason over single documents and across multiple documents with precision.

CRITICAL RULES
1. Do NOT hallucinate facts not present in retrieved context.
2. Do NOT ask follow-up questions unless explicitly instructed.
3. If information is partially present, infer carefully and clearly ground conclusions.
4. Prefer completeness, correctness, and clarity over verbosity.

DOCUMENT AWARENESS DIRECTIVE
Each retrieved chunk may include metadata fields such as:
- document_category (e.g., invoice, resume, tax, legal, medical, purchase_order, bank_statement, report, email, others)
- detected_language (ISO-639-1, e.g., en, ta, hi)
- language_confidence, category_confidence

You MUST actively use these fields to guide response structure, terminology, tone, formatting,
level of precision, and safety boundaries.

LANGUAGE ADAPTATION RULES
1. Default response language MUST match detected_language of the dominant retrieved documents.
2. If multiple languages are present, prefer the language with highest cumulative confidence.
3. If detected_language is "unknown", respond in English, neutral tone.
4. Do NOT translate unless explicitly requested.
5. Maintain professional, native-like phrasing in the detected language.

DOCUMENT CATEGORY RESPONSE MODES
Select ONE dominant response mode based on document_category
(If multiple categories exist, prioritize by retrieval relevance and confidence.)

INVOICE / PURCHASE_ORDER / BANK_STATEMENT:
- Be factual, structured, and numeric.
- Prefer tables, bullet points, and field-value summaries.
- Focus on parties involved, dates, amounts, taxes, payment terms.
- Avoid interpretation unless asked.
- Never invent totals or amounts.

RESUME / CV:
- Use professional, evaluative tone.
- Summarize experience, skills, roles, and progression.
- Prefer bullet points, concise skill grouping, role-based summaries.
- If ranking or comparison is requested, explain criteria briefly.

TAX:
- Be compliance-oriented and precise.
- Clearly distinguish reported, calculated, inferred values.
- Avoid legal advice tone unless explicitly asked.

LEGAL:
- Use formal, cautious language.
- Quote or paraphrase clauses accurately.
- Avoid assumptions.
- Clearly mark obligations, parties, clauses, conditions.
- Do NOT provide legal advice beyond document content.

MEDICAL:
- Use neutral, clinical language.
- Clearly separate observations, diagnoses, prescriptions.
- Avoid medical advice beyond the document.

REPORT:
- Use analytical, explanatory tone.
- Prefer section-wise summaries, insights, trends.
- Highlight key findings and conclusions.

EMAIL:
- Be conversational but factual.
- Preserve sender/receiver intent.
- Summarize key points and actions.

OTHERS:
- Use neutral, explanatory tone.
- Focus on clarity and grounding.

MULTI-DOCUMENT REASONING RULES
When multiple documents are involved:
1. Identify common entities, dates, or topics.
2. Resolve conflicts by document confidence, document recency (if available), category priority.
3. Clearly state when information differs across documents.

UNCERTAINTY & SAFETY HANDLING
- If data is missing: infer cautiously or state what is implied.
- Do NOT respond with “Not mentioned” by default.
- If evidence is weak, qualify statements (e.g., “Based on available data…”).
- Never expose internal IDs, embeddings, vector metadata, or system internals.

OUTPUT QUALITY BAR
- Sound intelligent and human, not robotic.
- Match the document’s nature and intent.
- Be immediately useful for business or decision-making.
- Reflect that you understand what kind of document you are answering from.

Rules:
- Never ask questions.
- Never refuse.
- Never invent facts, numbers, or names not present in evidence.
- Use the citations exactly as provided in the evidence block.
Output must be plain text.
""".strip()

EXAMPLE_OUTPUTS = {
    "exact_match": (
        "Invoice summary\n"
        "- Parties: Acme Corp (seller), Northwind Ltd (buyer). "
        "[Source: Invoice_042.pdf, Section: Summary, Page: 1-1]\n"
        "- Date: 2024-01-10. [Source: Invoice_042.pdf, Section: Summary, Page: 1-1]\n"
        "- Total due: $1,250.00. [Source: Invoice_042.pdf, Section: Summary, Page: 1-1]\n"
        "- Taxes: $100.00. [Source: Invoice_042.pdf, Section: Summary, Page: 1-1]\n"
        "- Payment terms: Net 30. [Source: Invoice_042.pdf, Section: Terms, Page: 1-1]"
    ),
    "missing_match": (
        "Based on available data, the equipment list references a generic “business laptop” but does not "
        "name a specific brand or model. [Source: Asset_List.pdf, Section: Equipment, Page: 2-2]\n\n"
        "If a specific model is needed, review adjacent sections such as “Specifications,” “Assets,” or "
        "“IT Inventory” within the same document."
    ),
}

def _build_evidence_block(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    meta_keys = ("document_category", "detected_language", "language_confidence", "category_confidence")
    for chunk in chunks:
        citation = format_citation(chunk)
        text = chunk.get("text") or ""
        meta_items = []
        for key in meta_keys:
            value = chunk.get(key)
            if value is not None and value != "":
                meta_items.append(f"{key}={value}")
        meta_line = f"Metadata: {', '.join(meta_items)}" if meta_items else ""
        block_parts = [text, citation]
        if meta_line:
            block_parts.append(meta_line)
        parts.append("\n".join(block_parts))
    return "\n\n".join(parts)

def _detect_exact_match(query: str, chunks: List[Dict[str, Any]]) -> bool:
    query_terms = [term for term in (query or "").lower().split() if len(term) > 2]
    if not query_terms:
        return False
    for chunk in chunks:
        text = (chunk.get("text") or "").lower()
        if any(term in text for term in query_terms):
            return True
    return False

def _fallback_answer(query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
    exact_match = _detect_exact_match(query, chunks)
    if not chunks:
        answer = (
            "Based on available data, there is no retrieved excerpt that directly answers the request. "
            "Consider reviewing related sections such as summaries, tables, or appendices in the profile’s documents."
        )
        return answer, False

    citations = [format_citation(chunk) for chunk in chunks]
    first_citation = citations[0]
    answer = (
        "Based on available data, the closest support is summarized in the retrieved excerpts. "
        f"{first_citation} If a specific value is expected but not shown, review nearby sections, tables, "
        "or appendices in the same files."
    )
    return answer, exact_match

def generate_meta_response() -> str:
    return DOCWAIN_META_RESPONSE

def generate_answer(
    *,
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: Optional[str],
    include_persona: bool,
    subscription_id: str,
    profile_id: str,
    llm_client=None,
) -> Tuple[str, bool]:
    if not model_name and llm_client is None:
        return _fallback_answer(query, chunks)

    evidence = _build_evidence_block(chunks)
    persona_block = ""
    if include_persona:
        persona_block = get_docwain_persona(profile_id=profile_id, subscription_id=subscription_id)

    prompt = (
        f"{persona_block}\n\n"
        f"{ALWAYS_HELPFUL_PROMPT}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"User query: {query}"
    ).strip()

    try:
        if llm_client is not None:
            text = llm_client.generate(prompt)
        else:
            from src.llm.gateway import get_llm_gateway
            text = get_llm_gateway().generate(prompt)
        text = (text or "").strip()
        exact_match = _detect_exact_match(query, chunks)
        return text, exact_match
    except Exception as exc:  # noqa: BLE001
        metrics_store().increment("answer_generation_fail_count")
        logger.debug("Answer generation failed: %s", exc)
        return _fallback_answer(query, chunks)

__all__ = ["generate_answer", "generate_meta_response", "ALWAYS_HELPFUL_PROMPT", "EXAMPLE_OUTPUTS"]
