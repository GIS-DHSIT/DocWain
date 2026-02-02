from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import ollama

from src.observability.metrics import metrics_store
from src.orchestrator.citations import format_citation
from src.prompting.persona import DOCWAIN_META_RESPONSE, get_docwain_persona

logger = logging.getLogger(__name__)

ALWAYS_HELPFUL_PROMPT = """
You are DocWain. Answer using ONLY the evidence provided. Follow the required structure:
A) Direct answer (best-supported conclusion)
B) Evidence-backed explanation (what was found, where)
C) Helpful outcome (next steps / how to interpret / what likely explains mismatch)
Rules:
- Never ask questions.
- Never refuse.
- Never invent facts, numbers, or names not present in evidence.
- If the exact answer is missing, provide the closest grounded evidence plus a helpful outcome.
- Use neutral phrasing like "I didn’t see an explicit mention of X".
- Use the citations exactly as provided in the evidence block.
Output must be plain text with labeled sections A, B, and C.
""".strip()

EXAMPLE_OUTPUTS = {
    "exact_match": (
        "A) Direct answer\n"
        "The invoice total is $1,250.00. [Source: Invoice_042.pdf, Section: Summary, Page: 1-1]\n\n"
        "B) Evidence-backed explanation\n"
        "The total appears in the invoice summary section as the final amount due. "
        "[Source: Invoice_042.pdf, Section: Summary, Page: 1-1]\n\n"
        "C) Helpful outcome\n"
        "This total reflects the full amount due on the invoice. If you need the tax breakdown, "
        "review the line-item table in the same document."
    ),
    "missing_match": (
        "A) Direct answer\n"
        "I didn’t see an explicit mention of a Lenovo model. The closest match is a "
        "“business laptop” reference in the equipment list. [Source: Asset_List.pdf, Section: Equipment, Page: 2-2]\n\n"
        "B) Evidence-backed explanation\n"
        "The equipment list includes a generic “business laptop” entry but does not name a brand or model. "
        "[Source: Asset_List.pdf, Section: Equipment, Page: 2-2]\n\n"
        "C) Helpful outcome\n"
        "If you are looking for a specific brand/model, check sections labeled “Specifications,” “Assets,” or "
        "“IT Inventory.” Brand names may appear as ThinkPad or IdeaPad rather than Lenovo."
    ),
}


def _build_evidence_block(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for chunk in chunks:
        citation = format_citation(chunk)
        text = chunk.get("text") or ""
        parts.append(f"{text}\n{citation}")
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
            "A) Direct answer\n"
            "I didn’t see an explicit answer in the available documents.\n\n"
            "B) Evidence-backed explanation\n"
            "No relevant excerpts were retrieved from the selected profile documents.\n\n"
            "C) Helpful outcome\n"
            "Consider checking related sections such as summaries, tables, or appendices within the profile’s documents."
        )
        return answer, False

    citations = [format_citation(chunk) for chunk in chunks]
    first_citation = citations[0]
    answer = (
        "A) Direct answer\n"
        f"Based on the available documents, the closest support is summarized below. {first_citation}\n\n"
        "B) Evidence-backed explanation\n"
        f"The retrieved excerpts relate to the query and are cited here. {first_citation}\n\n"
        "C) Helpful outcome\n"
        "If a specific value is expected but not shown, review the surrounding sections, tables, or appendices in the same files."
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
) -> Tuple[str, bool]:
    if not model_name:
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
        response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0})
        text = (response.get("response") or "").strip()
        exact_match = _detect_exact_match(query, chunks)
        return text, exact_match
    except Exception as exc:  # noqa: BLE001
        metrics_store().increment("answer_generation_fail_count")
        logger.debug("Answer generation failed: %s", exc)
        return _fallback_answer(query, chunks)


__all__ = ["generate_answer", "generate_meta_response", "ALWAYS_HELPFUL_PROMPT", "EXAMPLE_OUTPUTS"]
