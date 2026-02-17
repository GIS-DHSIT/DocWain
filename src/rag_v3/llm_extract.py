"""LLM-first generic extraction.

Replaces domain-specific regex extraction with a single LLM call that reads
the retrieved chunks and produces the final answer directly.  The answer is
returned inside an ``LLMResponseSchema`` so the pipeline can skip the separate
render step.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .types import LLMBudget, LLMResponseSchema

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────
LLM_EXTRACT_TIMEOUT_S = 60.0
LLM_EXTRACT_INTERMEDIATE_TIMEOUT_S = 30.0
LLM_MAX_OUTPUT_TOKENS = 2048
LLM_MAX_CONTEXT_CHARS = 6144
LLM_MAX_CHUNKS = 8  # Send only top-scored chunks to LLM
LLM_CHUNKED_TOKEN_THRESHOLD = 2000  # Estimated token count to trigger chunked extraction
_CHARS_PER_TOKEN = 4  # Rough char-to-token ratio for estimation


def llm_extract_and_respond(
    *,
    query: str,
    chunks: List[Any],
    llm_client: Any,
    budget: LLMBudget,
    correlation_id: Optional[str] = None,
    intent: Optional[str] = None,
    num_documents: int = 1,
) -> Optional[LLMResponseSchema]:
    """Answer *query* from *chunks* via a single LLM call with intent-adaptive prompts."""
    if not budget.consume():
        return None

    # Use expanded 8-type intent classification
    intent_type = classify_query_intent(query, intent_hint=intent)

    # Limit to top chunks by score to keep context manageable for LLM
    scored = [(getattr(c, "score", 0.0) or 0.0, c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:LLM_MAX_CHUNKS]]

    # Build grouped evidence from top chunks only
    evidence = _build_grouped_evidence(top_chunks)

    # For reasoning/cross_document intents, build structured evidence chain
    evidence_context = ""
    if intent_type in ("reasoning", "cross_document"):
        try:
            from .evidence_chain import build_evidence_chain
            chain = build_evidence_chain(query, top_chunks)
            evidence_context = chain.render_for_prompt()
        except Exception:
            pass  # evidence chain is optional enhancement

    # Build intent-adaptive prompt
    if evidence_context:
        full_evidence = f"{evidence_context}\n\nRAW EVIDENCE:\n{evidence}"
    else:
        full_evidence = evidence

    prompt = build_generation_prompt(
        query=query,
        evidence_text=full_evidence,
        intent=intent_type,
        num_documents=num_documents,
    )

    # Use GENERATOR role for multi-agent, else default
    _gen_role = None
    try:
        from src.llm.multi_agent import MultiAgentGateway, AgentRole
        if isinstance(llm_client, MultiAgentGateway):
            _gen_role = AgentRole.GENERATOR
    except ImportError:
        pass

    # Build simplified fallback prompt for intermediate timeout
    est_tokens = _estimate_tokens(full_evidence)
    fallback_prompt = None
    if est_tokens > LLM_CHUNKED_TOKEN_THRESHOLD:
        fallback_prompt = _build_simplified_prompt(query, full_evidence)
        logger.info(
            "LLM extract: large context (~%d tokens, %d chunks), fallback prompt ready",
            est_tokens, len(top_chunks),
            extra={"stage": "llm_extract", "correlation_id": correlation_id},
        )

    raw_text = _generate(llm_client, prompt, correlation_id, role=_gen_role, fallback_prompt=fallback_prompt)
    if not raw_text:
        logger.warning(
            "LLM extract returned no result: domain=%s chunks=%d est_tokens=%d",
            intent_type, len(top_chunks), est_tokens,
            extra={"stage": "llm_extract_timeout", "correlation_id": correlation_id},
        )
        return None

    return _parse_response(raw_text, top_chunks)


# ── intent detection ─────────────────────────────────────────────────

_RANK_COMPARE_RE = re.compile(
    r"\b(?:rank|compare|evaluate|assess|rate|score|benchmark)\b", re.IGNORECASE,
)
_CONTACT_RE = re.compile(
    r"\b(?:email|phone|contact|address|linkedin|reach|mobile)\b", re.IGNORECASE,
)
_LIST_EXTRACT_RE = re.compile(
    r"\b(?:list|extract|show|give me|what are|enumerate)\b", re.IGNORECASE,
)


def _classify_intent(query: str, intent_hint: Optional[str]) -> str:
    """Map query + optional hint to one of: rank, detail, contact, general."""
    hint = (intent_hint or "").lower()
    if hint in ("rank", "compare"):
        return "rank"
    if hint in ("contact", "email", "phone"):
        return "contact"

    if _RANK_COMPARE_RE.search(query):
        return "rank"
    if _CONTACT_RE.search(query):
        return "contact"
    if _LIST_EXTRACT_RE.search(query):
        return "detail"
    return "general"


# ── expanded intent classification (8 types) ─────────────────────────

_COMPARISON_RE = re.compile(
    r"\b(?:compare|versus|vs\.?|difference|similarities|contrast|side.by.side)\b", re.I,
)
_SUMMARY_RE = re.compile(
    r"\b(?:summarize|summary|overview|brief|highlights?|outline)\b", re.I,
)
_RANKING_RE = re.compile(
    r"\b(?:rank|rate|score|top|best|worst|order\s+by|sort\s+by|benchmark)\b", re.I,
)
_TIMELINE_RE = re.compile(
    r"\b(?:timeline|chronolog|progression|history|over\s+time|career\s+path|sequence)\b", re.I,
)
_MULTI_FIELD_RE = re.compile(
    r"\b(?:extract\s+all|line\s*items?|all\s+fields?|each\s+(?:item|entry|field)|fill\s+(?:in|out))\b", re.I,
)
_REASONING_RE = re.compile(
    r"\b(?:qualified|suitable|fit\s+for|should\s+we|recommend|assess\s+whether|evaluate\s+if|capable)\b", re.I,
)
_CROSS_DOC_RE = re.compile(
    r"\b(?:all\s+(?:candidates?|documents?|resumes?|invoices?)|across|shared?|common|each\s+(?:candidate|document))\b", re.I,
)
_ANALYTICS_RE = re.compile(
    r"\b(?:how many|total (?:amount|number)|average|sum of|count of|across all|in total|distribution)\b", re.I,
)

_INTENT_HINT_MAP = {
    "rank": "ranking", "compare": "comparison", "comparison": "comparison",
    "contact": "factual", "email": "factual", "phone": "factual",
    "summary": "summary", "summarize": "summary",
    "timeline": "timeline", "reasoning": "reasoning",
    "extraction": "multi_field", "cross_document": "cross_document",
    "analytics": "analytics", "aggregate": "analytics", "count": "analytics",
}


def classify_query_intent(query: str, *, intent_hint: str | None = None) -> str:
    """Classify query into one of 8 intent types for response template selection."""
    if intent_hint:
        mapped = _INTENT_HINT_MAP.get(intent_hint.lower())
        if mapped:
            return mapped

    if _COMPARISON_RE.search(query):
        return "comparison"
    if _RANKING_RE.search(query):
        return "ranking"
    if _SUMMARY_RE.search(query):
        return "summary"
    if _TIMELINE_RE.search(query):
        return "timeline"
    if _REASONING_RE.search(query):
        return "reasoning"
    if _MULTI_FIELD_RE.search(query):
        return "multi_field"
    if _ANALYTICS_RE.search(query):
        return "analytics"
    if _CROSS_DOC_RE.search(query):
        return "cross_document"
    return "factual"


# ── prompt construction ───────────────────────────────────────────────

_SYSTEM_BASE = (
    "You are an intelligent document analysis assistant. "
    "Answer the user's question using ONLY the document evidence provided below. "
    "Do not invent or hallucinate information. "
    "If a piece of information is not present in the evidence, say so.\n"
)

_PROMPT_TEMPLATES = {
    "rank": (
        "You are analyzing {num_documents} document(s). For each document:\n"
        "1. Identify the subject (person name, company, entity)\n"
        "2. Extract key attributes relevant to the ranking criteria\n"
        "3. Rank/compare them based on the user's criteria\n"
        "4. Present results in a clear ranked list with justification\n\n"
        "Format: For each candidate/document, show a numbered ranking with "
        "key strengths and the reasoning behind the ranking.\n"
    ),
    "contact": (
        "Extract the specific contact information requested. "
        "Present each item clearly on its own line. "
        "Include all variants found (e.g., multiple phone numbers).\n"
    ),
    "detail": (
        "Extract and present the specific information requested. "
        "Organize the response clearly with bullet points or sections. "
        "Include all relevant details found across the documents.\n"
    ),
    "general": (
        "Answer the question thoroughly based on the document evidence. "
        "Be specific, cite details from the documents, and organize "
        "your response clearly.\n"
    ),
}


# ── intent-adaptive generation templates (8 types) ───────────────────

_GENERATION_SYSTEM = (
    "You are an intelligent document analysis assistant. "
    "Answer ONLY from the provided evidence. "
    "Never invent information. If evidence is insufficient, say so explicitly. "
    "Cite specific details from the documents.\n"
)

_GENERATION_TEMPLATES = {
    "factual": (
        "Provide a direct, specific answer to the question. "
        "Include the exact value or fact requested. "
        "Cite the document source.\n"
    ),
    "comparison": (
        "Compare the {num_documents} document(s) systematically:\n"
        "1. Identify each entity/document subject\n"
        "2. Compare on the criteria mentioned in the question\n"
        "3. Present as a structured comparison (use a table if 2+ entities)\n"
        "4. End with key differences and similarities\n"
    ),
    "summary": (
        "Provide a structured summary:\n"
        "1. Opening statement (1-2 sentences capturing the essence)\n"
        "2. Key highlights (3-6 bullet points with specific details)\n"
        "3. Brief takeaway or notable observation\n"
    ),
    "ranking": (
        "Rank the {num_documents} document subjects based on the criteria:\n"
        "1. Identify each subject and extract relevant attributes\n"
        "2. Score/evaluate against the ranking criteria\n"
        "3. Present as a numbered ranked list\n"
        "4. Include justification for each ranking position\n"
    ),
    "timeline": (
        "Present information in chronological order:\n"
        "1. Identify dates, periods, and sequences\n"
        "2. Arrange events/experiences from earliest to latest\n"
        "3. Show progression or evolution over time\n"
        "4. Note any gaps in the timeline\n"
    ),
    "multi_field": (
        "Extract and present all requested fields systematically:\n"
        "1. Identify each field/item to extract\n"
        "2. Present in a structured format (table or labeled list)\n"
        "3. Mark any missing or unclear fields\n"
        "4. Include source document for each extraction\n"
    ),
    "reasoning": (
        "Reason through the question using evidence:\n"
        "1. Identify what evidence supports the question\n"
        "2. Identify what evidence contradicts or is missing\n"
        "3. Present supporting factors with specific citations\n"
        "4. Note gaps — explicitly state what the documents do NOT contain\n"
        "5. Provide a qualified conclusion based on available evidence\n"
    ),
    "cross_document": (
        "Analyze across all {num_documents} document(s):\n"
        "1. Extract relevant information from each document\n"
        "2. Identify patterns, commonalities, and differences\n"
        "3. Present per-document findings, then a synthesis\n"
        "4. Cite which document each fact comes from\n"
    ),
    "analytics": (
        "Compute aggregate statistics from the {num_documents} document(s):\n"
        "1. Count documents by type\n"
        "2. Sum/average any numeric fields mentioned\n"
        "3. Present statistics clearly with exact numbers\n"
        "4. Cite which documents contribute to each statistic\n"
    ),
}


def build_generation_prompt(
    *,
    query: str,
    evidence_text: str,
    intent: str,
    num_documents: int = 1,
) -> str:
    """Build a structured prompt for the LLM generation call."""
    template = _GENERATION_TEMPLATES.get(intent, _GENERATION_TEMPLATES["factual"])
    if "{num_documents}" in template:
        template = template.format(num_documents=num_documents)

    max_evidence = LLM_MAX_CONTEXT_CHARS - 1500
    if len(evidence_text) > max_evidence:
        evidence_text = evidence_text[:max_evidence]

    return (
        f"{_GENERATION_SYSTEM}\n"
        f"TASK:\n{template}\n"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EVIDENCE:\n{evidence_text}\n\n"
        "Answer thoroughly. Be specific and cite document details."
    )


def _build_prompt(
    query: str,
    chunks: List[Any],
    intent: Optional[str] = None,
    num_documents: int = 1,
) -> str:
    intent_type = _classify_intent(query, intent)
    task_instruction = _PROMPT_TEMPLATES.get(intent_type, _PROMPT_TEMPLATES["general"])
    if "{num_documents}" in task_instruction:
        task_instruction = task_instruction.format(num_documents=num_documents)

    # Group chunks by document for better context organization
    evidence = _build_grouped_evidence(chunks)

    return (
        f"{_SYSTEM_BASE}\n"
        f"TASK:\n{task_instruction}\n"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EVIDENCE:\n{evidence}\n\n"
        "Provide your answer directly. Be thorough but concise."
    )


def _build_grouped_evidence(chunks: List[Any]) -> str:
    """Group chunks by document and format with headers."""
    doc_groups: OrderedDict[str, List[dict]] = OrderedDict()

    for chunk in chunks:
        chunk_id = getattr(chunk, "id", "")
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue

        meta = getattr(chunk, "meta", None) or {}
        source = getattr(chunk, "source", None)
        doc_name = (
            meta.get("source_name")
            or meta.get("document_name")
            or (getattr(source, "document_name", "") if source else "")
            or "Document"
        )
        section_kind = meta.get("section_kind") or meta.get("section.kind") or ""

        # Truncate very long chunks
        snippet = text[:800] if len(text) > 800 else text

        entry = {"id": chunk_id, "section": section_kind, "text": snippet}
        doc_groups.setdefault(doc_name, []).append(entry)

    parts: List[str] = []
    total_chars = 0
    multi_doc = len(doc_groups) > 1

    for doc_name, entries in doc_groups.items():
        if multi_doc:
            header = f"\n=== Document: {doc_name} ==="
            parts.append(header)
            total_chars += len(header)

        for entry in entries:
            section_label = f"[{entry['section']}] " if entry["section"] else ""
            block = f"{section_label}{entry['text']}"
            if total_chars + len(block) > LLM_MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total_chars += len(block)

        if total_chars > LLM_MAX_CONTEXT_CHARS:
            break

    return "\n\n".join(parts)


# ── LLM call with timeout ────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate based on character length."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _build_simplified_prompt(query: str, evidence: str) -> str:
    """Shorter prompt for intermediate-timeout fallback."""
    truncated = evidence[:LLM_MAX_CONTEXT_CHARS // 2]
    return (
        f"Answer this question concisely using ONLY the evidence below.\n"
        f"Question: {query}\n\nEvidence:\n{truncated}\n\nAnswer:"
    )


def _generate(
    llm_client: Any,
    prompt: str,
    correlation_id: Optional[str],
    role: Optional[str] = None,
    *,
    fallback_prompt: Optional[str] = None,
) -> Optional[str]:
    options = {
        "num_predict": LLM_MAX_OUTPUT_TOKENS,
        "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
        "num_ctx": 4096,
        "stop": [],
    }

    def _call(p: str) -> str:
        # Multi-agent role-aware dispatch (isinstance avoids MagicMock false positives)
        if role:
            try:
                from src.llm.multi_agent import MultiAgentGateway
                if isinstance(llm_client, MultiAgentGateway):
                    text, _meta = llm_client.generate_with_metadata_for_role(
                        role, p, options=options, max_retries=1, backoff=0.4,
                    )
                    return text or ""
            except ImportError:
                pass
        if hasattr(llm_client, "generate_with_metadata"):
            text, _meta = llm_client.generate_with_metadata(
                p, options=options, max_retries=1, backoff=0.4,
            )
            return text or ""
        return llm_client.generate(p, max_retries=1, backoff=0.4) or ""

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call, prompt)
    try:
        # Try intermediate timeout first if we have a fallback prompt
        timeout = LLM_EXTRACT_INTERMEDIATE_TIMEOUT_S if fallback_prompt else LLM_EXTRACT_TIMEOUT_S
        result = future.result(timeout=timeout)
        return result if result and result.strip() else None
    except concurrent.futures.TimeoutError:
        future.cancel()
        if fallback_prompt:
            logger.warning(
                "LLM extract hit intermediate timeout (%.1fs), retrying with simplified prompt",
                LLM_EXTRACT_INTERMEDIATE_TIMEOUT_S,
                extra={"stage": "llm_extract", "correlation_id": correlation_id},
            )
            # Retry with simplified prompt and remaining time budget
            executor2 = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future2 = executor2.submit(_call, fallback_prompt)
            try:
                remaining = LLM_EXTRACT_TIMEOUT_S - LLM_EXTRACT_INTERMEDIATE_TIMEOUT_S
                result = future2.result(timeout=max(remaining, 5.0))
                return result if result and result.strip() else None
            except (concurrent.futures.TimeoutError, Exception) as exc:
                future2.cancel()
                logger.warning(
                    "LLM extract fallback also failed after %.1fs: %s",
                    LLM_EXTRACT_TIMEOUT_S,
                    exc,
                    extra={"stage": "llm_extract", "correlation_id": correlation_id},
                )
                return None
            finally:
                executor2.shutdown(wait=False)
        else:
            logger.warning(
                "LLM extract timed out after %.1fs",
                LLM_EXTRACT_TIMEOUT_S,
                extra={"stage": "llm_extract", "correlation_id": correlation_id},
            )
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM extract failed: %s",
            exc,
            extra={"stage": "llm_extract", "correlation_id": correlation_id},
        )
        return None
    finally:
        executor.shutdown(wait=False)


_METADATA_TOKENS = frozenset({
    "section_id", "chunk_type", "section_title", "section_kind",
    "page_start", "page_end", "start_page", "end_page",
    "canonical_text", "embedding_text", "doc_domain", "document_id",
})


def _looks_like_metadata(text: str) -> bool:
    """Return True if text looks like stringified metadata rather than content."""
    lowered = text.lower()
    hits = sum(1 for tok in _METADATA_TOKENS if tok in lowered)
    return hits >= 2


# ── response parsing ──────────────────────────────────────────────────

def _parse_response(raw: str, chunks: List[Any]) -> Optional[LLMResponseSchema]:
    payload = _extract_json(raw)

    answer = ""
    chunks_used: List[str] = []

    if isinstance(payload, dict):
        answer = str(payload.get("answer") or payload.get("response") or "").strip()
        raw_chunks = payload.get("chunks_used") or payload.get("sources") or []
        if isinstance(raw_chunks, list):
            chunks_used = [str(c) for c in raw_chunks]

    # Fallback: if JSON parsing failed but we got usable text, use it
    if not answer:
        cleaned = _clean_raw_response(raw)
        if cleaned and len(cleaned) >= 10 and not _looks_like_metadata(cleaned):
            answer = cleaned

    if not answer or len(answer) < 10:
        return None

    return LLMResponseSchema(text=answer, evidence_chunks=chunks_used)


def _clean_raw_response(raw: str) -> str:
    """Clean LLM response text: strip markdown fences, JSON artifacts, etc."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if len(lines) > 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # If it looks like a JSON object, try to extract the answer field
    if text.startswith("{"):
        parsed = _extract_json(text)
        if parsed:
            return str(parsed.get("answer") or parsed.get("response") or "").strip()
        return ""  # Malformed JSON — don't use raw

    # Strip leading "Answer:" or "Response:" labels
    text = re.sub(r"^(?:answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)

    return text.strip()


def _extract_json(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    if "{" in text and "}" in text:
        snippet = text[text.find("{"):text.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return {}


def _count_unique_documents(chunks: List[Any]) -> int:
    """Count unique document IDs across chunks."""
    doc_ids = set()
    for chunk in chunks:
        meta = getattr(chunk, "meta", None) or {}
        doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
        if doc_id:
            doc_ids.add(str(doc_id))
    return max(len(doc_ids), 1)
