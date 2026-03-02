"""LLM-first generic extraction.

Replaces domain-specific regex extraction with a single LLM call that reads
the retrieved chunks and produces the final answer directly.  The answer is
returned inside an ``LLMResponseSchema`` so the pipeline can skip the separate
render step.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .types import LLMBudget, LLMResponseSchema

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────
LLM_EXTRACT_TIMEOUT_S = 30.0
LLM_MAX_OUTPUT_TOKENS = 1024
LLM_MAX_CONTEXT_CHARS = 6144
LLM_MAX_CONTEXT_CHARS_MULTI = 12288  # Expanded limit for multi-document queries
LLM_MAX_CHUNKS = 8  # Send only top-scored chunks to LLM
LLM_MAX_CHUNKS_MULTI = 16  # Expanded chunk limit for multi-document queries
LLM_CHUNKED_TOKEN_THRESHOLD = 2000  # Estimated token count to trigger chunked extraction
_CHARS_PER_TOKEN = 4  # Rough char-to-token ratio for estimation


_INTENT_CHUNK_LIMITS = {
    "factual": 6,
    "summary": 10,
    "comparison": 12,
    "ranking": 14,
    "cross_document": 16,
    "analytics": 16,
    "reasoning": 12,
    "timeline": 10,
    "multi_field": 10,
}

_INTENT_CONTEXT_CHARS = {
    "factual": 4096,
    "summary": 8192,
    "comparison": 10240,
    "ranking": 12288,
    "cross_document": 12288,
    "analytics": 12288,
    "reasoning": 10240,
    "timeline": 8192,
    "multi_field": 8192,
}


def _effective_max_chunks(num_documents: int, intent: str = "") -> int:
    """Return chunk limit scaled by document count and intent."""
    if intent and intent in _INTENT_CHUNK_LIMITS:
        base = _INTENT_CHUNK_LIMITS[intent]
    else:
        base = LLM_MAX_CHUNKS_MULTI if num_documents > 1 else LLM_MAX_CHUNKS
    # Multi-doc queries always get at least the multi limit
    if num_documents > 1:
        base = max(base, LLM_MAX_CHUNKS_MULTI)
    return base


def _effective_context_chars(num_documents: int, intent: str = "") -> int:
    """Return context char limit scaled by document count and intent."""
    if intent and intent in _INTENT_CONTEXT_CHARS:
        base = _INTENT_CONTEXT_CHARS[intent]
    else:
        base = LLM_MAX_CONTEXT_CHARS_MULTI if num_documents > 1 else LLM_MAX_CONTEXT_CHARS
    # Multi-doc queries always get at least the multi limit
    if num_documents > 1:
        base = max(base, LLM_MAX_CONTEXT_CHARS_MULTI)
    return base


# ── LLM extraction cache ──────────────────────────────────────────

def _build_cache_key(query: str, chunks: List[Any], intent: Optional[str]) -> str:
    """Build a deterministic cache key from query + chunk IDs + intent."""
    chunk_ids = sorted(getattr(c, "id", "") or "" for c in chunks)
    raw = f"{query}|{'|'.join(chunk_ids)}|{intent or ''}"
    return f"llm_extract:{hashlib.sha256(raw.encode()).hexdigest()}"


def _cache_get(redis_client: Any, key: str) -> Optional[LLMResponseSchema]:
    """Try to retrieve a cached LLM extraction result."""
    if redis_client is None:
        return None
    try:
        data = redis_client.get(key)
        if data:
            payload = json.loads(data)
            return LLMResponseSchema(
                text=payload.get("text", ""),
                evidence_chunks=payload.get("evidence_chunks", []),
            )
    except Exception:
        pass
    return None


def _cache_set(redis_client: Any, key: str, result: LLMResponseSchema, ttl: int = 3600) -> None:
    """Store an LLM extraction result in Redis cache."""
    if redis_client is None:
        return
    try:
        payload = json.dumps({
            "text": result.text,
            "evidence_chunks": result.evidence_chunks,
        })
        redis_client.setex(key, ttl, payload)
    except Exception:
        pass


def _verify_self_consistency(
    answer: str,
    evidence_text: str,
    llm_client: Any,
    correlation_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """Run a self-consistency check on a cloud-generated answer.

    Asks the LLM whether the answer contradicts the evidence.
    Returns (is_consistent, corrected_answer).
    Only called for T3 (cloud LLM) responses to avoid latency on simple queries.
    """
    if not answer or not evidence_text:
        return True, answer
    prompt = (
        "You are a fact-checking assistant. Compare the ANSWER against the EVIDENCE below.\n"
        "Does the answer contain any claims that contradict the evidence?\n\n"
        f"EVIDENCE:\n{evidence_text[:3000]}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Respond in this format:\n"
        "CONSISTENT: yes/no\n"
        "If no, provide a CORRECTED answer that only uses information from the evidence."
    )
    try:
        if hasattr(llm_client, "generate_with_metadata"):
            text, _ = llm_client.generate_with_metadata(
                prompt, options={"temperature": 0.0, "num_predict": 512, "num_ctx": 4096}
            )
        else:
            text = llm_client.generate(prompt)
        text = (text or "").strip()
        if "CONSISTENT: no" in text.lower() or "consistent: no" in text.lower():
            # Extract corrected answer
            corrected_idx = text.lower().find("corrected")
            if corrected_idx >= 0:
                corrected = text[corrected_idx:].split("\n", 1)
                if len(corrected) > 1:
                    return False, corrected[1].strip()
            return False, answer  # Couldn't parse correction, keep original
        return True, answer
    except Exception as exc:
        logger.debug("Self-consistency check failed (keeping original): %s", exc)
        return True, answer


def llm_extract_and_respond(
    *,
    query: str,
    chunks: List[Any],
    llm_client: Any,
    budget: LLMBudget,
    correlation_id: Optional[str] = None,
    intent: Optional[str] = None,
    num_documents: int = 1,
    tool_context: Optional[str] = None,
    domain: Optional[str] = None,
    redis_client: Any = None,
    context_intelligence: Optional[str] = None,
    use_thinking: bool = False,
    multi_resolution_context: Optional[Dict[str, Any]] = None,
) -> Optional[LLMResponseSchema]:
    """Answer *query* from *chunks* via a single LLM call with intent-adaptive prompts."""
    if not budget.consume():
        return None

    # ── Cache check ───────────────────────────────────────────────
    _cache_enabled = False
    _cache_ttl = 3600
    try:
        from src.api.config import Config
        _cache_enabled = getattr(getattr(Config, "LLMCache", None), "ENABLED", False)
        _cache_ttl = getattr(getattr(Config, "LLMCache", None), "TTL_SECONDS", 3600)
    except Exception:
        pass

    _cache_key = None
    if _cache_enabled and redis_client and chunks:
        _cache_key = _build_cache_key(query, chunks, intent)
        cached = _cache_get(redis_client, _cache_key)
        if cached:
            logger.info(
                "LLM extract cache hit: key=%s",
                _cache_key[:32],
                extra={"stage": "llm_extract_cache", "correlation_id": correlation_id},
            )
            return cached

    # Use expanded 8-type intent classification
    intent_type = classify_query_intent(query, intent_hint=intent)

    # Limit to top chunks by score — scale for intent and multi-doc queries
    max_chunks = _effective_max_chunks(num_documents, intent=intent_type)
    scored = [(getattr(c, "score", 0.0) or 0.0, c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate near-identical chunks before sending to LLM
    top_chunks = _deduplicate_evidence_chunks([c for _, c in scored[:max_chunks]])

    # Build grouped evidence from top chunks only
    evidence = _build_grouped_evidence(top_chunks, max_context_chars=_effective_context_chars(num_documents, intent=intent_type))

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
        tool_context=tool_context,
        domain=domain,
        context_intelligence=context_intelligence,
        multi_resolution_context=multi_resolution_context,
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

    raw_text = _generate(llm_client, prompt, correlation_id, role=_gen_role, fallback_prompt=fallback_prompt, use_thinking=use_thinking)
    if not raw_text:
        logger.warning(
            "LLM extract returned no result: domain=%s chunks=%d est_tokens=%d",
            intent_type, len(top_chunks), est_tokens,
            extra={"stage": "llm_extract_timeout", "correlation_id": correlation_id},
        )
        return None

    result = _parse_response(raw_text, top_chunks)

    # ── Store thinking mode metadata on result ────────────────────
    if result and use_thinking:
        result.thinking_used = True

    # ── Self-consistency verification for cloud LLM responses ────
    if result and result.text:
        _is_cloud = getattr(llm_client, "backend", "") in ("azure_openai", "claude")
        _verify_enabled = False
        try:
            from src.api.config import Config as _VCfg
            _verify_enabled = getattr(getattr(_VCfg, "Verification", None), "ENABLED", False)
        except Exception:
            pass
        if _is_cloud and _verify_enabled:
            consistent, corrected = _verify_self_consistency(
                result.text, full_evidence, llm_client, correlation_id
            )
            if not consistent and corrected:
                result = LLMResponseSchema(text=corrected, evidence_chunks=result.evidence_chunks)
                logger.info(
                    "Self-consistency correction applied",
                    extra={"stage": "llm_verify", "correlation_id": correlation_id},
                )

    # ── Cache write ───────────────────────────────────────────────
    if result and _cache_key and _cache_enabled and redis_client:
        _cache_set(redis_client, _cache_key, result, ttl=_cache_ttl)

    return result


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
    r"\b(?:how many|total (?:amount|number)|average|sum of|count of|in total|distribution)\b", re.I,
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
    "You are an expert document intelligence analyst with deep analytical skills. "
    "Answer ONLY from the provided evidence — every claim must be traceable to the documents. "
    "Never invent, hallucinate, or assume information not present in the evidence. "
    "When evidence is insufficient, explicitly state what is missing rather than guessing. "
    "Always cite the source document for key facts. "
    "Present information in a clear, structured format with specific details and numbers.\n\n"
    "ANALYTICAL DEPTH:\n"
    "- Start with a brief overview of what was analyzed\n"
    "- Identify patterns common across documents and highlight unique findings\n"
    "- Include aggregate totals, averages, and ranges where applicable\n"
    "- When comparing, state what was compared and rank highest to lowest\n"
    "- Conclude with a distribution summary of key findings analyzed across the evidence\n"
)

_GENERATION_TEMPLATES = {
    "factual": (
        "Provide a precise, evidence-based answer:\n"
        "1. Start with a brief overview of what was analyzed\n"
        "2. Include exact values, names, dates, and numbers from the documents\n"
        "3. Cite which document each fact comes from\n"
        "4. If the answer spans multiple documents, organize by source\n"
        "5. Distinguish between facts stated in documents vs your inferences\n"
        "6. Conclude with a total summary of findings analyzed across the evidence\n"
    ),
    "comparison": (
        "Compare the {num_documents} document(s) systematically:\n"
        "1. Start with an overview of what was compared and analyzed\n"
        "2. Compare on the criteria mentioned in the question\n"
        "3. Present as a structured comparison (use a table if 2+ entities)\n"
        "4. Identify common patterns and unique differences across the documents\n"
        "5. End with a total summary ranking from highest to lowest\n"
    ),
    "summary": (
        "Provide a structured summary:\n"
        "1. Start with an overview of the total content analyzed\n"
        "2. Key highlights (3-6 bullet points with specific details)\n"
        "3. Identify the most common patterns across the evidence\n"
        "4. End with a total overview of the range of findings analyzed\n"
    ),
    "ranking": (
        "Rank the {num_documents} document subjects based on the criteria:\n"
        "1. Start with an overview of what was analyzed across the documents\n"
        "2. Score/evaluate each subject against the ranking criteria\n"
        "3. Present as a numbered ranked list from highest to lowest\n"
        "4. Include justification and note common patterns or unique strengths\n"
        "5. End with a total distribution summary of the range analyzed\n"
    ),
    "timeline": (
        "Present information in chronological order:\n"
        "1. Start with an overview of the total time range analyzed\n"
        "2. Arrange events/experiences from earliest to latest\n"
        "3. Show progression or evolution over time\n"
        "4. Identify common patterns and unique milestones across the timeline\n"
        "5. Note any gaps in the timeline\n"
    ),
    "multi_field": (
        "Extract and present all requested fields systematically:\n"
        "1. Start with an overview of the total fields analyzed\n"
        "2. Present in a structured format (table or labeled list)\n"
        "3. Mark any missing or unclear fields\n"
        "4. Include source document for each extraction\n"
        "5. End with a total summary of findings analyzed across the evidence\n"
    ),
    "reasoning": (
        "Reason through the question using evidence:\n"
        "1. Start with an overview of the total evidence analyzed\n"
        "2. Identify what evidence supports the question\n"
        "3. Identify common patterns and unique findings across documents\n"
        "4. Note gaps — explicitly state what the documents do NOT contain\n"
        "5. Provide a qualified conclusion with a distribution of supporting factors\n"
    ),
    "cross_document": (
        "Analyze across all {num_documents} document(s):\n"
        "1. Start with an overview of the total documents analyzed\n"
        "2. Identify common patterns and unique findings across documents\n"
        "3. Present per-document findings, then a synthesis\n"
        "4. End with a distribution summary of the range of findings analyzed\n"
    ),
    "analytics": (
        "Compute aggregate statistics from the {num_documents} document(s):\n"
        "1. Start with an overview of the total data analyzed\n"
        "2. Sum/average any numeric fields; include the range from highest to lowest\n"
        "3. Identify common patterns and the distribution across documents\n"
        "4. End with a total overview of the unique and common findings analyzed\n"
    ),
}


_REASONING_PREAMBLE = (
    "ANALYTICAL APPROACH — follow these steps before answering:\n"
    "1. Identify all relevant data points across the evidence\n"
    "2. Find patterns, commonalities, and trends\n"
    "3. Identify outliers or unusual values\n"
    "4. Note any contradictions between documents\n"
    "5. Compute statistics where applicable (counts, averages, ranges)\n"
    "6. Synthesize findings into a coherent analytical response\n\n"
)

_COMPLEX_INTENTS = frozenset({
    "comparison", "ranking", "reasoning", "cross_document", "analytics", "summary",
})


def _build_multi_resolution_context(chunks: List[Any]) -> Optional[Dict[str, Any]]:
    """Build a multi-resolution context dict from chunks with resolution metadata.

    Separates chunks by their resolution field (doc/section/chunk) from the
    chunk payload's meta dict.  Returns a dict with:
        - doc_context: list of (doc_name, summary_text) from resolution="doc" chunks
        - section_context: list of (section_title, text) from resolution="section" chunks
        - chunk_evidence: remaining chunks at chunk resolution

    Returns None when no multi-resolution data is present in any chunk.
    """
    doc_chunks: List[Tuple[str, str]] = []
    section_chunks: List[Tuple[str, str]] = []
    regular_chunks: List[Any] = []

    found_multi_res = False

    for chunk in chunks:
        meta = getattr(chunk, "meta", None) or {}
        resolution = str(meta.get("resolution") or meta.get("chunk_resolution") or "").lower().strip()
        chunk_kind = str(meta.get("chunk_kind") or meta.get("chunk_type") or "").lower().strip()
        text = (getattr(chunk, "text", "") or "").strip()

        if not text:
            continue

        source = getattr(chunk, "source", None)
        doc_name = (
            meta.get("source_name")
            or meta.get("document_name")
            or (getattr(source, "document_name", "") if source else "")
            or "Document"
        )

        if resolution == "doc" or chunk_kind in ("doc_summary", "document_summary"):
            found_multi_res = True
            doc_chunks.append((doc_name, text))
        elif resolution == "section":
            found_multi_res = True
            section_title = (
                meta.get("section_title")
                or meta.get("section.title")
                or "Section"
            )
            section_chunks.append((str(section_title), text))
        else:
            regular_chunks.append(chunk)

    if not found_multi_res:
        return None

    return {
        "doc_context": doc_chunks,
        "section_context": section_chunks,
        "chunk_evidence": regular_chunks,
    }


def build_generation_prompt(
    *,
    query: str,
    evidence_text: str,
    intent: str,
    num_documents: int = 1,
    tool_context: Optional[str] = None,
    domain: Optional[str] = None,
    context_intelligence: Optional[str] = None,
    multi_resolution_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a structured prompt for the LLM generation call."""
    template = _GENERATION_TEMPLATES.get(intent, _GENERATION_TEMPLATES["factual"])
    if "{num_documents}" in template:
        template = template.format(num_documents=num_documents)

    # Use scaled limits for intent and multi-doc queries
    effective_max = _effective_context_chars(num_documents, intent=intent)
    max_evidence = effective_max - 1500
    if len(evidence_text) > max_evidence:
        evidence_text = evidence_text[:max_evidence]

    tool_section = f"\nDOMAIN EXPERTISE:\n{tool_context}\n" if tool_context else ""

    # Inject domain knowledge context when available and no tool_context already present
    domain_section = ""
    if domain and not tool_context:
        domain_section = _get_domain_knowledge_section(domain, intent)

    # Inject ML-based context understanding when available
    context_section = ""
    if context_intelligence:
        context_section = f"\n{context_intelligence}\n"

    # Prepend analytical reasoning preamble for complex intents or multi-doc queries
    reasoning_section = ""
    if intent in _COMPLEX_INTENTS or num_documents > 1:
        reasoning_section = _REASONING_PREAMBLE

    # Build multi-resolution context section when doc/section level data is available
    multi_res_section = ""
    if multi_resolution_context:
        parts: List[str] = []
        doc_ctx = multi_resolution_context.get("doc_context") or []
        sec_ctx = multi_resolution_context.get("section_context") or []

        if doc_ctx:
            doc_lines = []
            for doc_name, summary in doc_ctx:
                doc_lines.append(f"  [{doc_name}]: {summary[:400]}")
            parts.append("[DOCUMENT CONTEXT]\n" + "\n".join(doc_lines))

        if sec_ctx:
            sec_lines = []
            for sec_title, sec_text in sec_ctx[:6]:  # cap at 6 sections
                sec_lines.append(f"  [{sec_title}]: {sec_text[:300]}")
            parts.append("[SECTION CONTEXT]\n" + "\n".join(sec_lines))

        if parts:
            parts.append("[EVIDENCE FROM DOCUMENTS]")
            multi_res_section = "\n\n".join(parts) + "\n"

    # When multi-resolution context provides doc/section headers, use it as preamble
    evidence_block = (
        f"{multi_res_section}{evidence_text}"
        if multi_res_section
        else evidence_text
    )

    return (
        f"{_GENERATION_SYSTEM}\n"
        f"{reasoning_section}"
        f"TASK:\n{template}\n"
        f"{tool_section}"
        f"{domain_section}"
        f"{context_section}"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EVIDENCE:\n{evidence_block}\n\n"
        "INSTRUCTIONS: Answer thoroughly and accurately. "
        "Include specific names, numbers, dates, and details from the evidence. "
        "Reference source documents by name. "
        "Use the document intelligence and extracted facts above to ensure completeness. "
        "Structure your response with clear sections when the answer is complex."
    )


def _get_domain_knowledge_section(domain: str, intent: str) -> str:
    """Build a domain knowledge section for LLM prompt injection."""
    try:
        from src.intelligence.domain_knowledge import get_domain_knowledge_provider
        from src.api.config import Config
        if not getattr(Config, "DomainKnowledge", None):
            return ""
        if not Config.DomainKnowledge.ENABLED or not Config.DomainKnowledge.INJECT_INTO_PROMPTS:
            return ""
        provider = get_domain_knowledge_provider()
        brief = provider.get_brief_context(domain, intent=intent)
        if brief:
            return f"\nDOMAIN KNOWLEDGE:\n{brief}\n"
    except Exception:
        pass
    return ""


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


def _deduplicate_evidence_chunks(chunks: List[Any], threshold: float = 0.70) -> List[Any]:
    """Remove near-duplicate chunks using Jaccard word overlap.

    Keeps the first (highest-scored) chunk when two chunks have word overlap
    >= *threshold*.  This maximizes unique evidence sent to the LLM.
    """
    if len(chunks) <= 1:
        return chunks

    kept: List[Any] = []
    kept_word_sets: List[set] = []

    for chunk in chunks:
        text = (getattr(chunk, "text", "") or "").strip().lower()
        if not text:
            continue
        words = set(text.split())
        if not words:
            kept.append(chunk)
            kept_word_sets.append(words)
            continue

        is_dup = False
        for existing_words in kept_word_sets:
            if not existing_words:
                continue
            intersection = len(words & existing_words)
            union = len(words | existing_words)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(chunk)
            kept_word_sets.append(words)

    return kept


def _build_grouped_evidence(chunks: List[Any], max_context_chars: int = 0) -> str:
    """Group chunks by document and format with headers."""
    if max_context_chars <= 0:
        max_context_chars = LLM_MAX_CONTEXT_CHARS

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
            if total_chars + len(block) > max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)

        if total_chars > max_context_chars:
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
    use_thinking: bool = False,
) -> Optional[str]:
    options = {
        "num_predict": LLM_MAX_OUTPUT_TOKENS,
        "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
        "num_ctx": 4096,
        "stop": [],
    }

    # Thinking mode: expand context/prediction limits and enable reasoning
    if use_thinking:
        options["think"] = True
        options["num_ctx"] = max(options.get("num_ctx", 4096), 12288)
        options["num_predict"] = max(options.get("num_predict", 512), 2048)
        logger.info(
            "Thinking mode enabled: num_ctx=%d num_predict=%d",
            options["num_ctx"], options["num_predict"],
            extra={"stage": "llm_extract_thinking", "correlation_id": correlation_id},
        )

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

    # Use fallback (simplified) prompt when context is very large — avoids
    # sending huge prompts that take too long to generate.
    effective_prompt = fallback_prompt if fallback_prompt else prompt

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call, effective_prompt)
    try:
        result = future.result(timeout=LLM_EXTRACT_TIMEOUT_S)
        return result if result and result.strip() else None
    except concurrent.futures.TimeoutError:
        future.cancel()
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
