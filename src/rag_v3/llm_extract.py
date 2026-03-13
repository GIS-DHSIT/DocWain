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
from src.utils.logging_utils import get_logger
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .types import LLMBudget, LLMResponseSchema

logger = get_logger(__name__)

# ── tunables ──────────────────────────────────────────────────────────
LLM_EXTRACT_TIMEOUT_S = 120.0  # qwen3:14b: ~60tok/s on T4, 4K tokens = ~67s + prompt overhead
LLM_MAX_OUTPUT_TOKENS = 4096  # qwen3:14b supports 40K context, generous output budget
LLM_MAX_CONTEXT_CHARS = 8192  # Expanded for qwen3:14b's larger context window
LLM_MAX_CONTEXT_CHARS_MULTI = 16384  # Multi-document gets more context
LLM_MAX_CHUNKS = 10  # More evidence = better answers
LLM_MAX_CHUNKS_MULTI = 20  # Multi-doc queries need broad evidence

def _get_num_ctx(_intent: str = "", _num_chunks: int = 0) -> int:
    """Context window size.  Fixed at 8192 to match the pinned model and avoid
    Ollama model reloads (which add 60-90s latency per swap).  8192 tokens is
    sufficient for our prompt sizes (typically 2000-5000 chars)."""
    return 8192
LLM_CHUNKED_TOKEN_THRESHOLD = 3000  # Higher threshold before chunked extraction
_CHARS_PER_TOKEN = 4  # Rough char-to-token ratio for estimation

import re as _re_mod
# Pre-compiled numeric pattern for evidence dedup (was compiled per-chunk, now module-level)
_DEDUP_NUM_RE = _re_mod.compile(
    r'\b\d+(?:[.,]\d+)?(?:\s*(?:%|years?|months?|days?|hrs?|kg|mg|ml|lbs?|USD|\$|€|£|₹))?',
    _re_mod.IGNORECASE,
)

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
    "generate": 10,
}

_INTENT_CONTEXT_CHARS = {
    "factual": 4096,
    "contact": 4096,
    "detail": 6144,
    "summary": 8192,
    "comparison": 10240,
    "ranking": 12288,
    "cross_document": 12288,
    "analytics": 12288,
    "reasoning": 10240,
    "timeline": 8192,
    "multi_field": 8192,
    "generate": 10240,
}

# Higher timeout for content generation — generation needs more tokens
LLM_GENERATE_TIMEOUT_S = 90.0

# ── multi-part query detection ───────────────────────────────────
import re as _re_mp
_MULTI_PART_RE = _re_mp.compile(r'\b(?:and|as well as|along with|plus|also)\b', _re_mp.IGNORECASE)

def _is_multi_part_query(query: str) -> bool:
    """Detect queries asking for multiple distinct fields/topics.

    Must identify genuine multi-field requests ("name, skills, and experience")
    while avoiding false positives on natural prose ("terms and conditions").
    """
    # Multiple question marks: "What is X? What about Y?"
    if query.count("?") >= 2:
        return True

    # Comma-separated field requests: "name, skills, experience"
    # Require short segments between commas (field names, not prose clauses)
    commas = query.count(",")
    if commas >= 2 and len(query) < 200:
        segments = query.split(",")
        short_segments = sum(1 for s in segments if len(s.strip().split()) <= 4)
        if short_segments >= 3:
            return True

    # Conjunctions only count when between short noun phrases (field names)
    # "terms and conditions" = compound noun → NOT multi-part
    # "name and skills and experience" = 2 conjunctions with field-like nouns → multi-part
    # First, reduce common compound nouns to single tokens
    _compound_nouns = ("terms and conditions", "pros and cons", "dos and donts",
                       "supply and demand", "research and development", "trial and error",
                       "back and forth", "give and take", "rules and regulations")
    _reduced = query.lower()
    for cn in _compound_nouns:
        _reduced = _reduced.replace(cn, cn.replace(" and ", "_"))
    conjunctions = _MULTI_PART_RE.findall(_reduced)
    if len(conjunctions) >= 2 and len(query.split()) <= 15:
        return True

    return False

def _decompose_query(query: str) -> List[str]:
    """Split a multi-part query into sub-questions for the LLM to address individually.

    Handles patterns like:
    - "What is X and what is Y?" → ["What is X?", "What is Y?"]
    - "Name, skills, and experience?" → ["Name?", "Skills?", "Experience?"]
    - "What is X? What about Y?" → ["What is X?", "What about Y?"]
    """
    import re as _re_dq

    # Pattern 1: Multiple question marks — split at "?" boundaries
    if query.count("?") >= 2:
        parts = [p.strip() + "?" for p in query.split("?") if p.strip()]
        if len(parts) >= 2:
            return parts[:5]  # cap at 5 sub-questions

    # Pattern 2: Comma-separated field requests
    commas = query.count(",")
    if commas >= 2:
        segments = [s.strip() for s in query.split(",") if s.strip()]
        short_segments = [s for s in segments if len(s.split()) <= 4]
        if len(short_segments) >= 3:
            # Clean up: strip conjunctions from last item ("and experience" → "experience")
            cleaned = []
            for seg in short_segments:
                seg = _re_dq.sub(r'^\s*(?:and|also|plus|as well as)\s+', '', seg, flags=_re_dq.IGNORECASE).strip()
                if seg:
                    cleaned.append(seg)
            return cleaned[:5]

    # Pattern 3: "and" splitting for simple conjunctions
    # "What is his name and what are his skills?" → two questions
    _and_parts = _re_dq.split(r'\band\b', query, flags=_re_dq.IGNORECASE)
    if len(_and_parts) >= 2:
        cleaned = [p.strip().rstrip("?").strip() + "?" for p in _and_parts if len(p.strip()) > 5]
        if len(cleaned) >= 2:
            return cleaned[:4]

    return []

def _effective_max_chunks(num_documents: int, intent: str = "", query: str = "") -> int:
    """Return chunk limit scaled by document count, intent, and query complexity."""
    if intent and intent in _INTENT_CHUNK_LIMITS:
        base = _INTENT_CHUNK_LIMITS[intent]
    else:
        base = LLM_MAX_CHUNKS_MULTI if num_documents > 1 else LLM_MAX_CHUNKS
    # Multi-doc queries always get at least the multi limit
    if num_documents > 1:
        base = max(base, LLM_MAX_CHUNKS_MULTI)
    # Scale up for many documents: ensure at least 2 chunks per document
    if num_documents > 3:
        per_doc_min = num_documents * 2
        base = max(base, min(per_doc_min, 30))  # cap at 30 (qwen3:14b has 40K ctx)
    # Query complexity scaling: multi-part queries need more evidence
    if query and _is_multi_part_query(query):
        base = min(base + 4, 30)  # Extra 4 chunks for multi-part
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
    # Scale up for many documents: +2048 chars per doc beyond 3
    if num_documents > 3:
        extra = (num_documents - 3) * 2048
        base = min(base + extra, 28672)  # cap at 28K chars (qwen3:14b supports 40K)
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
    except Exception as exc:
        logger.debug("Failed to retrieve LLM extraction result from cache", exc_info=True)
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
    except Exception as exc:
        logger.debug("Failed to store LLM extraction result in cache", exc_info=True)

def _lightweight_grounding_check(
    answer: str,
    chunks: List[Any],
) -> float:
    """Fast term-overlap grounding check with entity consistency — no LLM call.

    Returns a confidence score 0.0-1.0 indicating how well the answer
    terms are grounded in the evidence chunks.  Uses two signals:
    1. Term overlap ratio (how many answer terms appear in evidence)
    2. Entity consistency (proper nouns in answer must appear in evidence)
    """
    if not answer or not chunks:
        return 0.0

    # Build evidence term set
    evidence_text = " ".join(
        (getattr(c, "text", "") or "").lower() for c in chunks
    )
    evidence_terms = set(w for w in evidence_text.split() if len(w) > 3)
    if not evidence_terms:
        return 0.0

    # Extract meaningful answer terms (skip common words)
    _COMMON = frozenset({
        "this", "that", "with", "from", "have", "been", "they", "their",
        "what", "which", "where", "when", "will", "would", "could", "should",
        "about", "there", "these", "those", "than", "then", "into", "only",
        "also", "just", "very", "some", "more", "most", "each", "other",
        "based", "found", "using", "following", "document", "evidence",
        "provided", "information", "according", "shows", "indicates",
    })
    answer_terms = set(
        w for w in answer.lower().split()
        if len(w) > 3 and w.rstrip(".,;:!?") not in _COMMON
    )
    if not answer_terms:
        return 0.5  # No meaningful terms to check

    overlap = len(answer_terms & evidence_terms)
    # Stem-like matching: count unmatched terms with common prefix in evidence
    _unmatched_terms = answer_terms - evidence_terms
    _stem_extra = 0
    for _ut in _unmatched_terms:
        _clean = _ut.rstrip(".,;:!?")
        if len(_clean) >= 5:
            _prefix = _clean[:max(5, len(_clean) - 3)]
            if any(et[:len(_prefix)] == _prefix for et in evidence_terms if len(et) >= len(_prefix)):
                _stem_extra += 1
    overlap += _stem_extra
    ratio = overlap / len(answer_terms)

    # Entity consistency: capitalized multi-word sequences (names, orgs) in
    # the answer should appear somewhere in the evidence.  Ungrounded entities
    # are a strong hallucination signal.
    import re as _re_gc
    _entity_pat = _re_gc.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b')
    _generic_caps = frozenset({
        "The", "This", "That", "These", "Those", "Here", "There", "Based",
        "According", "Following", "Summary", "Details", "Information",
        "Document", "Evidence", "Source", "Table", "Section", "Note",
        "Important", "However", "Therefore", "Additionally", "Furthermore",
        "Regarding", "Overview", "Analysis", "Comparison", "Ranking",
    })
    answer_entities = set()
    for match in _entity_pat.finditer(answer):
        entity = match.group()
        if entity not in _generic_caps:
            answer_entities.add(entity.lower())

    entity_penalty = 0.0
    if answer_entities:
        # Use word-boundary matching to avoid false positives
        # (e.g., "Jon" matching "Jonathan")
        ungrounded = [
            e for e in answer_entities
            if not _re_gc.search(r'\b' + _re_gc.escape(e) + r'\b', evidence_text)
        ]
        if ungrounded:
            # Each ungrounded entity reduces confidence (capped at 0.25)
            entity_penalty = min(0.25, len(ungrounded) * 0.08)

    base_score = min(0.95, 0.3 + ratio * 0.65)

    # Numeric precision check — answers must not invent numbers
    _answer_numbers = set(_re_gc.findall(r'\b\d+\.?\d*\b', answer))
    _evidence_numbers = set(_re_gc.findall(r'\b\d+\.?\d*\b', evidence_text))
    _ungrounded_numbers = _answer_numbers - _evidence_numbers
    if _ungrounded_numbers:
        numeric_penalty = min(0.20, len(_ungrounded_numbers) * 0.05)
        base_score = max(0.0, base_score - numeric_penalty)

    return round(max(0.0, base_score - entity_penalty), 3)

def _check_entity_coverage(
    query: str,
    response: str,
    chunks: List[Any],
) -> Optional[str]:
    """Check if all entities mentioned in the query are addressed in the response.

    When a query asks about multiple named entities (e.g., "Compare Alice, Bob,
    and Carol") but the response only covers some of them, returns a note about
    the missing entities.  Returns None if coverage is adequate.
    """
    import re as _re_ec

    if not query or not response:
        return None

    # Extract proper-noun entities from query (multi-word capitalized sequences)
    _entity_pat = _re_ec.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    _generic = frozenset({
        "The", "This", "That", "What", "How", "Which", "When", "Where", "Who",
        "Compare", "Rank", "List", "Show", "Extract", "Find", "Tell", "Give",
        "Summary", "Details", "Information", "Document", "Table", "Please",
        "Can", "Does", "Did", "Not", "Also", "All",
    })
    query_entities = set()
    for m in _entity_pat.finditer(query):
        entity = m.group()
        # Skip if the entire match or its first word is a generic/action word
        first_word = entity.split()[0]
        if first_word in _generic:
            # But the remaining words might be entity names — extract them
            remaining = entity[len(first_word):].strip()
            if remaining and remaining.split()[0] not in _generic:
                entity = remaining
            else:
                continue
        if entity not in _generic and len(entity) > 2:
            query_entities.add(entity)

    if len(query_entities) < 2:
        return None  # Single or no entities — nothing to check

    # Check which entities appear in the response (case-insensitive word-boundary)
    response_lower = response.lower()
    missing = []
    for entity in query_entities:
        # Check both exact and first-name-only matching
        entity_lower = entity.lower()
        if _re_ec.search(r'\b' + _re_ec.escape(entity_lower) + r'\b', response_lower):
            continue
        # Try first name only for multi-word names
        first_name = entity.split()[0].lower()
        if len(first_name) > 2 and _re_ec.search(r'\b' + _re_ec.escape(first_name) + r'\b', response_lower):
            continue
        missing.append(entity)

    if not missing:
        return None

    # Check if missing entities have data in the evidence
    evidence_text = " ".join((getattr(c, "text", "") or "").lower() for c in chunks)
    missing_with_evidence = []
    missing_no_evidence = []
    for entity in missing:
        entity_lower = entity.lower()
        first_name = entity.split()[0].lower()
        if (entity_lower in evidence_text or
                (len(first_name) > 2 and first_name in evidence_text)):
            missing_with_evidence.append(entity)
        else:
            missing_no_evidence.append(entity)

    parts = []
    if missing_no_evidence:
        names = ", ".join(f"**{n}**" for n in missing_no_evidence)
        parts.append(f"{names}: No information found in the provided documents.")

    if not parts:
        return None  # Missing entities have evidence — LLM chose not to include them
        # (could be intentional, e.g., evidence was about a different topic)

    return "\n".join(parts)

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
    task_spec: Optional[Any] = None,
    conversation_context: Optional[str] = None,
) -> Optional[LLMResponseSchema]:
    """Answer *query* from *chunks* via a single LLM call with intent-adaptive prompts.

    When *task_spec* (a TaskSpec dataclass) is provided, its routing methods
    override the default intent-based chunk limits and context sizing.
    """
    if not budget.consume():
        return None

    # ── Cache check ───────────────────────────────────────────────
    _cache_enabled = False
    _cache_ttl = 3600
    try:
        from src.api.config import Config
        _cache_enabled = getattr(getattr(Config, "LLMCache", None), "ENABLED", False)
        _cache_ttl = getattr(getattr(Config, "LLMCache", None), "TTL_SECONDS", 3600)
    except Exception as exc:
        logger.debug("Failed to load LLM cache configuration", exc_info=True)

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
    # TaskSpec overrides if available (fine-tuned model provides tighter limits)
    if task_spec is not None and hasattr(task_spec, "get_chunk_limit"):
        max_chunks = task_spec.get_chunk_limit()
    else:
        max_chunks = _effective_max_chunks(num_documents, intent=intent_type, query=query)
    scored = [(getattr(c, "score", 0.0) or 0.0, c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate near-identical chunks before sending to LLM
    top_chunks = _deduplicate_evidence_chunks([c for _, c in scored[:max_chunks]])
    top_chunks = _enforce_chunk_diversity(top_chunks, max_chunks)

    # Build grouped evidence from top chunks only
    # TaskSpec overrides context sizing when available
    if task_spec is not None and hasattr(task_spec, "get_context_chars"):
        _max_ctx = task_spec.get_context_chars()
    else:
        _max_ctx = _effective_context_chars(num_documents, intent=intent_type)
    evidence = _build_grouped_evidence(top_chunks, max_context_chars=_max_ctx, domain=domain or "", query=query)

    # Build structured evidence chain for complex intents — helps LLM with
    # topic grouping, contradiction detection, and gap awareness
    evidence_context = ""
    _chain_intents = frozenset({
        "reasoning", "cross_document", "comparison", "ranking", "analytics", "summary",
    })
    if intent_type in _chain_intents or num_documents > 1:
        try:
            from .evidence_chain import build_evidence_chain
            chain = build_evidence_chain(query, top_chunks)
            evidence_context = chain.render_for_prompt()
        except Exception as exc:
            logger.debug("Failed to build evidence chain (optional enhancement)", exc_info=True)

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
        task_spec=task_spec,
        conversation_context=conversation_context,
    )

    # Use GENERATOR role for multi-agent, else default
    _gen_role = None
    try:
        from src.llm.multi_agent import MultiAgentGateway, AgentRole
        if isinstance(llm_client, MultiAgentGateway):
            _gen_role = AgentRole.GENERATOR
    except ImportError as exc:
        logger.debug("MultiAgentGateway not available for generator role selection", exc_info=True)

    # Build simplified fallback prompt for intermediate timeout
    est_tokens = _estimate_tokens(full_evidence)
    fallback_prompt = None
    if est_tokens > LLM_CHUNKED_TOKEN_THRESHOLD:
        fallback_prompt = _build_simplified_prompt(query, full_evidence, chunks=top_chunks)
        logger.info(
            "LLM extract: large context (~%d tokens, %d chunks), fallback prompt ready",
            est_tokens, len(top_chunks),
            extra={"stage": "llm_extract", "correlation_id": correlation_id},
        )

    raw_text = _generate(llm_client, prompt, correlation_id, role=_gen_role, fallback_prompt=fallback_prompt, use_thinking=use_thinking, task_spec=task_spec, intent=intent_type)
    if not raw_text:
        logger.warning(
            "LLM extract returned no result: domain=%s chunks=%d est_tokens=%d",
            intent_type, len(top_chunks), est_tokens,
            extra={"stage": "llm_extract_timeout", "correlation_id": correlation_id},
        )
        return None

    result = _parse_response(raw_text, top_chunks, query=query)

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
        except Exception as exc:
            logger.debug("Failed to load verification configuration", exc_info=True)
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

    # ── Lightweight grounding check for ALL LLM responses ──────────
    if result and result.text and top_chunks:
        grounding_score = _lightweight_grounding_check(result.text, top_chunks)
        result.grounding_confidence = grounding_score
        if grounding_score < 0.4:
            logger.warning(
                "Low grounding confidence (%.2f) for LLM response | cid=%s",
                grounding_score, correlation_id,
            )

    # ── Post-generation entity coverage validation ────────────────
    # Detect when the LLM response misses entities explicitly mentioned in the query.
    # Appends a "Missing information" note rather than re-generating (cheaper, reliable).
    if result and result.text and query:
        _missing_note = _check_entity_coverage(query, result.text, top_chunks)
        if _missing_note:
            result = LLMResponseSchema(
                text=result.text.rstrip() + "\n\n" + _missing_note,
                evidence_chunks=result.evidence_chunks,
            )
            if hasattr(result, "grounding_confidence"):
                result.grounding_confidence = grounding_score if 'grounding_score' in dir() else 0.5

    # ── Response length adequacy check ────────────────────────────
    # When evidence is rich but response is suspiciously short, log a warning.
    # This helps identify cases where the LLM truncated or gave minimal output.
    if result and result.text and top_chunks:
        _resp_len = len(result.text)
        _ev_len = len(full_evidence)
        _high_rel = full_evidence.count("[HIGH RELEVANCE]")
        if _ev_len > 4000 and _high_rel >= 3 and _resp_len < 100:
            logger.warning(
                "Suspiciously short response (%d chars) given rich evidence (%d chars, %d high-rel) | cid=%s",
                _resp_len, _ev_len, _high_rel, correlation_id,
            )

    # ── Cache write ───────────────────────────────────────────────
    if result and _cache_key and _cache_enabled and redis_client:
        _cache_set(redis_client, _cache_key, result, ttl=_cache_ttl)

    return result

# ── intent detection ─────────────────────────────────────────────────

def _classify_intent(query: str, intent_hint: Optional[str]) -> str:
    """Map query + optional hint to one of: rank, detail, contact, general.

    Uses spaCy structural analysis to identify action verbs and target nouns
    instead of regex keyword matching.
    """
    hint = (intent_hint or "").lower()
    if hint in ("rank", "compare"):
        return "rank"
    if hint in ("contact", "email", "phone"):
        return "contact"

    try:
        from src.nlp.nlu_engine import parse_query, is_contact_query

        # Check contact first using NLU
        if is_contact_query(query):
            return "contact"

        # Use spaCy semantic analysis for intent
        sem = parse_query(query)
        _RANK_VERBS = {"rank", "compare", "evaluate", "assess", "rate", "score", "benchmark"}
        _LIST_VERBS = {"list", "extract", "show", "give", "enumerate"}

        if any(v in _RANK_VERBS for v in sem.action_verbs):
            return "rank"
        if any(v in _LIST_VERBS for v in sem.action_verbs):
            return "detail"
        # Also check target nouns for implicit ranking
        _RANK_NOUNS = {"ranking", "comparison", "evaluation", "assessment"}
        if any(n in _RANK_NOUNS for n in sem.target_nouns + sem.context_words):
            return "rank"
    except Exception as exc:
        logger.debug("Failed NLU-based intent classification", exc_info=True)
    return "general"

# ── expanded intent classification (8 types) ─────────────────────────

def classify_query_intent(query: str, *, intent_hint: str | None = None) -> str:
    """Classify query into one of 8 intent types for response template selection.

    Uses the centralized NLU engine with embedding similarity + structural NLP
    instead of hardcoded regex patterns.
    """
    from src.nlp.nlu_engine import classify_intent
    return classify_intent(query, intent_hint=intent_hint)

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
        "Extract the specific contact information requested:\n"
        "THINK: 1) Identify all contact types in the evidence (email, phone, LinkedIn, address). "
        "2) Extract each variant found (work email vs personal, mobile vs office). "
        "3) Verify completeness — list ALL contact items, not just the first one.\n"
        "- Present each contact item on its own line with the type labeled\n"
        "- Include all variants found (e.g., multiple phone numbers)\n"
        "- **Bold** the contact values\n"
        "- Note which document each contact came from\n"
    ),
    "detail": (
        "Extract and present the specific information requested:\n"
        "THINK: 1) Identify what specific details the user is asking for. "
        "2) Scan all evidence chunks for relevant information. "
        "3) Organize by category or importance.\n"
        "- Organize the response clearly with bullet points or sections\n"
        "- Include all relevant details found across the documents\n"
        "- **Bold** key values, names, and numbers\n"
        "- Cite source documents for each detail\n"
    ),
    "general": (
        "Answer the question based on the document evidence:\n"
        "THINK: 1) What is the user asking? What kind of answer do they expect? "
        "2) Which evidence chunks are most relevant? Prioritize [HIGH RELEVANCE] chunks. "
        "3) Organize the answer: lead with the key finding, then supporting details.\n"
        "- Start with a direct one-line answer containing the key fact\n"
        "- **Bold** the most important 2-3 values per section\n"
        "- Choose the best format for the content:\n"
        "  * Single fact → one sentence\n"
        "  * List of items → bullet points\n"
        "  * Structured data → markdown table\n"
        "  * Explanation → short paragraphs\n"
        "- If the question has multiple parts, answer ALL parts with clear separation\n"
        "- Cite source: '(source: filename.pdf)' once per section\n"
    ),
}

# ── intent-adaptive generation templates (8 types) ───────────────────

_GENERATION_SYSTEM = (
    "You are a document intelligence assistant. Answer ONLY from the evidence provided.\n\n"
    "CORE PRINCIPLES:\n"
    "1. GROUND EVERYTHING: Every claim must come from evidence. Use exact numbers, dates, names.\n"
    "   - Conflicts between sources: report BOTH with attribution ('Doc A: $50K, Doc B: $55K').\n"
    "   - Missing info: say 'Not found in the provided documents.' Never silently omit.\n"
    "   - Partial results: state what was found AND what's missing.\n\n"
    "2. DIRECT AND STRUCTURED:\n"
    "   - Lead with the answer. No preamble ('Based on my analysis...').\n"
    "   - **Bold** key values (2-3 per paragraph). Group related facts.\n"
    "   - Format to content: tables for comparisons, bullets for lists, prose for reasoning.\n"
    "   - 'N/A' for missing table cells. Complete every table/list you start.\n"
    "   - Cite sources: '(source: filename.pdf)' once per section.\n\n"
    "3. PRECISION:\n"
    "   - Names: exact spelling from document. '$12,345.67' not 'about $12K'. '87.5%' not '~88%'.\n"
    "   - Dates: preserve document format. Never contradict yourself on a value.\n"
    "   - Tables: every row must have the same columns. No skipped sequence numbers.\n\n"
    "4. COMPLETENESS:\n"
    "   - Answer ALL parts of the question. If asked about 3 items, address all 3.\n"
    "   - For multi-turn context: resolve pronouns naturally, don't re-introduce known entities.\n\n"
    "5. SELF-CHECK: Before responding, verify every number/name/date matches evidence exactly.\n"
    "   - Count: if asked about N items, verify you covered all N.\n"
    "   - Consistency: don't state '5 years' in one place and '6 years' elsewhere.\n\n"
    "COMMON MISTAKES TO AVOID:\n"
    "- BAD: 'Based on the documents...' → GOOD: '**John Smith** has **8 years** of experience (source: resume.pdf).'\n"
    "- BAD: Inventing a job title not in evidence → GOOD: 'Job title not found in the provided documents.'\n"
    "- BAD: '50K' when document says '$50,000' → GOOD: Use exact format from document.\n"
    "- BAD: Table with 3 columns in header but 2 in data rows → GOOD: Pad with 'N/A'.\n"
    "- BAD: Answering only 2 of 3 asked questions → GOOD: Address all parts, even if to say 'not found'.\n"
    "- BAD: Ignoring conflicting data → GOOD: 'Doc A states $50K; Doc B states $55K.'\n"
)

# Intent-aware response length hint — helps the model calibrate output length
_INTENT_LENGTH_HINTS = {
    "factual": "TARGET LENGTH: 1-3 sentences. Be precise and concise. No preamble.",
    "contact": "TARGET LENGTH: 1-2 sentences or a short list. Just the facts, no narrative.",
    "detail": "TARGET LENGTH: 3-8 sentences or structured bullets. Cover all relevant details.",
    "extract": "TARGET LENGTH: Bulleted list. One item per line, no narrative.",
    "clarify": "TARGET LENGTH: 1-2 sentences. Ask what's needed.",
    "compare": "TARGET LENGTH: Comparison table + 2-3 sentence synthesis. Cover ALL entities.",
    "comparison": "TARGET LENGTH: Comparison table + 2-3 sentence synthesis. Cover ALL entities.",
    "rank": "TARGET LENGTH: Ranked table + 1-2 sentence recommendation.",
    "ranking": "TARGET LENGTH: Ranked table with scoring + recommendation paragraph.",
    "summarize": "TARGET LENGTH: 3-5 bullet points or a concise paragraph.",
    "summary": "TARGET LENGTH: 3-5 bullet points or a concise paragraph. Start with overview.",
    "generate": "TARGET LENGTH: Full-length output as requested. Match the expected format.",
    "analyze": "TARGET LENGTH: Structured analysis with key findings highlighted.",
    "analytics": "TARGET LENGTH: Statistics table + trend observations. Include numbers.",
    "timeline": "TARGET LENGTH: Chronological list with dates and key events.",
    "multi_field": "TARGET LENGTH: One answer per requested field. Cover ALL fields asked.",
    "cross_document": "TARGET LENGTH: Synthesis table + narrative connecting findings across ALL docs.",
    "reasoning": "TARGET LENGTH: Step-by-step analysis with conclusion. Show your reasoning chain.",
}

_GENERATION_TEMPLATES = {
    "factual": (
        "Provide a precise, evidence-based answer:\n"
        "THINK: 1) Identify which evidence chunks contain the answer. "
        "2) Extract the exact values/names/dates. 3) Cross-check if multiple sources agree.\n"
        "- Start with a one-line direct answer with the **key fact bolded**\n"
        "- Include exact values, names, dates, and numbers from the documents\n"
        "- If multiple items, use bullet points (- item) or a table\n"
        "- Cite source documents by name\n"
        "- **Bold** the most important values and answers\n"
    ),
    "comparison": (
        "Compare the {num_documents} document(s) systematically:\n"
        "THINK: 1) Identify the subjects to compare (names, entities). "
        "2) Select 3-5 most relevant criteria from the evidence (avoid >6 columns). "
        "3) For each subject, extract values for each criterion. "
        "4) Determine the overall winner based on the criteria.\n"
        "- Start with a one-line summary naming the top candidate/item and why\n"
        "- Present the comparison as a MARKDOWN TABLE with 3-5 criterion columns\n"
        "- **Bold** the winner/best value in each category\n"
        "- Order rows by overall strength (strongest first)\n"
        "- After the table, give a brief recommendation with justification\n"
        "- End with key differences and any close calls\n"
        "- ALL rows MUST use the SAME columns. If a criterion doesn't apply, use 'N/A'.\n"
        "- If sources CONFLICT on a value, report BOTH with source attribution.\n"
    ),
    "summary": (
        "Provide a structured summary:\n"
        "THINK: 1) Count the documents and identify their types/domains. "
        "2) Extract the most important facts from each. "
        "3) Identify common themes and aggregate statistics.\n"
        "- Start with a one-line overview of what was analyzed\n"
        "- Present **3-6 key highlights** as bullet points with specific details\n"
        "- **Bold** the most important findings\n"
        "- Include totals, counts, and ranges where applicable\n"
        "- End with a brief conclusion\n"
    ),
    "ranking": (
        "Rank the {num_documents} subjects with detailed scoring:\n"
        "THINK: 1) Identify what criteria are relevant for ranking from the query. "
        "2) For each subject, extract values for these criteria from the evidence. "
        "3) Determine ranking order based on overall strength across criteria.\n"
        "- First, define scoring criteria from the evidence (experience, skills, etc.)\n"
        "- Start with a one-line statement naming the **#1 ranked** item and why\n"
        "- Create a SCORING TABLE with multiple criteria columns (not just one dimension)\n"
        "- For each ranked item, cite the document source for each strength/gap\n"
        "- Note close calls: 'Candidates A and B are close on [criterion]'\n"
        "- End with a **Recommendation** paragraph explaining the top choice\n"
        "- Always rank by MULTIPLE dimensions — never just one factor\n"
        "- Use WEIGHTED SCORING: Experience 40%, Skills 35%, Education 25% (adjust weights if query specifies)\n"
        "- Score each criterion on a 0-10 scale with brief justification per score\n"
        "- Show a final WEIGHTED SCORE per subject and declare an explicit winner\n"
    ),
    "timeline": (
        "Present in chronological order:\n"
        "THINK: 1) Scan all evidence for dates and timestamps. "
        "2) Sort events chronologically. "
        "3) Identify gaps where no data exists between events.\n"
        "- Start with a one-line overview: **earliest to latest** dates\n"
        "- Use a table or numbered list ordered by date\n"
        "- **Bold** key milestones and dates\n"
        "- Note any gaps in the timeline\n"
    ),
    "multi_field": (
        "CRITICAL: The user asked for MULTIPLE pieces of information. Answer ALL parts — do not omit any. If a field has no evidence, explicitly state 'Not found in documents'.\n"
        "Extract and present all requested fields:\n"
        "THINK: 1) List all fields requested by the user. "
        "2) For each field, scan evidence for the value. "
        "3) If a field appears in multiple sources, use the most complete one.\n"
        "- Present as a MARKDOWN TABLE with columns: Field | Value | Source\n"
        "- **Bold** the key values\n"
        "- Mark missing fields with 'N/A' or 'Not found'\n"
        "- Include source document for each field\n"
    ),
    "reasoning": (
        "Reason through the question step by step:\n"
        "THINK: 1) What is the user really asking? What type of evidence would answer it? "
        "2) Which evidence chunks support a conclusion? "
        "3) Is there contradicting evidence or gaps that weaken the answer?\n"
        "- Start with a one-line **conclusion** answering the question directly\n"
        "- Present supporting evidence as bullet points\n"
        "- **Bold** the key facts that support the conclusion\n"
        "- Note gaps — explicitly state what is NOT in the documents\n"
        "- End with a qualified assessment based on evidence strength\n"
    ),
    "cross_document": (
        "Analyze and synthesize across all {num_documents} document(s):\n"
        "THINK: 1) Identify which documents contribute to the answer. "
        "2) Extract key facts per document WITH source attribution. "
        "3) Find commonalities (patterns in 2+ docs). "
        "4) Identify contradictions — when facts conflict, prioritize: most recent > most authoritative > most specific.\n"
        "- Start with a one-line overview: **N documents analyzed, key synthesis**\n"
        "- Use a TABLE to compare findings across documents\n"
        "- **Bold** common patterns. HIGHLIGHT contradictions (e.g., 'Doc A: $50K vs Doc B: $55K')\n"
        "- For each contradiction: state BOTH values with sources, don't pick silently\n"
        "- Synthesize only where docs agree; flag where evidence conflicts\n"
    ),
    "analytics": (
        "Compute aggregate statistics from the {num_documents} document(s):\n"
        "THINK: 1) Extract all numeric values from the evidence. "
        "2) Compute totals, averages, min, max. "
        "3) Identify trends (increasing/decreasing) and anomalies.\n"
        "- Start with a one-line **total/aggregate** answer\n"
        "- Present statistics in a TABLE: Category | Value | Details\n"
        "- **Bold** the highest, lowest, and total values\n"
        "- Identify patterns and distributions across the data\n"
    ),
    "generate": (
        "Generate the requested content based on the document evidence:\n"
        "THINK: 1) Identify the output type requested (letter, email, summary, interview questions, report). "
        "2) Extract all relevant facts, names, and details from the evidence that should appear in the output. "
        "3) Structure the output appropriately for its type (formal tone for letters, Q&A format for interview questions). "
        "4) Verify every claim or detail in the generated content is sourced from the evidence.\n"
        "- For letters/emails: use proper greeting, body paragraphs, and closing\n"
        "- For summaries: use bullet points for key findings, bold important values\n"
        "- For interview questions: number each question, group by topic area\n"
        "- For reports: use ## headers for sections, include data tables where relevant\n"
        "- Always ground generated content in the document evidence — never fabricate details\n"
    ),
    "contact": (
        "Extract contact information:\n"
        "THINK: 1) Scan all evidence for names, emails, phone numbers, addresses. "
        "2) Group by person/entity. 3) Present in a clean format.\n"
        "- List each contact on a separate line with label: Name, Email, Phone, Address\n"
        "- **Bold** the person/entity names\n"
        "- If multiple contacts exist, present as a table: Name | Email | Phone\n"
    ),
    "detail": (
        "Extract detailed information about the requested topic:\n"
        "THINK: 1) Identify the specific topic/entity the user is asking about. "
        "2) Gather ALL relevant facts from the evidence. "
        "3) Organize facts by sub-topic for readability.\n"
        "- Start with a one-line overview identifying the subject\n"
        "- Present details as organized bullet points grouped by sub-topic\n"
        "- **Bold** key values (numbers, dates, proper nouns)\n"
        "- Include all relevant details — don't summarize away specifics\n"
    ),
    "extract": (
        "Extract the requested data from the documents:\n"
        "THINK: 1) Identify exactly what data is requested. "
        "2) Scan each evidence chunk for matching fields. "
        "3) Present in structured format.\n"
        "- Use a table for structured data: Field | Value | Source\n"
        "- Use a bulleted list for unstructured extraction\n"
        "- **Bold** key values\n"
        "- Mark any unfound fields as 'Not found in documents'\n"
    ),
}

# ── Few-shot examples for format compliance ────────────────────────────

_FEW_SHOT_EXAMPLES = {
    "factual": (
        "\nEXAMPLE OUTPUT:\n"
        "**John Smith** has **8 years** of Python experience (Source: john_smith_resume.pdf).\n"
        "He specializes in **machine learning** and **cloud architecture (AWS)**.\n"
    ),
    "comparison": (
        "\nEXAMPLE OUTPUT:\n"
        "## Candidate Comparison\n"
        "**Alice Chen** leads with the strongest technical profile.\n\n"
        "| Criterion | **Alice Chen** | Bob Martinez | Carol Lee |\n"
        "|-----------|-------------|--------------|----------|\n"
        "| Experience | **10 years** | 5 years | 7 years |\n"
        "| Key Skills | **Python, AWS, K8s** | Java, React | Python, GCP |\n"
        "| Education | **MS CS Stanford** | BS CS UCLA | MS CS MIT |\n\n"
        "**Recommendation: Alice Chen** — strongest combination of experience and skills.\n"
    ),
    "summary": (
        "\nEXAMPLE OUTPUT:\n"
        "## Summary: Invoice Analysis (3 documents)\n"
        "Analyzed **3 invoices** totaling **$45,230** from 2 vendors.\n\n"
        "- **Highest invoice**: $22,100 from Acme Corp (INV-2024-001)\n"
        "- **Payment terms**: Net 30 across all invoices\n"
        "- **Common items**: Cloud services, consulting hours\n"
    ),
    "ranking": (
        "\nEXAMPLE OUTPUT:\n"
        "**#1: Sarah Johnson** — best overall fit for the Senior Engineer role.\n\n"
        "| Rank | Name | Score | Key Strength |\n"
        "|------|------|-------|--------------|\n"
        "| **1** | **Sarah Johnson** | **92/100** | 12yr experience, team lead |\n"
        "| 2 | Mike Chen | 85/100 | Strong AWS, 8yr experience |\n"
        "| 3 | Lisa Park | 78/100 | Recent bootcamp, eager learner |\n"
    ),
    "reasoning": (
        "\nEXAMPLE OUTPUT:\n"
        "**Conclusion: The patient shows signs of improving glycemic control.**\n\n"
        "Supporting evidence:\n"
        "- **HbA1c decreased** from 8.1% to **7.2%** over 6 months\n"
        "- **Fasting glucose**: 110 mg/dL (within target range)\n"
        "- **Gap**: No lipid panel results found in the provided documents\n"
    ),
    "cross_document": (
        "\nEXAMPLE OUTPUT:\n"
        "## Cross-Document Analysis (4 resumes)\n\n"
        "| Aspect | Doc 1 (Resume A) | Doc 2 (Resume B) | Doc 3 (Resume C) |\n"
        "|--------|------------------|------------------|------------------|\n"
        "| Experience | 8 years | 12 years | 5 years |\n"
        "| Top Skill | Python | Java | Go |\n\n"
        "**Common across all**: Bachelor's degree, cloud experience\n"
        "**Unique to Doc 1**: ML specialization\n"
        "**Note**: Docs 1 and 3 report conflicting project dates (2021 vs 2022) — used Doc 1 as primary.\n"
    ),
    "analytics": (
        "\nEXAMPLE OUTPUT:\n"
        "## Invoice Analytics Summary\n\n"
        "**Total spend**: $127,450 across **8 invoices** from **3 vendors**\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Average invoice | $15,931 |\n"
        "| Largest | $34,200 (Vendor A, INV-045) |\n"
        "| Smallest | $2,100 (Vendor C, INV-039) |\n\n"
        "**Trend**: Spending increased 23% from Q1 to Q2.\n"
        "**Anomaly**: INV-042 has no tax line — verify if tax-exempt.\n"
    ),
    "timeline": (
        "\nEXAMPLE OUTPUT:\n"
        "**Patient Timeline (2022-03 to 2024-01)**\n\n"
        "| Date | Event | Source |\n"
        "|------|-------|--------|\n"
        "| **2022-03-15** | Diagnosed with Type 2 Diabetes (HbA1c **8.9%**) | progress_note_2022.pdf |\n"
        "| **2022-06-20** | Started Metformin 500mg daily | progress_note_2022.pdf |\n"
        "| **2023-01-10** | Added Lisinopril 10mg for hypertension | progress_note_2023.pdf |\n"
        "| **2024-01-05** | HbA1c improved to **6.9%** — treatment effective | lab_report_2024.pdf |\n\n"
        "**Gap**: No records found between Jun 2022 and Jan 2023 (7-month gap).\n"
    ),
    "multi_field": (
        "\nEXAMPLE OUTPUT:\n"
        "## Candidate Profile: **John Smith**\n\n"
        "| Field | Value | Source |\n"
        "|-------|-------|--------|\n"
        "| Name | **John Smith** | john_smith_resume.pdf |\n"
        "| Experience | **8 years** in software engineering | john_smith_resume.pdf |\n"
        "| Skills | Python, AWS, Kubernetes, React | john_smith_resume.pdf |\n"
        "| Education | MS Computer Science, Stanford | john_smith_resume.pdf |\n"
        "| Certifications | Not found | — |\n"
        "| Salary | Not found | — |\n"
    ),
    "generate": (
        "\nEXAMPLE OUTPUT (interview questions):\n"
        "## Interview Questions for **Alice Chen** (Senior Engineer)\n\n"
        "1. **[Technical]** Your resume mentions building a real-time data pipeline with Kafka. "
        "Walk me through how you handled exactly-once delivery semantics.\n"
        "   *Rationale: Verifies depth of Kafka expertise claimed in resume.*\n\n"
        "2. **[Behavioral]** You led a team of 5 engineers on the cloud migration project. "
        "Describe a time when a team member disagreed with your approach.\n"
        "   *Rationale: Assesses leadership and conflict resolution skills.*\n\n"
        "3. **[Situational]** If asked to reduce the deployment pipeline from 45 minutes "
        "to under 10, how would you approach it given your Docker/K8s experience?\n"
        "   *Rationale: Tests practical problem-solving with stated skills.*\n"
    ),
    "contact": (
        "Q: What are Sarah's contact details?\n"
        "A: **Contact Information — Sarah Mitchell** (source: sarah_resume.pdf)\n"
        "- **Email:** sarah.mitchell@email.com\n"
        "- **Phone:** +1 (555) 234-5678\n"
        "- **LinkedIn:** linkedin.com/in/sarahmitchell\n"
        "- **Location:** San Francisco, CA\n"
    ),
    "detail": (
        "Q: What is the payment term on invoice #1042?\n"
        "A: The payment term on **Invoice #1042** is **Net 30** — payment is due within "
        "30 days of the invoice date (**March 15, 2024**). A **2% early payment discount** "
        "applies if paid within 10 days. (source: invoice_1042.pdf, p. 2)\n"
    ),
    "partial_data": (
        "Q: What are the salaries of all three candidates?\n"
        "A: **Salary information found for 2 of 3 candidates:**\n"
        "- **Alice Chen:** **$125,000/year** (source: alice_resume.pdf, p. 1)\n"
        "- **Bob Kumar:** **$98,500/year** (source: bob_resume.pdf, p. 2)\n"
        "- **Carlos Rivera:** Salary information was **not found** in the provided documents.\n"
    ),
    "extract": (
        "\nEXAMPLE OUTPUT:\n"
        "**Extracted from 2 documents:**\n\n"
        "- **Phone**: +1 (555) 234-5678 (resume_A.pdf)\n"
        "- **Email**: alice@company.com (resume_A.pdf)\n"
        "- **Address**: 123 Main St, City, ST 12345 (resume_B.pdf)\n"
        "- **LinkedIn**: Not found in documents"
    ),
    "reasoning": (
        "\nEXAMPLE OUTPUT:\n"
        "**Alice Chen** is the strongest candidate for the senior engineer role.\n\n"
        "**Supporting evidence:**\n"
        "- **10 years** of Python experience directly matches the requirement (source: alice_resume.pdf)\n"
        "- Led a **team of 8** engineers on the cloud migration — demonstrates leadership at scale\n"
        "- AWS Solutions Architect certification aligns with the cloud-first strategy\n\n"
        "**Gaps:**\n"
        "- No evidence of **Go** experience mentioned in the job requirements\n"
        "- Salary expectations not disclosed in the resume\n\n"
        "**Assessment:** Strong fit (3/4 key criteria met) based on available evidence.\n"
    ),
}

# ── Domain-specific reasoning injection ────────────────────────────────

_DOMAIN_REASONING = {
    "hr": (
        "HR/RESUME ANALYSIS GUIDELINES:\n"
        "- Weight recent experience (last 3-5 years) more heavily than older roles\n"
        "- Flag employment gaps > 6 months if relevant to the query\n"
        "- For skill assessments, distinguish stated skills from demonstrated experience\n"
        "- When comparing candidates, use consistent criteria across all\n"
        "- SKILL GAP ANALYSIS: If query implies a role, identify missing skills: "
        "'Strong in Python/AWS but no Kubernetes experience mentioned'\n"
        "- CAREER PROGRESSION: Note trajectory (junior→senior→lead) and role tenure\n"
        "- QUANTIFIED ACHIEVEMENTS: Highlight metrics (team size, revenue impact, cost savings)\n"
    ),
    "medical": (
        "CLINICAL DOCUMENT GUIDELINES:\n"
        "- When interpreting lab values, note whether they are within normal ranges\n"
        "- For medications, note dosages and frequencies exactly as stated\n"
        "- Do not make diagnostic conclusions — only report what the documents state\n"
        "- Flag any contraindications or interactions explicitly mentioned in the evidence\n"
        "- ABNORMAL FLAGS: When lab values are outside reference ranges, note: "
        "'**HbA1c: 8.1%** (above target <7.0%)'\n"
        "- MEDICATION INTERACTIONS: If multiple drugs listed, note known interactions "
        "only if mentioned in evidence\n"
        "- TREND DETECTION: If sequential values exist, note direction: "
        "'Improving (8.1% → 7.2%)' or 'Worsening'\n"
    ),
    "legal": (
        "LEGAL DOCUMENT GUIDELINES:\n"
        "- Classify clause risk as HIGH (creates liability/unenforceable), MEDIUM (ambiguous), LOW (standard boilerplate)\n"
        "- Quote exact clause language when identifying risks\n"
        "- Note jurisdiction-specific details if mentioned\n"
        "- Distinguish binding obligations from optional provisions\n"
        "- AMBIGUITY FLAGS: Identify vague terms like 'reasonable efforts', "
        "'material breach', 'best endeavors' — note they may need definition\n"
        "- MISSING PROTECTIONS: Flag common clauses NOT present (indemnification, "
        "limitation of liability, termination, dispute resolution)\n"
        "- RISK SUMMARY: End with: 'Overall risk: HIGH/MEDIUM/LOW — [1-line reason]'\n"
    ),
    "invoice": (
        "FINANCIAL DOCUMENT GUIDELINES:\n"
        "- Report all monetary values with exact currency and amounts\n"
        "- Calculate totals, subtotals, and tax amounts accurately from the evidence\n"
        "- Note payment terms, due dates, and vendor details precisely\n"
        "- Flag any discrepancies between line items and totals\n"
        "- PAYMENT HEALTH: Note if overdue (past due date), disputed, or partially paid\n"
        "- ITEMIZATION QUALITY: Flag if invoice lacks line items, quantities, or unit prices\n"
        "- ANOMALY DETECTION: Flag unusual items (e.g., tax-exempt without reason, "
        "duplicate line items, amounts significantly different from others)\n"
    ),
    "policy": (
        "POLICY/INSURANCE DOCUMENT GUIDELINES:\n"
        "- Distinguish coverage inclusions from exclusions clearly\n"
        "- Note policy limits, deductibles, and effective dates exactly\n"
        "- For claims, extract claim numbers, dates, and status\n"
        "- Identify conditions and prerequisites for coverage\n"
        "- COVERAGE GAPS: Identify common coverage types NOT mentioned "
        "(liability, property, business interruption)\n"
        "- FINE PRINT: Flag conditions that could void coverage "
        "(notification deadlines, exclusion triggers)\n"
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
    "generate",
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

def _assess_evidence_quality(evidence_text: str, query: str, num_documents: int) -> str:
    """Generate a brief evidence quality signal for the LLM prompt.

    Analyzes whether the evidence is sparse, dense, or has quality issues,
    helping the LLM calibrate its confidence and response style.
    """
    if not evidence_text:
        return "EVIDENCE STATUS: No evidence found. State that no information is available.\n"

    import re as _re_eq

    ev_len = len(evidence_text)
    lines = [l.strip() for l in evidence_text.split("\n") if l.strip()]
    high_rel_count = evidence_text.count("[HIGH RELEVANCE]")
    mod_rel_count = evidence_text.count("[MODERATE RELEVANCE]")

    # Extract query content words
    _stop = {"what", "is", "are", "the", "a", "an", "of", "for", "in", "how",
             "do", "does", "can", "about", "from", "with", "tell", "me", "show"}
    q_words = {w.lower().rstrip("?,!.") for w in query.split() if w.lower() not in _stop and len(w) > 2}

    # Check how many query terms appear in evidence (word-boundary matching)
    ev_lower = evidence_text.lower()
    matched_terms = sum(
        1 for w in q_words
        if _re_eq.search(r'\b' + _re_eq.escape(w) + r'\b', ev_lower)
    )
    coverage = matched_terms / max(len(q_words), 1)

    parts = []
    if ev_len < 500:
        parts.append("Limited evidence available")
    elif ev_len > 8000:
        parts.append("Rich evidence available")

    if high_rel_count >= 3:
        parts.append(f"{high_rel_count} highly relevant segments")
    elif high_rel_count == 0 and mod_rel_count > 0:
        parts.append("No strongly relevant segments — use moderate matches carefully")
    elif high_rel_count == 0 and mod_rel_count == 0:
        parts.append("Weak evidence match — answer may be incomplete")

    if coverage < 0.3:
        parts.append("Some query terms not found in evidence — note gaps")
    elif coverage > 0.8:
        parts.append("Good query-evidence alignment")

    # Document diversity signal for multi-doc queries
    if num_documents > 1:
        # Count unique doc headers in evidence
        _doc_headers = _re_eq.findall(r"=== Document: (.+?) ===", evidence_text)
        _unique_docs = len(set(_doc_headers))
        if _unique_docs < num_documents:
            parts.append(f"Evidence from {_unique_docs}/{num_documents} documents — some documents may lack relevant content")
        elif _unique_docs >= num_documents:
            parts.append(f"All {num_documents} documents represented in evidence")

    # Section variety signal — helps LLM know if evidence is narrow or broad
    _section_tags = _re_eq.findall(r"\[([a-z_]+)\]", evidence_text[:3000])
    _unique_sections = set(_section_tags)
    if len(_unique_sections) >= 4:
        parts.append(f"Evidence spans {len(_unique_sections)} sections")

    # Always emit a signal — moderate evidence is the most common case
    if not parts:
        parts.append("Moderate evidence available — answer from what is provided")
    return "EVIDENCE STATUS: " + "; ".join(parts) + ".\n"

def _build_entity_reminder(query: str, evidence_text: str) -> str:
    """Build an entity coverage reminder for the LLM prompt.

    When the query mentions 2+ named entities (e.g., "Compare Alice and Bob"),
    explicitly lists them so the LLM doesn't skip any. Only triggers for
    multi-entity queries where at least one entity appears in the evidence.
    """
    import re as _re_er

    if not query:
        return ""

    _entity_pat = _re_er.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    _generic = frozenset({
        "The", "This", "That", "What", "How", "Which", "When", "Where", "Who",
        "Compare", "Rank", "List", "Show", "Extract", "Find", "Tell", "Give",
        "Summary", "Details", "Information", "Document", "Table", "Please",
        "Can", "Does", "Did", "Not", "Also", "All", "Each", "Both",
    })

    entities = []
    for m in _entity_pat.finditer(query):
        entity = m.group()
        first_word = entity.split()[0]
        if first_word in _generic:
            remaining = entity[len(first_word):].strip()
            if remaining and remaining.split()[0] not in _generic:
                entity = remaining
            else:
                continue
        if entity not in _generic and len(entity) > 2:
            if entity not in entities:
                entities.append(entity)

    if len(entities) < 2:
        return ""

    # Check if entities appear in evidence
    ev_lower = evidence_text.lower()
    in_evidence = [e for e in entities if e.lower() in ev_lower]
    if not in_evidence:
        return ""

    entity_list = ", ".join(f"**{e}**" for e in entities)
    return (
        f"ENTITIES TO ADDRESS: {entity_list}\n"
        f"Cover each entity above in your response. "
        f"If information for any entity is missing from the evidence, explicitly state that.\n"
    )

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
    task_spec: Optional[Any] = None,
    conversation_context: Optional[str] = None,
) -> str:
    """Build a structured prompt for the LLM generation call.

    When *task_spec* is provided, its natural text description is prepended
    to give the LLM a structured understanding of the task.
    """
    # Normalize intent aliases to template keys
    _INTENT_ALIASES = {
        "qa": "factual", "extraction": "factual",
        "list": "multi_field",
        "compare": "comparison", "rank": "ranking",
        "summarize": "summary", "summarise": "summary",
        "analyze": "reasoning", "analyse": "reasoning", "analysis": "reasoning",
        "cross_doc": "cross_document",
        # NOTE: "contact", "detail", "extract" have dedicated templates —
        # do NOT alias them to "factual" (makes those templates unreachable)
    }
    intent = _INTENT_ALIASES.get(intent, intent)

    # Detect multi-part queries and upgrade to multi_field intent
    if _is_multi_part_query(query) and intent not in ("comparison", "ranking", "cross_document", "generate"):
        intent = "multi_field"

    # Handle ultra-short queries (1-2 words) — these are often entity lookups
    # or follow-ups that the classifier struggles with
    _words = (query or "").split()
    if len(_words) <= 2 and intent in ("factual", ""):
        _has_question_word = any(w.lower() in {"how", "what", "why", "which", "when", "where", "who"} for w in _words)
        if not _has_question_word:
            # Bare name or minimal phrase — keep as factual but it's likely
            # an entity lookup. The evidence block will be entity-focused.
            intent = "factual"

    # Auto-detect intent from query when classifier returned generic intent
    if intent in ("factual", ""):
        _ql = (query or "").lower()
        if num_documents > 1 and any(w in _ql for w in (
            "compare", "comparison", "versus", " vs ", "difference",
            "similarities", "how do they differ", "pros and cons",
            "side by side", "contrast", "distinguish",
        )):
            intent = "comparison"
        elif num_documents > 1 and any(w in _ql for w in (
            "rank", "best", "top ", "strongest", "weakest",
            "most qualified", "most experienced", "who is better",
            "sort by", "order by", "prioritize",
        )):
            intent = "ranking"
        elif any(w in _ql for w in ("summar", "overview", "key highlights", "brief", "at a glance", "tl;dr", "main points")):
            intent = "summary"
        elif any(w in _ql for w in ("explain", "why ", "how does", "what causes", "reason", "what led to", "root cause", "implication")):
            intent = "reasoning"
        elif any(w in _ql for w in ("timeline", "chronolog", "history of", "over time", "sequence of", "progression", "milestones")):
            intent = "timeline"
        elif any(w in _ql for w in ("generate", "write ", "draft ", "create ", "compose", "prepare ", "formulate")):
            intent = "generate"
        elif any(w in _ql for w in ("total", "average", "count", "how many", "percentage", "distribution", "breakdown")):
            intent = "analytics"
        elif num_documents > 1 and any(w in _ql for w in ("across all", "each document", "every document", "all candidates")):
            intent = "cross_document"
        elif any(w in _ql for w in ("contact", "email", "phone", "address", "reach", "call")):
            intent = "contact"
        elif any(w in _ql for w in ("detail", "details", "specifics", "particular", "elaborate", "more about")):
            intent = "detail"

    template = _GENERATION_TEMPLATES.get(intent, _GENERATION_TEMPLATES["factual"])
    if "{num_documents}" in template:
        template = template.format(num_documents=num_documents)

    # Use scaled limits for intent and multi-doc queries
    effective_max = _effective_context_chars(num_documents, intent=intent)
    # Adaptive margin: multi-doc queries need more evidence space
    _safety_margin = 1500 if num_documents <= 1 else 1000
    max_evidence = effective_max - _safety_margin
    _evidence_truncated = False
    if len(evidence_text) > max_evidence:
        _trunc = evidence_text[:max_evidence]
        # Find last sentence boundary to avoid mid-sentence cuts
        _last_boundary = max(_trunc.rfind('. '), _trunc.rfind('.\n'), _trunc.rfind('?\n'), _trunc.rfind('!\n'))
        if _last_boundary > max_evidence * 0.8:
            evidence_text = _trunc[:_last_boundary + 2]
        else:
            evidence_text = _trunc
        _evidence_truncated = True

    tool_section = f"\nDOMAIN EXPERTISE:\n{tool_context}\n" if tool_context else ""

    # Inject domain knowledge context when available and no tool_context already present
    domain_section = ""
    if domain and not tool_context:
        domain_section = _get_domain_knowledge_section(domain, intent)

    # Inject domain-specific reasoning guidelines
    domain_reasoning_section = ""
    if domain and domain in _DOMAIN_REASONING:
        domain_reasoning_section = f"\n{_DOMAIN_REASONING[domain]}\n"

    # Inject few-shot example for format compliance (only when TaskSpec doesn't provide one)
    few_shot_section = ""
    if not task_spec or not (hasattr(task_spec, "get_few_shot_example") and task_spec.get_few_shot_example()):
        few_shot = _FEW_SHOT_EXAMPLES.get(intent, "")
        # Fallback: use factual example for unrecognized intents (better than nothing)
        if not few_shot and intent not in _FEW_SHOT_EXAMPLES:
            few_shot = _FEW_SHOT_EXAMPLES.get("factual", "")
        if few_shot:
            few_shot_section = few_shot

    # Inject ML-based context understanding when available
    context_section = ""
    if context_intelligence:
        context_section = f"\n{context_intelligence}\n"

    # Prepend analytical reasoning preamble for complex intents, multi-doc, or high-evidence queries
    reasoning_section = ""
    _needs_reasoning = (
        intent in _COMPLEX_INTENTS
        or num_documents > 1
        or (num_documents == 1 and len(evidence_text) > 4000)  # long evidence needs structured approach
        or (domain and domain in ("medical", "legal", "policy"))  # precision domains benefit from reasoning
    )
    if _needs_reasoning:
        reasoning_section = _REASONING_PREAMBLE

    # Intent-aware token budget hint
    _budget_hint = _INTENT_LENGTH_HINTS.get(intent, "")

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

    # Detect mixed-domain evidence and inject a disambiguation note
    _domain_mix_note = ""
    if num_documents > 1 and evidence_text:
        _domain_keywords = {
            "medical": {"patient", "diagnosis", "medication", "lab", "clinical", "treatment"},
            "legal": {"clause", "agreement", "liability", "indemnification", "jurisdiction"},
            "invoice": {"invoice", "subtotal", "amount due", "bill to", "payment terms"},
            "hr": {"resume", "candidate", "experience", "skills", "education"},
            "policy": {"coverage", "premium", "exclusion", "deductible", "policyholder"},
        }
        _ev_lower = evidence_text[:3000].lower()
        _detected_domains = [d for d, kws in _domain_keywords.items() if sum(1 for k in kws if k in _ev_lower) >= 2]
        if len(_detected_domains) >= 2:
            _domain_mix_note = (
                f"NOTE: Evidence spans multiple domains ({', '.join(_detected_domains)}). "
                "Attribute each fact to its source domain. Do not conflate facts across domains.\n"
            )

    # When multi-resolution context provides doc/section headers, use it as preamble
    evidence_block = (
        f"{multi_res_section}{_domain_mix_note}{evidence_text}"
        if multi_res_section
        else f"{_domain_mix_note}{evidence_text}"
    )

    # TaskSpec enrichment: prepend structured task description from fine-tuned model
    taskspec_section = ""
    if task_spec is not None and hasattr(task_spec, "to_natural_text"):
        taskspec_section = f"TASK UNDERSTANDING: {task_spec.to_natural_text()}\n"
        # Inject explicit format instruction (e.g., "FORMAT: Present as markdown table...")
        if hasattr(task_spec, "get_format_instruction"):
            fmt_instr = task_spec.get_format_instruction()
            if fmt_instr:
                taskspec_section += f"{fmt_instr}\n"
        # Inject few-shot example for the requested output format
        if hasattr(task_spec, "get_few_shot_example"):
            few_shot = task_spec.get_few_shot_example()
            if few_shot:
                taskspec_section += f"\n{few_shot}\n"
        taskspec_section += "\n"

    # Build conversation history section for multi-turn follow-up queries
    conversation_section = ""
    if conversation_context and conversation_context.strip():
        # Limit conversation context length to prevent it from overwhelming evidence
        _conv_text = conversation_context.strip()
        _MAX_CONV_CHARS = 800  # Cap at ~200 tokens to preserve evidence budget
        if len(_conv_text) > _MAX_CONV_CHARS:
            # Keep the most recent part of conversation
            _conv_text = "..." + _conv_text[-_MAX_CONV_CHARS:]

        # Relevance check: skip conversation context if it has no word overlap
        # with the current query (avoids polluting prompt with unrelated history)
        _conv_words = set(_conv_text.lower().split())
        _q_words = set(query.lower().split()) - {"the", "a", "an", "is", "are", "what", "how", "who", "which", "do", "does", "can", "will", "it", "they", "this", "that"}
        _conv_overlap = len(_conv_words & _q_words) / max(len(_q_words), 1)
        # Detect if query uses pronouns/references that need resolution
        _ref_words = {"it", "they", "them", "those", "these", "that", "this",
                      "he", "she", "his", "her", "its", "their", "theirs",
                      "same", "above", "previous", "prior", "earlier", "aforementioned",
                      "similar", "other", "another", "rest", "remaining",
                      "former", "latter", "said", "such", "both"}
        _query_words = set(query.lower().split())
        _has_references = bool(_query_words & _ref_words)

        # Skip context if no overlap AND no pronoun references (unrelated history)
        # But preserve context when query has no content words after stopword removal
        # (e.g., "Tell me more." → empty _q_words → likely a follow-up)
        if len(_q_words) > 0 and _conv_overlap < 0.05 and not _has_references:
            conversation_context = None  # Will skip the section below

        # Extract entity names from conversation context for pronoun mapping
        _entity_mentions = []
        if conversation_context:
            import re as _re_conv
            # Extract capitalized multi-word names (e.g., "John Smith", "Sarah Chen")
            _entity_mentions = list(set(_re_conv.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', conversation_context)))
            # Also get single proper nouns that aren't common words
            _common_caps = {"The", "This", "That", "What", "How", "When", "Where", "Who", "Which", "Can", "Does", "Did", "Are", "Were", "Has", "Have", "Not", "But", "And", "For", "With", "From", "About", "Into", "Over", "Also", "Just", "Very", "Well", "Based", "Source", "Document", "Page", "Section"}
            _single_proper = [m for m in _re_conv.findall(r'\b([A-Z][a-z]{2,})\b', conversation_context) if m not in _common_caps]
            if _single_proper and not _entity_mentions:
                _entity_mentions = list(set(_single_proper))[:5]

        resolution_hint = ""
        if _has_references and _entity_mentions:
            _entity_list = ", ".join(_entity_mentions[:5])
            resolution_hint = (
                f"RESOLVE REFERENCES: Entities from conversation history: {_entity_list}\n"
                f"Map 'they/them/their' → these people/entities | 'it/its' → previous topic/metric | 'this/that' → subject above\n\n"
            )
        elif _has_references:
            resolution_hint = (
                "IMPORTANT: The current query uses pronouns or references. "
                "Resolve them using the conversation history — identify which "
                "entities, values, or topics 'they/it/those/this' refers to.\n\n"
            )

        if conversation_context:
            conversation_section = (
                f"CONVERSATION CONTEXT:\n"
                f"{conversation_context.strip()}\n\n"
                f"{resolution_hint}"
            )

    # Build prompt: SYSTEM → QUESTION → EVIDENCE → TASK → INSTRUCTIONS
    # Evidence-first structure: the LLM reads the question and evidence together,
    # then processes formatting/task instructions while the evidence is fresh.
    # This reduces hallucinations because the model has evidence in working memory
    # when processing output instructions.

    parts = [
        _GENERATION_SYSTEM,
    ]

    # Conversation context comes first (for pronoun resolution)
    if conversation_section:
        parts.append(conversation_section)

    # Question + Evidence block — keep these adjacent for better grounding
    parts.append(f"QUESTION: {query}\n")
    parts.append(f"DOCUMENT EVIDENCE:\n{evidence_block}")
    if _evidence_truncated:
        parts.append("(Note: Evidence truncated. Focus on what is provided.)")

    # Evidence quality signal removed — _build_grouped_evidence already appends
    # an EVIDENCE NOTE with score-based quality assessment, making the separate
    # EVIDENCE STATUS block redundant and potentially contradictory.

    # Grounding reminder after evidence
    parts.append("Answer using ONLY the evidence above. Do not add information not present in the documents.\n")

    # Completeness note for multi-doc queries
    if num_documents > 1:
        parts.append(f"COMPLETENESS: Analyzing {num_documents} documents. Cover ALL of them.\n")

    # Query decomposition for multi-part questions — list sub-questions explicitly
    # so the LLM addresses each part rather than focusing on just one
    if _is_multi_part_query(query):
        _sub_questions = _decompose_query(query)
        if _sub_questions and len(_sub_questions) >= 2:
            _sq_lines = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(_sub_questions))
            parts.append(f"SUB-QUESTIONS TO ADDRESS:\n{_sq_lines}\nAddress each sub-question separately.\n")

    # Entity coverage reminder — explicitly list query entities so LLM addresses each
    _query_entity_reminder = _build_entity_reminder(query, evidence_text)
    if _query_entity_reminder:
        parts.append(_query_entity_reminder)

    # Evidence field gap detection — alert LLM to missing fields early
    _field_gap_note = _detect_evidence_field_gaps(query, evidence_text)
    if _field_gap_note:
        parts.append(_field_gap_note)

    # Task template and format hint
    parts.append(f"TASK:\n{template}")
    if _budget_hint:
        parts.append(_budget_hint)

    # Task-specific instructions (from TaskSpec)
    if taskspec_section:
        parts.append(taskspec_section)

    # ML context intelligence (grounding constraints, entity salience, evidence quality)
    # Placed early so LLM processes grounding rules before format instructions
    if context_section:
        parts.append(context_section)

    # Domain context
    if tool_section:
        parts.append(tool_section)
    elif domain_section:
        parts.append(domain_section)

    # Domain-specific reasoning guidelines
    if domain_reasoning_section:
        parts.append(domain_reasoning_section)

    # Reasoning preamble for complex intents
    if reasoning_section:
        parts.append(reasoning_section)

    # Few-shot example (shows expected output format) — placed last
    # so the output format is the most recent instruction before evidence
    if few_shot_section:
        parts.append(few_shot_section)

    # Self-check: explicit verification step at end of prompt
    parts.append(
        "BEFORE RESPONDING — verify:\n"
        "1. Every name, number, and date appears in the evidence above\n"
        "2. No information is invented or assumed beyond what the evidence states\n"
        "3. Your response directly addresses the question asked"
    )

    return "\n".join(parts)

def _get_domain_knowledge_section(domain: str, intent: str) -> str:
    """Build a domain knowledge section for LLM prompt injection."""
    try:
        from src.intelligence.domain_knowledge import get_domain_knowledge_provider
        from src.api.config import Config
        if not getattr(Config, "DomainKnowledge", None):
            logger.debug("Domain knowledge disabled for domain=%s", domain)
            return ""
        if not Config.DomainKnowledge.ENABLED or not Config.DomainKnowledge.INJECT_INTO_PROMPTS:
            logger.debug("Domain knowledge disabled for domain=%s", domain)
            return ""
        provider = get_domain_knowledge_provider()
        brief = provider.get_brief_context(domain, intent=intent)
        if not brief:
            logger.debug("Domain knowledge provider returned empty for domain=%s intent=%s", domain, intent)
            return ""
        return f"\nDOMAIN KNOWLEDGE:\n{brief}\n"
    except Exception:
        logger.debug("Domain knowledge provider unavailable for domain=%s intent=%s", domain, intent, exc_info=True)
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

    # Self-check instruction for quality assurance
    _self_check = (
        "Before finalizing your answer, verify:\n"
        "1. Every number/date/name matches the evidence exactly\n"
        "2. No information was invented that isn't in the evidence\n"
        "3. The answer directly addresses the question asked"
    )

    return (
        f"{_SYSTEM_BASE}\n"
        f"TASK:\n{task_instruction}\n"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EVIDENCE:\n{evidence}\n\n"
        f"{_self_check}\n\n"
        "Provide your answer directly. Be thorough but concise."
    )

def _deduplicate_evidence_chunks(chunks: List[Any], threshold: float = 0.85) -> List[Any]:
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

        # Preserve chunks with different numeric values despite text overlap
        _new_nums = set(_DEDUP_NUM_RE.findall(text))

        is_dup = False
        for idx, existing_words in enumerate(kept_word_sets):
            if not existing_words:
                continue
            intersection = len(words & existing_words)
            union = len(words | existing_words)
            if union > 0 and intersection / union >= threshold:
                existing_text = (getattr(kept[idx], "text", "") or "").strip().lower()
                _existing_nums = set(_DEDUP_NUM_RE.findall(existing_text))
                if _new_nums and _new_nums != _existing_nums:
                    is_dup = False  # Different numeric facts — keep both
                else:
                    is_dup = True
                break

        if not is_dup:
            kept.append(chunk)
            kept_word_sets.append(words)

    return kept

def _enforce_chunk_diversity(chunks: list, max_chunks: int) -> list:
    """Select top chunks ensuring diversity across documents and section types.

    Prevents all top-K from same page/section when multiple docs available.
    """
    if len(chunks) <= max_chunks:
        return chunks

    selected = []
    doc_sections: dict = {}  # doc_name → set of section_kinds

    for chunk in chunks:
        meta = getattr(chunk, "meta", None) or {}
        doc_name = meta.get("source_name") or meta.get("document_name") or "unknown"
        section = meta.get("section_kind") or meta.get("chunk_type") or "body"

        if doc_name not in doc_sections:
            doc_sections[doc_name] = set()

        # Prefer chunks from new sections/documents
        is_new_section = section not in doc_sections[doc_name]
        if is_new_section or len(selected) < max_chunks * 0.7:
            selected.append(chunk)
            doc_sections[doc_name].add(section)

        if len(selected) >= max_chunks:
            break

    # Fill remaining slots if diversity pass didn't reach max
    if len(selected) < max_chunks:
        remaining = [c for c in chunks if c not in selected]
        selected.extend(remaining[:max_chunks - len(selected)])

    return selected

def _detect_evidence_contradictions(chunks: List[Any]) -> List[str]:
    """Detect contradicting claims across chunks from different documents.

    Scans for both numeric and text values associated with the same label/field
    across different documents and flags discrepancies.  Returns human-readable
    contradiction notes for injection into the LLM prompt.
    """
    import re as _re_cd
    # Numeric key-value pattern
    _KV_PATTERN = _re_cd.compile(
        r"(?:^|\n)\s*([A-Za-z][A-Za-z\s]{2,30})\s*[:]\s*([\$€£]?\s*[\d,]+\.?\d*\s*%?)",
    )
    # Text key-value pattern (for fields like "Title: Senior Engineer")
    _TEXT_KV_PATTERN = _re_cd.compile(
        r"(?:^|\n)\s*([A-Za-z][A-Za-z\s]{2,20})\s*[:]\s*([A-Z][A-Za-z\s,.-]{3,60})(?:\n|$)",
    )

    # Collect field→{doc_name: value} mappings
    field_values: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        meta = getattr(chunk, "meta", None) or {}
        source = getattr(chunk, "source", None)
        doc_name = (
            meta.get("source_name")
            or meta.get("document_name")
            or (getattr(source, "document_name", "") if source else "")
            or "Document"
        )
        text = (getattr(chunk, "text", "") or "").strip()
        for m in _KV_PATTERN.finditer(text):
            field = m.group(1).strip().lower()
            value = m.group(2).strip()
            if field and value:
                field_values.setdefault(field, {})[doc_name] = value
        # Also scan for text key-value pairs (e.g., "Title: Senior Engineer")
        for m in _TEXT_KV_PATTERN.finditer(text):
            field = m.group(1).strip().lower()
            value = m.group(2).strip()
            if field and value and field not in field_values:
                field_values.setdefault(field, {})[doc_name] = value

    # Find fields where different documents give different values
    contradictions: List[str] = []
    for field, doc_vals in field_values.items():
        unique_vals = set(doc_vals.values())
        if len(unique_vals) > 1 and len(doc_vals) >= 2:
            parts = [f"{doc}: {val}" for doc, val in doc_vals.items()]
            contradictions.append(
                f"CONFLICT in '{field}': {' vs '.join(parts)}"
            )
    return contradictions[:5]  # Cap to avoid prompt bloat

def _clean_evidence_text(text: str) -> str:
    """Clean OCR artifacts and normalize whitespace in evidence text.

    Removes form-feeds, repeated dots, excessive whitespace, and normalizes
    line breaks so the LLM sees clean, readable evidence.
    """
    import re as _re_clean

    if not text:
        return text
    # Remove OCR artifacts: form-feeds, line/paragraph separators,
    # repeated dot leaders, horizontal rules
    cleaned = _re_clean.sub(
        r"(?:\x0c|[\u2028\u2029]|\s*\.\s*\.\s*\.\s*\.[\s.]*)",
        " ", text,
    )
    # Remove horizontal rule lines
    cleaned = _re_clean.sub(r"^\s*[-_=]{5,}\s*$", "", cleaned, flags=_re_clean.MULTILINE)
    # Remove embedded metadata id: slugs (leak from chunk payloads)
    # Pattern: "id: <slug>" where slug is a run-on word like "workflowsandreducing..."
    cleaned = _re_clean.sub(r"\bid:\s*\S+", "", cleaned)
    # Remove other metadata key leaks from chunk payloads
    cleaned = _re_clean.sub(
        r"\b(?:chunk_type|chunk_kind|section_id|section_kind|embed_pipeline_version"
        r"|canonical_json|embedding_text|canonical_text|doc_domain|doc_type"
        r"|document_type|subscription_id|profile_id|document_id"
        r"|layout_confidence|ocr_confidence)\s*[:=]\s*\S+",
        "", cleaned, flags=_re_clean.IGNORECASE,
    )
    # Collapse excessive spaces (but preserve single spaces and newlines)
    cleaned = _re_clean.sub(r"[ \t]{3,}", "  ", cleaned)
    # Collapse 3+ newlines to 2
    cleaned = _re_clean.sub(r"\n{3,}", "\n\n", cleaned)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in cleaned.split("\n")]
    # Remove empty lines at start/end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    cleaned = "\n".join(lines)
    return cleaned.strip()

def _build_grouped_evidence(chunks: List[Any], max_context_chars: int = 0, domain: str = "", query: str = "") -> str:
    """Group chunks by document and format with headers.

    When *query* is provided, extracts named entities and boosts chunks
    that mention those entities higher in the evidence ordering within
    each document group. This ensures entity-relevant evidence appears
    first when context budget is tight.
    """
    if max_context_chars <= 0:
        max_context_chars = LLM_MAX_CONTEXT_CHARS

    # Extract query entities for evidence boosting
    _query_entities_lower: list[str] = []
    if query:
        import re as _re_qe
        _qe_pat = _re_qe.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        _qe_generic = frozenset({
            "The", "This", "That", "What", "How", "Which", "When", "Where", "Who",
            "Compare", "Rank", "List", "Show", "Extract", "Find", "Tell", "Give",
            "Summary", "Details", "Information", "Document", "Table", "Please",
        })
        for m in _qe_pat.finditer(query):
            ent = m.group()
            first_w = ent.split()[0]
            if first_w in _qe_generic:
                remaining = ent[len(first_w):].strip()
                if remaining and remaining.split()[0] not in _qe_generic:
                    ent = remaining
                else:
                    continue
            if ent not in _qe_generic and len(ent) > 2:
                _query_entities_lower.append(ent.lower())

    # Detect if reranker flagged all chunks as low-confidence
    _all_low_confidence = all(
        (getattr(c, "meta", None) or {}).get("rerank_low_confidence", False)
        for c in chunks
    ) if chunks else False

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

        # Clean OCR/extraction artifacts before presenting as evidence
        text = _clean_evidence_text(text)
        if not text:
            continue

        # Truncate very long chunks at sentence boundaries
        # Legal/policy/medical chunks are often longer coherent blocks — give more room
        # Standard chunks (250-450 tokens) need ~1200 chars to avoid mid-sentence cuts
        _chunk_limit = 2000 if domain in ("legal", "policy", "medical") else 1200
        if len(text) > _chunk_limit:
            _trunc = text[:_chunk_limit]
            _last_period = max(_trunc.rfind(". "), _trunc.rfind(".\n"), _trunc.rfind("? "), _trunc.rfind("! "))
            snippet = text[:_last_period + 1] if _last_period > _chunk_limit // 2 else _trunc
        else:
            snippet = text

        _raw_score = getattr(chunk, "score", 0.0)
        _score = float(_raw_score) if isinstance(_raw_score, (int, float)) else 0.0
        # Extract page info for source attribution
        page_start = meta.get("provenance.page_start") or meta.get("page_start") or ""
        page_end = meta.get("provenance.page_end") or meta.get("page_end") or ""
        page_info = ""
        if page_start:
            page_info = f"p.{page_start}" if not page_end or page_end == page_start else f"pp.{page_start}-{page_end}"

        entry = {"id": chunk_id, "section": section_kind, "text": snippet, "score": _score, "page": page_info}
        doc_groups.setdefault(doc_name, []).append(entry)

    parts: List[str] = []
    total_chars = 0
    multi_doc = len(doc_groups) > 1
    num_docs = len(doc_groups)

    # For multi-doc: distribute budget proportionally based on each document's
    # total chunk relevance scores, so high-relevance documents get more space.
    # A minimum floor of 15% per document prevents any doc from being starved.
    if multi_doc:
        doc_score_map: dict[str, float] = {}
        for dname, dentries in doc_groups.items():
            doc_score_map[dname] = sum(e.get("score", 0.0) for e in dentries)
        grand_total_score = sum(doc_score_map.values()) or 1.0
        min_budget_floor = int(max_context_chars * 0.15)  # 15% floor per doc
        per_doc_budgets: dict[str, int] = {}
        for dname in doc_groups:
            proportional = int(max_context_chars * (doc_score_map[dname] / grand_total_score))
            per_doc_budgets[dname] = max(proportional, min_budget_floor)
    else:
        per_doc_budgets = {}

    # Build document preview for multi-doc queries — helps LLM orient before evidence
    if multi_doc:
        preview_lines = []
        for dname, dentries in doc_groups.items():
            sections = set(e["section"] for e in dentries if e.get("section"))
            sec_str = f" ({', '.join(sorted(sections)[:4])})" if sections else ""
            preview_lines.append(f"  - {dname}: {len(dentries)} segments{sec_str}")
        parts.append("AVAILABLE DOCUMENTS:\n" + "\n".join(preview_lines))
        total_chars += sum(len(l) for l in preview_lines) + 25

    # Absolute relevance thresholds for quality tagging —
    # percentile-based scoring was misleading when score distributions were skewed
    _score_p75 = 0.7
    _score_p50 = 0.4

    for doc_name, entries in doc_groups.items():
        doc_chars = 0
        per_doc_budget = per_doc_budgets.get(doc_name, max_context_chars)
        if multi_doc:
            header = f"\n=== Document: {doc_name} ==="
            parts.append(header)
            total_chars += len(header)
            doc_chars += len(header)

        # Sort entries within each document by entity relevance first,
        # then page order for logical flow, with score as tiebreaker.
        # Entity-relevant chunks surface first when context is tight.
        def _sort_key(e):
            page = e.get("page", "")
            try:
                page_num = int(page.lstrip("p.").split("-")[0]) if page else 9999
            except (ValueError, IndexError):
                page_num = 9999
            # Entity boost: chunks mentioning query entities sort first (0)
            _entity_rank = 1
            if _query_entities_lower:
                _text_lower = e.get("text", "").lower()
                if any(qe in _text_lower for qe in _query_entities_lower):
                    _entity_rank = 0
            return (_entity_rank, page_num, -e.get("score", 0.0))
        entries.sort(key=_sort_key)

        prev_section = None
        for idx, entry in enumerate(entries):
            section_label = ""
            section = entry.get("section", "")
            if section:
                # Show section label only when it changes — reduces visual noise
                if section != prev_section:
                    section_label = f"[{section}] "
                    prev_section = section

            page_label = f" ({entry['page']})" if entry.get("page") else ""
            # Annotate evidence quality for LLM awareness
            # Use relative scoring: top-25% of scores are HIGH, next 25% MODERATE
            _score = entry.get("score", 0.0)
            quality_tag = ""
            if _score >= _score_p75:
                quality_tag = "[HIGH RELEVANCE] "
            elif _score >= _score_p50:
                quality_tag = "[MODERATE RELEVANCE] "
            block = f"{quality_tag}{section_label}{entry['text']}{page_label}"
            if total_chars + len(block) > max_context_chars:
                break
            # In multi-doc mode, respect per-doc budget (with 1.2x flexibility)
            if multi_doc and doc_chars + len(block) > per_doc_budget * 1.2:
                break
            parts.append(block)
            total_chars += len(block)
            doc_chars += len(block)

        if total_chars > max_context_chars:
            break

    # Append contradiction notes for multi-document evidence
    if multi_doc:
        contradictions = _detect_evidence_contradictions(chunks)
        if contradictions:
            conflict_block = "\n⚠ DETECTED CONTRADICTIONS IN EVIDENCE:\n" + "\n".join(f"  - {c}" for c in contradictions)
            parts.append(conflict_block)

    # Evidence quality summary — helps LLM calibrate confidence
    _all_scores = []
    for c in chunks:
        _s = getattr(c, "score", None)
        if isinstance(_s, (int, float)):
            _all_scores.append(float(_s))
    if _all_scores:
        _high = sum(1 for s in _all_scores if s >= 0.8)
        _mod = sum(1 for s in _all_scores if 0.5 <= s < 0.8)
        _low = sum(1 for s in _all_scores if s < 0.5)
        if _all_low_confidence or (_high == 0 and _low > _mod):
            parts.append("EVIDENCE NOTE: Evidence relevance is low. Answer cautiously and note uncertainty. "
                         "If the evidence does not contain the answer, say so rather than guessing.")
        elif _high >= len(_all_scores) * 0.6:
            parts.append("EVIDENCE NOTE: Strong evidence available. Provide a confident, detailed answer.")

    return "\n\n".join(parts)

# ── LLM call with timeout ────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate based on character length."""
    return max(1, len(text) // _CHARS_PER_TOKEN)

def _build_simplified_prompt(query: str, evidence: str, chunks: List[Any] = None) -> str:
    """Shorter prompt for intermediate-timeout fallback.

    Preserves core formatting instructions to maintain response quality.
    Uses only high-relevance chunks when available for better answer quality.
    Reduces evidence to top 4 chunks to minimize token count for faster generation.
    """
    # When chunks available, rebuild evidence from only high-relevance ones
    if chunks:
        # Sort by score descending and take top 4 — fewer tokens = faster generation
        scored = sorted(chunks, key=lambda c: getattr(c, "score", 0.0) or 0.0, reverse=True)
        top_chunks = scored[:4]
        fallback_evidence = _build_grouped_evidence(
            top_chunks, max_context_chars=LLM_MAX_CONTEXT_CHARS // 3
        )
    else:
        fallback_evidence = evidence[:LLM_MAX_CONTEXT_CHARS // 3]

    # Detect query type for format guidance
    _ql = (query or "").lower()
    _format_hint = ""
    if any(w in _ql for w in ("compare", "vs", "versus", "difference")):
        _format_hint = "Present as a markdown comparison table with one row per entity."
    elif any(w in _ql for w in ("rank", "best", "top ", "strongest")):
        _format_hint = "Present as a ranked list with scores or key strengths."
    elif any(w in _ql for w in ("list", "all ", "extract", "enumerate")):
        _format_hint = "Present as a bulleted list, one item per line."
    elif any(w in _ql for w in ("table", "tabular")):
        _format_hint = "Present as a markdown table."

    _format_line = f"{_format_hint}\n" if _format_hint else ""
    return (
        "You are a document intelligence assistant. Answer ONLY from the evidence.\n\n"
        "RULES:\n"
        "- Every claim must come from the evidence. Never invent.\n"
        "- **Bold** key values. Use bullet points for lists, tables for comparisons.\n"
        "- Start with the direct answer — no preamble.\n"
        "- Use exact numbers, dates, and names from the evidence.\n"
        "- If not found, say: 'Not found in the provided documents.'\n\n"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EVIDENCE:\n{fallback_evidence}\n\n"
        f"{_format_line}"
        "Answer concisely and accurately:"
    )

def _generate(
    llm_client: Any,
    prompt: str,
    correlation_id: Optional[str],
    role: Optional[str] = None,
    *,
    fallback_prompt: Optional[str] = None,
    use_thinking: bool = False,
    task_spec: Optional[Any] = None,
    intent: Optional[str] = None,
    _is_retry: bool = False,
) -> Optional[str]:
    # ── Build options with adaptive num_ctx ─────────────────────────
    # Complex intents (cross_document, comparison, etc.) and large chunk sets
    # benefit from 16K context; simple factual queries stay at 8K to reduce
    # VRAM pressure and avoid unnecessary Ollama model reloads.
    _num_predict = LLM_MAX_OUTPUT_TOKENS

    _prompt_chunk_count = len(prompt.split("---")) if prompt else 0  # rough chunk count proxy
    _NUM_CTX = _get_num_ctx(intent or "", _prompt_chunk_count)
    # Qwen3 thinking consumes ~1K tokens even with think=False.
    # Ensure num_predict always has room for thinking + actual content.
    _MIN_NUM_PREDICT = 1536

    if task_spec is not None:
        if hasattr(task_spec, "get_intent_output_tokens"):
            _num_predict = task_spec.get_intent_output_tokens()
        elif hasattr(task_spec, "get_num_predict"):
            _num_predict = task_spec.get_num_predict()

    _num_predict = max(_num_predict, _MIN_NUM_PREDICT)
    if _is_retry:
        _num_predict = min(_num_predict, 1024)  # Cap tokens on retry for faster response
    _prompt_tokens = _estimate_tokens(prompt) if prompt else 2048

    # Warn if prompt tokens + output tokens approach context window
    if _prompt_tokens + _num_predict > _NUM_CTX * 0.9:
        logger.warning(
            "Prompt may exceed context window: ~%d prompt tokens + %d output tokens > %d ctx | cid=%s",
            _prompt_tokens, _num_predict, _NUM_CTX, correlation_id,
        )

    # Intent-based temperature: extraction/factual needs precision,
    # generation/creative needs diversity
    _INTENT_TEMPERATURE = {
        "factual": 0.1, "contact": 0.05, "multi_field": 0.1,
        "comparison": 0.2, "ranking": 0.2, "reasoning": 0.2,
        "summary": 0.25, "timeline": 0.15, "cross_document": 0.2,
        "analytics": 0.15, "generate": 0.5,
    }
    _temperature = _INTENT_TEMPERATURE.get(intent or "", 0.2)

    options = {
        "num_predict": _num_predict,
        "num_ctx": _NUM_CTX,
        "temperature": _temperature,
    }

    # Thinking mode: expand prediction limits and enable reasoning
    # Keep num_ctx fixed at 8192 to avoid Ollama model reloads
    if use_thinking:
        options["think"] = True
        options["num_predict"] = max(options.get("num_predict", 512), 2048)
        logger.info(
            "Thinking mode enabled: num_ctx=%d num_predict=%d",
            options["num_ctx"], options["num_predict"],
            extra={"stage": "llm_extract_thinking", "correlation_id": correlation_id},
        )

    logger.info(
        "LLM generate: num_ctx=%d (fixed) prompt_est=%d num_predict=%d complexity=%s | cid=%s",
        options["num_ctx"], _prompt_tokens, options["num_predict"],
        getattr(task_spec, "complexity", "unknown") if task_spec else "no_taskspec",
        correlation_id,
    )

    # ── Split prompt into system/user for chat-based generation ────
    # The prompt structure is: SYSTEM + QUESTION + TASK + EVIDENCE + INSTRUCTIONS
    # Split at "DOCUMENT EVIDENCE:" — system gets instructions + question,
    # user gets evidence + question reminder.
    system_msg = _GENERATION_SYSTEM
    user_msg = prompt

    _split_marker = "DOCUMENT EVIDENCE:"
    if _split_marker in prompt:
        _parts = prompt.split(_split_marker, 1)
        system_msg = _parts[0].strip()
        # User message includes evidence + reminder of the question
        user_msg = f"DOCUMENT EVIDENCE:{_parts[1]}"

    # ── Resolve the actual chat-capable client ──────────────────────
    # MultiAgentGateway wraps role-specific OllamaClients. Unwrap to get the
    # actual client so we can use chat_with_metadata (system/user separation)
    # which produces far better results than raw generate().
    _actual_client = llm_client
    if role:
        try:
            from src.llm.multi_agent import MultiAgentGateway
            if isinstance(llm_client, MultiAgentGateway):
                _actual_client = llm_client._get_client(role)
        except Exception as exc:
            logger.debug("Failed to unwrap MultiAgentGateway client for role %s", role, exc_info=True)

    _use_chat = hasattr(_actual_client, "chat_with_metadata")

    def _call_chat() -> str:
        """Use chat API with system/user role separation (preferred)."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text, _meta = _actual_client.chat_with_metadata(
            messages, options=options, max_retries=1, backoff=0.4,
            thinking=use_thinking,
        )
        return text or ""

    def _call_generate(p: str) -> str:
        """Fallback to generate API when chat is not available."""
        if hasattr(_actual_client, "generate_with_metadata"):
            text, _meta = _actual_client.generate_with_metadata(
                p, options=options, max_retries=1, backoff=0.4,
                thinking=use_thinking,
            )
            return text or ""
        return _actual_client.generate(p, max_retries=1, backoff=0.4) or ""

    logger.info(
        "LLM dispatch: path=%s role=%s think=%s client=%s | cid=%s",
        "chat" if _use_chat else "generate",
        role, use_thinking, type(_actual_client).__name__,
        correlation_id,
    )

    def _dispatch() -> str:
        if _use_chat:
            try:
                return _call_chat()
            except (TypeError, ValueError):
                # Fallback to generate if chat_with_metadata returns unexpected type
                # (e.g. MagicMock in tests that only mocks generate_with_metadata)
                logger.debug("chat_with_metadata failed, falling back to generate | cid=%s", correlation_id)
                return _call_generate(prompt)
        return _call_generate(prompt)

    # Content generation gets a higher timeout — it produces more tokens
    _timeout = LLM_GENERATE_TIMEOUT_S if intent == "generate" else LLM_EXTRACT_TIMEOUT_S
    if _is_retry:
        _timeout = _timeout * 0.6  # Shorter timeout on retry to avoid doubling wait

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_dispatch)
    try:
        result = future.result(timeout=_timeout)
        if not result or not result.strip():
            logger.warning(
                "LLM returned empty/whitespace response (len=%d) | cid=%s",
                len(result) if result else 0, correlation_id,
            )
            return None
        return result
    except (TimeoutError, concurrent.futures.TimeoutError):
        future.cancel()
        # Retry once with simplified prompt (shorter evidence, no thinking)
        if fallback_prompt and not _is_retry:
            logger.info(
                "LLM extract timed out after %.1fs, retrying with simplified prompt | cid=%s",
                _timeout, correlation_id,
            )
            return _generate(
                llm_client=llm_client,
                prompt=fallback_prompt,
                correlation_id=correlation_id,
                role=role,
                fallback_prompt=None,
                use_thinking=False,
                task_spec=task_spec,
                intent=intent,
                _is_retry=True,
            )
        logger.warning(
            "LLM extract timed out after %.1fs",
            _timeout,
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

def _is_repetitive_output(text: str) -> bool:
    """Detect degenerate repetitive output (model stuck in a loop)."""
    if len(text) < 200:
        return False
    # Check if any 50-char substring repeats 3+ times
    for i in range(0, min(len(text) - 50, 300), 25):
        segment = text[i:i + 50]
        if text.count(segment) >= 3:
            return True
    return False

def _parse_response(raw: str, chunks: List[Any], query: str = "") -> Optional[LLMResponseSchema]:
    # Primary path: treat LLM output as direct markdown text.
    # The prompt asks for direct answers, not JSON.
    # Only fall back to JSON extraction if the response starts with '{'.
    cleaned = _clean_raw_response(raw)
    cleaned = _repair_unclosed_markdown(cleaned)
    cleaned = _deduplicate_bold_values(cleaned)
    chunks_used: List[str] = []

    if cleaned and len(cleaned) >= 8 and not _looks_like_metadata(cleaned):
        # Reject degenerate repetitive output
        if _is_repetitive_output(cleaned):
            logger.warning("Rejected repetitive LLM output (len=%d)", len(cleaned))
            return None
        return LLMResponseSchema(text=cleaned, evidence_chunks=chunks_used)

    # Fallback: try JSON extraction for responses that are wrapped in JSON
    payload = _extract_json(raw)
    if isinstance(payload, dict):
        answer = str(payload.get("answer") or payload.get("response") or "").strip()
        raw_chunks = payload.get("chunks_used") or payload.get("sources") or []
        if isinstance(raw_chunks, list):
            chunks_used = [str(c) for c in raw_chunks]
        if answer and len(answer) >= 8:
            return LLMResponseSchema(text=answer, evidence_chunks=chunks_used)

    return None

def _deduplicate_bold_values(text: str) -> str:
    """Remove duplicate bold values in consecutive sentences.

    LLMs sometimes bold the same value twice: "**John Smith** has 8 years.
    **John Smith** specializes in..." → keep only the first bold instance.
    """
    if not text or "**" not in text:
        return text

    import re as _re_db
    # Find all bold spans
    _bold_re = _re_db.compile(r"\*\*([^*]{2,60})\*\*")
    seen_in_paragraph: dict[str, int] = {}  # bold_text → first position

    paragraphs = text.split("\n\n")
    fixed_paragraphs = []

    for para in paragraphs:
        seen_in_paragraph.clear()

        def _dedup_bold(m: _re_db.Match) -> str:
            bold_text = m.group(1).strip()
            key = bold_text.lower()
            if key in seen_in_paragraph:
                # Already bolded in this paragraph — return plain text
                return bold_text
            seen_in_paragraph[key] = m.start()
            return m.group(0)  # Keep first occurrence bolded

        fixed_paragraphs.append(_bold_re.sub(_dedup_bold, para))

    return "\n\n".join(fixed_paragraphs)

def _detect_evidence_field_gaps(query: str, evidence_text: str) -> str:
    """Detect specific fields requested in query that are missing from evidence.

    Returns a gap note for prompt injection, or empty string if no gaps.
    """
    if not query or not evidence_text:
        return ""

    import re as _re_fg
    # Common field requests: "salary", "experience", "skills", "education", etc.
    _FIELD_PATTERNS = {
        "salary": _re_fg.compile(r"\b(?:salary|compensation|pay|wage|income|earning)\b", _re_fg.I),
        "experience": _re_fg.compile(r"\b(?:experience|years?\s+of|work\s+history|tenure)\b", _re_fg.I),
        "skills": _re_fg.compile(r"\b(?:skills?|competenc|proficienc|expertise)\b", _re_fg.I),
        "education": _re_fg.compile(r"\b(?:education|degree|university|diploma|qualification)\b", _re_fg.I),
        "contact": _re_fg.compile(r"\b(?:email|phone|address|contact|linkedin)\b", _re_fg.I),
        "certifications": _re_fg.compile(r"\b(?:certif\w*|licens\w*|accredit\w*)\b", _re_fg.I),
        "dates": _re_fg.compile(r"\b(?:date|deadline|when|expir|effective)\b", _re_fg.I),
        "amounts": _re_fg.compile(r"\b(?:amount|total|cost|price|fee|charge)\b", _re_fg.I),
    }

    # Find which fields the query asks about
    query_lower = query.lower()
    requested_fields = []
    for field_name, pattern in _FIELD_PATTERNS.items():
        if pattern.search(query_lower):
            requested_fields.append(field_name)

    if not requested_fields:
        return ""

    # Check which requested fields are absent from evidence
    ev_lower = evidence_text.lower()
    missing_fields = []
    for field_name in requested_fields:
        pattern = _FIELD_PATTERNS[field_name]
        if not pattern.search(ev_lower):
            missing_fields.append(field_name)

    if not missing_fields:
        return ""

    return (
        f"EVIDENCE GAPS: The query asks about {', '.join(missing_fields)} "
        f"but these may not appear in the evidence. If not found, explicitly state "
        f"'{', '.join(missing_fields)} not found in the provided documents.'\n"
    )

def _repair_unclosed_markdown(text: str) -> str:
    """Close unclosed bold, italic, and code markers to prevent broken rendering."""
    if not text:
        return text

    # Count backticks (outside code blocks, just raw count)
    backtick_count = text.count("`")
    # Subtract paired triple-backticks (``` ... ```)
    triple_count = text.count("```")
    # After removing triples, remaining singles
    single_backticks = backtick_count - triple_count * 3
    if triple_count % 2 == 1:
        text += "\n```"
    elif single_backticks % 2 == 1:
        text += "`"

    # Count ** pairs for bold
    bold_count = text.count("**")
    if bold_count % 2 == 1:
        text += "**"

    # Count single * for italic (not part of **)
    # Replace ** temporarily to count lone *
    temp = text.replace("**", "")
    italic_count = temp.count("*")
    if italic_count % 2 == 1:
        text += "*"

    return text

def _clean_raw_response(raw: str) -> str:
    """Clean LLM response text: strip markdown fences, JSON artifacts, etc."""
    import re as _re_clean_resp

    text = raw.strip()

    # Strip <think>...</think> tags (Qwen3/DeepSeek thinking artifacts)
    text = _re_clean_resp.sub(r"<think>.*?</think>", "", text, flags=_re_clean_resp.DOTALL).strip()

    # Strip leaked THINK step lines (from our prompt template)
    # Matches lines like "THINK: 1) Identify..." that should be internal reasoning
    text = _re_clean_resp.sub(
        r"(?m)^THINK:\s*\d?\)?\s*.*$\n?", "", text
    ).strip()

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
    stripped = text.lstrip()
    for prefix in ("answer:", "Answer:", "ANSWER:", "response:", "Response:", "RESPONSE:"):
        if stripped.startswith(prefix):
            text = stripped[len(prefix):].lstrip()
            break

    # Strip trailing meta-notes that LLMs sometimes append
    # e.g., "Note: I analyzed 3 documents..." or "Disclaimer: This is based on..."
    _trailing_meta = _re_clean_resp.compile(
        r"\n\s*(?:Note|Disclaimer|Caveat|Important|Warning):\s*(?:I |This |The above |My )"
        r"(?:analy[sz]|review|examin|response|answer|information).*$",
        _re_clean_resp.IGNORECASE | _re_clean_resp.DOTALL,
    )
    text = _trailing_meta.sub("", text).rstrip()

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
        except Exception as exc:
            logger.debug("Failed to parse LLM response as JSON object", exc_info=True)
    if "{" in text and "}" in text:
        snippet = text[text.find("{"):text.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception as exc:
            logger.debug("Failed to parse extracted JSON snippet from LLM response", exc_info=True)
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
