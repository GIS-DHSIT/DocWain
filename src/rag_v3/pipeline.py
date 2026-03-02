from __future__ import annotations

import json
import logging
import re
import time
import uuid
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.api import rag_state
from src.api.vector_store import build_collection_name, build_qdrant_filter
from src.intent.llm_intent import IntentParse, parse_intent
from src.metrics.quality_metrics import QueryMetrics, record_query_metrics

from .domain_router import DomainRouter
from .extract import extract_schema
from .judge import JudgeResult, judge
from .enterprise import render_enterprise as render
from .retrieve import (
    apply_quality_pipeline,
    deduplicate_by_content,
    expand_full_scan_by_document,
    expand_full_scan_by_profile,
    expand_full_scan_unscoped,
    filter_chunks_by_profile_scope,
    retrieve,
)
from .rerank import rerank
from .rewrite import rewrite_query
from .sanitize import FALLBACK_ANSWER, sanitize
from .types import (
    Chunk,
    ChunkSource,
    EvidenceSpan,
    FieldValue,
    FieldValuesField,
    GenericSchema,
    HRSchema,
    InvoiceSchema,
    LegalSchema,
    LLMBudget,
    LLMResponseSchema,
    MISSING_REASON,
    MedicalSchema,
    MultiEntitySchema,
    PolicySchema,
)

logger = logging.getLogger(__name__)


NO_CHUNKS_MESSAGE = "Not enough information in the documents to answer that."


def _no_results_message(query: str = "", scope: str = "") -> str:
    """Build a helpful no-results message that includes actionable suggestions."""
    parts = ["I couldn't find enough information in the documents to answer that."]
    if scope == "specific_document":
        parts.append("\nSuggestions:\n- The document may not contain this information\n- Try asking about the full profile instead")
    elif scope == "all_profile":
        parts.append("\nSuggestions:\n- Try asking about a specific document or person\n- Upload relevant documents if they haven't been added")
    else:
        parts.append("\nSuggestions:\n- Try rephrasing your question\n- Ask about a specific person or document by name\n- Use broader terms (e.g. 'skills' instead of a specific skill)")
    return "\n".join(parts)


# ── Smart Query Scope Inference ──────────────────────────────────────────────

@dataclass
class QueryScope:
    mode: str  # "all_profile", "specific_document", "targeted"
    document_id: Optional[str] = None
    entity_hint: Optional[str] = None


_DOCUMENT_ID_PATTERN = r"document_id\s*[:=]\s*([A-Za-z0-9_-]+)"

# Invoice/order number pattern (retained — this is a structured format, not entity name)
_INVOICE_NUMBER_RE = re.compile(r"(?:invoice|order|po)\s*#?\s*(\d+)", re.IGNORECASE)

# Intents that imply all_profile scope (used by ML-first path)
# For these intents, entity_hints are topic/skill filters, NOT document targets.
# "Who has the most Python experience?" → rank intent, entity_hint="Python" → all_profile
_SCOPE_ALL_INTENTS = frozenset({
    "compare", "rank", "list", "ranking", "comparison",
    "analytics", "summary",
})

# Words that strongly imply the user wants ALL documents, regardless of intent.
# Patterns allow optional articles ("all the resumes", "all candidates").
_ALL_SCOPE_SIGNALS = re.compile(
    r"\b(?:"
    r"all\s+(?:the\s+)?(?:candidates?|resumes?|documents?|invoices?|policies?|records?|profiles?)"
    r"|every\s+(?:candidate|resume|document|invoice|policy|record|profile)"
    r"|each\s+(?:candidate|resume|document|invoice|policy|record|profile)"
    r"|across\s+(?:all|every|the)\b"
    r"|who\s+(?:has|is)\s+the\s+(?:most|best|highest|lowest|least|top)"
    r"|which\s+candidate\s+(?:has|is)"
    r"|(?:multiple|several)\s+(?:candidates?|resumes?|documents?)"
    r"|(?:the|our|my)\s+(?:resumes?|candidates?|documents?|invoices?|policies?)"
    r"|(?:most\s+(?:common|important|frequent|shared))"
    r"|(?:shared\s+by\s+(?:multiple|all|several|the))"
    r"|(?:best\s+candidate)"
    r"|(?:comprehensive\s+analysis)"
    r"|(?:common\s+(?:skills?|patterns?|themes?))"
    r"|(?:list|find|show|get|rank|compare)\s+(?:all\s+)?(?:candidates?|people|applicants?)"
    r")\b",
    re.IGNORECASE,
)


# Multi-word phrases that are document types, NOT person/entity names.
# These slip through spaCy's single-word stopword filter but should never
# be treated as entity hints for chunk filtering.
_ENTITY_HINT_STOP_PHRASES = frozenset({
    "progress note", "progress notes", "pathology report", "radiology report",
    "lab results", "lab result", "vital signs", "medication list",
    "prescription list", "discharge summary", "clinical note", "clinical notes",
    "operative report", "consultation note", "patient report", "medical report",
    "medical record", "medical records", "insurance policy", "policy document",
    "invoice document", "invoice documents", "all documents", "the documents",
    "both documents", "sample invoice", "invoice number",
})


def _try_nlp_entity(query: str) -> Optional[str]:
    """Extract entity from query using spaCy NLP (helper for scope inference)."""
    try:
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query(query or "")
        if entity and len(entity) > 2:
            # Reject multi-word document-type phrases
            if entity.lower().strip() in _ENTITY_HINT_STOP_PHRASES:
                return None
            return entity
    except Exception:  # noqa: BLE001
        pass
    return None


def _infer_query_scope(
    query: str,
    explicit_document_id: Optional[str],
    intent_parse: Optional[IntentParse],
) -> QueryScope:
    """Detect whether the user wants ALL documents or a specific entity, and route accordingly.

    Uses ML-first approach: trained intent classifier signals are checked before
    any regex patterns. Regex is kept only for structured format extraction
    (document IDs, invoice numbers) — not for natural language interpretation.
    """
    # 1. Explicit document_id parameter (unchanged)
    if explicit_document_id:
        return QueryScope(mode="specific_document", document_id=str(explicit_document_id))

    lowered = (query or "").lower()

    # 2. ML-FIRST: intent_parse from trained MLP classifier
    if intent_parse:
        # For ranking/comparison/list intents, ALWAYS go all_profile.
        # Entity hints here are topic filters (e.g., "Python"), not document names.
        if intent_parse.intent in _SCOPE_ALL_INTENTS:
            return QueryScope(mode="all_profile")
        # "summarize" without entity → check for scope signals first,
        # then try spaCy (catches "summarize John's resume")
        if intent_parse.intent == "summarize" and not intent_parse.entity_hints:
            # If query has explicit all-profile signals, skip NLP entity extraction
            if _ALL_SCOPE_SIGNALS.search(query or ""):
                return QueryScope(mode="all_profile")
            entity = _try_nlp_entity(query)
            if entity:
                return QueryScope(mode="targeted", entity_hint=entity)
            return QueryScope(mode="all_profile")
        # Entity hints from classifier → targeted (for factual/extraction intents)
        if intent_parse.entity_hints:
            # BUT if query has explicit "all/every" scope signals, override to all_profile
            if _ALL_SCOPE_SIGNALS.search(query or ""):
                return QueryScope(mode="all_profile")
            return QueryScope(mode="targeted", entity_hint=intent_parse.entity_hints[0])

    # 2b. Explicit "all/every" scope signals in query text
    if _ALL_SCOPE_SIGNALS.search(query or ""):
        return QueryScope(mode="all_profile")

    # 3. Structured format extraction (not query analysis — these extract IDs from text)
    doc_match = re.search(_DOCUMENT_ID_PATTERN, query or "", flags=re.IGNORECASE)
    if doc_match:
        return QueryScope(mode="specific_document", document_id=doc_match.group(1).strip())

    inv_match = _INVOICE_NUMBER_RE.search(query or "")
    if inv_match:
        return QueryScope(mode="targeted", entity_hint=inv_match.group(1).strip())

    # 4. NLP-based entity extraction (spaCy dependency parsing)
    entity = _try_nlp_entity(query)
    if entity:
        logger.debug("NLP entity extraction result: query=%r entity=%r", query, entity)
        return QueryScope(mode="targeted", entity_hint=entity)

    # 5. Minimal safety net for when intent_parse is None (e.g. classifier unavailable)
    if re.search(r"\b(?:compare|rank|vs\.?|versus)\b", lowered):
        return QueryScope(mode="all_profile")

    # Default: targeted vector retrieval (standard behavior)
    return QueryScope(mode="targeted")


def _is_llm_response(extraction: Any) -> bool:
    """Check if extraction produced an LLMResponseSchema (skip render step)."""
    return isinstance(getattr(extraction, "schema", None), LLMResponseSchema)


def _has_valid_deterministic_extraction(schema: Any) -> bool:
    """Check if schema has valid deterministic data that doesn't need evidence_spans."""
    if isinstance(schema, HRSchema):
        return bool(
            schema.candidates
            and getattr(schema.candidates, "items", None)
            and any(c.name for c in schema.candidates.items)
        )
    if isinstance(schema, GenericSchema):
        if not (schema.facts and getattr(schema.facts, "items", None)):
            return False
        substantial = [
            f for f in schema.facts.items
            if (f.label or f.value) and len(str(f.value or "")) > 10
        ]
        # Require >= 2 facts with value > 10 chars, OR 1 fact with value > 50 chars
        return len(substantial) >= 2 or any(len(str(f.value or "")) > 50 for f in substantial)
    if isinstance(schema, (MedicalSchema, PolicySchema)):
        # Check if any field has items
        for field_name in type(schema).model_fields:
            field_val = getattr(schema, field_name, None)
            if field_val and getattr(field_val, "items", None):
                return True
        return False
    if isinstance(schema, InvoiceSchema):
        return bool(
            (schema.items and schema.items.items)
            or (schema.totals and schema.totals.items)
            or (schema.parties and schema.parties.items)
        )
    if isinstance(schema, LegalSchema):
        return bool(
            (schema.clauses and schema.clauses.items)
            or (schema.parties and schema.parties.items)
            or (schema.obligations and schema.obligations.items)
        )
    return False


# ── Agent-to-Domain mapping ────────────────────────────────────────────────
# When /ask receives tools=["resume-analysis"], the pipeline uses this to
# set an authoritative domain_hint that overrides query/chunk-based inference.

_AGENT_DOMAIN_MAP: Dict[str, str] = {
    "resume-analysis": "hr",
    "resume_analysis": "hr",
    "resumes": "hr",
    "resume": "hr",
    "medical": "medical",
    # lawhere intentionally omitted — it handles both legal and insurance/policy
    # docs; let the ML domain detector decide based on chunk content.
    "legal": "legal",
    "invoice": "invoice",
    "policy": "policy",
    "insurance": "policy",
    "cloud_platform": "cloud",
}
_TOOL_DOMAIN_MAP = _AGENT_DOMAIN_MAP  # backward-compat alias


def _resolve_domain_from_agents(tools: Optional[List[str]]) -> Optional[str]:
    """Map agent names to an authoritative domain hint.

    Returns the first matching domain or None if no agent maps to a domain.
    """
    if not tools:
        return None
    for tool in tools:
        domain = _AGENT_DOMAIN_MAP.get((tool or "").strip().lower())
        if domain:
            return domain
    return None


_resolve_domain_from_tools = _resolve_domain_from_agents  # backward-compat alias


# ── Document-agnostic agents ──────────────────────────────────────────────
# Agents that produce results from the query itself, not from document chunks.
# These are dispatched before the retrieval-empty guard so they still fire
# when Qdrant returns zero results.
_DOC_AGNOSTIC_AGENTS = frozenset({"web_search", "email_drafting", "creator", "tutor"})
_DOC_AGNOSTIC_TOOLS = _DOC_AGNOSTIC_AGENTS  # backward-compat alias


# ── Domain Agent Dispatch ─────────────────────────────────────────────────
# Detects queries that require specialized agent reasoning (e.g. "generate
# interview questions for this resume") and dispatches to the appropriate
# domain agent, which combines RAG-retrieved document context with LLM
# reasoning for higher-order analysis tasks.

def _try_domain_agent(
    query: str,
    domain: str,
    chunks: Optional[List["Chunk"]] = None,
    llm_client: Optional[Any] = None,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Detect and dispatch a domain agent task.

    Returns a complete answer dict if an agent handled the query, or None
    to fall through to the standard RAG pipeline.
    """
    try:
        from src.agentic.domain_agents import detect_agent_task, get_domain_agent
    except ImportError:
        return None

    detection = detect_agent_task(query, domain=domain)
    if detection is None:
        return None

    agent = get_domain_agent(detection["domain"], llm_client=llm_client)
    if agent is None:
        return None

    # Build context from retrieved chunks
    context: Dict[str, Any] = {"query": query}
    if chunks:
        chunk_texts = [c.text for c in chunks if hasattr(c, "text") and c.text]
        context["text"] = "\n\n".join(chunk_texts[:10])
        context["chunks"] = chunk_texts

    result = agent.execute(detection["task_type"], context)
    if not result.success or not result.output:
        logger.debug(
            "Domain agent %s/%s returned no output — falling through to RAG",
            detection["domain"], detection["task_type"],
        )
        return None

    logger.info(
        "Domain agent handled query: agent=%s task=%s | cid=%s",
        detection["domain"], detection["task_type"], request_id,
    )

    sources = []
    if chunks:
        sources = _collect_sources(chunks)

    return _build_answer(
        response_text=result.output,
        sources=sources,
        request_id=request_id,
        metadata={
            **(metadata or {}),
            "domain": detection["domain"],
            "agent_task": detection["task_type"],
            "agent_handled": True,
            "quality": "HIGH",
            "rag_v3": True,
        },
        query=query,
        chunks=chunks,
        llm_client=llm_client,
    )


def _time_remaining(pipeline_start: float, deadline_s: float) -> float:
    """Return seconds remaining before pipeline deadline.  Negative = overdue."""
    return deadline_s - (time.time() - pipeline_start)


# ── Chunk Translation ─────────────────────────────────────────────────────

def _detect_target_language(query: str) -> Optional[str]:
    """Detect target translation language from query using translator module patterns."""
    try:
        from src.tools.translator import (
            _TRANSLATE_PATTERN, _CONVERT_PATTERN, _TO_LANG_PATTERN,
            _LANG_CODE_MAP, _llm_detect_target_language,
        )
        for pattern in (_TRANSLATE_PATTERN, _CONVERT_PATTERN, _TO_LANG_PATTERN):
            m = pattern.search(query)
            if m:
                lang_name = m.group(1).lower().strip()
                return _LANG_CODE_MAP.get(lang_name, lang_name[:2] if len(lang_name) >= 2 else None)
        return _llm_detect_target_language(query)
    except Exception:
        return None


def _is_non_english(text: str) -> bool:
    """Quick heuristic to detect if text is likely non-English.

    Uses character analysis AND common non-English word detection to
    handle Latin-script languages (Spanish, French, German, etc.) that
    use mostly ASCII characters.
    """
    if not text or len(text) < 30:
        return False
    text_lower = text.lower()
    # Check for non-ASCII alphabetic characters (CJK, Arabic, accented, etc.)
    non_ascii = sum(1 for c in text if ord(c) > 127 and c.isalpha())
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count > 0 and non_ascii / alpha_count > 0.02:
        return True
    # Check for common non-English function words (Latin-script languages)
    _NON_EN_MARKERS = {
        "del", "los", "las", "una", "como", "para", "por", "pero", "desde",
        "también", "sobre", "este", "esta", "les", "des", "une", "dans",
        "pour", "avec", "sur", "mais", "sont", "cette", "nous", "vous",
        "leur", "chez", "und", "der", "die", "das", "ist", "ein", "eine",
        "für", "mit", "auf", "aus", "bei", "nach", "von", "bis", "oder",
        "wenn", "sich", "dos", "das", "uma", "com", "mas", "também", "pode",
        "tem", "onde", "het", "een", "van", "voor", "met", "niet", "zijn",
        "ook", "nog", "naar", "uit", "della", "degli", "delle", "sono",
        "anche", "questo", "questa", "tutti", "molto",
    }
    words = set(text_lower.split())
    marker_count = len(words & _NON_EN_MARKERS)
    return marker_count >= 3


def _translate_chunks(
    chunks: List[Chunk],
    target_lang: str,
    llm_client: Optional[Any],
    correlation_id: Optional[str] = None,
) -> List[Chunk]:
    """Translate non-English chunk text using a single LLM call.

    Only translates chunks detected as non-English via heuristic.
    Uses a single batched LLM call for efficiency.
    """
    if not chunks or not target_lang or llm_client is None:
        return chunks

    # Find non-English chunks
    non_english: list[tuple[int, str]] = []
    for i, chunk in enumerate(chunks[:10]):
        text = getattr(chunk, "text", "") or ""
        if text and len(text) >= 30 and _is_non_english(text):
            non_english.append((i, text[:800]))

    if not non_english:
        logger.info("No non-English chunks detected, skipping translation | cid=%s", correlation_id)
        return chunks

    lang_name = {
        "en": "English", "fr": "French", "es": "Spanish", "de": "German",
        "pt": "Portuguese", "nl": "Dutch", "it": "Italian",
    }.get(target_lang, target_lang)

    # Build compact batch prompt
    parts = [f"[{i+1}] {text}" for i, (_, text) in enumerate(non_english)]
    prompt = (
        f"Translate to {lang_name}. Keep [N] markers. Preserve numbers/formatting.\n\n"
        + "\n\n".join(parts)
    )

    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                llm_client.generate, prompt,
                {"temperature": 0.1, "num_predict": 2048, "num_ctx": 4096},
            )
            raw = future.result(timeout=20.0)

        if not raw:
            return chunks

        import re
        matches = re.findall(r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)', raw, re.DOTALL)

        translated_count = 0
        for num_str, translated_text in matches:
            try:
                idx = int(num_str) - 1
                if 0 <= idx < len(non_english):
                    orig_idx = non_english[idx][0]
                    cleaned = translated_text.strip()
                    if cleaned and len(cleaned) > 10:
                        chunks[orig_idx].text = cleaned
                        if chunks[orig_idx].meta is None:
                            chunks[orig_idx].meta = {}
                        chunks[orig_idx].meta["translated_to"] = target_lang
                        translated_count += 1
            except (ValueError, IndexError):
                continue

        logger.info(
            "Translated %d/%d non-English chunks to %s | cid=%s",
            translated_count, len(non_english), target_lang, correlation_id,
        )
    except concurrent.futures.TimeoutError:
        logger.warning("Batch translation timed out after 20s | cid=%s", correlation_id)
    except Exception as exc:
        logger.debug("Batch translation failed: %s", exc)

    return chunks


# ── Agent Dispatch ─────────────────────────────────────────────────────────

_TOOL_DISPATCH_TIMEOUT_S = 10.0
_MAX_TOOL_CONTEXT_CHUNKS = 10


def _dispatch_agents(
    tool_names: List[str],
    query: str,
    profile_id: str,
    subscription_id: str,
    tool_inputs: Optional[Dict[str, Any]],
    correlation_id: Optional[str],
    chunks: Optional[List[Chunk]] = None,
    llm_client: Optional[Any] = None,
    intent_hint: Optional[str] = None,
    domain_hint: Optional[str] = None,
) -> List[Chunk]:
    """Invoke registered agents and convert results to synthetic Chunk objects.

    Returns agent-result chunks (score=1.0) so they appear first in the
    extraction context.  Failures are logged and skipped gracefully.

    When *chunks* is provided, the reranked RAG context is serialized and
    included in the agent payload so agents can leverage document evidence.
    """
    if not tool_names:
        return []

    try:
        import asyncio
        from src.tools.base import registry
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent registry not available: %s", exc, extra={"correlation_id": correlation_id})
        return []

    # Serialize reranked chunks for tool consumption
    serialized_chunks: List[Dict[str, Any]] = []
    if chunks:
        for c in chunks[:_MAX_TOOL_CONTEXT_CHUNKS]:
            serialized_chunks.append({
                "id": c.id,
                "text": c.text,
                "score": c.score,
                "source": c.source.document_name if c.source else "",
                "meta": c.meta or {},
            })

    # Compose text from chunks for tools that expect a "text" field
    _tool_text = "\n\n".join(c.get("text", "") for c in serialized_chunks if c.get("text"))

    # Infer domain from chunk metadata for domain-aware tools (e.g. insights)
    _chunk_domains: Dict[str, int] = {}
    for c in serialized_chunks:
        _cd = str((c.get("meta") or {}).get("doc_domain", "")).lower().strip()
        if _cd:
            _chunk_domains[_cd] = _chunk_domains.get(_cd, 0) + 1
    _inferred_domain = max(_chunk_domains, key=_chunk_domains.get) if _chunk_domains else ""
    # Use caller-provided domain_hint when chunk metadata lacks domain info
    if not _inferred_domain and domain_hint:
        _inferred_domain = domain_hint

    result_chunks: List[Chunk] = []
    for tool_name in tool_names:
        payload = {
            "input": {"query": query, "chunks": serialized_chunks, "text": _tool_text, "domain": _inferred_domain},
            "context": {"profile_id": profile_id, "subscription_id": subscription_id},
            "options": {"requested_by": "rag_v3_pipeline"},
        }
        extra_input = (tool_inputs or {}).get(tool_name) if tool_inputs else None
        if isinstance(extra_input, dict):
            payload["input"].update(extra_input)
        elif extra_input is not None:
            payload["input"]["value"] = extra_input

        try:
            tool_resp = asyncio.run(
                asyncio.wait_for(
                    registry.invoke(tool_name, payload, correlation_id=correlation_id),
                    timeout=_TOOL_DISPATCH_TIMEOUT_S,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent %s dispatch failed: %s", tool_name, exc, extra={"correlation_id": correlation_id})
            continue

        if not isinstance(tool_resp, dict) or tool_resp.get("status") != "success":
            logger.warning("Agent %s returned status=%s", tool_name, (tool_resp or {}).get("status"))
            continue

        result = tool_resp.get("result") or {}

        # If the tool already produced a high-quality rendered response,
        # use it directly without expensive LLM enhancement.
        _pre_rendered = ""
        if isinstance(result, dict):
            _pre_rendered = result.get("rendered") or result.get("response") or ""
        if _pre_rendered and len(_pre_rendered) > 50:
            _tool_doc_name = ""
            if isinstance(result, dict):
                _tool_doc_name = result.get("document_name", "") or result.get("doc_name", "") or ""
            result_chunks.append(Chunk(
                id=f"tool_{tool_name}_{correlation_id or 'x'}",
                text=_pre_rendered[:2000],
                score=1.0,
                source=ChunkSource(document_name=_tool_doc_name),
                meta={"source": "tool_rendered", "tool_name": tool_name},
            ))
            logger.info("Agent %s used pre-rendered output (%d chars)", tool_name, len(_pre_rendered), extra={"correlation_id": correlation_id})
            continue

        # ── Tool Intelligence Enhancement ──────────────────────────────
        try:
            from src.tools.intelligence import enhance_tool_result
            enhanced = enhance_tool_result(
                tool_name=tool_name,
                raw_result=result if isinstance(result, dict) else {"value": result},
                query=query,
                chunks=chunks,
                llm_client=llm_client,
                intent_hint=intent_hint,
                correlation_id=correlation_id,
            )
            if enhanced and enhanced.get("enhanced_response"):
                enh_snippet = enhanced["enhanced_response"][:2000]
                _enh_doc_name = ""
                if isinstance(result, dict):
                    _enh_doc_name = result.get("document_name", "") or result.get("doc_name", "") or ""
                result_chunks.append(Chunk(
                    id=f"tool_{tool_name}_{correlation_id or 'x'}",
                    text=enh_snippet,
                    score=1.0,
                    source=ChunkSource(document_name=_enh_doc_name),
                    meta={"source": "tool_enhanced", "tool_name": tool_name, "domain": enhanced.get("domain")},
                ))
                logger.info("Agent %s enhanced with intelligence (%d chars)", tool_name, len(enh_snippet), extra={"correlation_id": correlation_id})
                continue  # Skip raw serialization
        except Exception as exc:
            logger.debug("Tool intelligence enhancement skipped: %s", exc)

        # Fallback: prefer rendered/response text from the result, else serialize
        if isinstance(result, dict):
            snippet = result.get("rendered") or result.get("response") or result.get("text") or ""
            if not snippet:
                snippet = json.dumps(result, default=str)
        else:
            snippet = str(result)
        snippet = snippet[:2000]

        _fb_doc_name = ""
        if isinstance(result, dict):
            _fb_doc_name = result.get("document_name", "") or result.get("doc_name", "") or ""
        result_chunks.append(Chunk(
            id=f"tool_{tool_name}_{correlation_id or 'x'}",
            text=snippet,
            score=1.0,
            source=ChunkSource(document_name=_fb_doc_name),
            meta={"source": "tool", "tool_name": tool_name},
        ))
        logger.info("Agent %s returned %d chars", tool_name, len(snippet), extra={"correlation_id": correlation_id})

    return result_chunks


_dispatch_tools = _dispatch_agents  # backward-compat alias


def _emergency_chunk_summary(chunks: List[Chunk], query: str) -> str:
    """Generate a factual summary from raw chunks when renderers return empty.

    Uses a multi-pass approach:
    1. Extract structured lines (key-value pairs via colon heuristic) relevant to query
    2. Sentence-level keyword overlap scoring for narrative content
    3. If keyword scoring finds nothing, present raw chunk content directly

    This is document-type-agnostic — works for invoices, medical, legal, etc.
    IMPORTANT: If chunks exist, this MUST return content. Never return empty
    when we have actual document chunks.
    """
    if not chunks:
        return ""
    # Extract query keywords for relevance filtering
    query_words = set(
        w.lower() for w in re.split(r'\W+', query or "")
        if len(w) > 2 and w.lower() not in _EMERGENCY_STOP_WORDS
    )

    # Pass 1: Extract structured key-value lines (colon-delimited)
    kv_lines: list[tuple[str, float]] = []
    narrative_lines: list[tuple[str, float]] = []
    # Scan ALL chunks, not just the first few — the relevant data may be anywhere
    _max_scan = min(len(chunks), 30)

    for c in chunks[:_max_scan]:
        text = (c.text or "").strip()
        if not text:
            continue

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if len(line) < 5 or len(line) > 500:
                continue

            line_words = set(w.lower() for w in re.split(r'\W+', line) if len(w) > 2)
            overlap = len(query_words & line_words)
            score = overlap / max(len(query_words), 1)

            # Structural heuristic: lines with a colon in the first half
            colon_pos = line.find(":")
            has_kv_structure = 0 < colon_pos < len(line) * 0.6 and len(line) > colon_pos + 2
            if has_kv_structure:
                kv_lines.append((line, score + 0.2))
            elif overlap > 0:
                narrative_lines.append((line, score + 0.05))

    # Merge and sort by relevance
    all_lines = kv_lines + narrative_lines
    if all_lines:
        all_lines.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        result = ["Based on the documents:"]
        for line, score in all_lines:
            normalized = line.lower().strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(f"- {line}")
            if len(result) >= 15:
                break
        return "\n".join(result)

    # Pass 2: No keyword overlap found — present raw chunk content directly.
    # This guarantees the user sees actual document data rather than "not found".
    content_parts: list[str] = []
    seen_text: set[str] = set()
    for c in chunks[:_max_scan]:
        text = (c.text or "").strip()
        if not text or len(text) < 10:
            continue
        # Deduplicate
        sig = text[:80].lower()
        if sig in seen_text:
            continue
        seen_text.add(sig)
        # Truncate long chunks
        snippet = text[:400].rstrip()
        if len(text) > 400:
            snippet += "..."
        content_parts.append(snippet)
        if len(content_parts) >= 5:
            break

    if content_parts:
        header = "Here's what I found in the documents:"
        body = "\n\n".join(content_parts)
        return f"{header}\n\n{body}"

    # Absolute last resort: first chunk raw text
    first_text = (chunks[0].text or "").strip()[:500]
    if first_text:
        return f"Based on available information:\n{first_text}"
    return ""


_EMERGENCY_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "with", "this", "that",
    "from", "what", "who", "how", "which", "tell", "show", "give",
    "about", "does", "can", "has", "have", "their", "been", "will",
    "would", "could", "should", "its", "you", "your", "they", "them",
    "not", "but", "all", "any", "each", "more", "some", "than",
})


def _normalize_llm_formatting(text: str) -> str:
    """Ensure LLM output has proper line breaks for readability.

    Ollama models sometimes generate single-line responses with inline list
    markers.  This inserts newlines before common structural patterns so the
    response renders properly in markdown-aware UIs.
    """
    if not text or "\n" in text:
        return text  # Already has newlines — leave it alone
    # Insert newline before list markers: "- ", "* ", "1. ", "2. ", etc.
    text = re.sub(r" (?=- )", "\n", text)
    text = re.sub(r" (?=\* )", "\n", text)
    text = re.sub(r" (?=\d+\. )", "\n", text)
    # Insert newline before markdown headers: "## ", "### "
    text = re.sub(r" (?=#{1,4} )", "\n\n", text)
    # Insert newline before bold section headers: "**Section:**"
    text = re.sub(r" (?=\*\*[A-Z])", "\n", text)
    # Insert newline before "| " table rows
    text = re.sub(r" (?=\| )", "\n", text)
    return text.strip()


def _extract_render_judge(
    *,
    extraction: Any,
    query: str,
    chunks: List[Chunk],
    llm_client: Any,
    budget: LLMBudget,
    correlation_id: Optional[str],
    query_focus: Any = None,
    pipeline_start: Optional[float] = None,
    pipeline_deadline: float = 45.0,
) -> Tuple[str, JudgeResult]:
    """Shared logic: render (or skip for LLM response) → sanitize → judge → evidence bypass."""
    if _is_llm_response(extraction):
        logger.info("Extract path: LLM response (schema type=%s)", type(extraction.schema).__name__)
        sanitized = sanitize(_normalize_llm_formatting(extraction.schema.text))
    else:
        logger.info("Extract path: deterministic render (schema type=%s, domain=%s)", type(extraction.schema).__name__, extraction.domain)
        rendered = render(
            domain=extraction.domain,
            intent=extraction.intent,
            schema=extraction.schema,
            strict=False,
            query=query,
            query_focus=query_focus,
        )
        logger.info("Rendered output: newlines=%d, len=%d, first100=%r", rendered.count("\n"), len(rendered), rendered[:100])
        sanitized = sanitize(rendered)
        logger.info("After sanitize: newlines=%d, len=%d", sanitized.count("\n"), len(sanitized))

    # Treat missing_reason as empty — it means structured extraction found nothing.
    # The emergency chunk summary will present raw chunk content relevant to the
    # query, which is more helpful than "Not explicitly mentioned."
    _is_missing = (
        not sanitized.strip()
        or sanitized.strip() == MISSING_REASON.strip()
        or sanitized.strip().lower().startswith("not explicitly mentioned")
        or sanitized.strip().lower().startswith("not enough information")
    )
    if _is_missing and chunks:
        logger.info(
            "Structured extraction empty/missing — using emergency chunk summary (%d chunks) | cid=%s",
            len(chunks), correlation_id,
        )
        rendered = _emergency_chunk_summary(chunks, query)
        sanitized = sanitize(rendered)
        if sanitized.strip():
            logger.info("Emergency chunk summary produced %d chars | cid=%s", len(sanitized), correlation_id)

    # Post-generation grounding verification — optionally blocks ungrounded answers
    # Skip grounding gate when extraction produced valid structured data (already grounded)
    _skip_grounding = _has_valid_deterministic_extraction(extraction.schema) or _is_llm_response(extraction)
    chunk_texts = [c.text for c in chunks if hasattr(c, "text") and c.text]
    grounding_blocked = False
    if sanitized and chunk_texts and not _skip_grounding:
        try:
            from src.quality.fast_grounding import evaluate_grounding
            from src.api.config import Config as _Cfg
            _gate_enabled = getattr(getattr(_Cfg, "Quality", None), "GROUNDING_GATE_ENABLED", True)
            _gate_th = getattr(getattr(_Cfg, "Quality", None), "GROUNDING_GATE_CRITICAL_TH", 0.30)
            grounding = evaluate_grounding(sanitized, chunk_texts)
            if grounding.critical_supported_ratio < _gate_th and grounding.unsupported_sentences:
                logger.warning(
                    "Grounding check failed: critical_support=%.2f unsupported=%d query=%r",
                    grounding.critical_supported_ratio,
                    len(grounding.unsupported_sentences),
                    query[:120],
                    extra={"stage": "grounding_gate", "correlation_id": correlation_id},
                )
                if _gate_enabled:
                    grounding_blocked = True
                    # Replace LLM answer with evidence-based fallback
                    fallback = _emergency_chunk_summary(chunks, query)
                    if fallback and fallback.strip():
                        sanitized = sanitize(fallback)
                    else:
                        sanitized = "I found relevant documents but couldn't verify a fully grounded answer. Here's what the documents contain."
                    logger.warning(
                        "Grounding gate blocked ungrounded response; using evidence fallback | cid=%s",
                        correlation_id,
                    )
        except Exception as exc:
            logger.debug("Grounding check error: %s", exc)
    elif _skip_grounding:
        logger.debug("Grounding gate skipped: valid deterministic extraction present")

    # Deterministic extraction with valid data is inherently grounded — it
    # was extracted directly from source chunks and cannot hallucinate.
    # Skip the (potentially slow) LLM judge entirely for deterministic results.
    _is_deterministic = _has_valid_deterministic_extraction(extraction.schema)
    # Deadline-aware: skip LLM judge/corrector when close to budget
    _erj_has_time = pipeline_start is None or _time_remaining(pipeline_start, pipeline_deadline) > 10.0
    if _is_deterministic:
        verdict = JudgeResult(status="pass", reason="deterministic_extraction")
    elif not _erj_has_time:
        verdict = JudgeResult(status="pass", reason="deadline_skip")
    else:
        verdict = judge(
            answer=sanitized,
            schema=extraction.schema,
            intent=extraction.intent,
            llm_client=llm_client,
            budget=budget,
            sources_present=bool(chunks),
            correlation_id=correlation_id,
        )

    # ── Hallucination self-correction ────────────────────────────────
    # Skip for deterministic extractions: they come from source data and
    # cannot hallucinate.  Running the corrector on them strips valid
    # synthesis because the Jaccard token overlap between rendered
    # summaries and raw chunk text is naturally low.
    if verdict.status != "pass" and sanitized and chunk_texts and not _is_deterministic and _erj_has_time:
        try:
            from src.api.config import Config as _HCCfg
            _hc_enabled = getattr(getattr(_HCCfg, "HallucinationCorrector", None), "ENABLED", False)
            if _hc_enabled:
                from src.intelligence.hallucination_corrector import correct_hallucinations
                _hc_threshold = getattr(_HCCfg.HallucinationCorrector, "SCORE_THRESHOLD", 0.5)
                _hc_max = min(getattr(_HCCfg.HallucinationCorrector, "MAX_CORRECTIONS", 2), 2)
                _hc_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                _hc_future = _hc_exec.submit(
                    correct_hallucinations,
                    sanitized, chunk_texts,
                    llm_client=llm_client,
                    score_threshold=_hc_threshold,
                    max_corrections=_hc_max,
                )
                try:
                    correction = _hc_future.result(timeout=8.0)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    _hc_future.cancel()
                    correction = None
                    logger.debug("Hallucination corrector timed out (8s) | cid=%s", correlation_id)
                finally:
                    _hc_exec.shutdown(wait=False)
                if correction and correction.was_modified:
                    sanitized = correction.corrected
                    logger.info(
                        "Hallucination corrector: %d corrections (%d removed, %d rewritten) | cid=%s",
                        correction.corrections_made,
                        len(correction.removed_sentences),
                        len(correction.corrected_sentences),
                        correlation_id,
                    )
        except Exception as _hc_exc:
            logger.debug("Hallucination corrector skipped: %s", _hc_exc)

    return sanitized, verdict


def _load_document_data_for_extraction(
    chunks: List[Any], query: str, correlation_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Load complete document data for accurate extraction when available.

    This function loads the full document content from pickle files, which enables
    more accurate information extraction compared to using fragmented vector chunks.

    Works with any document type (resumes, invoices, contracts, etc).

    Args:
        chunks: Retrieved chunks from Qdrant
        query: User query (for logging)
        correlation_id: Request correlation ID for logging

    Returns:
        Complete document data if available, None otherwise
    """
    from src.api.content_store import load_extracted_pickle

    document_data = None

    try:
        if not chunks:
            return None

        # Get document IDs from chunks
        doc_ids = set()
        for chunk in chunks:
            meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}
            doc_id = meta.get("document_id") or meta.get("doc_id")
            if doc_id:
                doc_ids.add(str(doc_id))

        # Load the first document's complete data
        if doc_ids:
            doc_id = next(iter(doc_ids))
            try:
                raw_pickle = load_extracted_pickle(doc_id)
                # If new payload structure with 'structured' exists, prefer first structured doc
                if isinstance(raw_pickle, dict) and ("structured" in raw_pickle or "raw" in raw_pickle):
                    structured = raw_pickle.get("structured") or {}
                    if structured:
                        # Take first structured entry and convert dataclass to dict if needed
                        first_key = next(iter(structured.keys()))
                        first_val = structured.get(first_key)
                        try:
                            from dataclasses import is_dataclass, asdict

                            if is_dataclass(first_val):
                                document_data = asdict(first_val)
                            else:
                                document_data = first_val
                        except Exception:
                            document_data = first_val
                    else:
                        # fallback: pass the raw payload
                        document_data = raw_pickle
                else:
                    document_data = raw_pickle
                logger.info(
                    "Loaded complete document data for extraction",
                    extra={"doc_id": doc_id, "correlation_id": correlation_id, "query_fragment": query[:50]}
                )
            except Exception as pickle_error:
                logger.debug(
                    "Could not load document pickle for %s: %s",
                    doc_id,
                    pickle_error,
                    extra={"correlation_id": correlation_id}
                )
    except Exception as load_error:
        logger.debug(
            "Error loading document data: %s",
            load_error,
            extra={"correlation_id": correlation_id}
        )

    return document_data


_CONTACT_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_CONTACT_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,15})")
_CONTACT_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE)


def _augment_chunks_with_contact_info(
    chunks: List[Any],
    document_data: Optional[Dict[str, Any]],
) -> List[Any]:
    """Augment chunk list with a synthetic contact-info chunk from pickle raw text.

    Resumes often have name/email/phone/LinkedIn in a header that the chunker
    may not preserve.  If the pickle ``raw_text`` contains contact info that
    does not appear in any chunk text, a synthetic chunk is prepended.
    """
    if not document_data or not isinstance(document_data, dict):
        return chunks

    raw = document_data.get("raw_text", "") or ""
    if not raw:
        return chunks

    # Un-wrap ExtractedDocument repr if needed
    prefix = "ExtractedDocument(full_text='"
    if raw.startswith(prefix):
        raw = raw[len(prefix):]
        if raw.endswith("')"):
            raw = raw[:-2]
        raw = raw.replace("\\n", "\n")

    # Only use the first ~600 chars (document header)
    header = raw[:600]
    emails = _CONTACT_EMAIL_RE.findall(header)
    phones = _CONTACT_PHONE_RE.findall(header)
    linkedins = _CONTACT_LINKEDIN_RE.findall(header)

    if not emails and not phones and not linkedins:
        return chunks

    # Check if the info already exists in any chunk
    all_chunk_text = " ".join(
        (getattr(c, "text", "") or "") for c in chunks
    )
    has_new_info = (
        (emails and not any(e in all_chunk_text for e in emails))
        or (linkedins and not any(li in all_chunk_text for li in linkedins))
    )
    if not has_new_info:
        return chunks

    # Build synthetic contact chunk from the header lines
    lines = header.split("\n")
    contact_lines = []
    for line in lines[:8]:
        line = line.strip()
        if not line:
            continue
        if _CONTACT_EMAIL_RE.search(line) or _CONTACT_PHONE_RE.search(line) or _CONTACT_LINKEDIN_RE.search(line):
            contact_lines.append(line)
        elif len(line) < 60 and not any(kw in line.lower() for kw in ("skills", "experience", "education", "project")):
            contact_lines.append(line)

    if not contact_lines:
        return chunks

    # Remove page markers and other artifacts
    contact_lines = [
        ln for ln in contact_lines
        if not re.match(r"^-{2,}\s*Page\s+\d+\s*-{2,}$", ln.strip())
    ]
    if not contact_lines:
        return chunks

    contact_text = "\n".join(contact_lines)

    # Infer doc_id from existing chunks
    doc_id = ""
    for c in chunks:
        meta = getattr(c, "meta", None) or getattr(c, "metadata", None) or {}
        doc_id = meta.get("document_id") or meta.get("doc_id") or ""
        if doc_id:
            break

    synthetic = Chunk(
        id=f"synthetic_contact_{doc_id}",
        text=contact_text,
        score=0.0,
        source=ChunkSource(document_name="Contact Header"),
        meta={
            "document_id": doc_id,
            "section_kind": "identity_contact",
            "section_title": "Contact Information",
            "synthetic": True,
        },
    )
    return [synthetic] + list(chunks)


_INSUFFICIENT_PHRASES = (
    "do not contain", "does not contain", "no information",
    "none of the", "sorry, but", "no relevant information",
    "not found in the", "cannot determine", "not mentioned in",
    "no evidence", "cannot find", "outside the scope",
    "not enough information", "no data available",
    "not covered in", "not available in",
    "does not mention", "do not mention",
)


def _web_search_fallback(
    answer_text: str,
    query: str,
    enable_internet: bool,
    metadata: Dict[str, Any],
    request_id: Optional[str],
    correlation_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Check if answer is insufficient and fall back to web search.

    Returns a web-search-based answer dict, or None if not applicable.
    """
    if not enable_internet or not answer_text:
        return None

    _lower = answer_text.lower().replace("\u2019", "'").replace("\u2018", "'")
    _phrase_match = any(p in _lower for p in _INSUFFICIENT_PHRASES)

    # Also check confidence dimensions
    _conf = metadata.get("confidence", {})
    _dims = _conf.get("dimensions", {}) if isinstance(_conf, dict) else {}
    _low_evidence = (
        _dims.get("evidence_coverage", 1.0) == 0.0
        and _dims.get("entity_grounding", 1.0) == 0.0
    )

    if not (_phrase_match or _low_evidence):
        return None

    logger.info(
        "Web search post-hoc triggered: phrase_match=%s low_evidence=%s | cid=%s",
        _phrase_match, _low_evidence, correlation_id,
    )
    try:
        from src.tools.web_search import search_web, format_web_results, build_web_sources
        _web_results = search_web(query)
        if _web_results:
            _web_text = format_web_results(_web_results, query=query)
            return _build_answer(
                response_text=_web_text,
                sources=build_web_sources(_web_results),
                request_id=request_id,
                metadata={**metadata, "web_search": True, "source_type": "web_search", "quality": "WEB"},
                query=query,
            )
    except Exception as exc:
        logger.debug("Web search post-hoc fallback failed: %s", exc)
    return None


def _run_all_profile_analysis(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    qdrant_client: Any,
    embedder: Any,
    cross_encoder: Any,
    llm_client: Any,
    budget: LLMBudget,
    intent_parse: Optional[IntentParse],
    correlation_id: Optional[str],
    request_id: Optional[str],
    tool_domain: Optional[str] = None,
    tools: Optional[List[str]] = None,
    tool_inputs: Optional[Dict[str, Any]] = None,
    redis_client: Any = None,
    pipeline_start: Optional[float] = None,
) -> Dict[str, Any]:
    """Retrieve ALL profile chunks, group by document, extract per-doc, then synthesize."""
    _ap_start = pipeline_start or time.time()
    _AP_DEADLINE_S = 90.0
    collection = build_collection_name(subscription_id)

    # 1. Get all chunks from profile
    all_chunks = expand_full_scan_by_profile(
        qdrant_client=qdrant_client,
        collection=collection,
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        correlation_id=correlation_id,
    )

    if not all_chunks:
        return _build_answer(
            response_text=_no_results_message(query, scope="all_profile"),
            sources=[],
            request_id=request_id,
            metadata={"scope": "all_profile", "rag_v3": True, "quality": "LOW"},
            query=query,
        )

    # 2. Group by document FIRST — needed to decide reranking strategy
    doc_chunks: Dict[str, List[Chunk]] = {}
    for chunk in all_chunks:
        doc_id = _chunk_document_id(chunk) or "unknown"
        doc_chunks.setdefault(doc_id, []).append(chunk)

    _num_docs = len(doc_chunks)

    # ALWAYS skip cross-encoder for all-profile queries.  Cross-encoder
    # on CPU takes 10-20s per query and adds no value when deterministic
    # extraction groups by doc_id anyway.  Bi-encoder scores from Qdrant
    # are sufficient for per-document chunk selection.
    _CHUNKS_PER_DOC = 5
    quality_chunks: List[Chunk] = []
    for _doc_id, _doc_chunk_list in doc_chunks.items():
        _doc_chunk_list.sort(key=lambda c: -c.score)
        quality_chunks.extend(_doc_chunk_list[:_CHUNKS_PER_DOC])
    quality_chunks = deduplicate_by_content(quality_chunks)
    reranked = quality_chunks
    logger.info(
        "All-profile fast path: %d docs, %d chunks (skipped cross-encoder) | cid=%s",
        _num_docs, len(quality_chunks), correlation_id,
    )

    # ── Chunk translation in all-profile path ────────────────────────
    if tools and "translator" in tools and quality_chunks:
        _target_lang = _detect_target_language(query)
        if _target_lang:
            quality_chunks = _translate_chunks(quality_chunks, _target_lang, llm_client, correlation_id)
            reranked = quality_chunks
            tools = [t for t in tools if t != "translator"]

    # ── Domain agent fast-path (all-profile) ────────────────────────
    try:
        from src.agentic.domain_agents import detect_agent_task as _dat_all
        _agent_det_all = _dat_all(query, domain=tool_domain or "")
        if _agent_det_all and quality_chunks:
            _agent_answer = _try_domain_agent(
                query=query,
                domain=tool_domain or "",
                chunks=quality_chunks,
                llm_client=llm_client,
                request_id=request_id,
                metadata={
                    "intent": intent_parse.intent if intent_parse else "factual",
                    "scope": "all_profile",
                    "document_count": len(doc_chunks),
                },
            )
            if _agent_answer is not None:
                return _agent_answer
    except Exception:
        pass

    # ── Agent dispatch inside all-profile path ─────────────────────────
    # Only dispatch content-generation agents (email_drafting, insights,
    # action_items) — these produce new content the extraction pipeline can't.
    # Domain-extraction agents (lawhere, resumes, medical, invoice) are
    # redundant with the deterministic extraction path and would be too slow
    # on large multi-document chunk sets.
    _GENERATION_AGENTS = {"email_drafting", "insights", "action_items", "web_search"}
    _GENERATION_TOOLS = _GENERATION_AGENTS  # backward-compat alias  # noqa: F841
    _gen_tools = [t for t in (tools or []) if t in _GENERATION_AGENTS]
    if _gen_tools:
        tool_chunks = _dispatch_agents(
            tool_names=_gen_tools,
            query=query,
            profile_id=str(profile_id),
            subscription_id=str(subscription_id),
            tool_inputs=tool_inputs,
            correlation_id=correlation_id,
            chunks=quality_chunks,
            llm_client=llm_client,
            intent_hint=intent_parse.intent if intent_parse else None,
            domain_hint=tool_domain,
        )
        if tool_chunks:
            # If a tool produced a pre-rendered response, use it directly
            for tc in tool_chunks:
                if tc.meta and tc.meta.get("source") == "tool_rendered" and tc.text and len(tc.text) > 50:
                    # Apply NER cleanup to tool output (render_enterprise cleanup
                    # only runs on the extract path, not tool path)
                    _clean_text = tc.text
                    try:
                        from .enterprise import _NER_LABEL_RE, _NER_INLINE_RE
                        _clean_text = _NER_LABEL_RE.sub("", _clean_text)
                        _clean_text = _NER_INLINE_RE.sub(" ", _clean_text)
                    except Exception:
                        pass
                    _tool_meta = {
                        "domain": tc.meta.get("domain") or tool_domain,
                        "intent": intent_parse.intent if intent_parse else "factual",
                        "scope": "all_profile",
                        "document_count": len(doc_chunks),
                        "quality": "HIGH",
                        "rag_v3": True,
                        "tool_rendered": True,
                        "tool_name": tc.meta.get("tool_name", ""),
                    }
                    return _build_answer(
                        response_text=_clean_text,
                        sources=_collect_sources(quality_chunks),
                        request_id=request_id,
                        metadata=_tool_meta,
                        query=query,
                        chunks=quality_chunks,
                    )
            # Otherwise merge tool chunks into quality_chunks for extraction
            quality_chunks = tool_chunks + quality_chunks

    # ── Cross-document intelligence routing ──────────────────────────
    from .document_context import assemble_document_contexts
    from .llm_extract import classify_query_intent

    doc_contexts = assemble_document_contexts(quality_chunks)
    intent_type = classify_query_intent(
        query, intent_hint=intent_parse.intent if intent_parse else None,
    )

    # Resolve majority domain from document contexts
    _domain_counts: Dict[str, int] = {}
    for _ctx in doc_contexts:
        _domain_counts[_ctx.doc_domain] = _domain_counts.get(_ctx.doc_domain, 0) + 1
    majority_domain = max(_domain_counts, key=_domain_counts.get) if _domain_counts else "generic"

    # ── ML-based context understanding — SKIPPED for all-profile ─────
    # Context understanding only enriches LLM prompts. All-profile queries
    # use deterministic extraction (no LLM), so this is pure wasted time
    # (~5-15s of embedding + clustering on all chunks).
    _all_profile_ctx_intel = None

    # Route analytics queries (how many, totals, averages)
    if intent_type == "analytics":
        from .corpus_analytics import is_analytics_query, compute_corpus_stats, answer_analytics_query
        if is_analytics_query(query):
            stats = compute_corpus_stats(doc_contexts)
            answer_text = answer_analytics_query(query, stats, doc_contexts)
            sources = _collect_sources(quality_chunks)
            metadata = {
                "domain": majority_domain,
                "intent": "analytics",
                "scope": "all_profile",
                "document_count": len(doc_contexts),
                "quality": "HIGH",
                "rag_v3": True,
            }
            return _build_answer(
                response_text=sanitize(answer_text),
                sources=sources,
                request_id=request_id,
                metadata=metadata,
                query=query,
                chunks=quality_chunks,
            )

    # Route comparison queries to comparator (2+ documents)
    # NOTE: ranking is deliberately excluded — it benefits from the full
    # HR extraction path which produces structured Candidate objects with
    # names, skills, and scoring.  The comparator only has raw chunk fields.
    if intent_type == "comparison" and 1 < len(doc_contexts) <= 5:
        # Only use comparator for small document sets (2-5 docs).
        # For large sets (>5), fall through to HR extraction + enterprise
        # rendering which creates structured Candidate objects per document.
        from .comparator import compare_documents, render_comparison

        # Try LLM-enhanced comparison first (skip if close to deadline)
        if llm_client and budget.allow() and _time_remaining(_ap_start, _AP_DEADLINE_S) > 15.0:
            from .llm_extract import llm_extract_and_respond
            _use_thinking = _should_use_thinking(
                query=query,
                intent_parse=intent_parse,
                chunk_count=len(quality_chunks),
                scope_mode="all_profile",
            )
            try:
                from src.llm.task_router import task_scope as _ts_gen, TaskType as _TT_gen
                _task_type = _TT_gen.COMPLEX_EXTRACTION if _use_thinking else _TT_gen.RESPONSE_GENERATION
                with _ts_gen(_task_type):
                    llm_result = llm_extract_and_respond(
                        query=query, chunks=quality_chunks, llm_client=llm_client,
                        budget=budget, intent=intent_type,
                        num_documents=len(doc_contexts),
                        correlation_id=correlation_id,
                        domain=majority_domain,
                        redis_client=redis_client,
                        context_intelligence=_all_profile_ctx_intel,
                        use_thinking=_use_thinking,
                    )
            except ImportError:
                llm_result = llm_extract_and_respond(
                    query=query, chunks=quality_chunks, llm_client=llm_client,
                    budget=budget, intent=intent_type,
                    num_documents=len(doc_contexts),
                    correlation_id=correlation_id,
                    domain=majority_domain,
                    redis_client=redis_client,
                    context_intelligence=_all_profile_ctx_intel,
                    use_thinking=_use_thinking,
                )
            if llm_result:
                sources = _collect_sources(quality_chunks)
                metadata = {
                    "domain": majority_domain,
                    "intent": intent_type,
                    "scope": "all_profile",
                    "document_count": len(doc_contexts),
                    "quality": "HIGH",
                    "rag_v3": True,
                    "thinking_used": getattr(llm_result, "thinking_used", False),
                }
                return _build_answer(
                    response_text=sanitize(llm_result.text),
                    sources=sources,
                    request_id=request_id,
                    metadata=metadata,
                    query=query,
                    chunks=quality_chunks,
                )

        # Deterministic comparison fallback
        comp_result = compare_documents(doc_contexts, query)
        rendered = render_comparison(comp_result, intent_type)
        if rendered:
            sources = _collect_sources(quality_chunks)
            metadata = {
                "domain": majority_domain,
                "intent": intent_type,
                "scope": "all_profile",
                "document_count": len(doc_contexts),
                "quality": "HIGH",
                "rag_v3": True,
            }
            return _build_answer(
                response_text=sanitize(rendered),
                sources=sources,
                request_id=request_id,
                metadata=metadata,
                query=query,
                chunks=quality_chunks,
            )

    # ── Existing extract path (unchanged) ─────────────────────────
    # 5. Extract with all quality chunks
    # NOTE: Do NOT pass document_data here.  _load_document_data_for_extraction()
    # only loads the first document's pickle, which triggers the hybrid single-doc
    # extraction path (Strategy 0) and returns 1 candidate.  By omitting it we fall
    # through to _extract_hr(), which groups chunks by doc_id and creates one
    # candidate PER document — exactly what multi-document queries need.
    # When a tool is selected (e.g., tools=resume-analysis), use the tool domain
    # as an authoritative hint to route extraction correctly.
    domain_hint = tool_domain  # None when no tool, "hr"/"legal"/etc. when tool-specified

    # ── Domain-aware chunk filtering for mixed-domain profiles ──────
    # When a specific domain is targeted in a mixed-domain profile
    # (resumes + invoices + medical records), filter quality_chunks to
    # only chunks from documents matching that domain.  Without this,
    # the deterministic extractor sees chunks from all document types
    # and fails to extract domain-specific data.
    _extract_chunks = quality_chunks
    _DOMAIN_ALIASES = {
        "hr": {"hr", "resume", "cv"},
        "invoice": {"invoice", "billing"},
        "medical": {"medical", "clinical", "health"},
        "legal": {"legal", "contract"},
        "policy": {"policy", "insurance"},
    }
    # Resolve effective domain: tool_domain, query ML domain, or majority
    _effective_domain = tool_domain
    if not _effective_domain:
        # Detect domain from query using ML classifier
        from .extract import _ml_query_domain
        _q_domain = _ml_query_domain(query, intent_parse=intent_parse)
        if _q_domain and _q_domain in _DOMAIN_ALIASES:
            _effective_domain = _q_domain
    # Only filter when we have multiple domain types (mixed profile)
    if _effective_domain and _effective_domain in _DOMAIN_ALIASES and len(_domain_counts) > 1:
        _target_domains = _DOMAIN_ALIASES[_effective_domain]
        _matching_doc_ids = set()
        for _ctx in doc_contexts:
            if _ctx.doc_domain in _target_domains:
                _matching_doc_ids.add(_ctx.document_id)
        if _matching_doc_ids:
            _domain_filtered = [
                c for c in quality_chunks
                if (_chunk_document_id(c) or "unknown") in _matching_doc_ids
            ]
            if _domain_filtered:
                _extract_chunks = _domain_filtered
                logger.info(
                    "Domain-filtered chunks for '%s': %d → %d (docs: %d) | cid=%s",
                    _effective_domain, len(quality_chunks), len(_extract_chunks),
                    len(_matching_doc_ids), correlation_id,
                )

    from .query_focus import build_query_focus as _bqf
    _all_profile_focus = _bqf(query, intent_hint=intent_parse.intent if intent_parse else None)

    # For multi-document all-profile queries (>5 docs), skip LLM extraction.
    # LLMs can't reliably process 10+ documents in a single context window —
    # they truncate and only return 1-3 candidates. Deterministic extraction
    # creates one Candidate PER document from chunks, which is exactly what
    # multi-doc queries (compare, rank, list) need.
    # Time budget: skip LLM extraction when pipeline deadline is near
    _ap_has_time = _time_remaining(_ap_start, _AP_DEADLINE_S) > 15.0
    _all_profile_llm = llm_client if len(doc_chunks) <= 5 and _ap_has_time else None
    _all_profile_budget = budget if _all_profile_llm else LLMBudget(llm_client=None, max_calls=0)

    _all_profile_thinking = _should_use_thinking(
        query=query,
        intent_parse=intent_parse,
        chunk_count=len(_extract_chunks),
        scope_mode="all_profile",
    )
    extraction = extract_schema(
        domain_hint,
        query=query,
        chunks=_extract_chunks,
        llm_client=_all_profile_llm,
        budget=_all_profile_budget,
        correlation_id=correlation_id,
        scope_document_id=None,
        intent_hint=intent_parse.intent if intent_parse else None,
        document_data=None,  # intentionally None for multi-doc
        query_focus=_all_profile_focus,
        tool_domain=bool(tool_domain),
        embedder=embedder,
        intent_parse=intent_parse,
        redis_client=redis_client,
        use_thinking=_all_profile_thinking,
    )

    sanitized, verdict = _extract_render_judge(
        extraction=extraction,
        query=query,
        chunks=quality_chunks,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
        query_focus=_all_profile_focus,
        pipeline_start=_ap_start,
        pipeline_deadline=_AP_DEADLINE_S,
    )

    # ── B5: Deterministic cross-document synthesis ──────────────────
    if len(doc_chunks) >= 2 and sanitized.strip():
        try:
            synthesis = _synthesize_cross_document(
                schemas=[extraction.schema],
                doc_contexts=doc_contexts,
                query=query,
                domain=extraction.domain,
            )
            if synthesis:
                sanitized = f"{synthesis}\n\n{sanitized}"
        except Exception as _synth_exc:
            logger.debug("Cross-document synthesis skipped: %s", _synth_exc)

    # ── B6: LLM synthesis post-processing ──────────────────────────
    # Only use LLM synthesis when deterministic rendering is insufficient.
    # Skip when we already have a substantial response (>300 chars) to avoid
    # degrading quality and adding 15-20s latency.
    _needs_synthesis = (
        len(doc_chunks) >= 2
        and sanitized.strip()
        and len(sanitized.strip()) < 300
        and intent_type in ("comparison", "reasoning", "analytics")
        and _time_remaining(_ap_start, _AP_DEADLINE_S) > 12.0
    )
    if _needs_synthesis:
        try:
            synth_budget = LLMBudget(llm_client=llm_client, max_calls=1)
            synthesized = _llm_synthesize(
                rendered_text=sanitized,
                query=query,
                domain=extraction.domain,
                intent=extraction.intent,
                num_documents=len(doc_chunks),
                llm_client=llm_client,
                budget=synth_budget,
                correlation_id=correlation_id,
            )
            if synthesized:
                sanitized = synthesized
        except Exception as _llm_synth_exc:
            logger.debug("LLM synthesis skipped: %s", _llm_synth_exc)

    sources = _collect_sources(quality_chunks)
    metadata = {
        "domain": extraction.domain,
        "intent": extraction.intent,
        "scope": "all_profile",
        "document_count": len(doc_chunks),
        "quality": "HIGH" if verdict.status == "pass" else "LOW",
        "rag_v3": True,
        "judge": {"status": verdict.status, "reason": verdict.reason},
    }
    # Last-resort: if sanitized is empty but we have chunks, show raw content
    _final_text = sanitized
    if not _final_text.strip() and quality_chunks:
        _final_text = _emergency_chunk_summary(quality_chunks, query)
    if not _final_text.strip():
        _final_text = NO_CHUNKS_MESSAGE
    return _build_answer(
        response_text=_final_text,
        sources=sources,
        request_id=request_id,
        metadata=metadata,
        query=query,
        chunks=quality_chunks,
        schema=extraction.schema,
        llm_client=llm_client,
    )


def run(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    document_id: Optional[str] = None,
    tool_hint: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: str = "api",
    request_id: Optional[str] = None,
    llm_client: Optional[Any] = None,
    qdrant_client: Optional[Any] = None,
    redis_client: Optional[Any] = None,
    embedder: Optional[Any] = None,
    cross_encoder: Optional[Any] = None,
    tools: Optional[List[str]] = None,
    tool_inputs: Optional[Dict[str, Any]] = None,
    enable_decomposition: bool = True,
    enable_internet: bool = False,
) -> Dict[str, Any]:
    correlation_id = request_id or str(uuid.uuid4())
    _pipeline_start = time.time()
    # Global pipeline deadline — hard cap to prevent UI timeouts.
    # Individual stages check this and early-exit when close to budget.
    # With embedding on CPU and gpt-oss having full GPU, generation takes 5-20s.
    # 90s budget allows retrieval + rerank + LLM extract + judge + render.
    _PIPELINE_DEADLINE_S = 90.0

    if any(dep is None for dep in (qdrant_client, embedder)) or llm_client is None or redis_client is None or cross_encoder is None:
        state = rag_state.get_app_state()
        if state:
            llm_client = llm_client or state.ollama_client
            qdrant_client = qdrant_client or state.qdrant_client
            redis_client = redis_client or state.redis_client
            embedder = embedder or state.embedding_model
            cross_encoder = cross_encoder or getattr(state.reranker, "cross_encoder", None) or state.reranker

    if qdrant_client is None or embedder is None:
        raise ValueError("RAG v3 dependencies missing: qdrant_client and embedder are required")

    # ── Clear per-request caches to prevent cross-request contamination ──
    try:
        clear_chunk_embed_cache()
    except Exception:  # noqa: BLE001
        pass

    # Separate budgets: infrastructure (rewrite) gets 2 calls,
    # extraction+judge gets 4 calls — ensures the LLM is always
    # available for the response-generating extraction step.
    infra_budget = LLMBudget(llm_client=llm_client, max_calls=2)
    budget = LLMBudget(llm_client=llm_client, max_calls=4)
    original_query = query or ""

    # ── URL pre-processing: fetch URLs embedded in the query ───────────
    _web_url_chunks: List[Any] = []
    if enable_internet:
        try:
            from src.tools.web_search import detect_urls_in_query, fetch_url_content
            _detected_urls, _cleaned_query = detect_urls_in_query(original_query)
            if _detected_urls:
                for _url in _detected_urls:
                    _fetched = fetch_url_content(_url)
                    if _fetched.get("text") and not _fetched.get("error"):
                        # Create a synthetic chunk-like dict for the fetched content
                        _web_url_chunks.append({
                            "text": _fetched["text"][:4000],
                            "source_name": _fetched.get("title") or _url,
                            "url": _url,
                            "type": "web",
                            "score": 0.5,
                        })
                if _cleaned_query:
                    original_query = _cleaned_query
                logger.info(
                    "URL pre-processing: fetched %d/%d URLs | cid=%s",
                    len(_web_url_chunks), len(_detected_urls), correlation_id,
                )
        except Exception as exc:
            logger.debug("URL pre-processing failed: %s", exc)

    intent_future = _start_intent_parse(
        query=original_query,
        llm_client=llm_client,
        redis_client=redis_client,
    )
    intent_parse = _resolve_intent_future(intent_future)
    intent_type = _infer_intent_type(original_query)
    scope = _infer_query_scope(original_query, document_id, intent_parse)
    scope_document_id = scope.document_id  # backwards compat for extraction

    logger.info(
        "RAG v3 query scope: mode=%s document_id=%s entity_hint=%s",
        scope.mode,
        scope.document_id,
        scope.entity_hint,
        extra={"stage": "scope_inference", "correlation_id": correlation_id},
    )

    # ── Build query focus (once, threaded through all stages) ────────
    from .query_focus import build_query_focus, filter_chunks_by_focus, clear_chunk_embed_cache
    focus = build_query_focus(original_query, intent_hint=intent_parse.intent if intent_parse else None, embedder=embedder)

    # Lazy-init field importance classifier (ML Enhancement 2)
    if focus.query_embedding is not None:
        try:
            from .field_classifier import get_field_classifier, ensure_field_classifier
            if get_field_classifier() is None and embedder is not None:
                ensure_field_classifier(embedder)
            _fc = get_field_classifier()
            if _fc is not None and focus.field_probabilities is None:
                focus.field_probabilities = _fc.predict(focus.query_embedding)
                focus.field_tags.update(focus.field_probabilities.keys())
        except Exception:
            pass  # classifier is optional

    # Lazy-init trained intent classifier
    if embedder is not None:
        try:
            from src.intent.intent_classifier import get_intent_classifier, ensure_intent_classifier
            if get_intent_classifier() is None:
                ensure_intent_classifier(embedder)
        except Exception:
            pass  # intent classifier is optional

    # Lazy-init line role classifier (ML-based KV extraction)
    if embedder is not None:
        try:
            from .line_classifier import get_line_classifier, ensure_line_classifier
            if get_line_classifier() is None:
                ensure_line_classifier(embedder)
        except Exception:
            pass  # line classifier is optional

    # ── Multi-agent classifier (advisory, never blocks) ─────────────
    _multi_agent_classification = None
    try:
        from src.api.config import Config as _Cfg
        from src.api.rag_state import get_app_state as _get_state
        _ma_state = _get_state()
        _ma_gw = getattr(_ma_state, "multi_agent_gateway", None) if _ma_state else None
        if _ma_gw is not None and getattr(_Cfg, "MultiAgent", None) and getattr(_Cfg.MultiAgent, "ENABLED", False):
            from src.llm.classifier import classify_query as _llm_classify
            from src.llm.task_router import task_scope, TaskType
            _conf_threshold = getattr(_Cfg.MultiAgent, "CLASSIFIER_CONFIDENCE_THRESHOLD", 0.7)
            _timeout = getattr(_Cfg.MultiAgent, "CLASSIFIER_TIMEOUT", 10.0)
            # Fetch profile domain hint for heuristic fallback
            _profile_domain = None
            try:
                from src.profiles.profile_domain_tagger import get_profile_domain
                _profile_domain = get_profile_domain(subscription_id, profile_id)
            except Exception:
                pass
            with task_scope(TaskType.QUERY_CLASSIFICATION):
                _multi_agent_classification = _llm_classify(
                    original_query, _ma_gw, timeout_s=_timeout, profile_domain=_profile_domain
                )
            if _multi_agent_classification and _multi_agent_classification.confidence >= _conf_threshold:
                # Advisory override: improve scope/entity when regex missed
                if _multi_agent_classification.entity and not scope.entity_hint:
                    scope = QueryScope(
                        mode=scope.mode,
                        document_id=scope.document_id,
                        entity_hint=_multi_agent_classification.entity,
                    )
                    logger.info("Multi-agent classifier detected entity: %s", _multi_agent_classification.entity)
    except Exception as _ma_exc:
        logger.debug("Multi-agent classifier skipped: %s", _ma_exc)

    # ── Multi-step query planner ─────────────────────────────────────
    if enable_decomposition:
        try:
            from src.api.config import Config as _PlannerCfg
            _planner_enabled = getattr(getattr(_PlannerCfg, "QueryPlanner", None), "ENABLED", False)
            if _planner_enabled:
                from src.intelligence.query_planner import is_multi_step_query, decompose_query, execute_plan
                if is_multi_step_query(original_query):
                    _max_steps = getattr(_PlannerCfg.QueryPlanner, "MAX_STEPS", 3)
                    plan = decompose_query(original_query, max_steps=_max_steps)
                    if plan.is_multi_step:
                        logger.info(
                            "Multi-step query detected (%d steps, strategy=%s) | cid=%s",
                            len(plan.steps), plan.synthesis_strategy, correlation_id,
                        )
                        plan_kwargs = dict(
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                            document_id=document_id,
                            tool_hint=tool_hint,
                            session_id=session_id,
                            user_id=user_id,
                            request_id=request_id,
                            llm_client=llm_client,
                            qdrant_client=qdrant_client,
                            redis_client=redis_client,
                            embedder=embedder,
                            cross_encoder=cross_encoder,
                            tools=tools,
                            tool_inputs=tool_inputs,
                        )
                        return execute_plan(plan, run, plan_kwargs)
        except Exception as _plan_exc:
            logger.debug("Query planner skipped: %s", _plan_exc)

    # ── Auto-select tools when none explicitly provided ──────────────
    # The ToolSelector normally runs only in AGENT mode.  We run it here
    # in the pipeline so that normal /api/ask requests also benefit from
    # automatic tool dispatch (e.g. "translate to english" → translator).
    if not tools:
        try:
            from src.agentic.tool_selector import ToolSelector
            _auto_tools = ToolSelector().select_tools(
                original_query,
                intent_parse=intent_parse,
                analysis=_multi_agent_classification,
            )
            if _auto_tools:
                tools = _auto_tools
                logger.info(
                    "Pipeline auto-selected agents: %s | cid=%s",
                    tools, correlation_id,
                )
        except Exception as _sel_exc:
            logger.debug("Agent auto-selection skipped: %s", _sel_exc)

    # ── Resolve authoritative domain from agents ─────────────────────
    tool_domain = _resolve_domain_from_agents(tools)
    if tool_domain:
        logger.info(
            "Agent domain resolved: agents=%s → domain=%s",
            tools, tool_domain,
            extra={"stage": "tool_domain", "correlation_id": correlation_id},
        )

    # ── Domain agent task detection ────────────────────────────────
    # Queries like "generate interview questions" or "check drug interactions"
    # are handled by specialized domain agents. We detect these early so the
    # agent can later receive RAG-retrieved chunks as context.
    _agent_detection = None
    try:
        from src.agentic.domain_agents import detect_agent_task as _dat
        _agent_detection = _dat(original_query, domain=tool_domain or "")
    except Exception:
        pass

    # ── Proactive web search when enable_internet=True ──────────────
    # If user explicitly requested internet and no document-domain tools
    # were selected, inject web_search so the early dispatch path fires.
    if enable_internet and not tool_domain and "web_search" not in (tools or []):
        tools = list(tools or []) + ["web_search"]
        logger.info(
            "Injected web_search tool (enable_internet=True, no domain tools) | cid=%s",
            correlation_id,
        )

    # ── EARLY EXIT: document-agnostic agents skip full RAG pipeline ──
    # Agents like translator, web_search, creator, tutor, email_drafting
    # don't need document chunks.  Dispatch them immediately to avoid
    # expensive rewrite → profile scan → retrieval → reranking.
    _early_agnostic = [t for t in (tools or []) if t in _DOC_AGNOSTIC_AGENTS]
    if _early_agnostic and not tool_domain:
        logger.info(
            "Early dispatch for document-agnostic agents: %s | cid=%s",
            _early_agnostic, correlation_id,
        )
        _early_chunks = _dispatch_agents(
            tool_names=_early_agnostic,
            query=original_query,
            profile_id=str(profile_id),
            subscription_id=str(subscription_id),
            tool_inputs=tool_inputs,
            correlation_id=correlation_id,
            chunks=[],
            llm_client=llm_client,
            intent_hint=intent_type,
            domain_hint=None,
        )
        if _early_chunks:
            for tc in _early_chunks:
                if tc.meta and tc.meta.get("source") == "tool_rendered" and tc.text and len(tc.text) > 50:
                    _clean_text = tc.text
                    try:
                        from .enterprise import _NER_LABEL_RE, _NER_INLINE_RE
                        _clean_text = _NER_LABEL_RE.sub("", _clean_text)
                        _clean_text = _NER_INLINE_RE.sub(" ", _clean_text)
                    except Exception:
                        pass
                    _elapsed = time.time() - _pipeline_start
                    logger.info(
                        "Early tool dispatch complete in %.1fs | tool=%s | cid=%s",
                        _elapsed, tc.meta.get("tool_name", ""), correlation_id,
                    )
                    return _build_answer(
                        response_text=_clean_text,
                        sources=_collect_sources([]),
                        request_id=request_id,
                        metadata={
                            "domain": None,
                            "intent": intent_type,
                            "intent_type": intent_type,
                            "scope": {"profile_id": profile_id},
                            "quality": "HIGH",
                            "rag_v3": True,
                            "tool_rendered": True,
                            "tool_name": tc.meta.get("tool_name", ""),
                        },
                        query=original_query,
                    )
            # Non-rendered tool results: add to context and continue pipeline
            # (rare — most agnostic tools produce rendered output)

    # ── All-profile path: multi-document analysis ────────────────────
    if scope.mode == "all_profile":
        return _run_all_profile_analysis(
            query=original_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            qdrant_client=qdrant_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            llm_client=llm_client,
            budget=budget,
            intent_parse=intent_parse,
            correlation_id=correlation_id,
            request_id=request_id,
            tool_domain=tool_domain,
            tools=tools,
            tool_inputs=tool_inputs,
            redis_client=redis_client,
            pipeline_start=_pipeline_start,
        )

    stage_start = time.time()
    try:
        from src.llm.task_router import task_scope as _ts_rewrite, TaskType as _TT_rewrite
        with _ts_rewrite(_TT_rewrite.QUERY_REWRITE):
            rewritten = rewrite_query(
                query=original_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                redis_client=redis_client,
                llm_client=llm_client,
                budget=infra_budget,  # infrastructure budget — don't eat extraction calls
                correlation_id=correlation_id,
            )
    except ImportError:
        rewritten = rewrite_query(
            query=original_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            redis_client=redis_client,
            llm_client=llm_client,
            budget=infra_budget,
            correlation_id=correlation_id,
        )
    _log_stage("rewrite", stage_start, correlation_id)

    profile_count, total_count = _profile_point_counts(subscription_id, profile_id, qdrant_client, correlation_id)
    # Skip unconditional profile scan when user wants a specific document —
    # the profile scan retrieves ALL profile chunks and would ignore document_id scoping.
    if _should_unconditional_profile_scan(profile_count) and scope.mode != "specific_document":
        logger.info(
            "RAG v3 unconditional profile scan triggered",
            extra={"stage": "profile_scan_unconditional", "correlation_id": correlation_id},
        )
        stage_start = time.time()
        profile_chunks = expand_full_scan_by_profile(
            qdrant_client=qdrant_client,
            collection=build_collection_name(subscription_id),
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            correlation_id=correlation_id,
        )
        _log_stage("profile_scan_retrieve", stage_start, correlation_id)
        if profile_chunks:
            reranked = rerank(query=original_query, chunks=profile_chunks,
                              cross_encoder=cross_encoder, top_k=16, correlation_id=correlation_id,
                              min_score=-100.0)  # scroll chunks have score=0; rerank for order, not filtering
            if not reranked:
                reranked = profile_chunks[:16]
            # Skip apply_quality_pipeline — scroll chunks have score=0.0 and the
            # quality filter (threshold 0.45/0.7) would drop everything.  Dedup only.
            reranked = deduplicate_by_content(reranked)
            reranked = filter_chunks_by_focus(reranked, focus, min_keep=6, top_k=16)
            clear_chunk_embed_cache()


            # Entity-hint filtering for targeted queries in profile scan path
            # Soft fallback: if entity filter returns empty, keep all chunks
            # (avoids hard early-return with 0 sources and domain="unknown")
            if scope.entity_hint and reranked:
                filtered_by_entity = _filter_chunks_by_entity_hint(reranked, scope.entity_hint, correlation_id)
                if filtered_by_entity:
                    reranked = filtered_by_entity
                else:
                    logger.info(
                        "Entity filter for '%s' returned empty in profile scan — keeping all %d chunks | cid=%s",
                        scope.entity_hint, len(reranked), correlation_id,
                    )

            stage_start = time.time()

            # ── Chunk translation in profile scan path ──────────────
            if tools and "translator" in tools and reranked:
                _target_lang = _detect_target_language(original_query)
                if _target_lang:
                    reranked = _translate_chunks(reranked, _target_lang, llm_client, correlation_id)
                    tools = [t for t in tools if t != "translator"]
                    # Recalculate tool_domain after removing translator
                    tool_domain = _resolve_domain_from_agents(tools)

            # Load complete document data for accurate extraction (works for any document type)
            document_data = _load_document_data_for_extraction(reranked, original_query, correlation_id)

            # Augment chunks with contact info from pickle header if missing
            reranked = _augment_chunks_with_contact_info(reranked, document_data)

            # ── Domain agent fast-path (profile scan) ────────────────
            if _agent_detection and reranked:
                _agent_answer = _try_domain_agent(
                    query=original_query,
                    domain=tool_domain or "",
                    chunks=reranked,
                    llm_client=llm_client,
                    request_id=request_id,
                    metadata={
                        "intent": intent_type,
                        "intent_type": intent_type,
                        "scope": {"profile_id": profile_id},
                    },
                )
                if _agent_answer is not None:
                    return _agent_answer

            _ps_thinking = _should_use_thinking(
                query=original_query,
                intent_parse=intent_parse,
                chunk_count=len(reranked),
                scope_mode="profile_scan",
            )
            # Time budget: skip LLM extraction when pipeline deadline is near
            _ps_llm = llm_client if _time_remaining(_pipeline_start, _PIPELINE_DEADLINE_S) > 15.0 else None
            _ps_budget = budget if _ps_llm else LLMBudget(llm_client=None, max_calls=0)
            extraction = extract_schema(
                tool_domain,  # None when no tool, authoritative domain when tool-specified
                query=original_query,
                chunks=reranked,
                llm_client=_ps_llm,
                budget=_ps_budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
                document_data=document_data,
                query_focus=focus,
                tool_domain=bool(tool_domain),
                embedder=embedder,
                intent_parse=intent_parse,
                redis_client=redis_client,
                use_thinking=_ps_thinking,
            )
            _log_stage("profile_scan_extract", stage_start, correlation_id)

            stage_start = time.time()
            sanitized, verdict = _extract_render_judge(
                extraction=extraction,
                query=original_query,
                chunks=reranked,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                query_focus=focus,
                pipeline_start=_pipeline_start,
                pipeline_deadline=_PIPELINE_DEADLINE_S,
            )
            _log_stage("profile_scan_judge", stage_start, correlation_id)

            sources = _collect_sources(reranked)
            # Fallback: if no sources from chunks but we have document_data, use its metadata
            if not sources and document_data:
                doc_meta = document_data.get("metadata", {}) if isinstance(document_data, dict) else {}
                doc_name = (doc_meta.get("file_name") or doc_meta.get("source_name") or "Document")
                sources = [{"file_name": doc_name, "page": 1, "snippet": ""}]
            metadata = {
                "domain": extraction.domain,
                "intent": extraction.intent,
                "intent_type": _infer_intent_type(original_query, intent_parse),
                "scope": {"profile_id": profile_id},
                "quality": "HIGH" if verdict.status == "pass" else "LOW",
                "rag_v3": True,
                "profile_scan": True,
                "judge": {"status": verdict.status, "reason": verdict.reason},
            }

            # Web search post-hoc fallback for profile scan path
            _ps_answer = sanitized
            if not _ps_answer.strip() and reranked:
                _ps_answer = _emergency_chunk_summary(reranked, original_query)
            if not _ps_answer.strip():
                _ps_answer = NO_CHUNKS_MESSAGE
            _web_result = _web_search_fallback(
                _ps_answer, original_query, enable_internet,
                metadata, request_id, correlation_id,
            )
            if _web_result:
                return _web_result

            return _build_answer(
                response_text=_ps_answer,
                sources=sources,
                request_id=request_id,
                metadata=metadata,
                query=original_query,
                chunks=reranked,
                schema=extraction.schema,
                llm_client=llm_client,
            )
    if profile_count == 0 and (total_count == 0 or total_count <= 80):
        logger.warning(
            "RAG v3 profile filter returned 0 points; using unscoped scan for small collection",
            extra={"stage": "profile_scan_unscoped", "correlation_id": correlation_id, "total_points": total_count},
        )
        stage_start = time.time()
        unscoped = expand_full_scan_unscoped(
            qdrant_client=qdrant_client,
            collection=build_collection_name(subscription_id),
            correlation_id=correlation_id,
        )
        reranked = filter_chunks_by_profile_scope(
            unscoped,
            profile_id=str(profile_id),
            subscription_id=str(subscription_id),
        )
        if unscoped and not reranked:
            logger.warning(
                "RAG v3 unscoped scan: profile filter dropped all chunks; returning empty rather than leaking cross-profile data",
                extra={"stage": "profile_scan_unscoped", "correlation_id": correlation_id},
            )
            reranked = []
        reranked = rerank(query=original_query, chunks=reranked,
                          cross_encoder=cross_encoder, top_k=16, correlation_id=correlation_id,
                          min_score=-100.0) or reranked[:16]
        reranked = deduplicate_by_content(reranked)
        reranked = filter_chunks_by_focus(reranked, focus, min_keep=6, top_k=16)
        clear_chunk_embed_cache()
        _log_stage("unscoped_scan_retrieve", stage_start, correlation_id)
        if reranked:
            stage_start = time.time()
            document_data = _load_document_data_for_extraction(reranked, original_query, correlation_id)
            reranked = _augment_chunks_with_contact_info(reranked, document_data)
            _unscoped_thinking = _should_use_thinking(
                query=original_query,
                intent_parse=intent_parse,
                chunk_count=len(reranked),
                scope_mode="unscoped_scan",
            )
            # Time budget: skip LLM extraction when pipeline deadline is near
            _us_llm = llm_client if _time_remaining(_pipeline_start, _PIPELINE_DEADLINE_S) > 15.0 else None
            _us_budget = budget if _us_llm else LLMBudget(llm_client=None, max_calls=0)
            extraction = extract_schema(
                tool_domain,  # None when no tool, authoritative domain when tool-specified
                query=original_query,
                chunks=reranked,
                llm_client=_us_llm,
                budget=_us_budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
                document_data=document_data,
                query_focus=focus,
                tool_domain=bool(tool_domain),
                embedder=embedder,
                intent_parse=intent_parse,
                redis_client=redis_client,
                use_thinking=_unscoped_thinking,
            )
            _log_stage("unscoped_scan_extract", stage_start, correlation_id)

            stage_start = time.time()
            sanitized, verdict = _extract_render_judge(
                extraction=extraction,
                query=original_query,
                chunks=reranked,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                query_focus=focus,
                pipeline_start=_pipeline_start,
                pipeline_deadline=_PIPELINE_DEADLINE_S,
            )
            _log_stage("unscoped_scan_judge", stage_start, correlation_id)

            sources = _collect_sources(reranked)
            metadata = {
                "domain": extraction.domain,
                "intent": extraction.intent,
                "intent_type": _infer_intent_type(original_query, intent_parse),
                "scope": {"profile_id": profile_id},
                "quality": "HIGH" if verdict.status == "pass" else "LOW",
                "rag_v3": True,
                "profile_scan": "unscoped",
                "judge": {"status": verdict.status, "reason": verdict.reason},
            }
            _us_answer = sanitized
            if not _us_answer.strip() and reranked:
                _us_answer = _emergency_chunk_summary(reranked, original_query)
            if not _us_answer.strip():
                _us_answer = NO_CHUNKS_MESSAGE
            return _build_answer(
                response_text=_us_answer,
                sources=sources,
                request_id=request_id,
                metadata=metadata,
                query=original_query,
                chunks=reranked,
                schema=extraction.schema,
                llm_client=llm_client,
            )
    # ── Knowledge Graph augmentation (non-blocking) ─────────────────
    graph_hints = None
    try:
        _kg_state = rag_state.get_app_state()
        _kg_augmenter = getattr(_kg_state, "graph_augmenter", None) if _kg_state else None
        if _kg_augmenter is not None:
            graph_hints = _kg_augmenter.augment(
                original_query,
                str(subscription_id),
                str(profile_id),
            )
            if graph_hints and graph_hints.query_expansion_terms:
                rewritten = f"{rewritten} {' '.join(graph_hints.query_expansion_terms)}"
                logger.info(
                    "KG expanded query with %d terms | cid=%s",
                    len(graph_hints.query_expansion_terms), correlation_id,
                )
            # Use KG doc_ids to improve entity-hint filtering
            if graph_hints and graph_hints.doc_ids and not scope.entity_hint:
                # KG knows which docs contain the entities — use as soft hint
                logger.debug("KG provided %d candidate doc_ids | cid=%s", len(graph_hints.doc_ids), correlation_id)
    except Exception as _kg_exc:  # noqa: BLE001
        logger.debug("KG augmentation skipped: %s", _kg_exc)

    # ── Query decomposition for complex queries ──
    # Skip decomposition for specific_document scope — the iterative retriever
    # doesn't pass document_id, so decomposed retrieval would lose scoping.
    stage_start = time.time()
    decomposed = None
    retrieved = None
    if enable_decomposition and rewritten and scope.mode != "specific_document":
        try:
            from src.rag_v3.query_decomposer import decompose_query
            decomposed = decompose_query(rewritten, llm_client=llm_client)
            if len(decomposed.sub_queries) <= 1:
                decomposed = None  # No decomposition needed
        except Exception as exc:
            logger.debug("Query decomposition failed: %s", exc)
            decomposed = None

    if decomposed and len(decomposed.sub_queries) > 1:
        # Multi-strategy retrieval for decomposed queries
        try:
            from src.rag_v3.iterative_retriever import iterative_retrieve
            collection = build_collection_name(subscription_id)
            entity_hints = [sq.entity_scope for sq in decomposed.sub_queries if sq.entity_scope]
            iter_result = iterative_retrieve(
                decomposed,
                collection=collection,
                subscription_id=subscription_id,
                profile_id=profile_id,
                entity_hints=entity_hints or None,
                max_hops=2,  # Keep latency reasonable
                embedder=embedder,
                qdrant_client=qdrant_client,
                correlation_id=correlation_id,
            )
            retrieved = iter_result.chunks
        except Exception as exc:
            logger.warning("Multi-strategy retrieval failed, falling back: %s", exc)
            decomposed = None  # Fall through to standard path

    if decomposed and retrieved:
        _log_stage("decomposed_retrieve", stage_start, correlation_id)
    else:
        stage_start = time.time()
        retrieved = retrieve(
            query=rewritten,
            raw_query=original_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            qdrant_client=qdrant_client,
            embedder=embedder,
            document_id=scope_document_id,
            correlation_id=correlation_id,
            intent_type=intent_type,
        )
        _log_stage("retrieve", stage_start, correlation_id)

    # ── Merge URL-fetched web chunks into retrieval results ────────────
    if _web_url_chunks:
        if not retrieved:
            retrieved = []
        # Convert web dicts to simple chunk-like objects for downstream use
        for _wc in _web_url_chunks:
            _synthetic = type("WebChunk", (), {
                "text": _wc["text"],
                "score": _wc.get("score", 0.5),
                "source": type("Source", (), {
                    "document_name": _wc.get("source_name", ""),
                    "url": _wc.get("url", ""),
                })(),
                "metadata": {"type": "web", "url": _wc.get("url", "")},
            })()
            retrieved.append(_synthetic)

    # ── Document-agnostic agent dispatch ────────────────────────────────
    # Agents like translator, web_search, creator don't require document chunks.
    # Dispatch them before the retrieval-empty guard so they can still produce results.
    _agnostic = [t for t in (tools or []) if t in _DOC_AGNOSTIC_AGENTS]
    if _agnostic and not retrieved:
        _agnostic_chunks = _dispatch_agents(
            tool_names=_agnostic,
            query=original_query,
            profile_id=str(profile_id),
            subscription_id=str(subscription_id),
            tool_inputs=tool_inputs,
            correlation_id=correlation_id,
            chunks=[],
            llm_client=llm_client,
            intent_hint=intent_type,
            domain_hint=None,
        )
        if _agnostic_chunks:
            for tc in _agnostic_chunks:
                if tc.meta and tc.meta.get("source") == "tool_rendered" and tc.text and len(tc.text) > 50:
                    _clean_text = tc.text
                    try:
                        from .enterprise import _NER_LABEL_RE, _NER_INLINE_RE
                        _clean_text = _NER_LABEL_RE.sub("", _clean_text)
                        _clean_text = _NER_INLINE_RE.sub(" ", _clean_text)
                    except Exception:
                        pass
                    return _build_answer(
                        response_text=_clean_text,
                        sources=_collect_sources([]),
                        request_id=request_id,
                        metadata={
                            "domain": None,
                            "intent": intent_type,
                            "intent_type": intent_type,
                            "scope": {"profile_id": profile_id},
                            "quality": "HIGH",
                            "rag_v3": True,
                            "tool_rendered": True,
                            "tool_name": tc.meta.get("tool_name", ""),
                        },
                        query=original_query,
                    )

    if not retrieved:
        # ── Web search fallback when no documents match ────────────────
        _try_web = enable_internet
        if not _try_web:
            try:
                from src.api.config import Config as _WsCfg
                _try_web = getattr(_WsCfg.WebSearch, "FALLBACK_ON_NO_RESULTS", False) and enable_internet
            except Exception:
                pass

        if _try_web:
            try:
                from src.tools.web_search import search_web, format_web_results, build_web_sources
                _web_results = search_web(original_query)
                if _web_results:
                    _web_text = format_web_results(_web_results, query=original_query)
                    _web_sources = build_web_sources(_web_results)
                    logger.info(
                        "Web search fallback returned %d results | cid=%s",
                        len(_web_results), correlation_id,
                    )
                    return _build_answer(
                        response_text=_web_text,
                        sources=_web_sources,
                        request_id=request_id,
                        metadata={
                            "domain": None,
                            "intent": intent_type,
                            "intent_type": intent_type,
                            "scope": {"profile_id": profile_id},
                            "quality": "WEB",
                            "rag_v3": True,
                            "web_search": True,
                            "source_type": "web_search",
                        },
                        query=original_query,
                        include_acknowledgement=False,
                    )
            except Exception as exc:
                logger.debug("Web search fallback failed: %s", exc)

        _scope_label = "specific_document" if scope_document_id else "targeted"
        return _build_answer(
            response_text=_no_results_message(original_query, scope=_scope_label),
            sources=[],
            request_id=request_id,
            metadata={
                "domain": None,
                "intent": None,
                "intent_type": intent_type,
                "scope": {"document_id": scope_document_id} if scope_document_id else {"profile_id": profile_id},
                "quality": "LOW",
                "rag_v3": True,
            },
            query=original_query,
            include_acknowledgement=False,  # No acknowledgement for no-results
        )

    stage_start = time.time()
    # Rerank more chunks for better accuracy - increased from 8 to 16
    reranked = rerank(
        query=rewritten,
        chunks=retrieved,
        cross_encoder=cross_encoder,
        top_k=16,
        correlation_id=correlation_id,
    )
    _log_stage("rerank", stage_start, correlation_id)

    # ── Entity-hint filtering: scope chunks to matching documents only ──
    if scope.entity_hint and reranked:
        filtered_by_entity = _filter_chunks_by_entity_hint(reranked, scope.entity_hint, correlation_id)
        if filtered_by_entity:
            reranked = filtered_by_entity
        else:
            # Entity not found in vector-retrieved chunks — the vector search likely
            # returned semantically-similar but wrong-candidate chunks.
            # Redirect to profile scan to search ALL documents for the entity.
            logger.info(
                "Entity filter for '%s' returned empty in vector results — trying profile scan | cid=%s",
                scope.entity_hint, correlation_id,
            )
            try:
                profile_chunks = expand_full_scan_by_profile(
                    qdrant_client=qdrant_client,
                    collection=build_collection_name(subscription_id),
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    correlation_id=correlation_id,
                )
                if profile_chunks:
                    entity_filtered = _filter_chunks_by_entity_hint(
                        profile_chunks, scope.entity_hint, correlation_id,
                    )
                    if entity_filtered:
                        logger.info(
                            "Entity '%s' found via profile scan: %d chunks | cid=%s",
                            scope.entity_hint, len(entity_filtered), correlation_id,
                        )
                        # Rerank the entity-filtered chunks
                        entity_reranked = rerank(
                            query=original_query, chunks=entity_filtered,
                            cross_encoder=cross_encoder, top_k=16,
                            correlation_id=correlation_id, min_score=-100.0,
                        )
                        reranked = entity_reranked or entity_filtered[:16]
            except Exception:  # noqa: BLE001
                logger.debug("Profile scan fallback for entity '%s' failed", scope.entity_hint)

    # ── KG score boosting: boost chunks supported by knowledge graph ──
    if graph_hints and reranked:
        try:
            from src.kg.score import GraphSupportScorer

            class _ChunkAdapter:
                """Adapt rag_v3 Chunk (.meta) to KG scorer (.metadata)."""
                __slots__ = ("_chunk",)
                def __init__(self, chunk):
                    self._chunk = chunk
                @property
                def text(self):
                    return self._chunk.text
                @property
                def score(self):
                    return self._chunk.score
                @score.setter
                def score(self, value):
                    self._chunk.score = value
                @property
                def metadata(self):
                    return self._chunk.meta or {}
                @metadata.setter
                def metadata(self, value):
                    self._chunk.meta = value

            scorer = GraphSupportScorer(
                alpha=getattr(Config.KnowledgeGraph, "GRAPH_SCORE_ALPHA", 0.7),
            )
            adapted = [_ChunkAdapter(c) for c in reranked]
            scorer.score_chunks(adapted, graph_hints)
            # Scores are written back via the adapter — chunks are now re-sorted
            reranked.sort(key=lambda c: float(c.score), reverse=True)
            logger.info("KG score boosting applied to %d chunks | cid=%s", len(reranked), correlation_id)
        except Exception as _kg_score_exc:  # noqa: BLE001
            logger.debug("KG score boosting skipped: %s", _kg_score_exc)

    sufficiency = None  # Evidence sufficiency — will be set if evaluator succeeds

    # ── Evidence sufficiency gate ──────────────────────────────────────
    try:
        from src.rag_v3.evidence_evaluator import evaluate_evidence
        entity_hints_for_eval = []
        if decomposed:
            entity_hints_for_eval = [sq.entity_scope for sq in decomposed.sub_queries if sq.entity_scope]
        elif scope.entity_hint:
            entity_hints_for_eval = [scope.entity_hint]

        sufficiency = evaluate_evidence(
            rewritten or original_query, reranked, entity_hints=entity_hints_for_eval or None,
        )
        if sufficiency.overall_score < 0.15 and not reranked:
            # Clearly insufficient: no evidence at all
            missing_desc = ""
            if sufficiency.missing_entities:
                missing_desc = f" about {', '.join(sufficiency.missing_entities)}"
            return _build_answer(
                response_text=f"I couldn't find specific information{missing_desc} in the documents.",
                sources=[],
                request_id=request_id,
                metadata={
                    "domain": None,
                    "intent": None,
                    "intent_type": intent_type,
                    "scope": {"document_id": scope_document_id} if scope_document_id else {"profile_id": profile_id},
                    "quality": "LOW",
                    "rag_v3": True,
                    "evidence_insufficient": True,
                    "evidence_score": sufficiency.overall_score,
                },
                query=original_query,
                include_acknowledgement=False,
            )
    except Exception:
        pass  # Evidence gate is best-effort, never blocks pipeline

    # ── Chunk translation: translate content before extraction ──────────
    # When translator is in the tool list, translate chunk text so that
    # downstream extraction/rendering works on translated content.
    _chunks_were_translated = False
    if tools and "translator" in tools and reranked:
        _target_lang = _detect_target_language(original_query)
        if _target_lang:
            reranked = _translate_chunks(reranked, _target_lang, llm_client, correlation_id)
            _chunks_were_translated = True
            # Remove translator from tools list — already handled as content transform
            tools = [t for t in tools if t != "translator"]

    # ── Agent dispatch: enrich context with agent results ──────────────
    if tools:
        tool_chunks = _dispatch_agents(
            tool_names=tools,
            query=original_query,
            profile_id=str(profile_id),
            subscription_id=str(subscription_id),
            tool_inputs=tool_inputs,
            correlation_id=correlation_id,
            chunks=reranked,
            llm_client=llm_client,
            intent_hint=intent_type,
            domain_hint=tool_domain,
        )
        if tool_chunks:
            # Check if any tool produced a direct pre-rendered response
            # (e.g., email drafts, legal summaries) — use it directly
            for tc in tool_chunks:
                if tc.meta and tc.meta.get("source") == "tool_rendered" and tc.text and len(tc.text) > 50:
                    # Apply NER cleanup to tool output
                    _clean_text = tc.text
                    try:
                        from .enterprise import _NER_LABEL_RE, _NER_INLINE_RE
                        _clean_text = _NER_LABEL_RE.sub("", _clean_text)
                        _clean_text = _NER_INLINE_RE.sub(" ", _clean_text)
                    except Exception:
                        pass
                    _tool_meta = {
                        "domain": tc.meta.get("domain") or tool_domain,
                        "intent": intent_type,
                        "intent_type": intent_type,
                        "scope": {"profile_id": profile_id},
                        "quality": "HIGH",
                        "rag_v3": True,
                        "tool_rendered": True,
                        "tool_name": tc.meta.get("tool_name", ""),
                    }
                    return _build_answer(
                        response_text=_clean_text,
                        sources=_collect_sources(reranked),
                        request_id=request_id,
                        metadata=_tool_meta,
                        query=original_query,
                        chunks=reranked,
                    )
            reranked = tool_chunks + reranked  # Tool chunks first (authoritative)

    # ── Domain agent fast-path (targeted retrieval) ─────────────────
    if _agent_detection and reranked:
        _agent_answer = _try_domain_agent(
            query=original_query,
            domain=tool_domain or "",
            chunks=reranked,
            llm_client=llm_client,
            request_id=request_id,
            metadata={
                "intent": intent_type,
                "intent_type": intent_type,
                "scope": {"profile_id": profile_id, "document_id": scope_document_id},
            },
        )
        if _agent_answer is not None:
            return _agent_answer

    # ── Translated-content fast-path ─────────────────────────────────────
    # When chunks were translated, return the translated text directly instead
    # of running through extract_schema (which may force a wrong domain schema).
    if _chunks_were_translated and reranked:
        _translated_text = "\n\n".join(
            c.text.strip() for c in reranked[:12] if c.text and c.text.strip()
        )
        if _translated_text:
            return _build_answer(
                response_text=_translated_text,
                sources=_collect_sources(reranked),
                request_id=request_id,
                metadata={
                    "domain": "generic",
                    "intent": intent_type,
                    "intent_type": intent_type,
                    "scope": {"profile_id": profile_id},
                    "quality": "HIGH",
                    "rag_v3": True,
                    "translated": True,
                },
                query=original_query,
            )

    # ── Unified extraction path ────────────────────────────────────────
    stage_start = time.time()
    document_data = _load_document_data_for_extraction(reranked, original_query, correlation_id)
    reranked = _augment_chunks_with_contact_info(reranked, document_data)

    # Build tool context for LLM-first extraction when a tool domain is active
    _tool_llm_context = None
    if tool_domain and tools:
        try:
            from src.tools.intelligence import build_tool_context_for_llm
            _tool_llm_context = build_tool_context_for_llm(tools, intent_type or "factual")
        except Exception:
            pass

    _targeted_thinking = _should_use_thinking(
        query=original_query,
        intent_parse=intent_parse,
        chunk_count=len(reranked),
        scope_mode=scope.mode,
    )
    # Time budget: skip LLM extraction when pipeline deadline is near
    _tgt_llm = llm_client if _time_remaining(_pipeline_start, _PIPELINE_DEADLINE_S) > 15.0 else None
    _tgt_budget = budget if _tgt_llm else LLMBudget(llm_client=None, max_calls=0)
    extraction = extract_schema(
        tool_domain,  # None when no tool, authoritative domain when tool-specified
        query=original_query,
        chunks=reranked,
        llm_client=_tgt_llm,
        budget=_tgt_budget,
        correlation_id=correlation_id,
        scope_document_id=scope_document_id,
        intent_hint=intent_parse.intent if intent_parse else None,
        document_data=document_data,
        query_focus=focus,
        tool_domain=bool(tool_domain),
        embedder=embedder,
        tool_context=_tool_llm_context,
        intent_parse=intent_parse,
        redis_client=redis_client,
        use_thinking=_targeted_thinking,
    )
    _log_stage("extract", stage_start, correlation_id)

    # Full-scan expansion if the deterministic extraction looks incomplete
    # (skipped when LLM-first extraction already produced a response)
    if not _is_llm_response(extraction):
        needs_full_scan = (
            _needs_full_scan_generic(extraction.schema)
            or (isinstance(extraction.schema, HRSchema) and _needs_full_scan_hr(extraction.schema, extraction.intent))
            or (isinstance(extraction.schema, InvoiceSchema) and _needs_full_scan_invoice(extraction.schema, extraction.intent))
            or (isinstance(extraction.schema, LegalSchema) and _needs_full_scan_legal(extraction.schema, extraction.intent))
        )
        if needs_full_scan:
            stage_start = time.time()
            expanded = expand_full_scan_by_document(
                qdrant_client=qdrant_client,
                collection=build_collection_name(subscription_id),
                base_chunks=reranked,
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
                correlation_id=correlation_id,
            )
            reranked = expanded or reranked
            if _needs_profile_scan(reranked) and not scope_document_id:
                reranked = expand_full_scan_by_profile(
                    qdrant_client=qdrant_client,
                    collection=build_collection_name(subscription_id),
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    correlation_id=correlation_id,
                ) or reranked
            extraction = extract_schema(
                tool_domain or extraction.domain,
                query=original_query,
                chunks=reranked,
                llm_client=None,
                budget=budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
                document_data=document_data,
                query_focus=focus,
                tool_domain=bool(tool_domain),
                embedder=embedder,
                intent_parse=intent_parse,
                use_thinking=_targeted_thinking,
            )
            _log_stage("full_scan_re_extract", stage_start, correlation_id)

    _log_extraction_diagnostics(
        extraction=extraction,
        intent_type=intent_type,
        chunks=reranked,
        correlation_id=correlation_id,
    )
    _maybe_log_hr_schema(extraction, correlation_id)

    # ── Render / Judge ────────────────────────────────────────────────
    stage_start = time.time()
    sanitized, verdict = _extract_render_judge(
        extraction=extraction,
        query=original_query,
        chunks=reranked,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
        query_focus=focus,
        pipeline_start=_pipeline_start,
        pipeline_deadline=_PIPELINE_DEADLINE_S,
    )
    _log_stage("render_judge", stage_start, correlation_id)

    final_answer = sanitized
    if verdict.status == "fail":
        if verdict.reason == "no_sources" and not reranked:
            final_answer = NO_CHUNKS_MESSAGE
        elif verdict.reason == "no_sources" and reranked:
            # Sources exist but judge says no_sources — show chunk content
            final_answer = _emergency_chunk_summary(reranked, original_query) or sanitized or NO_CHUNKS_MESSAGE
            if final_answer != NO_CHUNKS_MESSAGE:
                verdict = JudgeResult(status="pass", reason="chunk_content_fallback")
        elif not _is_llm_response(extraction):
            retry_answer = _retry_render(extraction, correlation_id, query=original_query)
            retry_sanitized = sanitize(retry_answer)
            retry_verdict = judge(
                answer=retry_sanitized,
                schema=extraction.schema,
                intent=extraction.intent,
                llm_client=None,
                budget=budget,
                sources_present=bool(reranked),
                correlation_id=correlation_id,
            )
            if retry_verdict.status == "pass":
                final_answer = retry_sanitized
                verdict = retry_verdict
            else:
                # Keep the original sanitized answer if it has content —
                # a partial answer is better than a stone-wall refusal
                if sanitized and sanitized.strip():
                    final_answer = sanitized
                    verdict = JudgeResult(status="pass", reason="content_preserved")
                elif reranked:
                    final_answer = _emergency_chunk_summary(reranked, original_query) or NO_CHUNKS_MESSAGE
                    if final_answer != NO_CHUNKS_MESSAGE:
                        verdict = JudgeResult(status="pass", reason="chunk_content_fallback")
                else:
                    final_answer = NO_CHUNKS_MESSAGE
                    verdict = JudgeResult(status="fail", reason="retry_failed")
        else:
            # Keep LLM response if it has content, even if judge rejected it
            if sanitized and sanitized.strip():
                final_answer = sanitized
                verdict = JudgeResult(status="pass", reason="llm_content_preserved")
            elif reranked:
                final_answer = _emergency_chunk_summary(reranked, original_query) or NO_CHUNKS_MESSAGE
                if final_answer != NO_CHUNKS_MESSAGE:
                    verdict = JudgeResult(status="pass", reason="chunk_content_fallback")
                else:
                    verdict = JudgeResult(status="fail", reason="llm_response_rejected")
            else:
                final_answer = NO_CHUNKS_MESSAGE
                verdict = JudgeResult(status="fail", reason="llm_response_rejected")

    sources = _collect_sources(reranked)
    # Fallback: if no sources from chunks but we have document_data, use its metadata
    if not sources and document_data:
        doc_meta = document_data.get("metadata", {}) if isinstance(document_data, dict) else {}
        doc_name = (doc_meta.get("file_name") or doc_meta.get("source_name") or "Document")
        sources = [{"file_name": doc_name, "page": 1, "snippet": ""}]
    metadata = {
        "domain": extraction.domain,
        "intent": extraction.intent,
        "intent_type": intent_type,
        "scope": {"document_id": scope_document_id} if scope_document_id else {"profile_id": profile_id},
        "quality": "HIGH" if verdict.status == "pass" else "LOW",
        "rag_v3": True,
        "judge": {"status": verdict.status, "reason": verdict.reason},
        "thinking_used": getattr(extraction.schema, "thinking_used", False) if extraction.schema else False,
    }

    # Evidence-based grounding annotation
    if sufficiency is not None and sufficiency.overall_score < 0.35:
        metadata["grounded"] = False
        metadata["evidence_score"] = sufficiency.overall_score

    # KG debug metadata
    if graph_hints:
        kg_debug = {}
        if graph_hints.entities_in_query:
            kg_debug["entities"] = [e.name for e in graph_hints.entities_in_query[:5]]
        if graph_hints.query_expansion_terms:
            kg_debug["expansion_terms"] = graph_hints.query_expansion_terms[:5]
        if graph_hints.doc_ids:
            kg_debug["doc_ids"] = graph_hints.doc_ids[:5]
        if kg_debug:
            metadata["kg"] = kg_debug

    # ── Multi-agent verifier (advisory, never blocks, disabled by default) ──
    try:
        from src.api.config import Config as _VCfg
        _verification_enabled = getattr(getattr(_VCfg, "Verification", None), "ENABLED", False)
        if _verification_enabled and _time_remaining(_pipeline_start, _PIPELINE_DEADLINE_S) > 12.0:
            _v_state = _get_state() if "_get_state" in dir() else None
            if _v_state is None:
                from src.api.rag_state import get_app_state as _get_v_state
                _v_state = _get_v_state()
            _v_gw = getattr(_v_state, "multi_agent_gateway", None) if _v_state else None
            if (
                _v_gw is not None
                and getattr(_VCfg, "MultiAgent", None)
                and getattr(_VCfg.MultiAgent, "ENABLED", False)
                and getattr(_VCfg.MultiAgent, "VERIFIER_ENABLED", True)
                and final_answer
                and final_answer not in (FALLBACK_ANSWER, NO_CHUNKS_MESSAGE)
                and reranked
            ):
                from src.llm.verifier import verify_grounding
                _v_timeout = min(getattr(_VCfg.MultiAgent, "VERIFIER_TIMEOUT", 10.0), 10.0)
                try:
                    from src.llm.task_router import task_scope as _ts_verify, TaskType as _TT_verify
                    with _ts_verify(_TT_verify.GROUNDING_VERIFY):
                        _v_result = verify_grounding(
                            answer=final_answer,
                            evidence_chunks=reranked,
                            query=original_query,
                            llm_client=_v_gw,
                            timeout_s=_v_timeout,
                        )
                except ImportError:
                    _v_result = verify_grounding(
                        answer=final_answer,
                        evidence_chunks=reranked,
                        query=original_query,
                        llm_client=_v_gw,
                        timeout_s=_v_timeout,
                    )
                if _v_result:
                    metadata["verifier"] = _v_result.to_dict()
                    if not _v_result.supported:
                        logger.warning(
                            "Verifier flagged answer as unsupported: issues=%s | cid=%s",
                            _v_result.issues, correlation_id,
                        )
    except Exception as _v_exc:
        logger.debug("Multi-agent verifier skipped: %s", _v_exc)

    # ── Web search post-hoc fallback ─────────────────────────────────
    _web_result = _web_search_fallback(
        final_answer, original_query, enable_internet,
        metadata, request_id, correlation_id,
    )
    if _web_result:
        return _web_result

    return _build_answer(
        response_text=final_answer,
        sources=sources,
        request_id=request_id,
        metadata=metadata,
        query=original_query,
        chunks=reranked,
        schema=extraction.schema,
        llm_client=llm_client,
    )


def run_docwain_rag_v3(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    session_id: Optional[str],
    user_id: str,
    request_id: Optional[str],
    llm_client: Optional[Any],
    qdrant_client: Any,
    redis_client: Optional[Any],
    embedder: Any,
    cross_encoder: Optional[Any] = None,
    document_id: Optional[str] = None,
    tools: Optional[List[str]] = None,
    tool_inputs: Optional[Dict[str, Any]] = None,
    enable_decomposition: bool = True,
    enable_internet: bool = False,
) -> Dict[str, Any]:
    return run(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
        tool_hint=None,
        session_id=session_id,
        user_id=user_id,
        request_id=request_id,
        llm_client=llm_client,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        tools=tools,
        tool_inputs=tool_inputs,
        enable_decomposition=enable_decomposition,
        enable_internet=enable_internet,
    )


def _retry_render(extraction: Any, correlation_id: Optional[str], query: str = "") -> str:
    try:
        return render(domain=extraction.domain, intent=extraction.intent, schema=extraction.schema, strict=True, query=query)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 retry render failed: %s",
            exc,
            extra={"stage": "render_retry", "correlation_id": correlation_id},
        )
        return ""


def _collect_retrieved_metadata(chunks: List[Chunk]) -> Dict[str, List[str]]:
    doc_types = set()
    doc_domains = set()
    document_ids = set()
    document_names = set()
    for chunk in chunks or []:
        meta = chunk.meta or {}
        doc_type = meta.get("doc_type") or meta.get("document.type") or meta.get("document_type")
        if doc_type:
            doc_types.add(str(doc_type).lower())
        doc_domain = meta.get("doc_domain") or meta.get("doc_type")
        if doc_domain:
            doc_domains.add(str(doc_domain).lower())
        doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
        if doc_id:
            document_ids.add(str(doc_id))
        doc_name = meta.get("source_name") or meta.get("document_name") or getattr(chunk.source, "document_name", None)
        if doc_name:
            document_names.add(str(doc_name))
    return {
        "doc_types": sorted(doc_types),
        "doc_domains": sorted(doc_domains),
        "document_ids": sorted(document_ids),
        "document_names": sorted(document_names),
    }


def _log_extraction_diagnostics(
    *,
    extraction: Any,
    intent_type: str,
    chunks: List[Chunk],
    correlation_id: Optional[str],
) -> None:
    if not Config.RagV3.DEBUG_LOGS:
        return
    domain = getattr(extraction, "domain", None)
    intent = getattr(extraction, "intent", None)
    schema = getattr(extraction, "schema", None)
    candidate_count = 0
    entity_count = 0
    if isinstance(schema, HRSchema):
        candidate_count = len((schema.candidates.items if schema.candidates else None) or [])
    if isinstance(schema, MultiEntitySchema):
        entity_count = len(schema.entities or [])
    doc_ids = {str(_chunk_document_id(chunk)) for chunk in chunks if _chunk_document_id(chunk)}
    logger.info(
        "RAG v3 extraction summary",
        extra={
            "stage": "extract_summary",
            "correlation_id": correlation_id,
            "domain": domain,
            "intent": intent,
            "intent_type": intent_type,
            "candidate_count": candidate_count,
            "entity_count": entity_count,
            "doc_count": len(doc_ids),
        },
    )
    if isinstance(schema, HRSchema):
        coverage = _hr_field_coverage(schema)
        logger.info(
            "RAG v3 HR field coverage",
            extra={
                "stage": "extract_hr_coverage",
                "correlation_id": correlation_id,
                "coverage": coverage,
            },
        )


def _maybe_log_hr_schema(extraction: Any, correlation_id: Optional[str]) -> None:
    if not Config.RagV3.DEBUG_SCHEMA:
        return
    schema = getattr(extraction, "schema", None)
    if not isinstance(schema, HRSchema):
        return
    redacted = _redact_hr_schema(schema)
    payload = json.dumps(redacted, ensure_ascii=True)
    payload = _truncate(payload, 4000)
    logger.info(
        "RAG v3 HR schema (redacted): %s",
        payload,
        extra={"stage": "extract_hr_schema", "correlation_id": correlation_id},
    )


def _hr_field_coverage(schema: HRSchema) -> Dict[str, Dict[str, int]]:
    fields = {
        "name": lambda c: bool(c.name),
        "total_years_experience": lambda c: bool(c.total_years_experience),
        "experience_summary": lambda c: bool(c.experience_summary),
        "technical_skills": lambda c: bool(c.technical_skills),
        "functional_skills": lambda c: bool(c.functional_skills),
        "certifications": lambda c: bool(c.certifications),
        "education": lambda c: bool(c.education),
        "achievements": lambda c: bool(c.achievements),
        "source_type": lambda c: bool(c.source_type),
    }
    coverage: Dict[str, Dict[str, int]] = {key: {"present": 0, "missing": 0} for key in fields}
    candidates = (schema.candidates.items if schema.candidates else None) or []
    for cand in candidates:
        for key, predicate in fields.items():
            if predicate(cand):
                coverage[key]["present"] += 1
            else:
                coverage[key]["missing"] += 1
    return coverage


def _redact_hr_schema(schema: HRSchema) -> Dict[str, Any]:
    data = schema.model_dump()
    candidates = ((data.get("candidates") or {}).get("items")) or []
    for idx, cand in enumerate(candidates, start=1):
        cand["name"] = f"Candidate {idx}"
        for key in (
            "role",
            "details",
            "total_years_experience",
            "experience_summary",
            "source_type",
        ):
            if cand.get(key):
                cand[key] = _redact_text(cand[key])
        for key in (
            "technical_skills",
            "functional_skills",
            "certifications",
            "education",
            "achievements",
        ):
            items = cand.get(key) or []
            cand[key] = [_redact_text(item) for item in items][:10]
        cand["evidence_spans"] = []
    return data


def _redact_text(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", "[email]", cleaned)
    cleaned = re.sub(r"\\+?\\d[\\d\\-\\s().]{6,}\\d", "[phone]", cleaned)
    cleaned = re.sub(r"\\b[A-Fa-f0-9]{8,}\\b", "[id]", cleaned)
    return cleaned


def _truncate(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _chunk_document_id(chunk: Chunk) -> Optional[str]:
    meta = chunk.meta or {}
    for key in ("document_id", "doc_id", "docId"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


# Tech abbreviation dictionary — maps stack acronyms to individual component terms
_TECH_ACRONYM_EXPANSIONS: Dict[str, List[str]] = {
    "mern": ["mongodb", "express", "react", "node"],
    "mean": ["mongodb", "express", "angular", "node"],
    "lamp": ["linux", "apache", "mysql", "php"],
    "lemp": ["linux", "nginx", "mysql", "php"],
    "mevn": ["mongodb", "express", "vue", "node"],
    "pern": ["postgresql", "express", "react", "node"],
    "jamstack": ["javascript", "api", "markup"],
    "devops": ["docker", "kubernetes", "ci/cd", "jenkins", "terraform"],
}

# Common tech terms that should NOT be treated as person names
_TECH_TERMS = frozenset({
    "mern", "mean", "lamp", "lemp", "mevn", "pern", "java", "react", "angular",
    "python", "django", "flask", "node", "rust", "swift", "kafka", "redis",
    "docker", "kubernetes", "terraform", "jenkins", "agile", "scrum", "devops",
    "jamstack", "graphql", "typescript", "golang", "spark", "hadoop",
})


def _is_tech_term(hint: str) -> bool:
    return hint.lower().strip() in _TECH_TERMS


def _expand_tech_hint(hint: str) -> List[str]:
    """Return expanded search terms for a tech abbreviation."""
    lower = hint.lower().strip()
    terms = [lower]
    if lower in _TECH_ACRONYM_EXPANSIONS:
        terms.extend(_TECH_ACRONYM_EXPANSIONS[lower])
    return terms


def _filter_chunks_by_entity_hint(
    chunks: List[Chunk],
    entity_hint: str,
    correlation_id: Optional[str] = None,
) -> List[Chunk]:
    """Filter chunks to only documents that mention the entity_hint in text, source, or metadata.

    Uses 3-tier matching:
      1. Exact substring (fastest)
      2. Fuzzy matching via rapidfuzz (handles OCR typos, name reordering)
      3. Token-level matching (handles split names like "Jaya Kumar" → "Jayakumar")

    Returns empty list when entity is not found in any chunk (caller handles empty).
    """
    # Graceful import — if rapidfuzz not installed, skip Tier 2
    try:
        from rapidfuzz.fuzz import token_sort_ratio, partial_ratio
        _has_rapidfuzz = True
    except ImportError:
        _has_rapidfuzz = False

    is_tech = _is_tech_term(entity_hint)
    search_terms = _expand_tech_hint(entity_hint) if is_tech else [entity_hint.lower()]
    hint_tokens = set(entity_hint.lower().split())

    # Find which doc_ids contain the entity name
    entity_doc_ids: set = set()
    for chunk in chunks:
        doc_id = _chunk_document_id(chunk) or ""
        if doc_id in entity_doc_ids:
            continue  # Already matched this doc

        text_lower = ((getattr(chunk, "text", "") or "").lower())
        source = getattr(chunk, "source", None)
        doc_name_lower = ((getattr(source, "document_name", "") or "").lower()) if source else ""
        meta = getattr(chunk, "meta", None) or {}
        meta_source = (str(meta.get("source_name") or "")).lower()
        meta_filename = (str(meta.get("filename") or "")).lower()
        meta_embed = (str(meta.get("embedding_text") or "")).lower()
        meta_doc_summary = (str(meta.get("doc_summary") or "")).lower()
        meta_sec_summary = (str(meta.get("section_summary") or "")).lower()
        searchable = (
            f"{text_lower} {doc_name_lower} {meta_source} {meta_filename} "
            f"{meta_embed} {meta_doc_summary} {meta_sec_summary}"
        )

        matched = False

        # Tier 1 — Exact substring (fastest)
        for term in search_terms:
            if term in searchable:
                matched = True
                break

        # Tier 2 — Fuzzy matching (OCR typos, name reordering)
        if not matched and _has_rapidfuzz and not is_tech:
            hint_lower = entity_hint.lower()
            # Limit fuzzy search to first 2000 chars for performance
            fuzzy_text = searchable[:2000]
            if token_sort_ratio(hint_lower, fuzzy_text) >= 85:
                matched = True
            elif partial_ratio(hint_lower, fuzzy_text) >= 88:
                matched = True

        # Tier 3 — Token-level matching (split names: "Jaya Kumar" → "Jayakumar")
        if not matched and len(hint_tokens) >= 2:
            if all(tok in searchable for tok in hint_tokens):
                matched = True
            else:
                # Check concatenated form (e.g., "jayakumar" from "jaya" + "kumar")
                concat = "".join(sorted(hint_tokens))
                if concat in searchable:
                    matched = True

        if matched:
            entity_doc_ids.add(doc_id)

    if not entity_doc_ids:
        hint_type = "technology" if is_tech else "person/entity"
        logger.info(
            "Entity hint '%s' (type=%s) not found in any chunk text/metadata — returning empty",
            entity_hint, hint_type,
            extra={"stage": "entity_filter", "correlation_id": correlation_id},
        )
        return []
    filtered = [c for c in chunks if (_chunk_document_id(c) or "") in entity_doc_ids]
    if filtered:
        logger.info(
            "Entity hint '%s' filtered chunks: %d → %d (docs: %s)",
            entity_hint, len(chunks), len(filtered), entity_doc_ids,
            extra={"stage": "entity_filter", "correlation_id": correlation_id},
        )
        return filtered
    return []


def _build_generic_schema_from_chunks(chunks: List[Chunk]) -> GenericSchema:
    items: List[FieldValue] = []
    for chunk in chunks or []:
        snippet = _snippet(chunk.text)
        if not snippet:
            continue
        items.append(
            FieldValue(
                label=None,
                value=snippet,
                evidence_spans=[EvidenceSpan(chunk_id=chunk.id, snippet=snippet)],
            )
        )
    if not items:
        return GenericSchema(facts=FieldValuesField(items=None, missing_reason=MISSING_REASON))
    return GenericSchema(facts=FieldValuesField(items=items))


def _collect_sources(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    for chunk in chunks:
        doc = chunk.source.document_name
        page = chunk.source.page
        snippet = _snippet(chunk.text)
        key = (doc, page, snippet)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"file_name": doc, "page": page, "snippet": snippet})
    return sources


def _synthesize_cross_document(
    schemas: List[Any],
    doc_contexts: List[Any],
    query: str,
    domain: str,
) -> str:
    """Deterministic cross-document synthesis paragraph. No LLM calls.

    Computes statistics and patterns from already-extracted structured data
    to produce an analytical summary paragraph.
    """
    from collections import Counter

    n_docs = len(doc_contexts) if doc_contexts else len(schemas)
    if n_docs < 2:
        return ""

    parts: List[str] = []

    if domain == "hr":
        # HR synthesis: candidate stats from schemas
        all_skills: Counter = Counter()
        years_values: List[float] = []
        for schema in schemas:
            if not hasattr(schema, "candidates"):
                continue
            cands = (schema.candidates.items if schema.candidates else None) or []
            for c in cands:
                for s in (c.technical_skills or []):
                    all_skills[s.lower()] += 1
                for s in (c.functional_skills or []):
                    all_skills[s.lower()] += 1
                if c.total_years_experience:
                    try:
                        y = float(c.total_years_experience.split()[0])
                        years_values.append(y)
                    except (ValueError, IndexError):
                        pass

        if years_values:
            avg = sum(years_values) / len(years_values)
            parts.append(
                f"Across {n_docs} candidates, experience ranges from "
                f"{min(years_values):.0f} to {max(years_values):.0f} years "
                f"(average {avg:.1f})."
            )

        if all_skills:
            shared = [s for s, c in all_skills.most_common(5) if c >= 2]
            if shared:
                parts.append(f"Shared skills: {', '.join(shared)}.")
            unique = [s for s, c in all_skills.items() if c == 1]
            if unique and len(unique) <= 5:
                parts.append(f"Distinctive skills: {', '.join(unique[:5])}.")

    elif domain == "invoice":
        # Invoice synthesis: sums and vendor frequency
        import re as _re
        total_amounts: List[float] = []
        vendors: Counter = Counter()
        for schema in schemas:
            totals_items = (getattr(schema, "totals", None) and schema.totals.items) or []
            for item in totals_items:
                nums = _re.findall(r'[\d,]+(?:\.\d+)?', (item.value or "").replace(",", ""))
                for n in nums:
                    try:
                        total_amounts.append(float(n))
                    except ValueError:
                        pass
            parties_items = (getattr(schema, "parties", None) and schema.parties.items) or []
            for item in parties_items:
                if item.value:
                    vendors[item.value.strip()] += 1

        if total_amounts:
            parts.append(
                f"Across {n_docs} invoices: total sum {sum(total_amounts):,.2f}, "
                f"average {sum(total_amounts)/len(total_amounts):,.2f}, "
                f"range {min(total_amounts):,.2f}–{max(total_amounts):,.2f}."
            )
        if vendors:
            top_vendors = [v for v, _ in vendors.most_common(3)]
            parts.append(f"Vendors: {', '.join(top_vendors)}.")

    else:
        # Generic synthesis: document count and shared themes
        from collections import Counter as _Counter
        themes: _Counter = _Counter()
        for schema in schemas:
            facts_items = []
            if hasattr(schema, "facts") and schema.facts:
                facts_items = schema.facts.items or []
            for f in facts_items:
                if f.label:
                    themes[f.label.lower()] += 1

        if themes:
            common = [t for t, c in themes.most_common(3) if c >= 2]
            if common:
                parts.append(f"Across {n_docs} documents, shared themes: {', '.join(common)}.")

    return " ".join(parts)


def _llm_synthesize(
    rendered_text: str,
    query: str,
    domain: str,
    intent: str,
    num_documents: int,
    llm_client: Any,
    budget: LLMBudget,
    correlation_id: Optional[str],
) -> Optional[str]:
    """Post-process rendered text through an LLM for analytical synthesis.

    Has its own budget (max_calls=1), separate timeout, and graceful fallback.
    Returns None on failure (caller keeps original text).
    """
    import concurrent.futures as _cf

    try:
        from src.api.config import Config as _SCfg
        _synth_enabled = getattr(getattr(_SCfg, "Synthesis", None), "ENABLED", False)
        _synth_timeout = getattr(getattr(_SCfg, "Synthesis", None), "TIMEOUT", 20.0)
        _synth_min_docs = getattr(getattr(_SCfg, "Synthesis", None), "MIN_DOCUMENTS", 2)
    except Exception:
        return None

    if not _synth_enabled or num_documents < _synth_min_docs:
        return None

    if not llm_client or not budget.allow():
        return None

    budget.consume()

    prompt = (
        "You are an analytical document intelligence assistant. "
        "The following is extracted data from multiple documents. "
        "Synthesize it into an intelligent analytical response. "
        "Highlight patterns, outliers, and key findings. "
        "Start with a synthesis statement, then provide details.\n\n"
        f"QUERY: {query}\n"
        f"DOMAIN: {domain}\n"
        f"DOCUMENTS ANALYZED: {num_documents}\n\n"
        f"EXTRACTED DATA:\n{rendered_text[:8000]}\n\n"
        "Provide a well-structured analytical response. "
        "Preserve all factual details from the extracted data. "
        "Add statistical observations and pattern analysis."
    )

    def _call() -> str:
        if hasattr(llm_client, "generate_with_metadata"):
            text, _ = llm_client.generate_with_metadata(
                prompt,
                options={"num_predict": 2048, "num_ctx": 4096},
                max_retries=1,
                backoff=0.3,
            )
            return text or ""
        return llm_client.generate(prompt, max_retries=1, backoff=0.3) or ""

    executor = _cf.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        result = future.result(timeout=_synth_timeout)
        if result and len(result.strip()) > 50:
            synthesized = result.strip()
            # Guard: don't replace a long deterministic response with a
            # shorter LLM synthesis — the LLM may have truncated details.
            if len(synthesized) < len(rendered_text) * 0.6:
                logger.info(
                    "LLM synthesis too short (%d vs %d orig) — keeping original",
                    len(synthesized), len(rendered_text),
                    extra={"stage": "llm_synthesis", "correlation_id": correlation_id},
                )
                return None
            logger.info(
                "LLM synthesis produced %d chars for %d-doc query",
                len(synthesized), num_documents,
                extra={"stage": "llm_synthesis", "correlation_id": correlation_id},
            )
            return synthesized
    except (_cf.TimeoutError, Exception) as exc:
        future.cancel()
        logger.debug("LLM synthesis failed/timed out: %s", exc)
    finally:
        executor.shutdown(wait=False)

    return None


def _should_use_thinking(
    query: str,
    intent_parse: Any,
    chunk_count: int,
    scope_mode: str,
) -> bool:
    """Decide whether to use thinking/reasoning mode for this query.

    Only triggers for genuinely complex multi-document reasoning tasks.
    Simple queries, summaries, and factual lookups do NOT need thinking mode.
    """
    intent = ""
    if intent_parse:
        intent = getattr(intent_parse, "intent", "") or ""

    # Only activate for explicit comparison/ranking across many documents
    heavy_intents = {"compare", "rank", "ranking", "comparison", "cross_document"}
    if intent in heavy_intents and chunk_count >= 8:
        return True

    # Very high chunk count + analytical intent
    if chunk_count > 20 and intent in heavy_intents | {"analytics", "reasoning"}:
        return True

    # Cross-document comparison scope (NOT summary — summary is fast-path)
    if scope_mode == "all_profile" and intent in {"compare", "rank"} and chunk_count >= 8:
        return True

    return False


def _build_answer(
    response_text: str,
    sources: List[Dict[str, Any]],
    request_id: Optional[str],
    metadata: Dict[str, Any],
    query: Optional[str] = None,
    include_acknowledgement: bool = False,
    chunks: Optional[List[Any]] = None,
    llm_client: Optional[Any] = None,
    schema: Any = None,
) -> Dict[str, Any]:
    """
    Build a RAG answer response with optional acknowledgement.

    Args:
        response_text: The answer text.
        sources: List of source documents.
        request_id: Request correlation ID.
        metadata: Additional metadata.
        query: Optional original query for acknowledgement generation.
        include_acknowledgement: Whether to include acknowledgement prefix.

    Returns:
        Response dictionary with answer and metadata.
    """
    final_response = response_text
    acknowledgement = None

    # Add acknowledgement if query is provided and response is valid
    _is_refusal = (
        response_text in (FALLBACK_ANSWER, NO_CHUNKS_MESSAGE)
        or (response_text and response_text.startswith("I couldn't find"))
    )
    # Skip acknowledgement for structured responses that are already direct answers
    # (rankings, comparisons, tables — these start with markdown headers)
    _is_structured = response_text and (
        response_text.startswith("**Top ")
        or response_text.startswith("**Ranking ")
        or response_text.startswith("**Comparison")
        or response_text.startswith("Here is the ranking")
        or response_text.startswith("| ")
    )
    if include_acknowledgement and query and response_text and not _is_refusal and not _is_structured:
        try:
            from src.intelligence.response_formatter import ResponseFormatter

            formatter = ResponseFormatter(include_confidence_note=False)
            formatted = formatter.format_response(
                query=query,
                content=response_text,
                sources=[s.get("file_name", "") for s in sources] if sources else None,
            )
            acknowledgement = formatted.acknowledgement
            # Prepend acknowledgement to response
            final_response = f"{acknowledgement}\n\n{response_text}"
            metadata["acknowledgement"] = acknowledgement
            metadata["query_intent"] = formatted.intent.value
        except Exception:
            # Fallback: don't modify response if formatter fails
            pass

    # ── Enterprise intelligence: follow-ups and confidence (parallel) ─
    chunk_texts = []
    if chunks:
        chunk_texts = [c.text for c in chunks if hasattr(c, "text") and c.text]

    def _compute_followups():
        try:
            from src.api.config import Config as _FUCfg
            _fu_enabled = getattr(getattr(_FUCfg, "FollowUp", None), "ENABLED", False)
            if _fu_enabled and query and final_response:
                from src.intelligence.followup_engine import generate_followups
                _fu_max = getattr(_FUCfg.FollowUp, "MAX_SUGGESTIONS", 3)
                _fu_timeout = getattr(_FUCfg.FollowUp, "LLM_TIMEOUT", 3.0)
                return generate_followups(
                    query=query,
                    response=final_response,
                    chunk_texts=chunk_texts or None,
                    domain=metadata.get("domain"),
                    intent_type=metadata.get("intent_type") or metadata.get("intent"),
                    llm_client=llm_client,
                    max_count=_fu_max,
                    llm_timeout=_fu_timeout,
                )
        except Exception as exc:
            logger.debug("Follow-up suggestions skipped: %s", exc)
        return None

    def _compute_confidence():
        try:
            from src.api.config import Config as _ConfCfg
            _conf_enabled = getattr(getattr(_ConfCfg, "Confidence", None), "ENABLED", False)
            if _conf_enabled and final_response:
                from src.intelligence.confidence_scorer import compute_confidence
                judge_meta_for_conf = metadata.get("judge", {})
                verdict_status = judge_meta_for_conf.get("status") if isinstance(judge_meta_for_conf, dict) else None
                return compute_confidence(
                    response=final_response,
                    chunk_texts=chunk_texts,
                    sources=sources,
                    schema=schema,
                    verdict_status=verdict_status,
                    domain=metadata.get("domain"),
                )
        except Exception as exc:
            logger.debug("Confidence scoring skipped: %s", exc)
        return None

    # Run follow-ups and confidence in parallel (non-critical — failures are silenced)
    followups = None
    confidence = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _intel_pool:
            _fu_future = _intel_pool.submit(_compute_followups)
            _conf_future = _intel_pool.submit(_compute_confidence)

            try:
                followups = _fu_future.result(timeout=1.5)
            except (TimeoutError, concurrent.futures.TimeoutError, Exception):
                _fu_future.cancel()
            try:
                confidence = _conf_future.result(timeout=1.5)
            except (TimeoutError, concurrent.futures.TimeoutError, Exception):
                _conf_future.cancel()
    except Exception as _intel_exc:
        logger.debug("Intelligence enrichment skipped: %s", _intel_exc)

    if followups:
        metadata["suggested_followups"] = followups
    if confidence:
        metadata["confidence"] = confidence.to_dict()

    # Domain knowledge metadata enrichment
    try:
        from src.api.config import Config as _DKCfg
        _dk_enabled = getattr(getattr(_DKCfg, "DomainKnowledge", None), "ENABLED", False)
        _resp_domain = metadata.get("domain", "")
        if _dk_enabled and _resp_domain:
            from src.intelligence.domain_knowledge import get_domain_knowledge_provider
            _dk_provider = get_domain_knowledge_provider()
            # For medical domain, detect region from chunks
            if _resp_domain in ("medical", "clinical") and chunk_texts:
                _sample_text = " ".join(chunk_texts[:5])[:8000]
                _code, _name, _region_ctx = _dk_provider.get_medical_region_context(_sample_text)
                metadata["domain_knowledge"] = {
                    "domain": _resp_domain,
                    "region_detected": _name,
                    "region_code": _code,
                }
            else:
                metadata["domain_knowledge"] = {"domain": _resp_domain}
    except Exception as _dk_exc:
        logger.debug("Domain knowledge metadata skipped: %s", _dk_exc)

    # ── Multi-resolution metadata enrichment ────────────────────────────────
    # When retrieved chunks carry schema_completeness or resolution-level metadata,
    # surface these into the debug/metadata dict so callers can inspect coverage.
    if chunks:
        try:
            _resolution_levels: set = set()
            _completeness_values: list = []
            for _rc in chunks:
                _rm = getattr(_rc, "meta", None) or {}
                _res = str(_rm.get("resolution") or _rm.get("chunk_resolution") or "").lower().strip()
                _ck = str(_rm.get("chunk_kind") or _rm.get("chunk_type") or "").lower().strip()
                if _res:
                    _resolution_levels.add(_res)
                elif _ck in ("doc_summary", "document_summary"):
                    _resolution_levels.add("doc")
                elif _ck == "summary":
                    _resolution_levels.add("section")
                _sc = _rm.get("schema_completeness")
                if _sc is not None:
                    try:
                        _completeness_values.append(float(_sc))
                    except (TypeError, ValueError):
                        pass
            if _resolution_levels:
                metadata["resolution_levels_used"] = sorted(_resolution_levels)
            if _completeness_values:
                _avg_completeness = sum(_completeness_values) / len(_completeness_values)
                metadata["schema_completeness"] = round(_avg_completeness, 3)
        except Exception as _mr_meta_exc:
            logger.debug("Multi-resolution metadata skipped: %s", _mr_meta_exc)

    # Query-adaptive keyword reinforcement: ensure response echoes key query terms
    # This catches cases where render_enterprise() output was sanitized or rebuilt
    if final_response and query:
        _ql_ba = (query or "").lower()
        _rl_ba = (final_response or "").lower()
        _domain_prefixes = {
            "vendor": "Vendor information from the invoices:\n\n",
            "invoice": "Invoice details:\n\n",
            "payment": "Payment information from the invoices:\n\n",
            "patient": "Patient information from the medical records:\n\n",
            "medication": "Medication details from the medical records:\n\n",
            "insurance": "Insurance policy details:\n\n",
            "candidate": "Candidate information from the resumes:\n\n",
            "premium": "Premium and payment details from the insurance policies:\n\n",
            "treatment": "Treatment and procedure details from the medical records:\n\n",
            "procedure": "Procedure details from the medical records:\n\n",
            "exclusion": "Exclusion details from the insurance policies:\n\n",
        }
        for _kw, _prefix in _domain_prefixes.items():
            if _kw in _ql_ba and _kw not in _rl_ba and len(final_response) > 50:
                final_response = f"{_prefix}{final_response}"
                break  # Only add one prefix

        # ── Query-term echo: ensure critical query nouns appear in response ──
        # Extract important query terms (nouns/skills) and check presence.
        # If missing, prepend a context line that echoes the query focus.
        import re as _re_ba
        _STOP_WORDS = frozenset({
            "what", "which", "who", "how", "when", "where", "why", "are",
            "is", "the", "a", "an", "in", "on", "for", "of", "and", "or",
            "all", "do", "does", "did", "has", "have", "had", "been", "be",
            "with", "from", "to", "by", "that", "this", "these", "those",
            "can", "could", "would", "should", "may", "might", "will",
            "me", "my", "i", "you", "your", "it", "its", "they", "them",
            "their", "we", "our", "most", "more", "any", "some", "find",
            "list", "show", "tell", "get", "give", "know", "across",
            "mentioned", "available", "than", "about",
        })
        _query_terms = [
            w for w in _re_ba.findall(r"[a-zA-Z]+", _ql_ba)
            if len(w) >= 3 and w not in _STOP_WORDS
        ]
        _missing_terms = [t for t in _query_terms if t not in _rl_ba]
        if _missing_terms and len(final_response) > 50:
            # Echo missing query terms so the response clearly addresses the question
            _echo_phrase = " ".join(_missing_terms[:4])
            if _echo_phrase:
                final_response = f"Regarding {_echo_phrase}: {final_response}"

    result = {
        "response": final_response,
        "sources": sources,
        "request_id": request_id,
        "context_found": bool(sources),
        "grounded": bool(sources),
        "metadata": metadata,
    }

    # Record quality metrics (best-effort, never blocks response)
    try:
        judge_meta = metadata.get("judge", {})
        llm_backend = "unknown"
        try:
            from src.llm.gateway import get_llm_gateway
            gw = get_llm_gateway()
            if gw:
                llm_backend = getattr(gw, "backend", "unknown")
        except Exception:
            pass
        record_query_metrics(QueryMetrics(
            query=query or "",
            latency_ms=metadata.get("_latency_ms", 0.0),
            retrieval_count=len(sources),
            entity_detected=bool(metadata.get("scope", {}).get("document_id")),
            llm_backend=llm_backend,
            answer_length=len(final_response),
            grounding_score=metadata.get("evidence_score", 0.0),
            judge_verdict=judge_meta.get("status", "unknown") if isinstance(judge_meta, dict) else "unknown",
            scope_mode=metadata.get("scope_mode", "unknown"),
            domain=metadata.get("domain") or "unknown",
        ))
    except Exception:
        pass  # Metrics never block pipeline

    return result


def _snippet(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()


def _log_stage(stage: str, start_time: float, correlation_id: Optional[str]) -> None:
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "RAG v3 stage %s completed in %.2f ms",
        stage,
        elapsed_ms,
        extra={"stage": stage, "correlation_id": correlation_id, "latency_ms": elapsed_ms},
    )


def _infer_intent_type(query: str, intent_parse: Optional[IntentParse] = None) -> str:
    if intent_parse:
        mapping = {
            "compare": "compare",
            "rank": "rank",
            "summarize": "summarize",
            "list": "extract",
            "extract": "extract",
            "contact": "extract",
            "qa": "answer",
        }
        mapped = mapping.get(intent_parse.intent)
        if mapped:
            return mapped
    lowered = (query or "").lower()
    if any(tok in lowered for tok in ("compare", "versus", "vs")):
        return "compare"
    if any(tok in lowered for tok in ("rank", "top ", "best ", "highest", "lowest")):
        return "rank"
    if any(tok in lowered for tok in ("summarize", "summary", "overview", "recap")):
        return "summarize"
    if any(tok in lowered for tok in ("extract", "list", "pull")):
        return "extract"
    return "answer"


def _start_intent_parse(
    *,
    query: str,
    llm_client: Optional[Any],
    redis_client: Optional[Any],
) -> Optional[concurrent.futures.Future]:
    if not query or llm_client is None:
        return None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(parse_intent, query=query, llm_client=llm_client, redis_client=redis_client)
    executor.shutdown(wait=False)
    return future


def _resolve_intent_future(future: Optional[concurrent.futures.Future]) -> Optional[IntentParse]:
    if future is None:
        return None
    try:
        return future.result(timeout=0.5)
    except Exception:
        return None


def _infer_scope_document_id(query: str, explicit_document_id: Optional[str]) -> Optional[str]:
    if explicit_document_id:
        return str(explicit_document_id)
    if not query:
        return None
    patterns = [
        r"document_id\s*[:=]\s*([A-Za-z0-9_-]+)",
        r"doc_id\s*[:=]\s*([A-Za-z0-9_-]+)",
        r"document\s+id\s*[:=]?\s*([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _needs_full_scan_generic(schema: Any) -> bool:
    """Check if generic extraction result is too sparse and needs more chunks."""
    if not isinstance(schema, GenericSchema):
        return False
    facts = (schema.facts.items if schema.facts else None) or []
    return len(facts) < 3


def _needs_full_scan_hr(schema: HRSchema, intent: str) -> bool:
    candidates = (schema.candidates.items if schema.candidates else None) or []
    if not candidates:
        return False
    if intent == "contact":
        missing = 0
        for cand in candidates:
            if not (cand.emails or cand.phones or cand.linkedins):
                missing += 1
        return missing >= max(1, len(candidates) // 2)

    missing = 0
    for cand in candidates:
        lacks_core = not (cand.technical_skills or cand.functional_skills or cand.certifications or cand.total_years_experience)
        if lacks_core:
            missing += 1
    return missing >= max(1, len(candidates) // 2)


def _needs_full_scan_invoice(schema: InvoiceSchema, intent: str) -> bool:
    _ = intent
    items = (schema.items.items if schema.items else None) or []
    totals = (schema.totals.items if schema.totals else None) or []
    parties = (schema.parties.items if schema.parties else None) or []
    terms = (schema.terms.items if schema.terms else None) or []
    missing_groups = sum(1 for group in (items, totals, parties, terms) if not group)
    return missing_groups >= 2


def _needs_full_scan_legal(schema: LegalSchema, intent: str) -> bool:
    _ = intent
    clauses = (schema.clauses.items if schema.clauses else None) or []
    return not clauses


def _needs_profile_scan(chunks: List[Chunk]) -> bool:
    if not chunks:
        return True
    unique_docs = {(_chunk_document_id(c) or "") for c in chunks}
    if len(unique_docs) < 2:
        return True
    top_score = max((c.score for c in chunks), default=0.0)
    return top_score < 0.2


def _profile_point_counts(
    subscription_id: str,
    profile_id: str,
    qdrant_client: Any,
    correlation_id: Optional[str],
) -> tuple[int, int]:
    count = 0
    total = 0
    try:
        response = qdrant_client.count(
            collection_name=build_collection_name(subscription_id),
            count_filter=build_qdrant_filter(
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            ),
            exact=True,
        )
        count = int(getattr(response, "count", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "RAG v3 profile scan count failed: %s",
            exc,
            extra={"stage": "profile_scan_check", "correlation_id": correlation_id},
        )
    try:
        response = qdrant_client.count(
            collection_name=build_collection_name(subscription_id),
            exact=True,
        )
        total = int(getattr(response, "count", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "RAG v3 total collection count failed: %s",
            exc,
            extra={"stage": "profile_scan_check", "correlation_id": correlation_id},
        )
    logger.info(
        "RAG v3 profile point count=%s total_count=%s (threshold=%s)",
        count,
        total,
        80,
        extra={"stage": "profile_scan_check", "correlation_id": correlation_id},
    )
    return count, total


def _should_unconditional_profile_scan(profile_count: int, threshold: int = 80) -> bool:
    # profile_count == 0 means the Qdrant filter matched nothing — still trigger
    # the scan so the retrieve fallback (unscoped scroll) has a chance to recover.
    return profile_count <= threshold


__all__ = ["run_docwain_rag_v3", "run"]
