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
from .renderers.router import render
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


_ALL_DOCS_PATTERNS = [
    r"\b(?:all|every|each)\s+(?:documents?|files?|resumes?|cvs?|candidates?|invoices?)\b",
    r"\b(?:compare|rank|summarize|overview|analyze)\s+(?:all|the|every)?\s*(?:documents?|candidates?|profiles?|resumes?|invoices?)\b",
    r"\b(?:across|between)\s+(?:all\s+)?(?:documents?|files?|resumes?|candidates?|profiles?)\b",
    r"\bhow many (?:documents?|candidates?|resumes?|invoices?)\b",
    r"\b(?:list|show)\s+(?:all\s+)?(?:candidates?|documents?|resumes?|invoices?)\b",
    # "mentioned in/across all resumes", "found across documents"
    r"\b(?:mentioned|found|listed)\s+(?:in|across)\s+(?:all\s+)?(?:resumes?|documents?|profiles?|candidates?)\b",
    # "among all candidates"
    r"\bamong\s+(?:all\s+)?(?:candidates?|resumes?|documents?|profiles?)\b",
    # "of candidates" (e.g. "linkedin profiles of candidates")
    r"\b(?:profiles?|info|information|details)\s+of\s+(?:all\s+)?(?:candidates?|resumes?)\b",
    # Superlative/ranking queries imply comparison across all candidates
    r"\bwho\s+(?:has|is|are)\s+(?:the\s+)?(?:most|best|highest|longest|strongest|top)\b",
    r"\b(?:most|best|highest|longest|strongest|top)\s+(?:experienced?|skilled|qualified|suitable)\b",
]

_DOCUMENT_ID_PATTERN = r"document_id\s*[:=]\s*([A-Za-z0-9_-]+)"

# Invoice/order number pattern (retained — this is a structured format, not entity name)
_INVOICE_NUMBER_RE = re.compile(r"(?:invoice|order|po)\s*#?\s*(\d+)", re.IGNORECASE)


def _infer_query_scope(
    query: str,
    explicit_document_id: Optional[str],
    intent_parse: Optional[IntentParse],
) -> QueryScope:
    """Detect whether the user wants ALL documents or a specific entity, and route accordingly."""
    if explicit_document_id:
        return QueryScope(mode="specific_document", document_id=str(explicit_document_id))

    lowered = (query or "").lower()

    # Check for "all documents" patterns
    for pattern in _ALL_DOCS_PATTERNS:
        if re.search(pattern, lowered):
            return QueryScope(mode="all_profile")

    # Check intent parse signals
    if intent_parse:
        if intent_parse.intent in ("compare", "rank", "list") and not intent_parse.entity_hints:
            return QueryScope(mode="all_profile")
        if intent_parse.entity_hints:
            return QueryScope(mode="targeted", entity_hint=intent_parse.entity_hints[0])

    # Check for explicit document_id in query text
    doc_match = re.search(_DOCUMENT_ID_PATTERN, query or "", flags=re.IGNORECASE)
    if doc_match:
        return QueryScope(mode="specific_document", document_id=doc_match.group(1).strip())

    # Check for comparison/ranking queries with multiple entities — these
    # should use all_profile mode so both entities get retrieved, not just the
    # first one captured by entity patterns.
    if re.search(r"\b(?:compare|rank|vs\.?|versus|between)\b", lowered):
        return QueryScope(mode="all_profile")

    # Check for invoice/order numbers (structured format, not a name)
    inv_match = _INVOICE_NUMBER_RE.search(query or "")
    if inv_match:
        return QueryScope(mode="targeted", entity_hint=inv_match.group(1).strip())

    # NLP-based entity extraction (spaCy dependency parsing + DPIE ML models)
    # Replaces all regex-based entity name patterns for robustness
    try:
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query(query or "")
        logger.debug("NLP entity extraction result: query=%r entity=%r", query, entity)
        if entity and len(entity) > 2:
            return QueryScope(mode="targeted", entity_hint=entity)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLP entity extraction failed: %s", exc)

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


# ── Tool-to-Domain mapping ─────────────────────────────────────────────────
# When /ask receives tools=["resume-analysis"], the pipeline uses this to
# set an authoritative domain_hint that overrides query/chunk-based inference.

_TOOL_DOMAIN_MAP: Dict[str, str] = {
    "resume-analysis": "hr",
    "resume_analysis": "hr",
    "resumes": "hr",
    "resume": "hr",
    "medical": "medical",
    "lawhere": "legal",
    "legal": "legal",
    "invoice": "invoice",
    "policy": "policy",
    "insurance": "policy",
}


def _resolve_domain_from_tools(tools: Optional[List[str]]) -> Optional[str]:
    """Map tool names to an authoritative domain hint.

    Returns the first matching domain or None if no tool maps to a domain.
    """
    if not tools:
        return None
    for tool in tools:
        domain = _TOOL_DOMAIN_MAP.get((tool or "").strip().lower())
        if domain:
            return domain
    return None


# ── Tool Dispatch ──────────────────────────────────────────────────────────

_TOOL_DISPATCH_TIMEOUT_S = 15.0
_MAX_TOOL_CONTEXT_CHUNKS = 10


def _dispatch_tools(
    tool_names: List[str],
    query: str,
    profile_id: str,
    subscription_id: str,
    tool_inputs: Optional[Dict[str, Any]],
    correlation_id: Optional[str],
    chunks: Optional[List[Chunk]] = None,
) -> List[Chunk]:
    """Invoke registered tools and convert results to synthetic Chunk objects.

    Returns tool-result chunks (score=1.0) so they appear first in the
    extraction context.  Failures are logged and skipped gracefully.

    When *chunks* is provided, the reranked RAG context is serialized and
    included in the tool payload so tools can leverage document evidence.
    """
    if not tool_names:
        return []

    try:
        import asyncio
        from src.tools.base import registry
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool registry not available: %s", exc, extra={"correlation_id": correlation_id})
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

    result_chunks: List[Chunk] = []
    for tool_name in tool_names:
        payload = {
            "input": {"query": query, "chunks": serialized_chunks},
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
            logger.warning("Tool %s dispatch failed: %s", tool_name, exc, extra={"correlation_id": correlation_id})
            continue

        if not isinstance(tool_resp, dict) or tool_resp.get("status") != "success":
            logger.warning("Tool %s returned status=%s", tool_name, (tool_resp or {}).get("status"))
            continue

        result = tool_resp.get("result") or {}
        snippet = json.dumps(result, default=str)[:2000] if isinstance(result, dict) else str(result)[:2000]

        result_chunks.append(Chunk(
            id=f"tool_{tool_name}_{correlation_id or 'x'}",
            text=snippet,
            score=1.0,
            source=ChunkSource(document_name=f"tool:{tool_name}"),
            meta={"source": "tool", "tool_name": tool_name},
        ))
        logger.info("Tool %s returned %d chars", tool_name, len(snippet), extra={"correlation_id": correlation_id})

    return result_chunks


def _emergency_chunk_summary(chunks: List[Chunk], query: str) -> str:
    """Generate a minimal factual summary from raw chunks when renderers return empty.

    Uses query keywords to select the most relevant sentences from chunks
    rather than blindly truncating.
    """
    if not chunks:
        return ""
    # Extract query keywords for relevance filtering
    query_words = set(
        w.lower() for w in re.split(r'\W+', query or "")
        if len(w) > 2 and w.lower() not in _EMERGENCY_STOP_WORDS
    )
    # Collect all sentences from top chunks
    all_sentences: list[tuple[str, float]] = []
    for c in chunks[:5]:
        text = (c.text or "").strip()
        if not text:
            continue
        # Split into sentences at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10 or len(sent) > 300:
                continue
            # Score by keyword overlap
            sent_words = set(w.lower() for w in re.split(r'\W+', sent) if len(w) > 2)
            overlap = len(query_words & sent_words)
            score = overlap / max(len(query_words), 1) + (0.1 if overlap else 0)
            all_sentences.append((sent, score))
    if not all_sentences:
        # Last resort: take first 200 chars of first chunk
        first_text = (chunks[0].text or "").strip()[:200]
        if first_text:
            return f"Based on available information:\n- {first_text}"
        return ""
    # Sort by relevance, take top 3-4 unique sentences
    all_sentences.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    lines = ["Based on the documents:"]
    for sent, score in all_sentences:
        normalized = sent.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(f"- {sent}")
        if len(lines) >= 5:
            break
    return "\n".join(lines)


_EMERGENCY_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "with", "this", "that",
    "from", "what", "who", "how", "which", "tell", "show", "give",
    "about", "does", "can", "has", "have", "their", "been", "will",
    "would", "could", "should", "its", "you", "your", "they", "them",
    "not", "but", "all", "any", "each", "more", "some", "than",
})


def _extract_render_judge(
    *,
    extraction: Any,
    query: str,
    chunks: List[Chunk],
    llm_client: Any,
    budget: LLMBudget,
    correlation_id: Optional[str],
    query_focus: Any = None,
) -> Tuple[str, JudgeResult]:
    """Shared logic: render (or skip for LLM response) → sanitize → judge → evidence bypass."""
    if _is_llm_response(extraction):
        sanitized = sanitize(extraction.schema.text)
    else:
        rendered = render(
            domain=extraction.domain,
            intent=extraction.intent,
            schema=extraction.schema,
            strict=False,
            query=query,
            query_focus=query_focus,
        )
        sanitized = sanitize(rendered)

    if not sanitized.strip():
        rendered = _emergency_chunk_summary(chunks, query)
        sanitized = sanitize(rendered)

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

    verdict = judge(
        answer=sanitized,
        schema=extraction.schema,
        intent=extraction.intent,
        llm_client=llm_client,
        budget=budget,
        sources_present=bool(chunks),
        correlation_id=correlation_id,
    )

    # Deterministic extraction with valid data passes without evidence_spans
    if verdict.status == "fail" and verdict.reason == "no_evidence_spans":
        if _has_valid_deterministic_extraction(extraction.schema):
            verdict = JudgeResult(status="pass", reason="deterministic_extraction")

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
) -> Dict[str, Any]:
    """Retrieve ALL profile chunks, group by document, extract per-doc, then synthesize."""
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

    # 2. Rerank all chunks against the query — use min_score=-100 to keep
    #    ALL chunks (cross-encoder threshold must not drop entire documents).
    #    The purpose here is sorting, not filtering.
    reranked = rerank(
        query=query,
        chunks=all_chunks,
        cross_encoder=cross_encoder,
        top_k=min(len(all_chunks), 100),
        correlation_id=correlation_id,
        min_score=-100.0,  # Keep everything — all-profile needs every doc
    )
    if not reranked:
        reranked = all_chunks

    # Soft floor: remove strongly irrelevant chunks but keep minimum 6
    if len(reranked) > 6:
        filtered = [c for c in reranked if c.score > -5.0]
        if len(filtered) >= 6:
            reranked = filtered

    # 3. Group by document and ensure per-document minimum representation
    doc_chunks: Dict[str, List[Chunk]] = {}
    for chunk in reranked:
        doc_id = _chunk_document_id(chunk) or "unknown"
        doc_chunks.setdefault(doc_id, []).append(chunk)

    # 4. Build quality_chunks: take top chunks but guarantee each document
    #    has at least MIN_PER_DOC chunks for extraction coverage.
    MIN_PER_DOC = 3
    # First pass: take top chunks per document
    quality_chunks: List[Chunk] = []
    doc_taken: Dict[str, int] = {}
    for chunk in reranked:
        doc_id = _chunk_document_id(chunk) or "unknown"
        doc_taken.setdefault(doc_id, 0)
        doc_taken[doc_id] += 1
        quality_chunks.append(chunk)
    # If any document has < MIN_PER_DOC chunks after dedup, ensure they survive
    quality_chunks = deduplicate_by_content(quality_chunks)
    # Limit to reasonable size but keep all doc coverage
    max_total = max(50, len(doc_chunks) * 10)
    quality_chunks = quality_chunks[:max_total]

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

        # Try LLM-enhanced comparison first
        if llm_client and budget.allow():
            from .llm_extract import llm_extract_and_respond
            llm_result = llm_extract_and_respond(
                query=query, chunks=quality_chunks, llm_client=llm_client,
                budget=budget, intent=intent_type,
                num_documents=len(doc_contexts),
                correlation_id=correlation_id,
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
                }
                return _build_answer(
                    response_text=sanitize(llm_result.text),
                    sources=sources,
                    request_id=request_id,
                    metadata=metadata,
                    query=query,
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

    from .query_focus import build_query_focus as _bqf
    _all_profile_focus = _bqf(query, intent_hint=intent_parse.intent if intent_parse else None)

    # For multi-document all-profile queries (>5 docs), skip LLM extraction.
    # LLMs can't reliably process 10+ documents in a single context window —
    # they truncate and only return 1-3 candidates. Deterministic extraction
    # creates one Candidate PER document from chunks, which is exactly what
    # multi-doc queries (compare, rank, list) need.
    _all_profile_llm = llm_client if len(doc_chunks) <= 5 else None
    _all_profile_budget = budget if len(doc_chunks) <= 5 else LLMBudget(llm_client=None, max_calls=0)

    extraction = extract_schema(
        domain_hint,
        query=query,
        chunks=quality_chunks,
        llm_client=_all_profile_llm,
        budget=_all_profile_budget,
        correlation_id=correlation_id,
        scope_document_id=None,
        intent_hint=intent_parse.intent if intent_parse else None,
        document_data=None,  # intentionally None for multi-doc
        query_focus=_all_profile_focus,
        tool_domain=bool(tool_domain),
        embedder=embedder,
    )

    sanitized, verdict = _extract_render_judge(
        extraction=extraction,
        query=query,
        chunks=quality_chunks,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
        query_focus=_all_profile_focus,
    )

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
    return _build_answer(
        response_text=sanitized if (verdict.status == "pass" or sanitized.strip()) else NO_CHUNKS_MESSAGE,
        sources=sources,
        request_id=request_id,
        metadata=metadata,
        query=query,
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
) -> Dict[str, Any]:
    correlation_id = request_id or str(uuid.uuid4())
    _pipeline_start = time.time()

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
        clear_dpie_cache()
    except Exception:  # noqa: BLE001
        pass

    # Lazy DPIE initialization — train on first query if not already ready.
    # Non-blocking: if training is already in progress from startup thread,
    # this returns immediately via the DPIERegistry lock.
    _ensure_dpie_ready(qdrant_client, embedder, subscription_id, profile_id)

    # Separate budgets: infrastructure (rewrite) gets 2 calls,
    # extraction+judge gets 4 calls — ensures the LLM is always
    # available for the response-generating extraction step.
    infra_budget = LLMBudget(llm_client=llm_client, max_calls=2)
    budget = LLMBudget(llm_client=llm_client, max_calls=4)
    original_query = query or ""
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
    from .query_focus import build_query_focus, filter_chunks_by_focus, clear_chunk_embed_cache, clear_dpie_cache
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
            _conf_threshold = getattr(_Cfg.MultiAgent, "CLASSIFIER_CONFIDENCE_THRESHOLD", 0.7)
            _timeout = getattr(_Cfg.MultiAgent, "CLASSIFIER_TIMEOUT", 10.0)
            _multi_agent_classification = _llm_classify(original_query, _ma_gw, timeout_s=_timeout)
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

    # ── Resolve authoritative domain from tools ──────────────────────
    tool_domain = _resolve_domain_from_tools(tools)
    if tool_domain:
        logger.info(
            "Tool domain resolved: tools=%s → domain=%s",
            tools, tool_domain,
            extra={"stage": "tool_domain", "correlation_id": correlation_id},
        )

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
        )

    stage_start = time.time()
    rewritten = rewrite_query(
        query=original_query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        redis_client=redis_client,
        llm_client=llm_client,
        budget=infra_budget,  # infrastructure budget — don't eat extraction calls
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
            clear_dpie_cache()

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

            # Load complete document data for accurate extraction (works for any document type)
            document_data = _load_document_data_for_extraction(reranked, original_query, correlation_id)

            # Augment chunks with contact info from pickle header if missing
            reranked = _augment_chunks_with_contact_info(reranked, document_data)

            extraction = extract_schema(
                tool_domain,  # None when no tool, authoritative domain when tool-specified
                query=original_query,
                chunks=reranked,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
                document_data=document_data,
                query_focus=focus,
                tool_domain=bool(tool_domain),
                embedder=embedder,
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
            return _build_answer(
                response_text=sanitized if (verdict.status == "pass" or sanitized.strip()) else NO_CHUNKS_MESSAGE,
                sources=sources,
                request_id=request_id,
                metadata=metadata,
                query=original_query,
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
        clear_dpie_cache()
        _log_stage("unscoped_scan_retrieve", stage_start, correlation_id)
        if reranked:
            stage_start = time.time()
            document_data = _load_document_data_for_extraction(reranked, original_query, correlation_id)
            reranked = _augment_chunks_with_contact_info(reranked, document_data)
            extraction = extract_schema(
                tool_domain,  # None when no tool, authoritative domain when tool-specified
                query=original_query,
                chunks=reranked,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
                document_data=document_data,
                query_focus=focus,
                tool_domain=bool(tool_domain),
                embedder=embedder,
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
            return _build_answer(
                response_text=sanitized if (verdict.status == "pass" or sanitized.strip()) else NO_CHUNKS_MESSAGE,
                sources=sources,
                request_id=request_id,
                metadata=metadata,
                query=original_query,
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

    if not retrieved:
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

    # ── Tool dispatch: enrich context with tool results ────────────────
    if tools:
        tool_chunks = _dispatch_tools(
            tool_names=tools,
            query=original_query,
            profile_id=str(profile_id),
            subscription_id=str(subscription_id),
            tool_inputs=tool_inputs,
            correlation_id=correlation_id,
            chunks=reranked,
        )
        if tool_chunks:
            reranked = tool_chunks + reranked  # Tool chunks first (authoritative)

    # ── Unified extraction path ────────────────────────────────────────
    stage_start = time.time()
    document_data = _load_document_data_for_extraction(reranked, original_query, correlation_id)
    reranked = _augment_chunks_with_contact_info(reranked, document_data)
    extraction = extract_schema(
        tool_domain,  # None when no tool, authoritative domain when tool-specified
        query=original_query,
        chunks=reranked,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
        scope_document_id=scope_document_id,
        intent_hint=intent_parse.intent if intent_parse else None,
        document_data=document_data,
        query_focus=focus,
        tool_domain=bool(tool_domain),
        embedder=embedder,
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
    )
    _log_stage("render_judge", stage_start, correlation_id)

    final_answer = sanitized
    if verdict.status == "fail":
        if verdict.reason == "no_sources":
            final_answer = NO_CHUNKS_MESSAGE
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
                else:
                    final_answer = NO_CHUNKS_MESSAGE
                    verdict = JudgeResult(status="fail", reason="retry_failed")
        else:
            # Keep LLM response if it has content, even if judge rejected it
            if sanitized and sanitized.strip():
                final_answer = sanitized
                verdict = JudgeResult(status="pass", reason="llm_content_preserved")
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

    # ── Multi-agent verifier (advisory, never blocks) ──────────────
    try:
        from src.api.config import Config as _VCfg
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
            _v_timeout = getattr(_VCfg.MultiAgent, "VERIFIER_TIMEOUT", 30.0)
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

    return _build_answer(
        response_text=final_answer,
        sources=sources,
        request_id=request_id,
        metadata=metadata,
        query=original_query,
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

    This prevents "tell me about John Doe" from returning chunks about other people.
    Returns empty list when entity is not found in any chunk (caller handles empty).
    """
    is_tech = _is_tech_term(entity_hint)
    search_terms = _expand_tech_hint(entity_hint) if is_tech else [entity_hint.lower()]

    # Find which doc_ids contain the entity name or expanded tech terms
    entity_doc_ids: set = set()
    for chunk in chunks:
        doc_id = _chunk_document_id(chunk) or ""
        text_lower = ((getattr(chunk, "text", "") or "").lower())
        source = getattr(chunk, "source", None)
        doc_name_lower = ((getattr(source, "document_name", "") or "").lower()) if source else ""
        # Also check metadata fields: source_name, filename, embedding_text, section_summary
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
        for term in search_terms:
            if term in searchable:
                entity_doc_ids.add(doc_id)
                break
    if not entity_doc_ids:
        # Entity not found in any chunk — log classification and return empty
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


def _build_answer(
    response_text: str,
    sources: List[Dict[str, Any]],
    request_id: Optional[str],
    metadata: Dict[str, Any],
    query: Optional[str] = None,
    include_acknowledgement: bool = True,
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


def _resolve_domain_from_chunks(
    chunks: List[Chunk],
    query: str,
    tool_hint: Optional[str] = None,
    intent_parse: Optional[IntentParse] = None,
) -> Optional[str]:
    """Determine domain_hint from chunk metadata, NOT from query keywords alone."""
    if not chunks:
        return "hr" if _query_is_hr(query) else None

    # 1. Check chunk metadata: majority vote on doc_domain / doc_type
    domain_counts: Dict[str, int] = {}
    for chunk in chunks:
        meta = chunk.meta or {}
        d = str(meta.get("doc_domain") or meta.get("doc_type") or "").lower().strip()
        if d and d not in ("generic", ""):
            domain_counts[d] = domain_counts.get(d, 0) + 1

    if domain_counts:
        best = max(domain_counts, key=domain_counts.get)
        total = sum(domain_counts.values())
        if domain_counts[best] > total * 0.5:
            domain_map = {"resume": "hr", "hr": "hr", "invoice": "invoice", "legal": "legal"}
            mapped = domain_map.get(best)
            if mapped:
                return mapped
            # Non-HR domain (medical, technical, etc.) → use generic extraction
            return None

    # 2. Fall back to DomainRouter
    if Config.Features.DOMAIN_SPECIFIC_ENABLED:
        retrieved_metadata = _collect_retrieved_metadata(chunks)
        decision = DomainRouter.route(query, tool_hint, retrieved_metadata)
        if decision.confidence >= 0.6 and decision.domain not in ("generic", "medical"):
            domain_map = {"resume": "hr", "invoice": "invoice", "legal": "legal"}
            return domain_map.get(decision.domain)

    # 3. Only use _query_is_hr as final fallback with tightened logic
    if _query_is_hr(query):
        return "hr"

    # 4. Check intent parse
    if intent_parse and intent_parse.domain not in ("generic", None):
        domain_map = {"resume": "hr", "invoice": "invoice", "legal": "legal"}
        return domain_map.get(intent_parse.domain)

    return None


def _query_is_hr(query: str) -> bool:
    lowered = (query or "").lower()
    # Strong HR signals — a single match is sufficient
    strong = ("resume", "cv", "curriculum vitae", "linkedin", "candidate")
    if any(token in lowered for token in strong):
        return True
    # Weak HR signals — need at least 2 to count as HR query
    weak = ("experience", "education", "skills", "certification", "certifications")
    return sum(1 for token in weak if token in lowered) >= 2


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


def _ensure_dpie_ready(
    qdrant_client: Any,
    embedder: Any,
    subscription_id: str,
    profile_id: str,
) -> None:
    """Lazily initialise DPIE models on first query.

    If the background startup thread already trained the models this is a
    no-op (``DPIERegistry.is_loaded`` check).  If not, trains synchronously
    on this first request so subsequent queries benefit from ML-based
    classification.  All errors are caught so the RAG pipeline is never
    blocked by DPIE failures.
    """
    try:
        from src.intelligence.dpie_integration import DPIERegistry

        registry = DPIERegistry.get()
        if registry.is_loaded:
            return

        collection_name = build_collection_name(subscription_id)
        registry.ensure_ready(
            qdrant_client=qdrant_client,
            sentence_model=embedder,
            collection_name=collection_name,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DPIE lazy init skipped: %s", exc)


__all__ = ["run_docwain_rag_v3", "run"]
