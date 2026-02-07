from __future__ import annotations

import json
import logging
import re
import time
import uuid
import concurrent.futures
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.api import rag_state
from src.api.vector_store import build_collection_name, build_qdrant_filter
from src.intent.llm_intent import IntentParse, parse_intent

from .domain_router import DomainRouter
from .extract import extract_schema
from .judge import JudgeResult, judge
from .renderers.router import render
from .retrieve import (
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
    EvidenceSpan,
    FieldValue,
    FieldValuesField,
    GenericSchema,
    HRSchema,
    InvoiceSchema,
    LegalSchema,
    LLMBudget,
    MISSING_REASON,
    MultiEntitySchema,
)

logger = logging.getLogger(__name__)


NO_CHUNKS_MESSAGE = "Not enough information in the documents to answer that."


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
) -> Dict[str, Any]:
    correlation_id = request_id or str(uuid.uuid4())

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

    budget = LLMBudget(llm_client=llm_client, max_calls=2)
    original_query = query or ""
    intent_future = _start_intent_parse(
        query=original_query,
        llm_client=llm_client,
        redis_client=redis_client,
    )
    intent_parse = _resolve_intent_future(intent_future)
    intent_type = _infer_intent_type(original_query)
    scope_document_id = _infer_scope_document_id(original_query, document_id)

    stage_start = time.time()
    rewritten = rewrite_query(
        query=original_query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        redis_client=redis_client,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
    )
    _log_stage("rewrite", stage_start, correlation_id)

    profile_count, total_count = _profile_point_counts(subscription_id, profile_id, qdrant_client, correlation_id)
    if _should_unconditional_profile_scan(profile_count):
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
            reranked = profile_chunks
            stage_start = time.time()
            extraction = extract_schema(
                "hr" if _query_is_hr(original_query) else None,
                query=original_query,
                chunks=reranked,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
            )
            _log_stage("profile_scan_extract", stage_start, correlation_id)
            rendered = render(
                domain="hr" if _query_is_hr(original_query) else "generic",
                intent=extraction.intent,
                schema=extraction.schema,
                strict=False,
            )
            sanitized = sanitize(rendered)
            sources = _collect_sources(reranked)
            metadata = {
                "domain": "resume" if _query_is_hr(original_query) else "generic",
                "intent": extraction.intent,
                "intent_type": _infer_intent_type(original_query, intent_parse),
                "scope": {"profile_id": profile_id},
                "quality": "HIGH",
                "rag_v3": True,
                "profile_scan": True,
            }
            return _build_answer(
                response_text=sanitized,
                sources=sources,
                request_id=request_id,
                metadata=metadata,
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
                "RAG v3 unscoped scan filter dropped all chunks; using unfiltered scan",
                extra={"stage": "profile_scan_unscoped", "correlation_id": correlation_id},
            )
            reranked = unscoped
        _log_stage("unscoped_scan_retrieve", stage_start, correlation_id)
        if reranked:
            stage_start = time.time()
            extraction = extract_schema(
                "hr" if _query_is_hr(original_query) else None,
                query=original_query,
                chunks=reranked,
                llm_client=llm_client,
                budget=budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
            )
            _log_stage("unscoped_scan_extract", stage_start, correlation_id)
            rendered = render(
                domain="hr" if _query_is_hr(original_query) else "generic",
                intent=extraction.intent,
                schema=extraction.schema,
                strict=False,
            )
            sanitized = sanitize(rendered)
            sources = _collect_sources(reranked)
            metadata = {
                "domain": "resume" if _query_is_hr(original_query) else "generic",
                "intent": extraction.intent,
                "intent_type": _infer_intent_type(original_query, intent_parse),
                "scope": {"profile_id": profile_id},
                "quality": "HIGH",
                "rag_v3": True,
                "profile_scan": "unscoped",
            }
            return _build_answer(
                response_text=sanitized,
                sources=sources,
                request_id=request_id,
                metadata=metadata,
            )
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
        return _build_answer(
            response_text=NO_CHUNKS_MESSAGE,
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
        )

    retrieved_metadata = _collect_retrieved_metadata(retrieved)
    domain = (
        DomainRouter.resolve(original_query, tool_hint, retrieved_metadata)
        if Config.Features.DOMAIN_SPECIFIC_ENABLED
        else "generic"
    )
    if intent_parse and intent_parse.domain in {"resume"} and domain != "resume":
        domain = "resume"

    stage_start = time.time()
    reranked = rerank(
        query=rewritten,
        chunks=retrieved,
        cross_encoder=cross_encoder,
        top_k=8,
        correlation_id=correlation_id,
    )
    _log_stage("rerank", stage_start, correlation_id)

    if Config.Features.DOMAIN_SPECIFIC_ENABLED and domain == "resume":
        stage_start = time.time()
        extraction = extract_schema(
            "hr",
            query=original_query,
            chunks=reranked,
            llm_client=llm_client,
            budget=budget,
            correlation_id=correlation_id,
            scope_document_id=scope_document_id,
            intent_hint=intent_parse.intent if intent_parse else None,
        )
        _log_stage("resume_extract_hr", stage_start, correlation_id)

        if _needs_full_scan_hr(extraction.schema, extraction.intent):
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
            if _needs_profile_scan(reranked):
                reranked = expand_full_scan_by_profile(
                    qdrant_client=qdrant_client,
                    collection=build_collection_name(subscription_id),
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    correlation_id=correlation_id,
                ) or reranked
            extraction = extract_schema(
                "hr",
                query=original_query,
                chunks=reranked,
                llm_client=None,
                budget=budget,
                correlation_id=correlation_id,
                scope_document_id=scope_document_id,
                intent_hint=intent_parse.intent if intent_parse else None,
            )
            _log_stage("resume_full_scan_extract_hr", stage_start, correlation_id)

        _log_extraction_diagnostics(
            extraction=extraction,
            intent_type=intent_type,
            chunks=reranked,
            correlation_id=correlation_id,
        )
        _maybe_log_hr_schema(extraction, correlation_id)

        stage_start = time.time()
        rendered = render(domain="hr", intent=extraction.intent, schema=extraction.schema, strict=False)
        _log_stage("resume_render_hr", stage_start, correlation_id)

        stage_start = time.time()
        sanitized = sanitize(rendered)
        _log_stage("sanitize", stage_start, correlation_id)

        stage_start = time.time()
        verdict = judge(
            answer=sanitized,
            schema=extraction.schema,
            intent=extraction.intent,
            llm_client=llm_client,
            budget=budget,
            sources_present=bool(reranked),
            correlation_id=correlation_id,
        )
        _log_stage("judge", stage_start, correlation_id)

        final_answer = _apply_verdict(sanitized, verdict)
        sources = _collect_sources(reranked)
        metadata = {
            "domain": "resume",
            "intent": extraction.intent,
            "intent_type": intent_type,
            "scope": {"document_id": scope_document_id} if scope_document_id else {"profile_id": profile_id},
            "quality": "HIGH" if verdict.status == "pass" else "LOW",
            "rag_v3": True,
            "judge": {"status": verdict.status, "reason": verdict.reason},
        }

        return _build_answer(
            response_text=final_answer,
            sources=sources,
            request_id=request_id,
            metadata=metadata,
        )

    stage_start = time.time()
    extraction = extract_schema(
        intent_parse.domain if intent_parse and intent_parse.domain != "generic" else None,
        query=original_query,
        chunks=reranked,
        llm_client=llm_client,
        budget=budget,
        correlation_id=correlation_id,
        scope_document_id=scope_document_id,
        intent_hint=intent_parse.intent if intent_parse else None,
    )
    _log_stage("extract", stage_start, correlation_id)

    if isinstance(extraction.schema, HRSchema) and _needs_full_scan_hr(extraction.schema, extraction.intent):
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
        if _needs_profile_scan(reranked):
            reranked = expand_full_scan_by_profile(
                qdrant_client=qdrant_client,
                collection=build_collection_name(subscription_id),
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
                correlation_id=correlation_id,
            ) or reranked
        extraction = extract_schema(
            "hr",
            query=original_query,
            chunks=reranked,
            llm_client=None,
            budget=budget,
            correlation_id=correlation_id,
            scope_document_id=scope_document_id,
            intent_hint=intent_parse.intent if intent_parse else None,
        )
        _log_stage("full_scan_extract_hr", stage_start, correlation_id)
    elif isinstance(extraction.schema, InvoiceSchema) and _needs_full_scan_invoice(extraction.schema, extraction.intent):
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
        if _needs_profile_scan(reranked):
            reranked = expand_full_scan_by_profile(
                qdrant_client=qdrant_client,
                collection=build_collection_name(subscription_id),
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
                correlation_id=correlation_id,
            ) or reranked
        extraction = extract_schema(
            "invoice",
            query=original_query,
            chunks=reranked,
            llm_client=None,
            budget=budget,
            correlation_id=correlation_id,
            scope_document_id=scope_document_id,
            intent_hint=intent_parse.intent if intent_parse else None,
        )
        _log_stage("full_scan_extract_invoice", stage_start, correlation_id)
    elif isinstance(extraction.schema, LegalSchema) and _needs_full_scan_legal(extraction.schema, extraction.intent):
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
        if _needs_profile_scan(reranked):
            reranked = expand_full_scan_by_profile(
                qdrant_client=qdrant_client,
                collection=build_collection_name(subscription_id),
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
                correlation_id=correlation_id,
            ) or reranked
        extraction = extract_schema(
            "legal",
            query=original_query,
            chunks=reranked,
            llm_client=None,
            budget=budget,
            correlation_id=correlation_id,
            scope_document_id=scope_document_id,
            intent_hint=intent_parse.intent if intent_parse else None,
        )
        _log_stage("full_scan_extract_legal", stage_start, correlation_id)

    if Config.Features.DOMAIN_SPECIFIC_ENABLED and extraction.domain == "multi" and _query_is_hr(original_query):
        stage_start = time.time()
        extraction = extract_schema(
            "hr",
            query=original_query,
            chunks=reranked,
            llm_client=None,
            budget=budget,
            correlation_id=correlation_id,
            scope_document_id=scope_document_id,
            intent_hint=intent_parse.intent if intent_parse else None,
        )
        _log_stage("extract_hr_fallback", stage_start, correlation_id)

    _log_extraction_diagnostics(
        extraction=extraction,
        intent_type=intent_type,
        chunks=reranked,
        correlation_id=correlation_id,
    )
    _maybe_log_hr_schema(extraction, correlation_id)

    stage_start = time.time()
    rendered = render(domain=extraction.domain, intent=extraction.intent, schema=extraction.schema, strict=False)
    _log_stage("render", stage_start, correlation_id)

    stage_start = time.time()
    sanitized = sanitize(rendered)
    _log_stage("sanitize", stage_start, correlation_id)

    stage_start = time.time()
    verdict = judge(
        answer=sanitized,
        schema=extraction.schema,
        intent=extraction.intent,
        llm_client=llm_client,
        budget=budget,
        sources_present=bool(reranked),
        correlation_id=correlation_id,
    )
    _log_stage("judge", stage_start, correlation_id)

    final_answer = sanitized
    if verdict.status == "fail":
        if verdict.reason in {"no_sources", "no_evidence_spans"}:
            final_answer = NO_CHUNKS_MESSAGE
            verdict = JudgeResult(status="fail", reason=verdict.reason)
        else:
            retry_answer = _retry_render(extraction, correlation_id)
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
                final_answer = FALLBACK_ANSWER
                verdict = JudgeResult(status="fail", reason="retry_failed")

    sources = _collect_sources(reranked)
    metadata = {
        "domain": extraction.domain,
        "intent": extraction.intent,
        "intent_type": intent_type,
        "scope": {"document_id": scope_document_id} if scope_document_id else {"profile_id": profile_id},
        "quality": "HIGH" if verdict.status == "pass" else "LOW",
        "rag_v3": True,
        "judge": {"status": verdict.status, "reason": verdict.reason},
    }

    return _build_answer(
        response_text=final_answer,
        sources=sources,
        request_id=request_id,
        metadata=metadata,
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
    )


def _retry_render(extraction: Any, correlation_id: Optional[str]) -> str:
    try:
        return render(domain=extraction.domain, intent=extraction.intent, schema=extraction.schema, strict=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 retry render failed: %s",
            exc,
            extra={"stage": "render_retry", "correlation_id": correlation_id},
        )
        return ""


def _apply_verdict(answer: str, verdict: JudgeResult) -> str:
    if verdict.status == "fail":
        if verdict.reason in {"no_sources", "no_evidence_spans"}:
            return NO_CHUNKS_MESSAGE
        return FALLBACK_ANSWER
    return answer


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


def _build_answer(response_text: str, sources: List[Dict[str, Any]], request_id: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response": response_text,
        "sources": sources,
        "request_id": request_id,
        "context_found": bool(sources),
        "grounded": bool(sources),
        "metadata": metadata,
    }


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
        return future.result(timeout=0.05)
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


def _query_is_hr(query: str) -> bool:
    lowered = (query or "").lower()
    return any(
        token in lowered
        for token in (
            "resume",
            "cv",
            "curriculum vitae",
            "linkedin",
            "candidate",
            "experience",
            "education",
            "skills",
            "certification",
            "certifications",
        )
    )


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
    return bool(profile_count and profile_count <= threshold)


__all__ = ["run_docwain_rag_v3", "run"]
