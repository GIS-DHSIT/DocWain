from __future__ import annotations

import hashlib
import os
import json
from src.utils.logging_utils import get_logger
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.agent.limits import check_and_count
from src.agent.prompts import EXTRACTION_PROMPT, FINALIZER_PROMPT, VALIDATION_PROMPT
from src.agent.graph_worker import GraphWorker
from src.api.dw_chat import add_message_to_history
from src.api.dw_newron import create_llm_client, get_redis_client
from src.api.vector_store import build_collection_name
from src.prompting.persona import enforce_docwain_identity, get_docwain_persona, sanitize_response
from src.prompting.response_contract import format_docwain_response
from src.utils.payload_utils import get_canonical_text, get_document_type, get_source_name

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 8000
PER_DOC_CHUNKS = 3

@dataclass
class DocumentContext:
    doc_id: str
    doc_name: str
    category: str
    language: str
    chunks: List[Dict[str, Any]]

@dataclass
class DocumentSummary:
    doc_name: str
    category: str
    language: str
    summary: str
    key_points: List[str]
    fields: Dict[str, Any]

class AgentOrchestrator:
    @staticmethod
    def run(request: Any) -> Dict[str, Any]:
        session_id = AgentOrchestrator._resolve_session_id(request)
        request.session_id = session_id

        redis_client = None
        try:
            redis_client = get_redis_client()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent orchestrator: Redis unavailable: %s", exc, exc_info=True)

        lock_key = AgentOrchestrator._lock_key(
            subscription_id=request.subscription_id,
            user_id=request.user_id,
            session_id=session_id,
            query=request.query,
        )
        if not AgentOrchestrator._acquire_lock(redis_client, lock_key):
            return AgentOrchestrator._deny_response(
                request,
                "Agent mode is already processing this request.",
            )

        limit_result = check_and_count(
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
            user_id=request.user_id,
            session_id=session_id,
            query=request.query,
            redis_client=redis_client,
        )
        if not limit_result.allowed:
            return AgentOrchestrator._deny_response(request, limit_result.message)

        query_text = (request.query or "").strip()
        auto_scope = AgentOrchestrator._needs_auto_scope(query_text)
        effective_query = query_text if not auto_scope else "profile summary across documents"

        backend_model = AgentOrchestrator._resolve_model(request.model_name)
        llm_client = create_llm_client(backend_model)

        graph_worker = GraphWorker.from_config()
        graph_result = None
        if graph_worker:
            graph_result = graph_worker.run(query_text, request.subscription_id, request.profile_id)

        documents, raw_chunks = AgentOrchestrator._retrieve_profile_context(
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
            query=effective_query,
            auto_scope=auto_scope,
            graph_result=graph_result,
        )

        categories = [doc.category for doc in documents if doc.category]
        language = AgentOrchestrator._dominant_language(documents)
        plan = AgentOrchestrator._build_plan(categories=categories, language=language)

        summaries = AgentOrchestrator._run_extraction_workers(documents, llm_client)
        summaries = AgentOrchestrator._validate_summaries(documents, summaries, llm_client)

        final_answer = AgentOrchestrator._finalize_response(
            summaries=summaries,
            documents=documents,
            language=language,
            llm_client=llm_client,
        )
        sources = AgentOrchestrator._build_sources(raw_chunks)
        metadata_hint = {
            "route_plan": {
                "task_type": "summarize" if auto_scope else "qa",
                "scope": "profile_all_docs",
                "domain_hint": categories[0] if categories else "generic",
            }
        }
        persona_text = get_docwain_persona(request.profile_id, request.subscription_id, None)
        final_answer = enforce_docwain_identity(final_answer, request.query, persona_text)
        final_answer = format_docwain_response(
            response_text=final_answer,
            query=request.query,
            sources=sources,
            metadata=metadata_hint,
            context_found=bool(raw_chunks),
            grounded=bool(raw_chunks),
        )
        final_answer = sanitize_response(final_answer)

        answer_payload = {
            "response": final_answer,
            "sources": sources,
            "grounded": bool(raw_chunks),
            "context_found": bool(raw_chunks),
            "metadata": {
                "agent": {
                    "mode": "agent",
                    "auto_scope": auto_scope,
                    "plan": plan,
                    "model_name": request.model_name,
                    "backend_model": backend_model,
                    "limit_reason": limit_result.reason,
                }
            },
        }

        _, active_session_id = add_message_to_history(
            request.user_id,
            request.query,
            answer_payload,
            session_id=session_id,
            new_session=request.new_session,
        )

        return {"answer": answer_payload, "current_session_id": active_session_id, "debug": {}}

    @staticmethod
    def _resolve_session_id(request: Any) -> Optional[str]:
        if request.session_id:
            return request.session_id
        if request.new_session:
            return str(uuid.uuid4())
        return None

    @staticmethod
    def _resolve_model(model_name: str) -> str:
        normalized = str(model_name).strip().lower()
        if normalized in ("docwain-agent", "gpt-oss"):
            return "qwen3:14b"
        return model_name

    @staticmethod
    def _lock_key(subscription_id: str, user_id: str, session_id: Optional[str], query: str) -> str:
        base = f"{subscription_id}:{user_id}:{session_id or ''}:{query}"
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
        return f"agentlock:{subscription_id}:{user_id}:{session_id}:{digest}"

    @staticmethod
    def _acquire_lock(redis_client: Any, lock_key: str) -> bool:
        if redis_client is None:
            return True
        try:
            return bool(redis_client.set(lock_key, "1", nx=True, ex=120))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent lock failure: %s", exc, exc_info=True)
            return True

    @staticmethod
    def _needs_auto_scope(query: str) -> bool:
        if not query:
            return True
        normalized = query.strip().lower()
        if len(normalized) <= 3:
            return True
        summarize_variants = {
            "summarize",
            "summary",
            "summarise",
            "summarization",
            "summarise this",
            "summarize this",
        }
        return normalized in summarize_variants

    @staticmethod
    def _retrieve_profile_context(
        *,
        subscription_id: str,
        profile_id: str,
        query: str,
        auto_scope: bool,
        graph_result: Optional[Any] = None,
    ) -> Tuple[List[DocumentContext], List[Dict[str, Any]]]:
        from src.api.dataHandler import encode_with_fallback, get_qdrant_client

        client = get_qdrant_client()
        collection = build_collection_name(subscription_id)
        from src.api.vector_store import build_qdrant_filter

        query_filter = build_qdrant_filter(
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
        )

        chunks: List[Dict[str, Any]] = []
        effective_query = query
        if graph_result and getattr(graph_result, "graph_hints", None):
            expansion_terms = graph_result.graph_hints.query_expansion_terms
            if expansion_terms and not auto_scope:
                effective_query = f"{query} {' '.join(expansion_terms)}"

        if auto_scope:
            chunks = AgentOrchestrator._scroll_sample(client, collection, query_filter)
        else:
            query_vec = encode_with_fallback([effective_query], normalize_embeddings=True, convert_to_numpy=False)[0]
            results = client.search(
                collection_name=collection,
                query_vector=("content_vector", list(query_vec)),
                query_filter=query_filter,
                limit=60,
                with_payload=True,
                with_vectors=False,
            )
            if graph_result and graph_result.candidate_doc_ids:
                base = build_qdrant_filter(
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                )
                must = list(getattr(base, "must", []) or [])
                must.append(FieldCondition(key="document_id", match=MatchAny(any=graph_result.candidate_doc_ids)))
                candidate_filter = Filter(must=must)
                candidate_results = client.search(
                    collection_name=collection,
                    query_vector=("content_vector", list(query_vec)),
                    query_filter=candidate_filter,
                    limit=40,
                    with_payload=True,
                    with_vectors=False,
                )
                combined = list(candidate_results or []) + list(results or [])
                seen_ids = set()
                deduped = []
                for hit in combined:
                    hit_id = getattr(hit, "id", None)
                    if hit_id in seen_ids:
                        continue
                    seen_ids.add(hit_id)
                    deduped.append(hit)
                results = deduped
            for hit in results:
                payload = hit.payload or {}
                chunks.append({"payload": payload, "score": float(hit.score), "text": get_canonical_text(payload)})

        selected_chunks = AgentOrchestrator._select_per_document(chunks)
        documents = AgentOrchestrator._build_document_contexts(selected_chunks)
        return documents, selected_chunks

    @staticmethod
    def _scroll_sample(client: Any, collection: str, query_filter: Filter) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        next_offset = None
        attempts = 0
        while attempts < 3:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=120,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            for pt in points:
                payload = pt.payload or {}
                chunks.append({"payload": payload, "score": 0.0, "text": get_canonical_text(payload)})
            if not next_offset:
                break
            attempts += 1
        return chunks

    @staticmethod
    def _select_per_document(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            payload = chunk.get("payload") or {}
            doc_id = str(payload.get("document_id") or "")
            if not doc_id:
                doc_id = get_source_name(payload) or "unknown"
            grouped[doc_id].append(chunk)

        deduped: List[Dict[str, Any]] = []
        seen_hashes: set[str] = set()
        total_chars = 0
        for doc_chunks in grouped.values():
            doc_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
            for chunk in doc_chunks[:PER_DOC_CHUNKS]:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_hash in seen_hashes:
                    continue
                if total_chars + len(text) > MAX_CONTEXT_CHARS:
                    continue
                seen_hashes.add(text_hash)
                total_chars += len(text)
                deduped.append(chunk)
        return deduped

    @staticmethod
    def _build_document_contexts(chunks: List[Dict[str, Any]]) -> List[DocumentContext]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        doc_meta: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            payload = chunk.get("payload") or {}
            doc_id = str(payload.get("document_id") or "")
            if not doc_id:
                doc_id = get_source_name(payload) or "unknown"
            grouped[doc_id].append(chunk)
            if doc_id not in doc_meta:
                doc_meta[doc_id] = {
                    "doc_name": get_source_name(payload) or f"Document {doc_id[:6]}",
                    "category": AgentOrchestrator._doc_category(payload),
                    "language": AgentOrchestrator._doc_language(payload),
                }

        documents: List[DocumentContext] = []
        for doc_id, doc_chunks in grouped.items():
            meta = doc_meta.get(doc_id, {})
            documents.append(
                DocumentContext(
                    doc_id=doc_id,
                    doc_name=meta.get("doc_name") or f"Document {doc_id[:6]}",
                    category=meta.get("category") or "others",
                    language=meta.get("language") or "unknown",
                    chunks=doc_chunks,
                )
            )
        return documents

    @staticmethod
    def _doc_category(payload: Dict[str, Any]) -> str:
        return (
            payload.get("document_category")
            or (payload.get("document") or {}).get("category")
            or get_document_type(payload)
            or "others"
        )

    @staticmethod
    def _doc_language(payload: Dict[str, Any]) -> str:
        return (
            payload.get("detected_language")
            or payload.get("language")
            or payload.get("lang")
            or "unknown"
        )

    @staticmethod
    def _dominant_language(documents: List[DocumentContext]) -> str:
        counts = Counter(
            [doc.language for doc in documents if doc.language and doc.language != "unknown"]
        )
        if not counts:
            return "en"
        return counts.most_common(1)[0][0]

    @staticmethod
    def _build_plan(categories: List[str], language: str) -> Dict[str, Any]:
        plan = {
            "plan_id": f"plan-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "tasks": [
                {
                    "id": "T1",
                    "type": "discover_documents",
                    "goal": "List documents, types, and categories.",
                    "notes": "Use profile metadata and retrieval evidence.",
                },
                {
                    "id": "T2",
                    "type": "summarize_each_document",
                    "goal": "Provide short summaries and key points per document.",
                    "notes": "Respect document_category and detected_language when present.",
                },
                {
                    "id": "T3",
                    "type": "merge_profile_summary",
                    "goal": "Combine per-document insights into a profile-level summary.",
                    "notes": "Group findings by category.",
                },
                {
                    "id": "T4",
                    "type": "quality_check",
                    "goal": "Ensure statements are grounded in evidence.",
                    "notes": "Drop ungrounded points.",
                },
            ],
            "language": language,
            "categories": sorted({c for c in categories if c}),
        }
        return json.loads(json.dumps(plan))

    @staticmethod
    def _run_extraction_workers(
        documents: List[DocumentContext],
        llm_client: Any,
    ) -> List[DocumentSummary]:
        summaries: List[DocumentSummary] = []
        for doc in documents:
            context = AgentOrchestrator._format_document_context(doc)
            prompt = EXTRACTION_PROMPT.format(context=context)
            raw = AgentOrchestrator._safe_llm_json(llm_client, prompt)
            summary = DocumentSummary(
                doc_name=raw.get("doc_name") or doc.doc_name,
                category=raw.get("category") or doc.category,
                language=raw.get("language") or doc.language,
                summary=raw.get("summary") or "",
                key_points=[str(p) for p in raw.get("key_points") or []],
                fields=raw.get("fields") or {},
            )
            if not summary.summary:
                summary.summary = AgentOrchestrator._fallback_summary(doc)
            summaries.append(summary)
        return summaries

    @staticmethod
    def _validate_summaries(
        documents: List[DocumentContext],
        summaries: List[DocumentSummary],
        llm_client: Any,
    ) -> List[DocumentSummary]:
        context_map = {doc.doc_name: AgentOrchestrator._format_document_context(doc) for doc in documents}
        cleaned: List[DocumentSummary] = []
        for summary in summaries:
            context = context_map.get(summary.doc_name, "")
            prompt = VALIDATION_PROMPT.format(context=context)
            validation = AgentOrchestrator._safe_llm_json(llm_client, prompt)
            supported = validation.get("supported")
            if isinstance(supported, list) and supported:
                summary.key_points = [str(p) for p in supported]
            else:
                summary.key_points = summary.key_points[:3]
            cleaned.append(summary)
        return cleaned

    @staticmethod
    def _finalize_response(
        *,
        summaries: List[DocumentSummary],
        documents: List[DocumentContext],
        language: str,
        llm_client: Any,
    ) -> str:
        context = AgentOrchestrator._format_final_context(summaries, documents)
        prompt = FINALIZER_PROMPT.format(context=context, language=language)
        output = AgentOrchestrator._safe_llm_text(llm_client, prompt)
        if output.strip():
            return output
        return AgentOrchestrator._fallback_final(summaries, documents)

    @staticmethod
    def _format_document_context(doc: DocumentContext) -> str:
        lines = [f"Document: {doc.doc_name}", f"Category: {doc.category}", f"Language: {doc.language}"]
        for chunk in doc.chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"- {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_final_context(
        summaries: List[DocumentSummary],
        documents: List[DocumentContext],
    ) -> str:
        parts = []
        for summary in summaries:
            parts.append(
                json.dumps(
                    {
                        "doc_name": summary.doc_name,
                        "category": summary.category,
                        "summary": summary.summary,
                        "key_points": summary.key_points,
                        "fields": summary.fields,
                    },
                    ensure_ascii=False,
                )
            )
        doc_index = [
            {
                "doc_name": doc.doc_name,
                "category": doc.category,
                "language": doc.language,
            }
            for doc in documents
        ]
        return "\n".join(["DOCUMENT_INDEX:", json.dumps(doc_index, ensure_ascii=False), "SUMMARIES:", "\n".join(parts)])

    @staticmethod
    def _safe_llm_json(llm_client: Any, prompt: str) -> Dict[str, Any]:
        try:
            raw = llm_client.generate(prompt, max_retries=2)
            if not raw:
                return {}
            return json.loads(AgentOrchestrator._extract_json(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent JSON parse failed: %s", exc, exc_info=True)
            return {}

    @staticmethod
    def _safe_llm_text(llm_client: Any, prompt: str) -> str:
        try:
            return llm_client.generate(prompt, max_retries=2) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent LLM generation failed: %s", exc, exc_info=True)
            return ""

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return "{}"

    @staticmethod
    def _fallback_summary(doc: DocumentContext) -> str:
        snippets = [chunk.get("text", "").strip() for chunk in doc.chunks]
        joined = " ".join(s for s in snippets if s)
        return joined[:280] or f"Summary unavailable for {doc.doc_name}."

    @staticmethod
    def _fallback_final(
        summaries: List[DocumentSummary],
        documents: List[DocumentContext],
    ) -> str:
        overview_parts = [s.summary for s in summaries if s.summary]
        overview = " ".join(overview_parts)[:400]
        lines = ["SECTIONED SUMMARY", "Overview", overview or "Profile summary prepared from available documents."]
        lines.append("Documents Covered")
        for doc in documents:
            lines.append(f"- {doc.doc_name} — {doc.category} — summarized")
        lines.append("Key Findings")
        grouped: Dict[str, List[str]] = defaultdict(list)
        for summary in summaries:
            grouped[summary.category].extend(summary.key_points or [])
        for category, points in grouped.items():
            lines.append(f"{category}:")
            for point in points[:5]:
                lines.append(f"- {point}")
        lines.append("Evidence")
        for doc in documents:
            lines.append(f"- {doc.doc_name}: evidence from retrieved sections")
        return "\n".join(lines)

    @staticmethod
    def _build_sources(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for chunk in chunks:
            payload = chunk.get("payload") or {}
            name = get_source_name(payload) or "Document"
            name = os.path.basename(str(name))
            if name in seen:
                continue
            seen.add(name)
            page = payload.get("page") or payload.get("page_start") or payload.get("page_end")
            sources.append(
                {
                    "source_name": name,
                    "page": page,
                }
            )
        return sources

    @staticmethod
    def _deny_response(request: Any, message: str) -> Dict[str, Any]:
        persona_text = get_docwain_persona(request.profile_id, request.subscription_id, None)
        response_text = enforce_docwain_identity(message, request.query, persona_text)
        response_text = format_docwain_response(
            response_text=response_text,
            query=request.query,
            sources=[],
            metadata={"route_plan": {"task_type": "info", "scope": "profile_all_docs"}},
            context_found=False,
            grounded=False,
        )
        response_text = sanitize_response(response_text)
        answer_payload = {
            "response": response_text,
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {"agent": {"mode": "agent", "allowed": False}},
        }
        _, active_session_id = add_message_to_history(
            request.user_id,
            request.query,
            answer_payload,
            session_id=request.session_id,
            new_session=request.new_session,
        )
        return {"answer": answer_payload, "current_session_id": active_session_id, "debug": {}}

__all__ = ["AgentOrchestrator"]
