from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.api.vector_store import build_collection_name
from src.ask.models import EvidenceChunk
from src.ask.retriever import DocWainRetriever
from src.metadata.normalizer import ALLOWED_CHUNK_KINDS
from src.services.retrieval.reranker import Reranker, RerankerConfig
from src.utils.redis_cache import RedisJsonCache, hash_query

logger = get_logger(__name__)

@dataclass
class RoutePlan:
    scope: str
    task: str
    requested_fields: List[str]
    retrieval_plan: Dict[str, Any]
    assumption: Optional[str] = None

def _route_query(query: str, *, has_document_scope: bool) -> RoutePlan:
    lowered = (query or "").lower()
    task = "summarize"
    if any(word in lowered for word in ["compare", "difference", "versus"]):
        task = "compare"
    elif any(word in lowered for word in ["rank", "ranking", "best fit"]):
        task = "rank"
    elif any(word in lowered for word in ["verify", "check"]):
        task = "verify"
    elif any(word in lowered for word in ["list", "show"]):
        task = "list"
    elif any(word in lowered for word in ["extract", "pull"]):
        task = "extract"

    scope = "document" if has_document_scope else "profile"

    requested_fields: List[str] = []
    if any(word in lowered for word in ["cert", "certification", "certificate", "certified"]):
        requested_fields.append("certifications")
    if any(word in lowered for word in ["skill", "skills", "technologies", "tools", "stack"]):
        requested_fields.append("skills")
    if any(word in lowered for word in ["experience", "years"]):
        requested_fields.append("experience")
    if any(word in lowered for word in ["education", "degree"]):
        requested_fields.append("education")
    if not requested_fields:
        requested_fields.append("summary")

    retrieval_plan = {
        "general_query": query,
        "field_queries": _field_queries(requested_fields),
    }
    return RoutePlan(scope=scope, task=task, requested_fields=requested_fields, retrieval_plan=retrieval_plan)

def _field_queries(fields: List[str]) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    for field in fields:
        if field == "certifications":
            queries[field] = "certification OR certified OR certificate"
        elif field == "skills":
            queries[field] = "skills OR technical skills OR tools OR technologies"
        elif field == "experience":
            queries[field] = "experience OR years OR roles"
        elif field == "education":
            queries[field] = "education OR degree OR university"
        else:
            queries[field] = field
    return queries

def _dedupe_chunks(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    seen = set()
    deduped: List[EvidenceChunk] = []
    for chunk in chunks:
        chunk_id = (chunk.metadata or {}).get("chunk_id") or chunk.snippet_sha
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk)
    return deduped

def _group_by_document(chunks: List[EvidenceChunk]) -> Dict[str, List[EvidenceChunk]]:
    grouped: Dict[str, List[EvidenceChunk]] = {}
    for chunk in chunks:
        doc_id = chunk.document_id or "document"
        grouped.setdefault(doc_id, []).append(chunk)
    return grouped

def _extract_field_values(field: str, chunks: List[EvidenceChunk]) -> List[str]:
    values: List[str] = []
    seen = set()
    for chunk in chunks:
        text = chunk.text or ""
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            candidate = line
            if field == "certifications":
                if "cert" in line.lower():
                    candidate = re.sub(r"^certifications?\s*[:–\-]\s*", "", line, flags=re.IGNORECASE)
                else:
                    continue
            if field == "skills":
                if any(token in line.lower() for token in ["skill", "tool", "technology"]):
                    # Strip the label prefix (e.g. "Skills:" or "Technical Skills:")
                    candidate = re.sub(r"^(?:technical\s+|functional\s+)?skills?\s*[:–\-]\s*", "", line, flags=re.IGNORECASE)
            # Split common delimiters
            parts = [p.strip("-• ") for p in re.split(r"[,;/]\s*", candidate) if p.strip()]
            for part in parts:
                if part.lower() == "professional summary":
                    continue
                if part in seen:
                    continue
                seen.add(part)
                values.append(part)
    return values

def _validate_values(values: List[str], chunks: List[EvidenceChunk]) -> List[str]:
    if not values:
        return []
    evidence_text = "\n".join([c.text or "" for c in chunks]).lower()
    validated: List[str] = []
    for value in values:
        if value.lower() in evidence_text:
            validated.append(value)
    return validated

def _compose_response(
    *,
    plan: RoutePlan,
    chunks_by_doc: Dict[str, List[EvidenceChunk]],
    field_chunks: Dict[str, List[EvidenceChunk]],
    document_names: Dict[str, str],
) -> str:
    lines: List[str] = []
    scope_label = "this document" if plan.scope == "document" else "this profile"
    ack = f"I understand you want {plan.task} for {scope_label}."
    if plan.assumption:
        ack = f"{ack} {plan.assumption}"
    lines.append(ack)

    for doc_id, chunks in chunks_by_doc.items():
        file_name = document_names.get(doc_id) or "document"
        line_parts: List[str] = []
        for field in plan.requested_fields:
            relevant_chunks = field_chunks.get(field) or chunks
            values = _extract_field_values(field, relevant_chunks)
            values = _validate_values(values, relevant_chunks)
            if values:
                line_parts.append(f"{field.title()}: {', '.join(values)}")
            else:
                line_parts.append(f"{field.title()}: Not found in the retrieved evidence.")
        lines.append(f"- {file_name}: " + "; ".join(line_parts))

    return "\n".join(lines).strip()

def run_agentic_rag(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    llm_client: Optional[Any] = None,
    qdrant_client: Any,
    redis_client: Optional[Any],
    embedder: Any,
    cross_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    _ = (session_id, llm_client)
    start_time = time.time()
    plan = _route_query(query, has_document_scope=bool(document_id))

    cache = RedisJsonCache(redis_client, default_ttl=120)
    collection_name = build_collection_name(subscription_id)

    retriever = DocWainRetriever(qdrant_client, embedder)

    def _cached_retrieve(key: str, q: str, chunk_kinds: Optional[List[str]] = None) -> List[EvidenceChunk]:
        cached = cache.get_json(key, feature="retrieval")
        if cached:
            return [EvidenceChunk(**item) for item in cached]
        results = retriever.retrieve(
            query=q,
            subscription_id=subscription_id,
            profile_id=profile_id,
            top_k=50,
            document_ids=[document_id] if document_id else None,
            chunk_kinds=chunk_kinds,
            collection_name=collection_name,
        )
        cache.set_json(key, [c.__dict__ for c in results], feature="retrieval", ttl=120)
        return results

    general_key = f"gen:{subscription_id}:{profile_id}:{document_id}:{hash_query(plan.retrieval_plan['general_query'])}"
    initial = _cached_retrieve(general_key, plan.retrieval_plan["general_query"], None)

    field_chunks: Dict[str, List[EvidenceChunk]] = {}
    for field, q in plan.retrieval_plan["field_queries"].items():
        chunk_kinds = list(ALLOWED_CHUNK_KINDS)
        key = f"field:{field}:{subscription_id}:{profile_id}:{document_id}:{hash_query(q)}"
        field_chunks[field] = _cached_retrieve(key, q, chunk_kinds)

    merged = _dedupe_chunks(initial + [c for chunks in field_chunks.values() for c in chunks])

    reranker = Reranker(cross_encoder=cross_encoder, llm_client=None, config=RerankerConfig(top_k=12, llm_fallback=False))
    try:
        reranked = reranker.rerank(plan.retrieval_plan["general_query"], merged, top_k=12)
    except Exception:  # noqa: BLE001
        reranked = merged

    chunks_by_doc = _group_by_document(reranked)
    document_names = {
        chunk.document_id: (chunk.file_name or "document")
        for chunk in reranked
    }

    response_text = _compose_response(
        plan=plan,
        chunks_by_doc=chunks_by_doc,
        field_chunks=field_chunks,
        document_names=document_names,
    )

    sources = [
        {
            "file_name": chunk.file_name,
            "page": chunk.page,
            "snippet": chunk.snippet,
        }
        for chunk in reranked
    ]

    metadata = {
        "task": plan.task,
        "scope": plan.scope,
        "fields": plan.requested_fields,
        "retrieval": {
            "initial": len(initial),
            "field": {k: len(v) for k, v in field_chunks.items()},
            "merged": len(merged),
            "reranked": len(reranked),
        },
        "timing_ms": int((time.time() - start_time) * 1000),
    }

    return {
        "response": response_text,
        "sources": sources,
        "request_id": request_id,
        "context_found": bool(reranked),
        "grounded": bool(reranked),
        "metadata": metadata,
    }

__all__ = ["run_agentic_rag"]
