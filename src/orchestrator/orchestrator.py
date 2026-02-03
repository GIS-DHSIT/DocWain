from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.observability.metrics import metrics_store
from src.utils.payload_utils import get_source_name
from src.orchestrator.answer import generate_answer, generate_meta_response
from src.orchestrator.citations import build_citations
from src.orchestrator.grounding_guard import apply_grounding_guard
from src.orchestrator.retrieval import retrieve_chunks
from src.orchestrator.rerank import rerank_chunks
from src.router.router import route


def _unique_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for chunk in chunks:
        key = (get_source_name(chunk) or chunk.get("file_name"), chunk.get("section_title"), chunk.get("page_start"), chunk.get("text"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _diversify_by_doc(chunks: List[Dict[str, Any]], max_per_doc: int) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    diversified: List[Dict[str, Any]] = []
    for chunk in chunks:
        file_name = get_source_name(chunk) or chunk.get("file_name") or "Unknown"
        count = counts.get(file_name, 0)
        if count >= max_per_doc:
            continue
        counts[file_name] = count + 1
        diversified.append(chunk)
    return diversified


def run_query(
    *,
    subscription_id: str,
    profile_id: str,
    profile_name: str,
    query: str,
    model_name: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    decision = route(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        model_name=model_name,
    )

    if decision.intent.category == "meta":
        answer = generate_meta_response()
        return {
            "intent": decision.intent.dict(),
            "answer": answer,
            "citations": [],
        }

    retrieval_start = time.perf_counter()
    queries = list(dict.fromkeys(decision.retrieval_plan.query_rewrites + [query]))
    chunk_kinds = decision.retrieval_plan.chunk_kinds

    raw_chunks = retrieve_chunks(
        subscription_id=subscription_id,
        profile_id=profile_id,
        queries=queries,
        document_filters=decision.scope.document_filters,
        chunk_kinds=chunk_kinds,
        top_k=top_k,
    )
    metrics_store().observe_ms("retrieval_latency_ms", (time.perf_counter() - retrieval_start) * 1000)

    if decision.retrieval_plan.strategy == "multi_doc":
        raw_chunks = _diversify_by_doc(raw_chunks, max_per_doc=2)

    chunks = _unique_chunks(raw_chunks)
    metrics_store().set_gauge("evidence_found_count", float(len(chunks)))

    reranked_chunks = rerank_chunks(chunks, query)

    answer_start = time.perf_counter()
    answer_text, exact_match = generate_answer(
        query=query,
        chunks=reranked_chunks,
        model_name=model_name,
        include_persona=decision.response_policy.include_persona,
        subscription_id=subscription_id,
        profile_id=profile_id,
    )
    metrics_store().observe_ms("answer_latency_ms", (time.perf_counter() - answer_start) * 1000)
    metrics_store().set_gauge("exact_match_found", 1.0 if exact_match else 0.0)

    evidence_text = "\n\n".join(chunk.get("text") or "" for chunk in reranked_chunks)
    answer_text = apply_grounding_guard(answer_text, evidence_text)

    citations = build_citations(reranked_chunks)
    return {
        "intent": decision.intent.dict(),
        "answer": answer_text,
        "citations": citations,
    }


__all__ = ["run_query"]
