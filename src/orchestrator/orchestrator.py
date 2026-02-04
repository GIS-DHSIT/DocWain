from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.observability.metrics import metrics_store
from src.orchestrator.answer import generate_meta_response
from src.orchestrator.citations import build_citations
from src.orchestrator.structured_answer import generate_structured_answer
from src.retrieval.deterministic_retrieval import fetch_document_corpus, route_query
from src.retrieval.profile_document_index import build_profile_document_index
from src.retrieval.profile_evidence import build_profile_evidence_graph
from src.router.router import route


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
    pdi = build_profile_document_index(subscription_id, profile_id)
    plan = route_query(query, pdi)
    corpora = {doc_id: fetch_document_corpus(subscription_id, profile_id, doc_id) for doc_id in plan.target_document_ids}
    metrics_store().observe_ms("retrieval_latency_ms", (time.perf_counter() - retrieval_start) * 1000)

    answer_start = time.perf_counter()
    evidence_graph = build_profile_evidence_graph(corpora)
    answer_text = generate_structured_answer(
        user_query=query,
        intent=decision.intent.category,
        retrieval_scope=plan.scope,
        target_document_ids=plan.target_document_ids,
        evidence_graph=evidence_graph,
        model_name=model_name,
    )
    metrics_store().observe_ms("answer_latency_ms", (time.perf_counter() - answer_start) * 1000)
    metrics_store().set_gauge("exact_match_found", 0.0)

    citations = build_citations(_flatten_citations(corpora))
    return {
        "intent": decision.intent.dict(),
        "answer": answer_text,
        "citations": citations,
    }


def _flatten_citations(corpora: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for chunks in corpora.values():
        for chunk in chunks:
            citations.append(
                {
                    "source": {"name": chunk.get("source_name")},
                    "section_title": chunk.get("section_title"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                }
            )
    return citations[:10]


__all__ = ["run_query"]
