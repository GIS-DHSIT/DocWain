from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from src.api.vector_store import build_collection_name
from src.ask.generator import ResponseGenerator
from src.ask.planner import Planner
from src.ask.repair_loop import RepairLoop
from src.ask.retriever import DocWainRetriever
from src.ask.synthesizer import EvidenceSynthesizer
from src.ask.verifier import EvidenceVerifier
from src.services.retrieval.reranker import Reranker, RerankerConfig

logger = get_logger(__name__)

def run_docwain_rag_v2(
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
) -> Dict[str, Any]:
    collection_name = build_collection_name(subscription_id)
    planner = Planner(llm_client=llm_client, redis_client=redis_client)
    plan = planner.plan(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=None,
        session_id=session_id,
    )

    if plan.intent == "greet":
        generator = ResponseGenerator()
        response = generator.generate(plan=plan, bundle={"documents": []}, quality=_quality("HIGH"))
        return _build_answer(
            response_text=response["response"],
            sources=[],
            request_id=request_id,
            metadata={"intent": plan.intent, "quality": "HIGH", "acknowledged": True},
        )

    retriever = DocWainRetriever(qdrant_client, embedder)
    try:
        initial = retriever.retrieve(
            query=plan.query_rewrites[0],
            subscription_id=subscription_id,
            profile_id=profile_id,
            top_k=50,
            collection_name=collection_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v2 retrieval failed; using empty evidence: %s",
            exc,
            extra={
                "stage": "retrieve",
                "correlation_id": request_id,
                "session_id": session_id,
                "provider": "qdrant",
            },
            exc_info=True,
        )
        initial = []

    reranker = Reranker(cross_encoder=cross_encoder, llm_client=None, config=RerankerConfig(top_k=60, llm_fallback=False))
    try:
        reranked = reranker.rerank(plan.query_rewrites[0], initial, top_k=60)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v2 rerank failed; using retrieval order: %s",
            exc,
            extra={
                "stage": "rerank",
                "correlation_id": request_id,
                "session_id": session_id,
                "provider": "cross_encoder",
            },
            exc_info=True,
        )
        reranked = list(initial)

    repair_loop = RepairLoop(redis_client=redis_client)
    repaired, quality, repair_meta = repair_loop.run(
        plan=plan,
        evidence=reranked,
        retriever=retriever,
        subscription_id=subscription_id,
        profile_id=profile_id,
        collection_name=collection_name,
        top_k=60,
        max_iter=2,
        log_context={"request_id": request_id, "session_id": session_id},
    )

    filtered = _filter_by_hints(repaired, plan.entity_hints)
    if filtered:
        repaired = filtered

    synthesizer = EvidenceSynthesizer()
    bundle = synthesizer.synthesize(evidence=repaired)

    verifier = EvidenceVerifier()
    verified_bundle, verify_meta = verifier.verify(bundle)

    generator = ResponseGenerator()
    generated = generator.generate(plan=plan, bundle=verified_bundle, quality=quality)

    sources = _collect_sources(repaired)
    metadata = {
        "intent": plan.intent,
        "quality": quality.quality,
        "quality_stats": quality.stats,
        "repair": repair_meta,
        "verifier": verify_meta,
        "acknowledged": True,
    }

    return _build_answer(
        response_text=generated["response"],
        sources=sources,
        request_id=request_id,
        metadata=metadata,
    )

def _collect_sources(evidence: List[Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    for chunk in evidence:
        key = (chunk.file_name, chunk.page, chunk.snippet_sha)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"file_name": chunk.file_name, "page": chunk.page, "snippet": chunk.snippet})
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

def _filter_by_hints(evidence: List[Any], hints: List[str]) -> List[Any]:
    if not evidence or not hints:
        return []
    filtered = []
    for chunk in evidence:
        text = (chunk.text or "").lower()
        if any(hint.lower() in text for hint in hints):
            filtered.append(chunk)
    return filtered

def _quality(level: str):
    class _Q:
        def __init__(self, level: str):
            self.quality = level
            self.stats = {}

    return _Q(level)

__all__ = ["run_docwain_rag_v2"]
