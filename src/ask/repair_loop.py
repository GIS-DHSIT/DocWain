from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.cache.redis_keys import RedisKeys
from src.cache.redis_store import RedisStore
from src.kg.kg_store import KGStore

from .models import EvidenceChunk, EvidenceQuality, Plan

logger = get_logger(__name__)

class RepairLoop:
    def __init__(self, redis_client: Optional[Any] = None, kg_store: Optional[KGStore] = None) -> None:
        self.redis_client = redis_client
        self.kg_store = kg_store or KGStore()

    def score(self, evidence: List[EvidenceChunk], plan: Plan) -> EvidenceQuality:
        return _score_evidence(evidence, plan)

    def run(
        self,
        *,
        plan: Plan,
        evidence: List[EvidenceChunk],
        retriever: Any,
        subscription_id: str,
        profile_id: str,
        collection_name: str,
        top_k: int,
        max_iter: int = 2,
        log_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[EvidenceChunk], EvidenceQuality, Dict[str, Any]]:
        quality = self.score(evidence, plan)
        meta: Dict[str, Any] = {"iterations": [], "started_at": int(time.time())}
        if quality.quality != "LOW":
            return evidence, quality, meta

        current = list(evidence)
        for iteration in range(max_iter):
            iteration_meta: Dict[str, Any] = {"iteration": iteration + 1, "actions": [], "quality_before": quality.quality}
            improved = False
            retrieval_errors: List[str] = []

            rewrite = _next_rewrite(plan.query_rewrites, iteration)
            if rewrite and rewrite != plan.query_rewrites[0]:
                iteration_meta["actions"].append("rewrite")
                retrieved, error = _safe_retrieve(
                    retriever,
                    query=rewrite,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=top_k,
                    collection_name=collection_name,
                    log_context=log_context,
                    stage="rewrite",
                )
                if error:
                    retrieval_errors.append(error)
                if retrieved:
                    current = _merge_evidence(current, retrieved)
                    improved = True

            entity_targets = self._kg_targets(subscription_id, profile_id, plan.entity_hints)
            if entity_targets:
                iteration_meta["actions"].append("kg_expand")
                retrieved, error = _safe_retrieve(
                    retriever,
                    query=plan.query_rewrites[0],
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=max(10, min(top_k, 80)),
                    document_ids=entity_targets.get("document_ids"),
                    section_ids=entity_targets.get("section_ids"),
                    collection_name=collection_name,
                    log_context=log_context,
                    stage="kg_expand",
                )
                if error:
                    retrieval_errors.append(error)
                if retrieved:
                    current = _merge_evidence(current, retrieved)
                    improved = True

            redis_targets = self._redis_targets(subscription_id, profile_id, plan.entity_hints)
            if redis_targets:
                iteration_meta["actions"].append("redis_target")
                retrieved, error = _safe_retrieve(
                    retriever,
                    query=plan.query_rewrites[0],
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=max(10, min(top_k, 80)),
                    document_ids=redis_targets.get("document_ids"),
                    section_ids=redis_targets.get("section_ids"),
                    collection_name=collection_name,
                    log_context=log_context,
                    stage="redis_target",
                )
                if error:
                    retrieval_errors.append(error)
                if retrieved:
                    current = _merge_evidence(current, retrieved)
                    improved = True

            if not improved:
                iteration_meta["actions"].append("increase_top_k")
                retrieved, error = _safe_retrieve(
                    retriever,
                    query=plan.query_rewrites[0],
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=min(top_k + 20, 80),
                    collection_name=collection_name,
                    log_context=log_context,
                    stage="increase_top_k",
                )
                if error:
                    retrieval_errors.append(error)
                if retrieved:
                    current = _merge_evidence(current, retrieved)
                    improved = True

            quality = self.score(current, plan)
            iteration_meta["quality_after"] = quality.quality
            if retrieval_errors:
                iteration_meta["retrieval_errors"] = retrieval_errors
            meta["iterations"].append(iteration_meta)
            if retrieval_errors and not improved:
                break
            if quality.quality in {"MEDIUM", "HIGH"}:
                break

        meta["completed_at"] = int(time.time())
        return current, quality, meta

    def _kg_targets(self, subscription_id: str, profile_id: str, entity_hints: List[str]) -> Dict[str, List[str]]:
        if not entity_hints:
            return {}
        rows = self.kg_store.find_sections_for_entities(subscription_id, profile_id, entity_hints, limit=20)
        document_ids = [row.get("document_id") for row in rows if row.get("document_id")]
        section_ids = [row.get("section_id") for row in rows if row.get("section_id")]
        return {
            "document_ids": list(dict.fromkeys(document_ids)),
            "section_ids": list(dict.fromkeys(section_ids)),
        }

    def _redis_targets(self, subscription_id: str, profile_id: str, entity_hints: List[str]) -> Dict[str, List[str]]:
        if not (self.redis_client and entity_hints):
            return {}
        store = RedisStore(self.redis_client)
        keys = RedisKeys(subscription_id=str(subscription_id), profile_id=str(profile_id))
        entity_index = store.get_entity_index(keys) or {}
        entities = entity_index.get("entities") or {}
        document_ids: List[str] = []
        section_ids: List[str] = []
        for hint in entity_hints:
            for entry in entities.get(hint, []):
                doc_id = entry.get("document_id")
                sec_id = entry.get("section_id")
                if doc_id:
                    document_ids.append(str(doc_id))
                if sec_id:
                    section_ids.append(str(sec_id))
        return {
            "document_ids": list(dict.fromkeys(document_ids)),
            "section_ids": list(dict.fromkeys(section_ids)),
        }

def _score_evidence(evidence: List[EvidenceChunk], plan: Plan) -> EvidenceQuality:
    if not evidence:
        return EvidenceQuality(quality="LOW", reasons=["no_evidence"], stats={"documents": 0, "coverage": 0.0})
    doc_ids = {chunk.document_id for chunk in evidence if chunk.document_id}
    doc_count = len(doc_ids)
    hints = plan.entity_hints or []
    coverage = _entity_coverage(evidence, hints)
    duplicates = _duplicate_ratio(evidence)

    reasons: List[str] = []
    if doc_count <= 1:
        reasons.append("single_document")
    if coverage < 0.3 and hints:
        reasons.append("low_entity_coverage")
    if duplicates > 0.4:
        reasons.append("high_redundancy")

    if doc_count >= 2 and coverage >= 0.6:
        quality = "HIGH"
    elif doc_count >= 1 and coverage >= 0.3:
        quality = "MEDIUM"
    else:
        quality = "LOW"

    return EvidenceQuality(
        quality=quality,
        reasons=reasons,
        stats={
            "documents": doc_count,
            "coverage": coverage,
            "redundancy": duplicates,
            "evidence": len(evidence),
        },
    )

def _entity_coverage(evidence: List[EvidenceChunk], hints: Iterable[str]) -> float:
    hints_list = [h for h in hints if h]
    if not hints_list:
        return 1.0
    found = 0
    for hint in hints_list:
        hint_lower = hint.lower()
        for chunk in evidence:
            if hint_lower in (chunk.text or "").lower():
                found += 1
                break
    return found / max(len(hints_list), 1)

def _duplicate_ratio(evidence: List[EvidenceChunk]) -> float:
    if not evidence:
        return 0.0
    seen = set()
    dup = 0
    for chunk in evidence:
        if chunk.snippet_sha in seen:
            dup += 1
        else:
            seen.add(chunk.snippet_sha)
    return dup / max(len(evidence), 1)

def _merge_evidence(current: List[EvidenceChunk], new_items: List[EvidenceChunk]) -> List[EvidenceChunk]:
    merged = list(current)
    seen = {chunk.snippet_sha for chunk in merged}
    for item in new_items:
        if item.snippet_sha in seen:
            continue
        seen.add(item.snippet_sha)
        merged.append(item)
    return merged

def _next_rewrite(rewrites: List[str], iteration: int) -> Optional[str]:
    if not rewrites:
        return None
    idx = min(iteration + 1, len(rewrites) - 1)
    return rewrites[idx]

def _safe_retrieve(
    retriever: Any,
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    top_k: int,
    collection_name: str,
    document_ids: Optional[List[str]] = None,
    section_ids: Optional[List[str]] = None,
    log_context: Optional[Dict[str, Any]] = None,
    stage: str = "retrieve",
) -> Tuple[List[EvidenceChunk], Optional[str]]:
    try:
        results = retriever.retrieve(
            query=query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            top_k=top_k,
            document_ids=document_ids,
            section_ids=section_ids,
            collection_name=collection_name,
        )
        return results, None
    except Exception as exc:  # noqa: BLE001
        ctx = log_context or {}
        logger.warning(
            "RAG v2 retrieval attempt failed: %s",
            exc,
            extra={
                "stage": "retrieve",
                "detail": stage,
                "correlation_id": ctx.get("request_id"),
                "session_id": ctx.get("session_id"),
                "provider": "qdrant",
            },
            exc_info=True,
        )
        return [], str(exc)

__all__ = ["RepairLoop"]
