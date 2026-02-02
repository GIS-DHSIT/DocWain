from __future__ import annotations

import time
from typing import Optional

from src.observability.metrics import metrics_store
from src.router.fallback import fallback_decision
from src.router.heuristics import is_meta_query
from src.router.ollama_router import route_with_ollama
from src.router.schema import (
    DocumentFilters,
    IntentModel,
    RetrievalFilter,
    RetrievalPlan,
    ResponsePolicy,
    RouterDecision,
    ScopeModel,
)


def _meta_decision(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    profile_name: str,
) -> RouterDecision:
    return RouterDecision(
        intent=IntentModel(category="meta", confidence=1.0, user_goal=query),
        scope=ScopeModel(
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_name=profile_name,
            document_filters=DocumentFilters(),
        ),
        retrieval_plan=RetrievalPlan(strategy="none", chunk_kinds=[], query_rewrites=[], filters=[]),
        response_policy=ResponsePolicy(
            include_persona=True,
            no_questions=True,
            no_refusals=True,
            style="direct",
        ),
    )


def _enforce_profile_filters(decision: RouterDecision, subscription_id: str, profile_id: str) -> RouterDecision:
    if decision.intent.category == "meta":
        return decision
    filters = decision.retrieval_plan.filters
    fields = {f.field for f in filters}
    if "subscription_id" not in fields:
        filters.append(RetrievalFilter(field="subscription_id", op="==", value=subscription_id))
    if "profile_id" not in fields:
        filters.append(RetrievalFilter(field="profile_id", op="==", value=profile_id))
    decision.retrieval_plan.filters = filters
    decision.scope.subscription_id = subscription_id
    decision.scope.profile_id = profile_id
    return decision


def route(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    profile_name: str,
    model_name: Optional[str],
) -> RouterDecision:
    start = time.perf_counter()
    if is_meta_query(query):
        decision = _meta_decision(
            query=query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_name=profile_name,
        )
        metrics_store().observe_ms("route_latency_ms", (time.perf_counter() - start) * 1000)
        return decision

    decision = route_with_ollama(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        model_name=model_name,
    ) or fallback_decision(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
    )
    decision = _enforce_profile_filters(decision, subscription_id, profile_id)
    metrics_store().observe_ms("route_latency_ms", (time.perf_counter() - start) * 1000)
    return decision


__all__ = ["route"]
