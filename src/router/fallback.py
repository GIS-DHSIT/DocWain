from __future__ import annotations

from src.router.schema import (
    DocumentFilters,
    IntentModel,
    RetrievalFilter,
    RetrievalPlan,
    ResponsePolicy,
    RouterDecision,
    ScopeModel,
)


def fallback_decision(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    profile_name: str,
) -> RouterDecision:
    return RouterDecision(
        intent=IntentModel(category="qa", confidence=0.0, user_goal=query),
        scope=ScopeModel(
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_name=profile_name,
            document_filters=DocumentFilters(),
        ),
        retrieval_plan=RetrievalPlan(
            strategy="semantic",
            chunk_kinds=["section_text", "table_text", "doc_summary"],
            query_rewrites=[],
            filters=[
                RetrievalFilter(field="subscription_id", op="==", value=subscription_id),
                RetrievalFilter(field="profile_id", op="==", value=profile_id),
            ],
        ),
        response_policy=ResponsePolicy(
            include_persona=False,
            no_questions=True,
            no_refusals=True,
            style="explanatory",
        ),
    )


__all__ = ["fallback_decision"]
