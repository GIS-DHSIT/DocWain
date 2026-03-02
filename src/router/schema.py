from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, root_validator


IntentCategory = Literal[
    "meta",
    "qa",
    "summarize",
    "extract",
    "compare",
    "rank",
    "timeline",
    "compute",
    "transform",
    "unknown",
]

RetrievalStrategy = Literal[
    "none",
    "semantic",
    "hybrid",
    "summary_first",
    "table_first",
    "multi_doc",
]

ChunkKind = Literal[
    "section_text",
    "table_text",
    "image_caption",
    "doc_summary",
    "section_summary",
    "structured_field",
]

ResponseStyle = Literal["direct", "explanatory", "executive"]


class IntentModel(BaseModel):
    category: IntentCategory = Field(..., description="Intent category")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    user_goal: str = Field("", description="User goal summary")


class DocumentFilters(BaseModel):
    document_type_hints: List[str] = Field(default_factory=list)
    file_name_hints: List[str] = Field(default_factory=list)
    must_use_tables: bool = False
    must_use_images: bool = False


class ScopeModel(BaseModel):
    subscription_id: str = ""
    profile_id: str = ""
    profile_name: str = ""
    document_filters: DocumentFilters = Field(default_factory=DocumentFilters)


class RetrievalFilter(BaseModel):
    field: str
    op: str
    value: str


class RetrievalPlan(BaseModel):
    strategy: RetrievalStrategy = "semantic"
    chunk_kinds: List[ChunkKind] = Field(default_factory=list)
    query_rewrites: List[str] = Field(default_factory=list)
    filters: List[RetrievalFilter] = Field(default_factory=list)


class ResponsePolicy(BaseModel):
    include_persona: bool = False
    no_questions: bool = True
    no_refusals: bool = True
    style: ResponseStyle = "explanatory"


class RouterDecision(BaseModel):
    intent: IntentModel
    scope: ScopeModel
    retrieval_plan: RetrievalPlan
    response_policy: ResponsePolicy

    @root_validator(skip_on_failure=True)
    def _enforce_rules(cls, values: dict) -> dict:
        intent: Optional[IntentModel] = values.get("intent")
        retrieval: Optional[RetrievalPlan] = values.get("retrieval_plan")
        policy: Optional[ResponsePolicy] = values.get("response_policy")
        scope: Optional[ScopeModel] = values.get("scope")

        if intent and policy:
            if intent.category == "meta":
                policy.include_persona = True
                if retrieval:
                    retrieval.strategy = "none"
            else:
                policy.include_persona = False

        if policy:
            policy.no_questions = True
            policy.no_refusals = True

        if intent and intent.category != "meta" and scope:
            if not scope.profile_id:
                scope.profile_id = ""

        return values


__all__ = [
    "IntentCategory",
    "RetrievalStrategy",
    "ChunkKind",
    "ResponseStyle",
    "IntentModel",
    "DocumentFilters",
    "ScopeModel",
    "RetrievalFilter",
    "RetrievalPlan",
    "ResponsePolicy",
    "RouterDecision",
]
