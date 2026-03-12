"""Agentic API router — provides HTTP endpoints for domain agent invocation.

Replaces the old per-tool HTTP endpoints with a unified agent-based API.
Agents can be invoked:
1. Directly via POST /api/agents/{agent_name}/execute
2. Via the /api/ask endpoint with agent_name parameter
3. Via POST /api/agents/run (generic dispatcher)
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = get_logger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])

class AgentRequest(BaseModel):
    """Request body for agent execution."""
    task_type: str = Field(..., description="The specific capability to invoke")
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context (documents, chunks)")
    options: Dict[str, Any] = Field(default_factory=dict, description="Execution options")
    subscription_id: Optional[str] = None
    profile_id: Optional[str] = None

class AgentRunRequest(BaseModel):
    """Request body for the generic agent run endpoint."""
    agent_name: str = Field(..., description="Name of the agent to invoke")
    task_type: str = Field(..., description="Capability to invoke")
    input: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    subscription_id: Optional[str] = None
    profile_id: Optional[str] = None

# GET /api/agents — list all agents and their capabilities
@router.get("")
def list_agents():
    """List all available domain agents and their capabilities."""
    from src.agentic.domain_agents import list_available_agents
    agents = list_available_agents()
    return {
        "agents": [
            {
                "name": name,
                "capabilities": caps,
                "description": f"Specialized agent for {name} domain tasks",
            }
            for name, caps in agents.items()
        ],
        "total": len(agents),
    }

# GET /api/agents/{agent_name} — get specific agent details
@router.get("/{agent_name}")
def get_agent_info(agent_name: str):
    """Get details about a specific agent."""
    from src.agentic.domain_agents import get_domain_agent
    agent = get_domain_agent(agent_name)
    if agent is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Agent '{agent_name}' not found", "available": _get_agent_names()},
        )
    return {
        "name": agent_name,
        "domain": agent.domain,
        "capabilities": agent.get_capabilities(),
        "description": f"Specialized agent for {agent.domain} domain tasks",
    }

# POST /api/agents/{agent_name}/execute — invoke a specific agent
@router.post("/{agent_name}/execute")
def execute_agent(
    agent_name: str,
    request: AgentRequest,
    x_correlation_id: str | None = Header(None),
):
    """Execute a specific agent capability."""
    correlation_id = x_correlation_id or str(uuid.uuid4())
    from src.agentic.domain_agents import get_domain_agent

    agent = get_domain_agent(agent_name)
    if agent is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": f"Agent '{agent_name}' not found",
                "available_agents": _get_agent_names(),
                "correlation_id": correlation_id,
            },
        )

    if not agent.can_handle(request.task_type):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": f"Agent '{agent_name}' does not support task '{request.task_type}'",
                "available_tasks": agent.get_capabilities(),
                "correlation_id": correlation_id,
            },
        )

    # Build context from request
    context = {**request.input, **request.context}
    if request.subscription_id:
        context["subscription_id"] = request.subscription_id
    if request.profile_id:
        context["profile_id"] = request.profile_id

    # Optionally retrieve RAG context if profile_id is provided and no text given
    if request.profile_id and request.subscription_id and not context.get("text"):
        _rag_text = _retrieve_rag_context(
            query=context.get("query", ""),
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
        )
        if _rag_text:
            context["text"] = _rag_text

    try:
        result = agent.execute(request.task_type, context)
        return {
            "status": "success" if result.success else "error",
            "agent": agent_name,
            "task_type": request.task_type,
            "output": result.output,
            "structured_data": result.structured_data,
            "sources": result.sources,
            "reasoning": result.reasoning,
            "error": result.error,
            "correlation_id": correlation_id,
        }
    except Exception as exc:
        logger.exception("Agent execution failed: %s/%s", agent_name, request.task_type)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(exc),
                "agent": agent_name,
                "task_type": request.task_type,
                "correlation_id": correlation_id,
            },
        )

# POST /api/agents/run — generic agent dispatcher (similar to /api/tools/run)
@router.post("/run")
def run_agent(
    request: AgentRunRequest,
    x_correlation_id: str | None = Header(None),
):
    """Generic agent dispatcher — invoke any agent by name."""
    agent_request = AgentRequest(
        task_type=request.task_type,
        input=request.input,
        context=request.context,
        options=request.options,
        subscription_id=request.subscription_id,
        profile_id=request.profile_id,
    )
    return execute_agent(request.agent_name, agent_request, x_correlation_id)

def _get_agent_names() -> List[str]:
    """Get list of available agent names."""
    try:
        from src.agentic.domain_agents import list_available_agents
        return list(list_available_agents().keys())
    except Exception:
        return []

def _retrieve_rag_context(query: str, subscription_id: str, profile_id: str) -> Optional[str]:
    """Retrieve document context from RAG for agent use."""
    if not query:
        return None
    try:
        from src.api.rag_state import get_app_state
        state = get_app_state()
        if not state or not state.qdrant_client or not state.embedding_model:
            return None
        from src.rag_v3.retrieve import retrieve
        from src.api.vector_store import build_collection_name
        collection = build_collection_name(subscription_id)
        results = retrieve(
            query=query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            qdrant_client=state.qdrant_client,
            embedder=state.embedding_model,
            top_k=5,
        )
        if results:
            texts = [r.text for r in results if hasattr(r, "text") and r.text]
            return "\n\n".join(texts[:5])
    except Exception as exc:
        logger.debug("RAG context retrieval for agent failed: %s", exc)
    return None

# Backward-compat: tools/run shim
# This can be mounted separately to handle old /api/tools/run requests
# by mapping tool_name to agent_name

_TOOL_TO_AGENT_MAP = {
    "resumes": "hr",
    "medical": "medical",
    "lawhere": "legal",
    "insights": "analytics",
    "action_items": "analytics",
    "creator": "content",
    "email_drafting": "content",
    "code_docs": "content",
    "translator": "translation",
    "tutor": "education",
    "image_analysis": "image",
    "web_search": "web",
    "web_extract": "web",
    "screen_pii": "screening",
    "screen_ai_authorship": "screening",
    "screen_resume": "screening",
    "screen_readability": "screening",
    "content_generate": "content",
    "cloud_platform": "cloud",
    "sharepoint": "cloud",
}

_TOOL_TO_TASK_MAP = {
    "resumes": "candidate_summary",
    "medical": "clinical_summary",
    "lawhere": "key_terms_extraction",
    "insights": "detect_anomalies",
    "action_items": "extract_action_items",
    "creator": "generate_content",
    "email_drafting": "draft_email",
    "code_docs": "generate_documentation",
    "translator": "translate_text",
    "tutor": "create_lesson",
    "image_analysis": "analyze_image",
    "web_search": "search_web",
    "web_extract": "fetch_url",
    "screen_pii": "screen_pii",
    "screen_ai_authorship": "detect_ai_content",
    "screen_resume": "screen_resume",
    "screen_readability": "assess_readability",
    "content_generate": "generate_content",
    "cloud_platform": "manage_cloud_resource",
    "sharepoint": "manage_sharepoint",
}

agents_router = router
