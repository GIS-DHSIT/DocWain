"""Agent router — backward-compatible /api/tools/run endpoint.

Individual tool HTTP endpoints have been replaced by the unified agent API
at /api/agents/{agent_name}/execute. This router maintains /api/tools/run
for backward compatibility and delegates to the agent registry.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, registry, standard_response

# Import agent modules so their handlers are registered via @register_agent.
# The handlers are still used by the RAG pipeline internally.
from src.tools import (  # noqa: F401
    stt,
    tts,
    translator,
    image_analysis,
    tutor,
    creator,
    email_drafting,
    db_connector,
    code_docs,
    medical,
    lawhere,
    resumes,
    jira_confluence,
    web_extract,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/tools", tags=["Agents"])

class ToolRunRequest(BaseModel):
    tool_name: str = Field(default="", description="Registered tool/agent name to execute (deprecated, use agent_name)")
    agent_name: Optional[str] = Field(default=None, description="Agent name to execute (takes priority over tool_name)")
    input: Dict[str, Any] | None = Field(default=None, description="Input payload for the agent")
    context: Dict[str, Any] | None = Field(default=None, description="Context (documents, urls, text)")
    options: Dict[str, Any] | None = Field(default=None, description="Execution options")
    subscription_id: Optional[str] = None
    profile_id: Optional[str] = None
    user_id: Optional[str] = None

# Alias: callers may use AgentRunRequest as the canonical name
AgentRunRequest = ToolRunRequest

@router.post("/run")
async def run_tool(request: ToolRunRequest, x_correlation_id: str | None = Header(None)):
    """Execute a registered agent by name.

    .. deprecated::
        This endpoint is maintained for backward compatibility.
        Prefer ``POST /api/agents/{agent_name}/execute`` for new integrations.
    """
    correlation_id = generate_correlation_id(x_correlation_id)
    resolved_name = request.agent_name or request.tool_name
    payload = {
        "input": request.input or {},
        "context": request.context or {},
        "options": request.options or {},
        "subscription_id": request.subscription_id,
        "profile_id": request.profile_id,
        "user_id": request.user_id,
    }
    try:
        result = await registry.invoke(
            resolved_name,
            payload,
            correlation_id=correlation_id,
        )
        return result
    except ToolError as exc:
        error_response = standard_response(
            resolved_name,
            status="error",
            grounded=False,
            context_found=False,
            warnings=[],
            error=exc.as_dict(),
            correlation_id=correlation_id,
        )
        return JSONResponse(status_code=exc.status_code, content=error_response)

# Alias for import convenience
tools_router = router
