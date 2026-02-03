from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, registry, standard_response

# Import tool modules so their routers + handlers are registered.
from src.tools import (  # noqa: F401
    stt,
    tts,
    translator,
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

router = APIRouter(prefix="/tools", tags=["Tools"])


class ToolRunRequest(BaseModel):
    tool_name: str = Field(..., description="Registered tool name to execute")
    input: Dict[str, Any] | None = Field(default=None, description="Input payload for the tool")
    context: Dict[str, Any] | None = Field(default=None, description="Context (documents, urls, text)")
    options: Dict[str, Any] | None = Field(default=None, description="Execution options")
    subscription_id: Optional[str] = None
    profile_id: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/run")
async def run_tool(request: ToolRunRequest, x_correlation_id: str | None = Header(None)):
    correlation_id = generate_correlation_id(x_correlation_id)
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
            request.tool_name,
            payload,
            correlation_id=correlation_id,
        )
        return result
    except ToolError as exc:
        error_response = standard_response(
            request.tool_name,
            status="error",
            grounded=False,
            context_found=False,
            warnings=[],
            error=exc.as_dict(),
            correlation_id=correlation_id,
        )
        return JSONResponse(status_code=exc.status_code, content=error_response)


# Mount individual routers under /tools/<name>/...
router.include_router(stt.router)
router.include_router(tts.router)
router.include_router(translator.router)
router.include_router(tutor.router)
router.include_router(creator.router)
router.include_router(email_drafting.router)
router.include_router(db_connector.router)
router.include_router(code_docs.router)
router.include_router(medical.router)
router.include_router(lawhere.router)
router.include_router(resumes.router)
router.include_router(jira_confluence.router)
router.include_router(web_extract.router)

# Alias for import convenience
tools_router = router

