"""Gateway HTTP endpoint — POST /screen for document screening."""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from .unified_executor import ScreeningExecutor

logger = get_logger(__name__)

gateway_router = APIRouter(prefix="/gateway", tags=["Gateway"])

_executor = ScreeningExecutor()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScreenRequest(BaseModel):
    category: List[str] = Field(
        ...,
        description=(
            "Screening categories to run. Accepts one or more of: "
            "'integrity', 'compliance', 'quality', 'language', 'security', "
            "'ai_authorship', 'resume', 'legality', 'all', "
            "or 'run' (batch — screens all docs in the session's profile). "
            "Example: [\"security\", \"integrity\"]"
        ),
    )
    doc_ids: Optional[List[str]] = Field(
        None,
        description="Document IDs to screen. Not needed for 'run' (batch) category.",
    )

class ScreenResponse(BaseModel):
    status: str  # "success", "partial", "error"
    action: str  # e.g. "screen:integrity"
    correlation_id: str
    timestamp: str
    result: Optional[Dict[str, Any]] = None
    documents: Optional[List[Dict]] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    grounded: bool = True
    warnings: List[str] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    duration_ms: int = 0
    metadata: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Session-based context resolution
# ---------------------------------------------------------------------------

def _resolve_session_context(
    session_id: Optional[str],
    subscription_id: Optional[str],
) -> Dict[str, Optional[str]]:
    """Resolve profile_id and subscription_id from session state.

    Returns dict with 'profile_id' and 'subscription_id' (may be None).
    """
    resolved: Dict[str, Optional[str]] = {
        "profile_id": None,
        "subscription_id": subscription_id,
    }

    if not session_id:
        return resolved

    try:
        from src.api.dw_newron import get_redis_client
        from src.intelligence.redis_intel_cache import RedisIntelCache

        redis_client = get_redis_client()
        cache = RedisIntelCache(redis_client)

        sub_id = subscription_id or "default"
        state = cache.get_session_state(sub_id, session_id)
        if state and state.active_profile_id:
            resolved["profile_id"] = str(state.active_profile_id)
        if not resolved["subscription_id"]:
            resolved["subscription_id"] = sub_id
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve session context for session %s", session_id)

    return resolved

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@gateway_router.post("/screen", response_model=ScreenResponse)
async def screen_documents(
    request: ScreenRequest,
    x_correlation_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    x_subscription_id: Optional[str] = Header(None),
):
    """
    Screening endpoint for document analysis.

    **Body** only requires ``category`` and ``doc_ids``.
    All other context (profile, subscription) is resolved automatically
    from the session via ``x-session-id`` and ``x-subscription-id`` headers.

    Modes:
    - **Category screening**: ``category`` + ``doc_ids`` — runs the specified
      screening category on each document.
    - **Batch screening**: ``category="run"`` — resolves the profile from the
      session and screens all documents belonging to that profile.
    """
    correlation_id = x_correlation_id or str(uuid.uuid4())

    # Resolve profile/subscription from session headers
    ctx = _resolve_session_context(x_session_id, x_subscription_id)

    # For batch ("run"), build profile_ids from session context
    profile_ids: Optional[List[str]] = None
    if "run" in request.category and ctx["profile_id"]:
        profile_ids = [ctx["profile_id"]]

    result = await _executor.execute_screening(
        categories=request.category,
        doc_ids=request.doc_ids,
        profile_ids=profile_ids,
        options={},
        correlation_id=correlation_id,
    )
    return result
