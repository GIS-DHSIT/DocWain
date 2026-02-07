from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    state = getattr(request.app.state, "rag_state", None)
    index_status = getattr(request.app.state, "qdrant_index_status", {}) if state else {}
    overall = "healthy"
    for entry in (index_status or {}).values():
        if isinstance(entry, dict) and entry.get("status") == "unhealthy":
            overall = "degraded"
            break
    if state is None:
        overall = "unhealthy"

    return {
        "status": overall,
        "qdrant_indexes": index_status,
        "instance_ids": getattr(request.app.state, "instance_ids", {}),
    }


__all__ = ["health_router"]
