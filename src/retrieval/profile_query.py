from __future__ import annotations

from typing import Any, Dict, Optional

from src.orchestrator.orchestrator import run_query
from src.profiles.profile_store import resolve_profile_name


def query_profile(
    *,
    subscription_id: str,
    profile_id: str,
    query: str,
    model_name: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    profile_name = resolve_profile_name(subscription_id=subscription_id, profile_id=profile_id)
    return run_query(
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        query=query,
        model_name=model_name,
        top_k=top_k,
    )


__all__ = ["query_profile"]
