from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    """
    Immutable request-scoped metadata. A new instance must be created per request
    to prevent accidental cross-request state reuse.
    """

    request_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    query: str
    timestamp: float
    mode: str = "normal"
    index_version: Optional[str] = None
    debug: bool = False
    profile_id: Optional[str] = None
    subscription_id: Optional[str] = None
    model_name: Optional[str] = None
    persona: Optional[str] = None
    filters: Dict[str, Any] | None = None

    @classmethod
    def build(
        cls,
        *,
        query: str,
        session_id: Optional[str],
        user_id: Optional[str],
        mode: str,
        debug: bool = False,
        profile_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        model_name: Optional[str] = None,
        persona: Optional[str] = None,
        index_version: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> "RequestContext":
        return cls(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            query=query,
            timestamp=time.time(),
            mode=mode,
            index_version=index_version,
            debug=debug,
            profile_id=profile_id,
            subscription_id=subscription_id,
            model_name=model_name,
            persona=persona,
            filters=filters or {},
        )
