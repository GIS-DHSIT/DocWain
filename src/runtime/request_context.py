from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    tools: Optional[list[str]] = None
    use_tools: bool = False
    tool_inputs: Optional[Dict[str, Any]] = None
    enable_internet: bool = False

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
        tools: Optional[list[str]] = None,
        use_tools: bool = False,
        tool_inputs: Optional[Dict[str, Any]] = None,
        enable_internet: bool = False,
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
            tools=tools,
            use_tools=use_tools,
            tool_inputs=tool_inputs,
            enable_internet=enable_internet,
        )

    def with_tools(self, tool_names: List[str]) -> None:
        """Inject auto-selected tools into this context."""
        existing = self.tools or []
        self.tools = list(dict.fromkeys(existing + tool_names))
        if self.tools:
            self.use_tools = True
