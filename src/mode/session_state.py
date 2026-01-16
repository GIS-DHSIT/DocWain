from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.mode.execution_mode import ExecutionMode


@dataclass
class SessionState:
    preferred_execution_mode: Optional[ExecutionMode] = None
    last_docs_used: List[str] = field(default_factory=list)
    last_intent: Optional[str] = None
    last_request_fingerprint: Optional[str] = None
    last_evidence_fingerprint: Optional[str] = None
    last_answer_hash: Optional[str] = None
    last_request_id: Optional[str] = None
    last_query: Optional[str] = None


class SessionStateStore:
    """
    In-memory store for session-scoped execution preferences.
    Lifetime is bound to the process; no persisted answers or caches are kept.
    """

    def __init__(self):
        self._store: Dict[str, SessionState] = {}

    def get(self, session_id: Optional[str]) -> SessionState:
        if not session_id:
            return SessionState()
        if session_id not in self._store:
            self._store[session_id] = SessionState()
        return self._store[session_id]

    def set_preferred_mode(self, session_id: Optional[str], mode: ExecutionMode) -> None:
        if not session_id:
            return
        state = self._store.setdefault(session_id, SessionState())
        state.preferred_execution_mode = mode

    def clear(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        self._store.pop(session_id, None)
