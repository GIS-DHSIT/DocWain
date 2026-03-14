from __future__ import annotations

from src.utils.logging_utils import get_logger
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = get_logger(__name__)

_APP_STATE: Optional["AppState"] = None
_SINGLETON_GUARD = False

@dataclass
class AppState:
    embedding_model: Any
    reranker: Any
    qdrant_client: Any
    redis_client: Any
    ollama_client: Any
    rag_system: Any
    llm_gateway: Any = None  # src.llm.gateway.LLMGateway — canonical LLM entry point
    multi_agent_gateway: Any = None  # src.llm.multi_agent.MultiAgentGateway — role-specific models
    graph_augmenter: Any = None  # src.kg.retrieval.GraphAugmenter — KG-augmented retrieval
    instance_ids: Dict[str, str] = field(default_factory=dict)
    qdrant_index_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)

def _assign_instance_id(name: str, instance: Any, registry: Dict[str, str]) -> str:
    if instance is None:
        return ""
    existing = getattr(instance, "__docwain_instance_id__", None)
    if existing:
        registry[name] = existing
        return existing
    instance_id = f"{name}-{uuid.uuid4().hex[:10]}"
    try:
        setattr(instance, "__docwain_instance_id__", instance_id)
    except Exception:
        pass
    registry[name] = instance_id
    return instance_id

def set_app_state(state: AppState) -> AppState:
    global _APP_STATE
    _APP_STATE = state
    return state

def get_app_state() -> Optional[AppState]:
    return _APP_STATE

def require_app_state() -> AppState:
    if _APP_STATE is None:
        raise RuntimeError("AppState not initialized")
    return _APP_STATE

def activate_singleton_guard() -> None:
    global _SINGLETON_GUARD
    _SINGLETON_GUARD = True

def singleton_guard_active() -> bool:
    return _SINGLETON_GUARD

def register_instance_ids(state: AppState) -> None:
    _assign_instance_id("embedding_model", state.embedding_model, state.instance_ids)
    _assign_instance_id("reranker", state.reranker, state.instance_ids)
    _assign_instance_id("qdrant_client", state.qdrant_client, state.instance_ids)
    _assign_instance_id("redis_client", state.redis_client, state.instance_ids)
    _assign_instance_id("ollama_client", state.ollama_client, state.instance_ids)
    _assign_instance_id("llm_gateway", state.llm_gateway, state.instance_ids)
    _assign_instance_id("rag_system", state.rag_system, state.instance_ids)
    logger.info("Singleton instance IDs: %s", state.instance_ids)

__all__ = [
    "AppState",
    "activate_singleton_guard",
    "get_app_state",
    "register_instance_ids",
    "require_app_state",
    "set_app_state",
    "singleton_guard_active",
]
