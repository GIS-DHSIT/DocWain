"""
Request execution router.
Routes all /api/ask requests through the DocWain Core Agent.
"""
import logging
from typing import Any

from src.execution.common import ExecutionResult
from src.mode.execution_mode import ExecutionMode

logger = logging.getLogger(__name__)

# Lazy singleton
_core_agent = None


def _get_core_agent():
    """Lazy-initialize the Core Agent singleton."""
    global _core_agent
    if _core_agent is None:
        from src.agent.core_agent import CoreAgent
        from src.llm.gateway import get_llm_gateway
        from src.api.config import Config
        from src.embedding.model_loader import get_embedding_model
        from qdrant_client import QdrantClient
        from pymongo import MongoClient

        llm = get_llm_gateway()
        qdrant = QdrantClient(
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
        )
        embedder, _ = get_embedding_model()
        mongo_client = MongoClient(Config.MongoDB.URI)
        mongodb = mongo_client[Config.MongoDB.DB][Config.MongoDB.DOCUMENTS]

        try:
            from src.intelligence.kg_query import KGQueryService
            kg_service = KGQueryService()
        except Exception:
            kg_service = None

        # Get cross-encoder from app state (loaded at startup on CPU)
        ce = None
        try:
            from src.api.rag_state import get_app_state
            app_state = get_app_state()
            if app_state:
                ce = getattr(app_state, "reranker", None)
        except Exception:
            pass

        _core_agent = CoreAgent(
            llm_gateway=llm,
            qdrant_client=qdrant,
            embedder=embedder,
            mongodb=mongodb,
            kg_query_service=kg_service,
            cross_encoder=ce,
        )
    return _core_agent


def execute_request(
    request: Any,
    session_state: Any,
    ctx: Any,
    *,
    stream: bool = False,
    debug: bool = False,
) -> ExecutionResult:
    """Execute an /api/ask request through the Core Agent."""
    agent = _get_core_agent()

    # Load conversation history
    conversation_history = None
    if hasattr(ctx, "session_id") and ctx.session_id:
        try:
            from src.api.dw_chat import get_current_session_context
            conversation_history = get_current_session_context(
                user_id=getattr(request, "user_id", ""),
                session_id=ctx.session_id,
                max_messages=5,
            )
        except Exception as e:
            logger.debug("[ROUTER] Could not load conversation history: %s", e)

    answer = agent.handle(
        query=getattr(request, "query", ctx.query),
        subscription_id=getattr(request, "subscription_id", ctx.subscription_id),
        profile_id=getattr(request, "profile_id", ctx.profile_id),
        user_id=getattr(request, "user_id", getattr(ctx, "user_id", "")),
        session_id=getattr(ctx, "session_id", None),
        conversation_history=conversation_history,
        agent_name=getattr(request, "agent_name", None),
        document_id=getattr(request, "document_id", None),
        debug=debug,
    )

    debug_info = answer.get("metadata", {}) if debug else {}

    # When streaming is requested, convert the completed answer to a text stream
    # so the UI receives chunked content instead of an empty response.
    stream_iter = None
    if stream:
        from src.execution.common import chunk_text_stream_with_metadata
        response_text = answer.get("response", "")
        metadata = {
            "grounded": answer.get("grounded", False),
            "context_found": answer.get("context_found", False),
            "sources": answer.get("sources", []),
        }
        stream_iter = chunk_text_stream_with_metadata(response_text, metadata=metadata)

    return ExecutionResult(
        answer=answer,
        mode=ExecutionMode.AGENT,
        debug=debug_info,
        stream=stream_iter,
    )
