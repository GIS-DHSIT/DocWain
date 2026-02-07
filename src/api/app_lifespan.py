from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI

from src.api import dw_newron
from src.api.config import Config
from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS, ensure_payload_indexes
from src.api.rag_state import AppState, activate_singleton_guard, register_instance_ids, set_app_state

logger = logging.getLogger(__name__)


def _bootstrap_qdrant_indexes(qdrant_client) -> Dict[str, Dict[str, object]]:
    index_status: Dict[str, Dict[str, object]] = {}
    try:
        collections = qdrant_client.get_collections().collections
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list Qdrant collections: %s", exc)
        return {"__all__": {"status": "unhealthy", "error": str(exc)}}

    for col in collections or []:
        name = getattr(col, "name", None) or str(col)
        try:
            ensure_payload_indexes(
                client=qdrant_client,
                collection_name=name,
                required_fields=REQUIRED_PAYLOAD_INDEX_FIELDS,
                create_missing=True,
            )
            index_status[name] = {"status": "healthy", "missing": []}
        except Exception as exc:  # noqa: BLE001
            logger.error("Payload index bootstrap failed for %s: %s", name, exc)
            index_status[name] = {"status": "unhealthy", "error": str(exc)}

    return index_status


def initialize_app_state(app: FastAPI) -> AppState:
    default_model = os.getenv("DOCWAIN_DEFAULT_MODEL", "DocWain-Agent")

    embedding_model = None
    cross_encoder = None
    qdrant_client = None
    redis_client = None
    ollama_client = None
    rag_system = None

    try:
        embedding_model = dw_newron.get_model()
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding model init failed: %s", exc)
    try:
        cross_encoder = dw_newron.get_cross_encoder()
    except Exception as exc:  # noqa: BLE001
        logger.error("Reranker init failed: %s", exc)
    try:
        qdrant_client = dw_newron.get_qdrant_client()
    except Exception as exc:  # noqa: BLE001
        logger.error("Qdrant client init failed: %s", exc)
    try:
        redis_client = dw_newron.get_redis_client()
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis client init failed: %s", exc)
    if redis_client and getattr(Config.Redis, "CLEAR_UNSAFE_ON_STARTUP", False):
        try:
            from src.utils.redis_startup import clear_unsafe_keys, parse_unsafe_patterns

            patterns = parse_unsafe_patterns(getattr(Config.Redis, "UNSAFE_KEY_PATTERNS", ""))
            result = clear_unsafe_keys(redis_client, patterns)
            logger.info("Redis unsafe key cleanup: cleared=%s patterns=%s", result.get("cleared"), result.get("patterns"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unsafe key cleanup skipped: %s", exc)
    try:
        ollama_client = dw_newron.create_llm_client(default_model)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama client init failed: %s", exc)

    if qdrant_client and embedding_model and ollama_client:
        try:
            rag_system = dw_newron.EnterpriseRAGSystem(
                model_name=default_model,
                llm_client=ollama_client,
                qdrant_client=qdrant_client,
                embedder=embedding_model,
                cross_encoder=cross_encoder,
                redis_client=redis_client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG system init failed: %s", exc)

    # Ensure globals are aligned for legacy call sites.
    if rag_system is not None:
        dw_newron._RAG_SYSTEM = rag_system
        dw_newron._RAG_MODEL = getattr(rag_system, "model_name", default_model)
        dw_newron._RAG_PROFILE = None
        dw_newron._RAG_BACKEND = getattr(ollama_client, "backend", None) if ollama_client else None
        dw_newron._RAG_MODEL_PATH = None

    try:
        from src.api.document_status import normalize_error_fields

        result = normalize_error_fields()
        logger.info(
            "Mongo error normalization: updated=%s paths=%s skipped=%s",
            result.get("updated"),
            len(result.get("paths") or []),
            result.get("skipped"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Mongo error normalization skipped: %s", exc)

    state = AppState(
        embedding_model=embedding_model,
        reranker=cross_encoder,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        ollama_client=ollama_client,
        rag_system=rag_system,
        qdrant_index_status=_bootstrap_qdrant_indexes(qdrant_client) if qdrant_client else {"__all__": {"status": "unhealthy", "error": "qdrant_unavailable"}},
    )
    register_instance_ids(state)

    app.state.rag_state = state
    app.state.rag_system = rag_system
    app.state.instance_ids = state.instance_ids
    app.state.qdrant_index_status = state.qdrant_index_status
    set_app_state(state)
    activate_singleton_guard()

    logger.info("AppState initialized with model=%s", default_model)
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_state(app)
    yield


__all__ = ["initialize_app_state", "lifespan"]
