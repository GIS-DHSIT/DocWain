from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

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


def _bootstrap_dpie_background(
    qdrant_client: Any,
    embedding_model: Any,
) -> None:
    """Discover all Qdrant collections and train DPIE models in the background.

    Runs in a daemon thread so it does not block server startup.
    Skips silently if DPIE dependencies are not available or if
    models are already trained.
    """
    try:
        from src.intelligence.dpie_integration import DPIERegistry
        from src.api.vector_store import build_collection_name

        registry = DPIERegistry.get()
        if registry.is_loaded:
            logger.info("DPIE models already loaded; skipping background training")
            return

        # Discover all collections (each is a subscription_id)
        collections = qdrant_client.get_collections().collections
        if not collections:
            logger.info("DPIE: no Qdrant collections found; skipping")
            return

        for col in collections:
            subscription_id = getattr(col, "name", None) or str(col)
            # Try to find a profile_id by scrolling a few points
            profile_id = _discover_profile_id(qdrant_client, subscription_id)
            if not profile_id:
                logger.debug("DPIE: no profile found in collection %s; skipping", subscription_id)
                continue

            logger.info("DPIE: auto-training for subscription=%s profile=%s", subscription_id, profile_id)
            try:
                registry.ensure_ready(
                    qdrant_client=qdrant_client,
                    sentence_model=embedding_model,
                    collection_name=subscription_id,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                )
                logger.info("DPIE: models ready for subscription=%s", subscription_id)
                break  # Train on the first valid collection
            except Exception as exc:  # noqa: BLE001
                logger.warning("DPIE: training failed for %s: %s", subscription_id, exc)

    except ImportError:
        logger.debug("DPIE: intelligence.dpie_integration not available; skipping")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DPIE: background bootstrap failed: %s", exc)


def _discover_profile_id(qdrant_client: Any, collection_name: str) -> Optional[str]:
    """Scroll a few points to discover a profile_id in the collection."""
    try:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
        points = result[0] if result else []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            pid = payload.get("profile_id")
            if pid:
                return str(pid)
    except Exception:
        pass
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = initialize_app_state(app)

    # Kick off DPIE background training (non-blocking)
    if state.qdrant_client and state.embedding_model:
        dpie_thread = threading.Thread(
            target=_bootstrap_dpie_background,
            args=(state.qdrant_client, state.embedding_model),
            daemon=True,
            name="dpie-bootstrap",
        )
        dpie_thread.start()
        logger.info("DPIE background training thread started")

    yield


__all__ = ["initialize_app_state", "lifespan"]
