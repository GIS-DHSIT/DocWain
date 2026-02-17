from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from src.api import dw_newron
from src.api.config import Config
from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS, ensure_payload_indexes
from src.api.rag_state import AppState, activate_singleton_guard, register_instance_ids, set_app_state
from src.llm.gateway import create_llm_gateway, set_llm_gateway

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
    llm_gateway = None
    try:
        llm_gateway = create_llm_gateway(model_name=default_model)
        set_llm_gateway(llm_gateway)
        # Backward compat: ollama_client points to the gateway (duck-typed)
        ollama_client = llm_gateway
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM gateway init failed, trying direct Ollama: %s", exc)
        try:
            ollama_client = dw_newron.create_llm_client(default_model)
        except Exception as exc2:  # noqa: BLE001
            logger.error("Ollama client init also failed: %s", exc2)

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

    # Multi-agent gateway (role-specific Ollama models)
    multi_agent_gateway = None
    if getattr(Config, "MultiAgent", None) and getattr(Config.MultiAgent, "ENABLED", False):
        try:
            from src.llm.multi_agent import create_multi_agent_gateway
            multi_agent_gateway = create_multi_agent_gateway(fallback_gateway=llm_gateway)
            logger.info("Multi-agent gateway initialized with role-specific models")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multi-agent gateway init failed (continuing with single-model): %s", exc)

    state = AppState(
        embedding_model=embedding_model,
        reranker=cross_encoder,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        ollama_client=ollama_client,
        rag_system=rag_system,
        llm_gateway=llm_gateway,
        multi_agent_gateway=multi_agent_gateway,
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
            # Discover all profile_ids (only fetches identity field, not full payloads)
            profile_ids = _discover_profile_ids(qdrant_client, subscription_id)
            if not profile_ids:
                logger.debug("DPIE: no profiles found in collection %s; skipping", subscription_id)
                continue

            for profile_id in profile_ids:
                logger.info("DPIE: auto-training for subscription=%s profile=%s", subscription_id, profile_id)
                try:
                    registry.ensure_ready(
                        qdrant_client=qdrant_client,
                        sentence_model=embedding_model,
                        collection_name=subscription_id,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                    )
                    logger.info("DPIE: models ready for subscription=%s profile=%s", subscription_id, profile_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DPIE: training failed for %s/%s: %s", subscription_id, profile_id, exc)

    except ImportError:
        logger.debug("DPIE: intelligence.dpie_integration not available; skipping")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DPIE: background bootstrap failed: %s", exc)


def _discover_profile_ids(qdrant_client: Any, collection_name: str) -> List[str]:
    """Scroll points to discover all unique profile_ids in the collection.

    Only fetches the ``profile_id`` payload field (never full payloads) to
    avoid leaking document content into memory during bootstrap.
    """
    try:
        seen: set[str] = set()
        offset = None
        while len(seen) < 50:  # cap to avoid unbounded scroll
            result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=20,
                with_payload=["profile_id"],
                with_vectors=False,
                offset=offset,
            )
            points, next_offset = result if result else ([], None)
            for point in points:
                payload = getattr(point, "payload", None) or {}
                pid = payload.get("profile_id")
                if pid:
                    seen.add(str(pid))
            if not next_offset or not points:
                break
            offset = next_offset
        return list(seen)
    except Exception:
        pass
    return []


def _precreate_subscription_collections(qdrant_client) -> None:
    """Pre-create Qdrant collections for active subscriptions to avoid 404s on first document."""
    try:
        from src.api.vector_store import build_collection_name, VectorStoreClient
        from src.api.config import Config
        db = Config.MongoDB.get_db()
        if db is None:
            return
        subs_coll = db.get_collection("subscriptions")
        active_subs = subs_coll.find({"status": {"$in": ["active", "ACTIVE", None]}}, {"_id": 1})
        existing = set()
        try:
            for col in (qdrant_client.get_collections().collections or []):
                existing.add(getattr(col, "name", str(col)))
        except Exception:
            return
        vec_size = int(getattr(Config.Model, "EMBEDDING_DIM", 1024))
        vs = VectorStoreClient(client=qdrant_client)
        created = 0
        for sub in active_subs:
            sub_id = str(sub["_id"])
            coll_name = build_collection_name(sub_id)
            if coll_name not in existing:
                try:
                    vs.ensure_collection(coll_name, vec_size)
                    created += 1
                except Exception as exc:
                    logger.warning("Failed to pre-create collection %s: %s", coll_name, exc)
        if created:
            logger.info("Pre-created %d Qdrant collections for active subscriptions", created)
    except Exception as exc:
        logger.debug("Subscription collection pre-creation skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = initialize_app_state(app)

    # Pre-create Qdrant collections for active subscriptions (non-blocking)
    if state.qdrant_client:
        try:
            _precreate_subscription_collections(state.qdrant_client)
        except Exception as exc:
            logger.debug("Subscription collection pre-creation skipped: %s", exc)

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

    # Run startup checks (migrated from @app.on_event("startup") in main.py)
    try:
        from src.api.logging_config import configure_logging
        configure_logging(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            json_format=os.getenv("JSON_LOGGING", "false").lower() in {"1", "true", "yes"},
            include_correlation_id=True,
        )
    except Exception as exc:
        logger.debug("Logging configuration skipped: %s", exc)
    try:
        from src.storage.blob_persistence import validate_storage_configured_once, validate_containers_once
        validate_storage_configured_once()
    except Exception as exc:
        logger.warning("Azure blob storage configuration check skipped: %s", exc)
    try:
        validate_containers_once()
    except Exception as exc:
        logger.warning("Azure blob container validation skipped: %s", exc)
    try:
        from src.api.dataHandler import clear_legacy_vetting_metadata, log_legacy_vetting_notice_if_missing
        clear_legacy_vetting_metadata()
    except Exception as exc:
        logger.warning("Legacy metadata cleanup skipped: %s", exc)
    try:
        log_legacy_vetting_notice_if_missing()
    except Exception as exc:
        logger.warning("Legacy config notice skipped: %s", exc)
    logger.info("Startup checks completed")

    yield

    # Shutdown (currently no-op)
    logger.info("DocWain API shutting down")


__all__ = ["initialize_app_state", "lifespan"]
