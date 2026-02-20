"""
Health check endpoints for DocWain API.

Provides:
- /health - Basic health check
- /health/detailed - Detailed component status
- /health/ready - Kubernetes readiness probe
- /health/live - Kubernetes liveness probe
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["Health"])


class ReindexRequest(BaseModel):
    subscription_id: str
    profile_id: str
    document_id: Optional[str] = None  # None = all documents in profile


def _check_qdrant_connection() -> Dict[str, Any]:
    """Check Qdrant vector database connectivity."""
    try:
        from src.api.config import Config
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
            timeout=5,
        )
        # Quick health check - just verify we can connect
        collections = client.get_collections()
        return {
            "status": "healthy",
            "collections_count": len(collections.collections),
            "url": Config.Qdrant.URL[:50] + "..." if len(Config.Qdrant.URL) > 50 else Config.Qdrant.URL,
        }
    except Exception as e:
        logger.warning("Qdrant health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e)[:100],
        }


def _check_mongodb_connection() -> Dict[str, Any]:
    """Check MongoDB connectivity."""
    try:
        from src.api.config import Config
        from pymongo import MongoClient

        client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
        # Ping the database
        client.admin.command("ping")
        return {
            "status": "healthy",
            "database": Config.MongoDB.DB,
        }
    except Exception as e:
        logger.warning("MongoDB health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e)[:100],
        }


def _check_redis_connection() -> Dict[str, Any]:
    """Check Redis cache connectivity."""
    try:
        from src.api.config import Config
        import redis

        client = redis.Redis(
            host=Config.Redis.HOST,
            port=Config.Redis.PORT,
            password=Config.Redis.PASSWORD,
            ssl=Config.Redis.SSL,
            socket_timeout=5,
        )
        client.ping()
        return {
            "status": "healthy",
            "host": Config.Redis.HOST,
        }
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e)[:100],
        }


def _check_ollama_connection() -> Dict[str, Any]:
    """Check Ollama LLM service connectivity."""
    try:
        import ollama

        models = ollama.list()
        model_count = len(models.get("models", []))
        return {
            "status": "healthy",
            "models_available": model_count,
        }
    except Exception as e:
        logger.warning("Ollama health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e)[:100],
        }


def _validate_configuration() -> Dict[str, Any]:
    """Validate critical configuration settings."""
    issues: List[str] = []

    try:
        from src.api.settings import get_settings
        settings = get_settings()
        missing = settings.validate_required()
        if missing:
            issues.extend(f"Missing: {key}" for key in missing)
    except Exception as e:
        issues.append(f"Settings error: {str(e)[:50]}")

    return {
        "status": "healthy" if not issues else "warning",
        "issues": issues if issues else None,
    }


@health_router.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns overall health status based on app state.
    """
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


@health_router.get("/health/detailed")
def health_check_detailed(request: Request) -> Dict[str, Any]:
    """
    Detailed health check with component status.

    Checks connectivity to all critical services:
    - Qdrant (vector database)
    - MongoDB (document store)
    - Redis (cache)
    - Ollama (LLM)

    Returns:
        Detailed health status for each component.
    """
    start_time = time.time()

    # Get correlation ID from request state
    correlation_id = getattr(request.state, "correlation_id", None)

    # Check each component
    components = {
        "qdrant": _check_qdrant_connection(),
        "mongodb": _check_mongodb_connection(),
        "redis": _check_redis_connection(),
        "ollama": _check_ollama_connection(),
        "configuration": _validate_configuration(),
    }

    # Determine overall status
    statuses = [c.get("status", "unknown") for c in components.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "status": overall,
        "timestamp": time.time(),
        "check_duration_ms": round(elapsed_ms, 2),
        "correlation_id": correlation_id,
        "components": components,
    }


@health_router.get("/health/ready")
def readiness_check(request: Request, response: Response) -> Dict[str, Any]:
    """
    Kubernetes readiness probe.

    Returns 200 if the application is ready to receive traffic.
    Returns 503 if critical services are unavailable.

    This endpoint checks:
    - Qdrant connectivity (required for RAG)
    - MongoDB connectivity (required for document access)
    """
    issues: List[str] = []

    # Check Qdrant
    qdrant_status = _check_qdrant_connection()
    if qdrant_status.get("status") != "healthy":
        issues.append("Qdrant unavailable")

    # Check MongoDB
    mongodb_status = _check_mongodb_connection()
    if mongodb_status.get("status") != "healthy":
        issues.append("MongoDB unavailable")

    if issues:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ready": False,
            "issues": issues,
        }

    return {
        "ready": True,
    }


@health_router.get("/health/live")
def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe.

    Returns 200 if the application process is alive.
    This is a simple check that always returns healthy
    unless the process itself is dead.
    """
    return {
        "alive": True,
        "timestamp": time.time(),
    }


@health_router.post("/api/admin/reindex")
async def reindex_profile(body: ReindexRequest) -> Dict[str, Any]:
    """
    Clear Qdrant data for a profile (or single document) and re-embed from pickle.

    This forces re-ingestion through the clean embedding pipeline so that
    payloads match the new slim schema (section_kind, doc_domain, etc.).
    """
    from src.api.embedding_service import embed_documents
    from src.api.vector_store import QdrantVectorStore, build_collection_name, build_qdrant_filter

    subscription_id = body.subscription_id
    profile_id = body.profile_id
    document_id = body.document_id

    deleted_count = 0
    try:
        qdrant_client = QdrantVectorStore()
        collection = build_collection_name(subscription_id)

        if document_id:
            # Single document re-index
            qdrant_client.delete_document(
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=document_id,
            )
            deleted_count = 1
            logger.info(
                "Reindex: deleted Qdrant points for document %s",
                document_id,
            )
        else:
            # All documents in profile: delete via profile filter
            delete_filter = build_qdrant_filter(
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            )
            qdrant_client.client.delete(
                collection_name=collection,
                points_selector=delete_filter,
                wait=True,
            )
            logger.info(
                "Reindex: deleted all Qdrant points for profile %s/%s",
                subscription_id,
                profile_id,
            )
    except Exception as exc:
        logger.error("Reindex: Qdrant delete failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "phase": "delete",
            "error": str(exc)[:200],
        }

    # Re-embed from pickle via the standard embedding pipeline
    try:
        result = embed_documents(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )
        logger.info("Reindex: embedding complete for %s/%s", subscription_id, profile_id)
        return {
            "status": "success",
            "deleted_from_qdrant": True,
            "embed_result": result,
        }
    except Exception as exc:
        logger.error("Reindex: embedding failed: %s", exc, exc_info=True)
        return {
            "status": "partial",
            "phase": "embed",
            "deleted_from_qdrant": True,
            "error": str(exc)[:200],
        }


class DPIETrainRequest(BaseModel):
    subscription_id: str
    profile_id: str


@health_router.post("/api/admin/dpie/train")
async def train_dpie(body: DPIETrainRequest) -> Dict[str, Any]:
    """
    Manually trigger DPIE model training from Qdrant data.

    Bootstraps training data from the specified profile's Qdrant vectors
    and trains all DPIE ML models (document classifier, section detector,
    section kind classifier, entity recognizer).
    """
    from src.api import dw_newron
    from src.api.rag_state import get_app_state

    subscription_id = body.subscription_id
    profile_id = body.profile_id

    state = get_app_state()
    qdrant_client = (state.qdrant_client if state else None) or dw_newron.get_qdrant_client()
    embedding_model = (state.embedding_model if state else None) or dw_newron.get_model()

    if not qdrant_client or not embedding_model:
        return {"status": "error", "error": "qdrant_client or embedding_model not available"}

    try:
        from src.intelligence.dpie_integration import DPIERegistry

        registry = DPIERegistry.get()
        stats = registry.train_and_save(
            qdrant_client=qdrant_client,
            sentence_model=embedding_model,
            collection_name=subscription_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )

        return {
            "status": "success",
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "is_loaded": registry.is_loaded,
            "training_stats": {
                k: v if not isinstance(v, dict) else {
                    sk: sv[:5] if isinstance(sv, list) and len(sv) > 5 else sv
                    for sk, sv in v.items()
                }
                for k, v in stats.items()
            },
        }
    except Exception as exc:
        logger.error("DPIE training failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc)[:300],
        }


@health_router.get("/api/admin/dpie/status")
async def dpie_status() -> Dict[str, Any]:
    """Return the current DPIE model status."""
    try:
        from src.intelligence.dpie_integration import DPIERegistry

        registry = DPIERegistry.get()
        return {
            "is_loaded": registry.is_loaded,
            "model_dir": getattr(registry, "_model_dir", ""),
        }
    except ImportError:
        return {"is_loaded": False, "error": "dpie_integration not available"}


@health_router.get("/api/admin/metrics/quality")
async def quality_metrics(hours: int = 24) -> Dict[str, Any]:
    """Return pipeline quality metrics aggregated over the last N hours."""
    try:
        from src.metrics.quality_metrics import get_quality_summary
        return get_quality_summary(hours=hours)
    except Exception as exc:
        return {"error": str(exc)}


@health_router.get("/api/admin/llm/status")
async def llm_status() -> Dict[str, Any]:
    """Return the current LLM gateway status and health."""
    try:
        from src.llm.gateway import get_llm_gateway
        gateway = get_llm_gateway()
        if gateway is None:
            return {"status": "not_initialized"}
        info: Dict[str, Any] = {
            "status": "initialized",
            "backend": getattr(gateway, "backend", "unknown"),
            "model": getattr(gateway, "model_name", "unknown"),
            "name": getattr(gateway, "name", "unknown"),
        }
        if hasattr(gateway, "get_stats"):
            info["stats"] = gateway.get_stats()
        return info
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@health_router.get("/api/admin/teams/status")
async def teams_status() -> Dict[str, Any]:
    """Return Teams bot configuration status."""
    from src.api.config import Config

    app_id = getattr(Config.Teams, "BOT_APP_ID", None) or ""
    app_password = getattr(Config.Teams, "BOT_APP_PASSWORD", None) or ""
    shared_secret = getattr(Config.Teams, "SHARED_SECRET", None) or ""
    masked_app_id = (app_id[:8] + "...") if len(app_id) > 8 else app_id

    bot_adapter_ready = False
    try:
        from src.teams.bot_app import BOT_CREDENTIALS_CONFIGURED, bot_adapter as _ba

        bot_adapter_ready = _ba is not None and BOT_CREDENTIALS_CONFIGURED
    except ImportError:
        pass

    redis_status = "unknown"
    try:
        from src.teams.state import TeamsStateStore

        store = TeamsStateStore()
        if store.client:
            store.client.ping()
            redis_status = "healthy"
        else:
            redis_status = "in-memory fallback"
    except Exception as exc:
        redis_status = f"unhealthy: {str(exc)[:80]}"

    return {
        "bot_credentials_configured": bool(app_id and app_password),
        "app_id_masked": masked_app_id or "(not set)",
        "shared_secret_configured": bool(shared_secret),
        "bot_adapter_ready": bot_adapter_ready,
        "state_store_redis": redis_status,
        "diag_mode": getattr(Config.Teams, "DIAG_MODE", False),
        "session_as_subscription": getattr(Config.Teams, "SESSION_AS_SUBSCRIPTION", True),
        "profile_per_user": getattr(Config.Teams, "PROFILE_PER_USER", True),
        "default_model": getattr(Config.Teams, "DEFAULT_MODEL", ""),
        "default_persona": getattr(Config.Teams, "DEFAULT_PERSONA", ""),
    }


@health_router.get("/api/admin/multi-agent/status")
async def multi_agent_status() -> Dict[str, Any]:
    """Return multi-agent gateway status and per-role statistics."""
    from src.api.config import Config

    enabled = getattr(Config, "MultiAgent", None) and getattr(Config.MultiAgent, "ENABLED", False)
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    try:
        from src.api.rag_state import get_app_state
        state = get_app_state()
        gw = getattr(state, "multi_agent_gateway", None) if state else None
        if gw is None:
            return {"enabled": True, "status": "not_initialized"}
        stats = gw.get_stats() if hasattr(gw, "get_stats") else {}
        return {
            "enabled": True,
            "status": "initialized",
            **stats,
        }
    except Exception as exc:
        return {"enabled": True, "status": "error", "error": str(exc)[:200]}


@health_router.get("/api/admin/intent-classifier/status")
async def intent_classifier_status() -> Dict[str, Any]:
    """Return the trained intent classifier status."""
    try:
        from src.intent.intent_classifier import get_intent_classifier
        clf = get_intent_classifier()
        if clf is None:
            return {"status": "not_initialized", "trained": False}
        return {
            "status": "ready" if clf._trained else "untrained",
            "trained": clf._trained,
            "input_dim": clf.input_dim,
            "hidden_dim": clf.hidden_dim,
            "intent_classes": clf.intent_names,
            "domain_classes": clf.domain_names,
        }
    except ImportError:
        return {"status": "unavailable", "error": "intent_classifier module not found"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


@health_router.get("/api/admin/line-classifier/status")
async def line_classifier_status() -> Dict[str, Any]:
    """Return the trained line role classifier status."""
    try:
        from src.rag_v3.line_classifier import get_line_classifier, HEAD_NAMES
        clf = get_line_classifier()
        if clf is None:
            return {"status": "not_initialized", "trained": False}
        return {
            "status": "ready" if clf._trained else "untrained",
            "trained": clf._trained,
            "input_dim": clf.input_dim,
            "hidden_dim": clf.hidden_dim,
            "heads": list(HEAD_NAMES.keys()),
        }
    except ImportError:
        return {"status": "unavailable", "error": "line_classifier module not found"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


@health_router.get("/api/admin/kg/status")
async def kg_status() -> Dict[str, Any]:
    """Return Knowledge Graph status, connectivity, and statistics."""
    from src.api.config import Config

    enabled = getattr(Config.KnowledgeGraph, "ENABLED", False)
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    result: Dict[str, Any] = {"enabled": True}

    try:
        from src.kg.neo4j_store import Neo4jStore
        store = Neo4jStore()

        # Connection test
        store.run_query("RETURN 1 AS ok")
        result["neo4j_connected"] = True

        # Entity and document counts
        stats_rows = store.run_query(
            "OPTIONAL MATCH (d:Document) "
            "WITH count(d) AS doc_count "
            "OPTIONAL MATCH (e:Entity) "
            "WITH doc_count, count(e) AS entity_count "
            "OPTIONAL MATCH ()-[r:MENTIONS]->() "
            "RETURN doc_count, entity_count, count(r) AS mention_count"
        )
        if stats_rows:
            row = stats_rows[0]
            result["documents"] = row.get("doc_count", 0)
            result["entities"] = row.get("entity_count", 0)
            result["mentions"] = row.get("mention_count", 0)

        # Last sync timestamp
        try:
            state = store.get_state("kg_sync")
            result["last_sync_at"] = getattr(state, "last_sync_at", None)
        except Exception:  # noqa: BLE001
            result["last_sync_at"] = None

        # Graph augmenter availability
        from src.api.rag_state import get_app_state
        app_state = get_app_state()
        result["graph_augmenter_ready"] = getattr(app_state, "graph_augmenter", None) is not None

        result["status"] = "healthy"
        store.close()
    except Exception as exc:  # noqa: BLE001
        result["status"] = "unhealthy"
        result["neo4j_connected"] = False
        result["error"] = str(exc)[:200]

    return result


__all__ = ["health_router"]
