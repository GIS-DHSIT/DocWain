"""
Health check endpoints for DocWain API.

Provides:
- /health - Basic health check
- /health/detailed - Detailed component status
- /health/ready - Kubernetes readiness probe
- /health/live - Kubernetes liveness probe
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

logger = get_logger(__name__)

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

def _check_neo4j_connection() -> Dict[str, Any]:
    """Check Neo4j graph database connectivity."""
    try:
        from src.api.config import Config
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            Config.Neo4j.URI,
            auth=(Config.Neo4j.USER, Config.Neo4j.PASSWORD),
        )
        driver.verify_connectivity()
        driver.close()
        return {
            "status": "healthy",
            "uri": Config.Neo4j.URI,
        }
    except Exception as e:
        logger.warning("Neo4j health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e)[:100],
        }

def _get_celery_queue_stats() -> Dict[str, Any]:
    """Get Celery worker and queue statistics."""
    try:
        from src.celery_app import app as celery_app

        inspect = celery_app.control.inspect(timeout=3.0)
        stats = inspect.stats()
        if stats is None:
            return {"status": "unavailable", "reason": "no workers responding"}

        active_queues = inspect.active_queues() or {}
        active_tasks = inspect.active() or {}

        worker_count = len(stats)
        queue_names = set()
        for worker_queues in active_queues.values():
            for q in worker_queues:
                queue_names.add(q.get("name", "unknown"))

        total_active = sum(len(tasks) for tasks in active_tasks.values())

        return {
            "status": "ok",
            "workers": worker_count,
            "queues": sorted(queue_names),
            "active_tasks": total_active,
        }
    except Exception as e:
        logger.warning("Celery health check failed: %s", e)
        return {
            "status": "unavailable",
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
    Production health check endpoint.

    Checks all backend services concurrently and returns per-component status.
    Each check is independent — one failing service will not prevent
    the others from being reported.
    """
    import concurrent.futures

    def _safe_check(fn):
        try:
            result = fn()
            return result.get("status", "ok") if isinstance(result, dict) else "ok"
        except Exception as e:
            return f"error: {str(e)[:80]}"

    def _safe_check_full(fn):
        try:
            return fn()
        except Exception as e:
            return {"status": "error", "error": str(e)[:80]}

    # Run all checks concurrently to reduce total latency from ~6s to ~1s
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        redis_future = executor.submit(_safe_check, _check_redis_connection)
        mongodb_future = executor.submit(_safe_check, _check_mongodb_connection)
        qdrant_future = executor.submit(_safe_check, _check_qdrant_connection)
        neo4j_future = executor.submit(_safe_check, _check_neo4j_connection)
        ollama_future = executor.submit(_safe_check, _check_ollama_connection)
        celery_future = executor.submit(_safe_check_full, _get_celery_queue_stats)

    return {
        "api": "ok",
        "redis": redis_future.result(timeout=6),
        "mongodb": mongodb_future.result(timeout=6),
        "qdrant": qdrant_future.result(timeout=6),
        "neo4j": neo4j_future.result(timeout=6),
        "ollama_local": ollama_future.result(timeout=6),
        "celery": celery_future.result(timeout=6),
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

@health_router.post("/admin/reindex")
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

@health_router.get("/admin/metrics/quality")
async def quality_metrics(hours: int = 24) -> Dict[str, Any]:
    """Return pipeline quality metrics aggregated over the last N hours."""
    try:
        from src.metrics.quality_metrics import get_quality_summary
        return get_quality_summary(hours=hours)
    except Exception as exc:
        return {"error": str(exc)}

@health_router.get("/admin/llm/status")
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

@health_router.get("/admin/teams/status")
async def teams_status() -> Dict[str, Any]:
    """Return Teams bot configuration status."""
    from src.api.config import Config

    app_id = getattr(Config.Teams, "BOT_APP_ID", None) or ""
    app_password = getattr(Config.Teams, "BOT_APP_PASSWORD", None) or ""
    shared_secret = getattr(Config.Teams, "SHARED_SECRET", None) or ""
    tenant_id = getattr(Config.Teams, "BOT_APP_TENANT_ID", None) or ""
    app_type = getattr(Config.Teams, "BOT_APP_TYPE", None) or "unknown"
    masked_app_id = (app_id[:8] + "...") if len(app_id) > 8 else app_id
    masked_tenant = (tenant_id[:8] + "...") if len(tenant_id) > 8 else tenant_id

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
        "app_type": app_type,
        "tenant_id_masked": masked_tenant or "(not set)",
        "tenant_id_configured": bool(tenant_id),
        "shared_secret_configured": bool(shared_secret),
        "bot_adapter_ready": bot_adapter_ready,
        "state_store_redis": redis_status,
        "diag_mode": getattr(Config.Teams, "DIAG_MODE", False),
        "session_as_subscription": getattr(Config.Teams, "SESSION_AS_SUBSCRIPTION", True),
        "profile_per_user": getattr(Config.Teams, "PROFILE_PER_USER", True),
        "default_model": getattr(Config.Teams, "DEFAULT_MODEL", ""),
        "default_persona": getattr(Config.Teams, "DEFAULT_PERSONA", ""),
    }

@health_router.get("/admin/multi-agent/status")
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

@health_router.get("/admin/intent-classifier/status")
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

@health_router.get("/admin/line-classifier/status")
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

@health_router.get("/admin/kg/status")
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

@health_router.get("/admin/task-routing/status")
async def task_routing_status() -> Dict[str, Any]:
    """Return task-aware model routing status, discovered models, and per-task statistics."""
    from src.api.config import Config

    enabled = getattr(Config, "TaskRouting", None) and getattr(Config.TaskRouting, "ENABLED", False)
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    try:
        from src.llm.model_registry import get_model_registry
        from src.llm.task_router import TaskType

        registry = get_model_registry()
        if registry is None:
            return {"enabled": True, "status": "registry_not_initialized"}

        # Check if the active gateway is task-aware
        from src.llm.gateway import get_llm_gateway
        gateway = get_llm_gateway()
        is_task_aware = hasattr(gateway, "_router")

        models = [
            {
                "name": m.name,
                "size_bytes": m.size_bytes,
                "speed_tier": m.speed_tier,
                "strengths": m.strengths,
                "available": m.available,
            }
            for m in registry.get_available()
        ]

        routing_table = {}
        if is_task_aware:
            for task in TaskType:
                routing_table[task.value] = gateway._router.explain(task)

        task_stats = {}
        if is_task_aware and hasattr(gateway, "get_task_stats"):
            task_stats = gateway.get_task_stats()

        return {
            "enabled": is_task_aware,
            "status": "active" if is_task_aware else "registry_only",
            "models_discovered": len(models),
            "models": models,
            "routing_table": routing_table,
            "task_stats": task_stats,
        }
    except Exception as exc:
        return {"enabled": True, "status": "error", "error": str(exc)[:200]}

@health_router.get("/admin/tools/status")
async def tools_status() -> Dict[str, Any]:
    """Return registered tool count and intelligence profile summary."""
    try:
        from src.tools.intelligence import TOOL_PROFILES
        profiles_info = [
            {"name": p.name, "domain": p.domain, "intelligence": bool(p.system_prompt)}
            for p in TOOL_PROFILES.values()
        ]
    except ImportError:
        profiles_info = []

    registered_count = 0
    try:
        from src.tools.base import registry
        registered_count = len(registry._registry)
    except Exception:
        pass

    return {
        "registered_tools": registered_count,
        "intelligence_profiles": len(profiles_info),
        "tools": profiles_info,
    }

@health_router.get("/admin/enterprise-intelligence/status")
async def enterprise_intelligence_status() -> Dict[str, Any]:
    """Return status of enterprise intelligence features."""
    from src.api.config import Config

    features = {}

    # Feature 1: Follow-Up Suggestions
    fu_cls = getattr(Config, "FollowUp", None)
    features["followup_suggestions"] = {
        "enabled": getattr(fu_cls, "ENABLED", False) if fu_cls else False,
        "max_suggestions": getattr(fu_cls, "MAX_SUGGESTIONS", 3) if fu_cls else 3,
    }

    # Feature 2: Insights Tool
    try:
        from src.tools.base import registry
        features["insights_tool"] = {
            "registered": "insights" in registry._registry,
        }
    except Exception:
        features["insights_tool"] = {"registered": False}

    # Feature 3: Action Items Tool
    try:
        from src.tools.base import registry
        features["action_items_tool"] = {
            "registered": "action_items" in registry._registry,
        }
    except Exception:
        features["action_items_tool"] = {"registered": False}

    # Feature 4: Query Planner
    qp_cls = getattr(Config, "QueryPlanner", None)
    features["query_planner"] = {
        "enabled": getattr(qp_cls, "ENABLED", False) if qp_cls else False,
        "max_steps": getattr(qp_cls, "MAX_STEPS", 3) if qp_cls else 3,
    }

    # Feature 5: Hallucination Corrector
    hc_cls = getattr(Config, "HallucinationCorrector", None)
    features["hallucination_corrector"] = {
        "enabled": getattr(hc_cls, "ENABLED", False) if hc_cls else False,
        "score_threshold": getattr(hc_cls, "SCORE_THRESHOLD", 0.5) if hc_cls else 0.5,
        "max_corrections": getattr(hc_cls, "MAX_CORRECTIONS", 3) if hc_cls else 3,
    }

    # Feature 6: Confidence Scoring
    conf_cls = getattr(Config, "Confidence", None)
    features["confidence_scoring"] = {
        "enabled": getattr(conf_cls, "ENABLED", False) if conf_cls else False,
    }

    enabled_count = sum(
        1 for f in features.values()
        if f.get("enabled") or f.get("registered")
    )

    return {
        "status": "active" if enabled_count > 0 else "disabled",
        "features_enabled": enabled_count,
        "features_total": len(features),
        "features": features,
    }

@health_router.get("/admin/web-search/status")
async def web_search_status() -> Dict[str, Any]:
    """Return web search configuration status."""
    from src.api.config import Config

    ws = getattr(Config, "WebSearch", None)
    if ws is None:
        return {"enabled": False, "status": "not_configured"}

    tavily_configured = bool(getattr(ws, "TAVILY_API_KEY", ""))

    return {
        "enabled": getattr(ws, "ENABLED", False),
        "engine": getattr(ws, "ENGINE", "duckduckgo"),
        "tavily_configured": tavily_configured,
        "max_results": getattr(ws, "MAX_RESULTS", 5),
        "timeout": getattr(ws, "TIMEOUT", 10.0),
        "max_url_fetch_chars": getattr(ws, "MAX_URL_FETCH_CHARS", 6000),
        "fallback_on_no_results": getattr(ws, "FALLBACK_ON_NO_RESULTS", True),
    }

@health_router.get("/admin/vision-ocr/status")
async def vision_ocr_status() -> Dict[str, Any]:
    """Return vision OCR engine status."""
    from src.api.config import Config

    cfg = getattr(Config, "VisionOCR", None)
    if cfg is None:
        return {"enabled": False, "status": "not_configured"}

    result: Dict[str, Any] = {
        "enabled": getattr(cfg, "ENABLED", False),
        "model": getattr(cfg, "MODEL", "glm-ocr:latest"),
        "ocr_content_images": getattr(cfg, "OCR_CONTENT_IMAGES", True),
        "fallback_to_traditional": getattr(cfg, "FALLBACK_TO_TRADITIONAL", True),
    }

    try:
        from src.llm.vision_ocr import get_vision_ocr_client
        client = get_vision_ocr_client()
        result["available"] = client.is_available() if client else False
        result["status"] = "active" if result["available"] else "model_not_available"
    except Exception:
        result["available"] = False
        result["status"] = "not_initialized"

    return result

@health_router.get("/admin/domain-knowledge/status")
async def domain_knowledge_status() -> Dict[str, Any]:
    """Return domain knowledge engine status."""
    from src.api.config import Config

    dk = getattr(Config, "DomainKnowledge", None)
    if dk is None:
        return {"enabled": False, "status": "not_configured"}

    result: Dict[str, Any] = {
        "enabled": getattr(dk, "ENABLED", False),
        "web_enrichment": getattr(dk, "WEB_ENRICHMENT", False),
        "inject_into_prompts": getattr(dk, "INJECT_INTO_PROMPTS", True),
        "cache_ttl": getattr(dk, "CACHE_TTL", 3600),
    }

    try:
        from src.intelligence.domain_knowledge import get_domain_knowledge_provider
        provider = get_domain_knowledge_provider()
        result["supported_domains"] = provider.supported_domains
        result["status"] = "active"
    except Exception:
        result["status"] = "not_initialized"

    return result

@health_router.get("/admin/context-understanding/status")
async def context_understanding_status() -> Dict[str, Any]:
    """Return ML-based context understanding module status."""
    result: Dict[str, Any] = {"enabled": True, "status": "active"}
    try:
        from src.intelligence.context_understanding import understand_context
        result["capabilities"] = [
            "semantic_clustering",
            "query_evidence_alignment",
            "entity_salience",
            "cross_document_relationships",
            "structured_fact_extraction",
            "context_distillation",
        ]
    except ImportError:
        result["enabled"] = False
        result["status"] = "not_available"
    return result

@health_router.get("/admin/domain-agents/status")
async def domain_agents_status() -> Dict[str, Any]:
    """Return domain agent availability and capabilities."""
    result: Dict[str, Any] = {"enabled": True, "status": "active"}
    try:
        from src.agentic.domain_agents import list_available_agents
        agents = list_available_agents()
        result["agents"] = {domain: caps for domain, caps in agents.items()}
        result["agent_count"] = len(agents)
        result["total_capabilities"] = sum(len(caps) for caps in agents.values())
    except ImportError:
        result["enabled"] = False
        result["status"] = "not_available"
    except Exception:
        result["status"] = "error"
    return result

@health_router.get("/admin/profile-domain/status")
async def profile_domain_status() -> Dict[str, Any]:
    """Return profile domain tagging system status."""
    result: Dict[str, Any] = {"enabled": True, "status": "active"}
    try:
        from src.api.config import Config
        cfg = getattr(Config, "ProfileDomain", None)
        if cfg is None or not getattr(cfg, "ENABLED", True):
            result["enabled"] = False
            result["status"] = "disabled"
            return result
        result["majority_threshold"] = getattr(cfg, "MAJORITY_THRESHOLD", 0.80)
        result["min_signal_score"] = getattr(cfg, "MIN_SIGNAL_SCORE", 0.25)
        from src.profiles.profile_domain_tagger import compute_profile_domain
        result["module_available"] = True
    except ImportError:
        result["enabled"] = False
        result["status"] = "not_available"
        result["module_available"] = False
    except Exception:
        result["status"] = "error"
    return result

@health_router.get("/admin/thinking-model/status")
async def thinking_model_status() -> Dict[str, Any]:
    """Return lfm2.5-thinking MoE sub-agent status."""
    from src.api.config import Config

    cfg = getattr(Config, "ThinkingModel", None)
    if cfg is None:
        return {"enabled": False, "status": "not_configured"}

    result: Dict[str, Any] = {
        "enabled": getattr(cfg, "ENABLED", False),
        "model": getattr(cfg, "MODEL", "lfm2.5-thinking:latest"),
        "use_for_judging": getattr(cfg, "USE_FOR_JUDGING", True),
        "use_for_agent_steps": getattr(cfg, "USE_FOR_AGENT_STEPS", True),
        "use_for_verification": getattr(cfg, "USE_FOR_VERIFICATION", True),
        "keep_alive": getattr(cfg, "KEEP_ALIVE", "24h"),
        "temperature": getattr(cfg, "DEFAULT_TEMPERATURE", 0.05),
    }

    try:
        import ollama
        ollama.show(result["model"])
        result["available"] = True
        result["status"] = "active"
    except Exception:
        result["available"] = False
        result["status"] = "model_not_available"

    return result

@health_router.get("/admin/vision-analysis/status")
async def vision_analysis_status() -> Dict[str, Any]:
    """Return glm-ocr enhanced vision analysis status."""
    from src.api.config import Config

    cfg = getattr(Config, "VisionAnalysis", None)
    if cfg is None:
        return {"enabled": False, "status": "not_configured"}

    result: Dict[str, Any] = {
        "enabled": getattr(cfg, "ENABLED", False),
        "model": getattr(cfg, "MODEL", "glm-ocr:latest"),
        "chart_analysis": getattr(cfg, "CHART_ANALYSIS", True),
        "table_analysis": getattr(cfg, "TABLE_ANALYSIS", True),
        "diagram_analysis": getattr(cfg, "DIAGRAM_ANALYSIS", True),
        "photo_analysis": getattr(cfg, "PHOTO_ANALYSIS", True),
        "max_image_tokens": getattr(cfg, "MAX_IMAGE_TOKENS", 4096),
    }

    try:
        from src.llm.vision_ocr import get_vision_ocr_client
        client = get_vision_ocr_client()
        result["available"] = client.is_available() if client else False
        result["status"] = "active" if result["available"] else "model_not_available"
    except Exception:
        result["available"] = False
        result["status"] = "not_initialized"

    return result

__all__ = ["health_router"]
