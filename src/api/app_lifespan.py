from __future__ import annotations

from src.utils.logging_utils import get_logger
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

logger = get_logger(__name__)

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

    # Load embedding model + cross-encoder in parallel (they're CPU-bound and independent)
    def _load_embedding():
        nonlocal embedding_model
        try:
            embedding_model = dw_newron.get_model()
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding model init failed: %s", exc)

    def _load_cross_encoder():
        nonlocal cross_encoder
        try:
            cross_encoder = dw_newron.get_cross_encoder()
        except Exception as exc:  # noqa: BLE001
            logger.error("Reranker init failed: %s", exc)

    t_embed = threading.Thread(target=_load_embedding, daemon=True)
    t_rerank = threading.Thread(target=_load_cross_encoder, daemon=True)
    t_embed.start()
    t_rerank.start()

    # While models load, initialize clients (I/O bound, fast)
    try:
        qdrant_client = dw_newron.get_qdrant_client()
    except Exception as exc:  # noqa: BLE001
        logger.error("Qdrant client init failed: %s", exc)
    try:
        redis_client = dw_newron.get_redis_client()
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis client init failed: %s", exc)

    # Wait for model loading to finish
    t_embed.join()
    t_rerank.join()
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
        # Warm up Ollama so the model is loaded before the first query
        if hasattr(llm_gateway, "warm_up"):
            def _do_warmup():
                try:
                    llm_gateway.warm_up()
                    logger.info("LLM model warm-up completed")
                except Exception:  # noqa: BLE001
                    logger.debug("LLM warm-up skipped (model will load on first query)")
            _warmup_thread = threading.Thread(target=_do_warmup, daemon=True)
            _warmup_thread.start()
            _warmup_thread.join(timeout=30)
            if _warmup_thread.is_alive():
                logger.warning("LLM warm-up timed out after 30s — model will load on first query")
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

    # Recover zombie documents stuck in TRAINING_STARTED (e.g. server restart killed threads)
    try:
        from src.api.document_status import recover_zombie_documents
        recovered = recover_zombie_documents()
        if recovered:
            logger.info("Recovered %d zombie training documents at startup", recovered)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Zombie document recovery skipped: %s", exc)

    # Recover documents stuck in extraction IN_PROGRESS (server killed during extraction)
    try:
        from src.api.document_status import recover_zombie_extractions
        extraction_recovered = recover_zombie_extractions()
        if extraction_recovered:
            logger.info("Reset %d zombie extraction documents at startup", extraction_recovered)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Zombie extraction recovery skipped: %s", exc)

    # Auto-extraction at startup DISABLED — pipeline is HITL-driven.
    # Documents are only extracted when the UI triggers it (upload auto-extracts,
    # or user clicks extract). Server restart should not re-process documents.
    # Zombie recovery above already resets stuck documents so users can retry.

    # Legacy task-aware routing disabled — single vLLM model handles all tasks.
    # ModelRegistry, TaskRouter, TaskAwareGateway are no longer used.
    # vLLM pre-loads the model at server start, no GPU pinning needed.

    # Multi-agent gateway (role-specific Ollama models)
    multi_agent_gateway = None
    if getattr(Config, "MultiAgent", None) and getattr(Config.MultiAgent, "ENABLED", False):
        try:
            from src.llm.multi_agent import create_multi_agent_gateway
            multi_agent_gateway = create_multi_agent_gateway(fallback_gateway=llm_gateway)
            logger.info("Multi-agent gateway initialized with role-specific models")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multi-agent gateway init failed (continuing with single-model): %s", exc)

    # Knowledge Graph augmenter (Neo4j-backed entity graph for retrieval)
    graph_augmenter = None
    if getattr(Config.KnowledgeGraph, "ENABLED", False):
        try:
            from src.kg.neo4j_store import Neo4jStore
            from src.kg.retrieval import GraphAugmenter
            neo4j_store = Neo4jStore()
            graph_augmenter = GraphAugmenter(neo4j_store=neo4j_store, enabled=True)
            logger.info("GraphAugmenter initialized with Neo4j backend")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j unavailable; KG augmentation disabled: %s", exc)

    state = AppState(
        embedding_model=embedding_model,
        reranker=cross_encoder,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        ollama_client=ollama_client,
        rag_system=rag_system,
        llm_gateway=llm_gateway,
        multi_agent_gateway=multi_agent_gateway,
        graph_augmenter=graph_augmenter,
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

def _precreate_subscription_collections(qdrant_client) -> None:
    """Pre-create Qdrant collections for active subscriptions to avoid 404s on first document."""
    try:
        from src.api.vector_store import build_collection_name, VectorStoreClient
        from src.api.config import Config
        from src.api.dataHandler import db
        if db is None:
            return
        subs_coll = db.get_collection("subscriptions")
        active_subs = subs_coll.find({"status": {"$in": ["active", "ACTIVE", None]}}, {"_id": 1})
        existing = set()
        try:
            for col in (qdrant_client.get_collections().collections or []):
                existing.add(getattr(col, "name", str(col)))
        except Exception as exc:
            logger.warning("Failed to list Qdrant collections for pre-creation", exc_info=True)
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

    # Pre-load spaCy model for NLP entity extraction (avoids first-request race condition)
    try:
        from src.nlp.query_entity_extractor import preload_spacy
        if preload_spacy():
            logger.info("spaCy model pre-loaded at startup")
    except Exception as exc:  # noqa: BLE001
        logger.debug("spaCy pre-load skipped: %s", exc)

    # Validate embedding dimension at startup
    if state.embedding_model:
        try:
            test_vec = state.embedding_model.encode("dimension validation test",
                                                     convert_to_numpy=True,
                                                     normalize_embeddings=True)
            actual_dim = len(test_vec)
            expected_dim = int(getattr(Config.Model, "EMBEDDING_DIM", 1024))
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: model produces {actual_dim}d vectors "
                    f"but Config.Model.EMBEDDING_DIM={expected_dim}. "
                    f"Fix EMBEDDING_DIM env var or change the embedding model."
                )
            logger.info("Embedding dimension validated: %dd", actual_dim)
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("Embedding dimension validation skipped: %s", exc)

    # Pre-train intent classifier (fast, <2s on CPU)
    if state.embedding_model:
        try:
            from src.intent.intent_classifier import ensure_intent_classifier
            ensure_intent_classifier(state.embedding_model)
            logger.info("Intent classifier ready at startup")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Intent classifier startup training skipped: %s", exc)

    # Pre-train line role classifier (fast, <3s on CPU)
    if state.embedding_model:
        try:
            from src.rag_v3.line_classifier import ensure_line_classifier
            ensure_line_classifier(state.embedding_model)
            logger.info("Line role classifier ready at startup")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Line role classifier startup training skipped: %s", exc)

    # Pre-train field importance classifier (fast, <2s on CPU)
    if state.embedding_model:
        try:
            from src.rag_v3.field_classifier import ensure_field_classifier
            ensure_field_classifier(state.embedding_model)
            logger.info("Field importance classifier ready at startup")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Field importance classifier startup training skipped: %s", exc)

    # Pre-warm NLU registries (batch-encodes all category descriptions once)
    try:
        from src.nlp.nlu_engine import classify_intent, classify_domain_task
        classify_intent("warmup")
        classify_domain_task("warmup")
        logger.info("NLU registries pre-warmed at startup")
    except Exception as exc:  # noqa: BLE001
        logger.debug("NLU registry pre-warm skipped: %s", exc)

    # Pre-warm domain classifier centroids
    try:
        from src.intelligence.domain_classifier import _build_centroids
        if state.embedding_model:
            _build_centroids(state.embedding_model)
            logger.info("Domain classifier centroids pre-warmed at startup")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Domain classifier pre-warm skipped: %s", exc)

    # Initialize Vision OCR client (lazy availability check)
    if getattr(Config, "VisionOCR", None) and getattr(Config.VisionOCR, "ENABLED", True):
        try:
            from src.llm.vision_ocr import get_vision_ocr_client
            client = get_vision_ocr_client()
            if client and client.is_available():
                logger.info("Vision OCR ready: %s", Config.VisionOCR.MODEL)
            else:
                logger.debug("Vision OCR model not available; traditional OCR will be used")
        except Exception as exc:
            logger.warning("Vision OCR init failed: %s", exc)

    # Initialize domain knowledge provider (fast, no I/O)
    try:
        from src.intelligence.domain_knowledge import ensure_domain_knowledge_provider
        _dk_web = getattr(getattr(Config, "DomainKnowledge", None), "WEB_ENRICHMENT", False)
        ensure_domain_knowledge_provider(web_enrichment=_dk_web)
        logger.info("Domain knowledge provider initialized (web_enrichment=%s)", _dk_web)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Domain knowledge provider init skipped: %s", exc)

    # Initialize intelligence router (cloud LLM tiered chain)
    try:
        from src.llm.intelligence_router import IntelligenceRouter, set_intelligence_router
        from src.llm.clients import OllamaClient, GeminiClient, OpenAIClient, ClaudeClient
        _local = OllamaClient()
        _gemini = None
        _openai = None
        _claude = None
        _cloud_cfg = getattr(Config, "CloudLLM", None)
        _cloud_enabled = getattr(_cloud_cfg, "ENABLED", False) if _cloud_cfg else False
        if _cloud_enabled:
            try:
                _gemini = GeminiClient()
            except Exception as exc:
                logger.debug("Failed to initialize Gemini client", exc_info=True)
            _az_ep = getattr(_cloud_cfg, "AZURE_OPENAI_ENDPOINT", "")
            _az_key = getattr(_cloud_cfg, "AZURE_OPENAI_API_KEY", "")
            if _az_ep and _az_key:
                try:
                    _openai = OpenAIClient(
                        endpoint=_az_ep, api_key=_az_key,
                        deployment=getattr(_cloud_cfg, "AZURE_DEPLOYMENT", "gpt-4.1"),
                        api_version=getattr(_cloud_cfg, "AZURE_API_VERSION", "2024-05-01-preview"),
                    )
                except Exception as exc:
                    logger.debug("Failed to initialize Azure OpenAI client", exc_info=True)
            _cl_key = getattr(_cloud_cfg, "CLAUDE_API_KEY", "")
            if _cl_key:
                try:
                    _claude = ClaudeClient(api_key=_cl_key, model=getattr(_cloud_cfg, "CLAUDE_MODEL", "claude-sonnet-4-20250514"))
                except Exception as exc:
                    logger.debug("Failed to initialize Claude client", exc_info=True)
        router = IntelligenceRouter(_local, _gemini, _openai, _claude)
        if _cloud_enabled and _cloud_cfg:
            router.configure(
                enabled=True,
                t2_threshold=getattr(_cloud_cfg, "COMPLEXITY_THRESHOLD_T2", 0.4),
                t3_threshold=getattr(_cloud_cfg, "COMPLEXITY_THRESHOLD_T3", 0.7),
            )
        set_intelligence_router(router)
        logger.info("Intelligence router initialized (cloud=%s)", _cloud_enabled)
    except Exception as exc:
        logger.warning("Intelligence router init skipped: %s", exc)

    # Initialize background document analyzer
    try:
        from src.doc_understanding.background_analyzer import BackgroundAnalyzer, set_background_analyzer
        _deep_cfg = getattr(Config, "DeepAnalysis", None)
        if _deep_cfg and getattr(_deep_cfg, "BACKGROUND_ENABLED", True):
            bg = BackgroundAnalyzer()
            if state.redis_client:
                bg.set_redis(state.redis_client)
                bg.start_worker()
            set_background_analyzer(bg)
            logger.info("Background analyzer initialized and worker started")
    except Exception as exc:
        logger.warning("Background analyzer init skipped: %s", exc)

    # Run startup checks (migrated from @app.on_event("startup") in main.py)
    try:
        from src.utils.logging_utils import configure_logging
        configure_logging(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            json_format=os.getenv("JSON_LOGGING", "true").lower() in {"1", "true", "yes"},
            include_correlation_id=True,
        )
    except Exception as exc:
        logger.debug("Logging configuration skipped: %s", exc)
    try:
        from src.storage.blob_persistence import validate_storage_configured_once
        validate_storage_configured_once()
    except (ImportError, AttributeError) as exc:
        logger.debug("Azure blob storage validation not available (function not yet implemented)", exc_info=True)
    except Exception as exc:
        logger.warning("Azure blob storage configuration check skipped: %s", exc)
    try:
        from src.storage.blob_persistence import validate_containers_once
        validate_containers_once()
    except (ImportError, AttributeError) as exc:
        logger.debug("Azure blob container validation not available (function not yet implemented)", exc_info=True)
    except Exception as exc:
        logger.warning("Azure blob container validation skipped: %s", exc)
    try:
        from src.api.dataHandler import clear_legacy_vetting_metadata
        clear_legacy_vetting_metadata()
    except Exception as exc:
        logger.warning("Legacy metadata cleanup skipped: %s", exc)
    logger.info("Startup checks completed")

    yield

    # Shutdown
    try:
        from src.doc_understanding.background_analyzer import get_background_analyzer
        bg = get_background_analyzer()
        if bg:
            bg.stop_worker()
    except Exception as exc:
        logger.debug("Failed to stop background analyzer worker during shutdown", exc_info=True)
    logger.info("DocWain API shutting down")

__all__ = ["initialize_app_state", "lifespan"]
