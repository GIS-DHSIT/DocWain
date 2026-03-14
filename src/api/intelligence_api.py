"""Intelligence monitoring and tuning API endpoints.

Provides visibility into the knowledge graph hot cache, query metrics,
intelligence configuration, and runtime tuning for each profile.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# ---------------------------------------------------------------------------
# Lazy Redis client
# ---------------------------------------------------------------------------

_redis_client = None


def _get_redis():
    """Lazily obtain the shared Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            from src.api.dw_newron import get_redis_client
            _redis_client = get_redis_client()
        except Exception:
            logger.warning("[IntelAPI] Unable to obtain Redis client")
    return _redis_client


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ProfileHealth(BaseModel):
    profile_id: str
    entity_count: int = 0
    fact_count: int = 0
    relationship_count: int = 0
    document_count: int = 0
    domain: str = "general"
    last_updated: Optional[str] = None
    cache_status: str = "unknown"


class SummaryResponse(BaseModel):
    total_profiles: int = 0
    total_entities: int = 0
    total_facts: int = 0
    total_relationships: int = 0
    total_documents: int = 0
    profile_domains: List[Dict[str, str]] = Field(default_factory=list)


class QueryMetrics(BaseModel):
    profile_id: str
    total_queries: int = 0
    avg_confidence: float = 0.0
    grounded_ratio: float = 0.0
    top_intents: List[str] = Field(default_factory=list)


class IntelligenceConfig(BaseModel):
    enabled: bool = True
    session_ttl_seconds: int = 604800
    catalog_ttl_seconds: int = 2592000
    summary_ttl_seconds: int = 2592000
    entities_ttl_seconds: int = 2592000
    reasoning_engine_enabled: bool = True
    verify_confidence_threshold: float = 0.8
    section_retrieval_enabled: bool = True


class TuneRequest(BaseModel):
    profile_id: str
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    grounding_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class TuneResponse(BaseModel):
    profile_id: str
    updated_fields: List[str] = Field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KG_PREFIX = "kg"


def _kg_key(profile_id: str, *parts: str) -> str:
    return ":".join([_KG_PREFIX, profile_id] + list(parts))


def _count_list_keys(r, pattern: str) -> int:
    """Count total items across all Redis list keys matching *pattern*."""
    total = 0
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
        for key in keys:
            total += r.llen(key)
        if cursor == 0:
            break
    return total


def _discover_profile_ids(r) -> List[str]:
    """Discover profile IDs that have knowledge cached (via doc_domains hash)."""
    profile_ids: set = set()
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="kg:*:doc_domains", count=200)
        for key in keys:
            key_str = key if isinstance(key, str) else key.decode("utf-8")
            parts = key_str.split(":")
            if len(parts) >= 3:
                profile_ids.add(parts[1])
        if cursor == 0:
            break
    return sorted(profile_ids)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/monitor/summary", response_model=SummaryResponse)
async def intelligence_summary():
    """Get aggregate intelligence health across all profiles."""
    r = _get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    try:
        profile_ids = _discover_profile_ids(r)
        total_entities = 0
        total_facts = 0
        total_relationships = 0
        total_documents = 0
        profile_domains: List[Dict[str, str]] = []

        for pid in profile_ids:
            ent_count = r.hlen(_kg_key(pid, "entities"))
            rel_count = r.zcard(_kg_key(pid, "relations"))
            doc_count = r.hlen(_kg_key(pid, "doc_domains"))
            fact_count = _count_list_keys(r, _kg_key(pid, "facts", "*"))

            total_entities += ent_count
            total_relationships += rel_count
            total_documents += doc_count
            total_facts += fact_count

            domain_raw = r.get(_kg_key(pid, "profile_domain"))
            domain = (domain_raw if isinstance(domain_raw, str) else "general") if domain_raw else "general"
            profile_domains.append({"profile_id": pid, "domain": domain})

        return SummaryResponse(
            total_profiles=len(profile_ids),
            total_entities=total_entities,
            total_facts=total_facts,
            total_relationships=total_relationships,
            total_documents=total_documents,
            profile_domains=profile_domains,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[IntelAPI] Summary failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to build summary") from exc


@router.get("/monitor/{profile_id}", response_model=ProfileHealth)
async def intelligence_monitor(profile_id: str):
    """Get intelligence health for a single profile."""
    r = _get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    try:
        entity_count = r.hlen(_kg_key(profile_id, "entities"))
        relationship_count = r.zcard(_kg_key(profile_id, "relations"))
        document_count = r.hlen(_kg_key(profile_id, "doc_domains"))
        fact_count = _count_list_keys(r, _kg_key(profile_id, "facts", "*"))

        domain_raw = r.get(_kg_key(profile_id, "profile_domain"))
        domain = (domain_raw if isinstance(domain_raw, str) else "general") if domain_raw else "general"

        # Determine cache status
        if entity_count == 0 and document_count == 0:
            cache_status = "empty"
        else:
            cache_status = "active"

        # last_updated: check TTL on entities hash as a proxy
        last_updated = None
        ttl = r.ttl(_kg_key(profile_id, "entities"))
        if ttl and ttl > 0:
            last_updated = f"TTL {ttl}s remaining"
        elif entity_count > 0:
            last_updated = "persistent (no TTL)"

        return ProfileHealth(
            profile_id=profile_id,
            entity_count=entity_count,
            fact_count=fact_count,
            relationship_count=relationship_count,
            document_count=document_count,
            domain=domain,
            last_updated=last_updated,
            cache_status=cache_status,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[IntelAPI] Monitor failed for profile=%s: %s", profile_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read profile health") from exc


@router.get("/metrics/{profile_id}", response_model=QueryMetrics)
async def intelligence_metrics(profile_id: str):
    """Get query metrics for a profile."""
    r = _get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    metrics_prefix = f"intel:metrics:{profile_id}"

    try:
        total_queries = int(r.get(f"{metrics_prefix}:total_queries") or 0)

        # Confidence scores stored as a list
        raw_confidences = r.lrange(f"{metrics_prefix}:confidences", 0, -1)
        if raw_confidences:
            confidences = [float(c) for c in raw_confidences]
            avg_confidence = round(sum(confidences) / len(confidences), 4)
        else:
            avg_confidence = 0.0

        # Grounded ratio
        grounded_count = int(r.get(f"{metrics_prefix}:grounded_count") or 0)
        grounded_ratio = round(grounded_count / total_queries, 4) if total_queries > 0 else 0.0

        # Top intents (sorted set, descending by count)
        raw_intents = r.zrevrange(f"{metrics_prefix}:intents", 0, 9)
        top_intents = []
        for intent in raw_intents:
            intent_str = intent if isinstance(intent, str) else intent.decode("utf-8")
            top_intents.append(intent_str)

        return QueryMetrics(
            profile_id=profile_id,
            total_queries=total_queries,
            avg_confidence=avg_confidence,
            grounded_ratio=grounded_ratio,
            top_intents=top_intents,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[IntelAPI] Metrics failed for profile=%s: %s", profile_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read query metrics") from exc


@router.get("/config/{profile_id}", response_model=IntelligenceConfig)
async def intelligence_config(profile_id: str):
    """Get current intelligence configuration for a profile.

    Returns the base Config.Intelligence values, overlaid with any
    profile-specific overrides stored in Redis.
    """
    from src.api.config import Config

    r = _get_redis()

    # Start with global defaults
    config = IntelligenceConfig(
        enabled=Config.Intelligence.ENABLED,
        session_ttl_seconds=Config.Intelligence.SESSION_TTL_SECONDS,
        catalog_ttl_seconds=Config.Intelligence.CATALOG_TTL_SECONDS,
        summary_ttl_seconds=Config.Intelligence.SUMMARY_TTL_SECONDS,
        entities_ttl_seconds=Config.Intelligence.ENTITIES_TTL_SECONDS,
        reasoning_engine_enabled=Config.Intelligence.REASONING_ENGINE_ENABLED,
        verify_confidence_threshold=Config.Intelligence.VERIFY_CONFIDENCE_THRESHOLD,
        section_retrieval_enabled=Config.Intelligence.SECTION_RETRIEVAL_ENABLED,
    )

    # Apply per-profile overrides from Redis
    if r:
        try:
            override_key = f"intel:config:{profile_id}"
            raw = r.get(override_key)
            if raw:
                overrides = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                for field_name in config.model_fields:
                    if field_name in overrides:
                        setattr(config, field_name, overrides[field_name])
        except Exception as exc:
            logger.warning("[IntelAPI] Failed to load config overrides for profile=%s: %s", profile_id, exc)

    return config


@router.post("/tune", response_model=TuneResponse)
async def intelligence_tune(request: TuneRequest):
    """Apply intelligence tuning parameters for a profile.

    Stores overrides in Redis so they take effect on the next query
    without requiring a restart.
    """
    r = _get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    override_key = f"intel:config:{request.profile_id}"
    updated_fields: List[str] = []

    try:
        # Load existing overrides
        raw = r.get(override_key)
        overrides: Dict[str, Any] = {}
        if raw:
            try:
                overrides = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except (json.JSONDecodeError, TypeError):
                overrides = {}

        # Apply requested changes
        if request.confidence_threshold is not None:
            overrides["verify_confidence_threshold"] = request.confidence_threshold
            updated_fields.append("verify_confidence_threshold")

        if request.grounding_threshold is not None:
            overrides["grounding_threshold"] = request.grounding_threshold
            updated_fields.append("grounding_threshold")

        if not updated_fields:
            return TuneResponse(
                profile_id=request.profile_id,
                updated_fields=[],
                message="No parameters provided; nothing changed.",
            )

        overrides["_updated_at"] = datetime.now(timezone.utc).isoformat()
        r.set(override_key, json.dumps(overrides))

        logger.info(
            "[IntelAPI] Tuned profile=%s fields=%s",
            request.profile_id,
            updated_fields,
        )

        return TuneResponse(
            profile_id=request.profile_id,
            updated_fields=updated_fields,
            message=f"Updated {len(updated_fields)} parameter(s) for profile {request.profile_id}.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[IntelAPI] Tune failed for profile=%s: %s", request.profile_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to apply tuning") from exc
