from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from src.api.dw_newron import get_qdrant_client, get_redis_client
from src.api.vector_store import REQUIRED_PAYLOAD_INDEX_FIELDS, QdrantVectorStore
from src.intelligence.kg_query import KGQueryService
from src.intelligence.redis_intel_cache import RedisIntelCache

debug_router = APIRouter(prefix="/debug", tags=["Debug"])


@debug_router.get("/redis_state")
def debug_redis_state(
    subscription_id: str = Query(...),
    session_id: str = Query(...),
    profile_id: str = Query(...),
) -> Dict[str, Any]:
    redis_client = get_redis_client()
    cache = RedisIntelCache(redis_client)
    return {
        "session": cache.get_json(cache.session_key(subscription_id, session_id)),
        "catalog": cache.get_json(cache.catalog_key(subscription_id, profile_id)),
        "entities": cache.get_json(cache.entities_key(subscription_id, profile_id)),
    }


@debug_router.get("/kg")
def debug_kg(
    profile_id: str = Query(...),
    entity: str = Query(...),
    subscription_id: str = Query("default"),
) -> Dict[str, Any]:
    service = KGQueryService()
    entities = service.extract_entities(entity)
    result = service.query(
        subscription_id=subscription_id,
        profile_id=profile_id,
        domain_hint=None,
        entities=entities,
    )
    return {
        "doc_ids": result.doc_ids,
        "chunk_ids": result.chunk_ids,
        "entities": result.entities,
    }


@debug_router.get("/qdrant_indexes")
def debug_qdrant_indexes(
    collection: str = Query(...),
) -> Dict[str, Any]:
    client = get_qdrant_client()
    store = QdrantVectorStore(client=client)
    existing = store.get_payload_indexes(collection)
    required = list(REQUIRED_PAYLOAD_INDEX_FIELDS)
    missing = [field for field in required if field not in set(existing)]
    return {
        "collection": collection,
        "existing_indexes": existing,
        "required_indexes": required,
        "missing_indexes": missing,
    }


__all__ = ["debug_router"]
