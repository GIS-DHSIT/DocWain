"""Redis hot knowledge cache for instant query-time retrieval.

Written at document ingestion time with extracted knowledge (entities, facts,
relationships, claims, summaries). Read at query time as the first tier of
the three-tier retrieval system (Redis → Neo4j → Qdrant).

All keys are profile-scoped for strict tenant isolation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Key prefix
_PREFIX = "kg"


def _key(profile_id: str, *parts: str) -> str:
    """Build a Redis key with profile scoping."""
    return ":".join([_PREFIX, profile_id] + list(parts))


# ---------------------------------------------------------------------------
# Write operations (called at ingestion time)
# ---------------------------------------------------------------------------

def cache_document_knowledge(
    redis_client,
    profile_id: str,
    doc_id: str,
    entities: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    domain: str = "general",
    summary: str = "",
) -> None:
    """Cache extracted knowledge for a document in Redis.

    Called after LLM knowledge extraction during document ingestion.
    """
    if not redis_client:
        logger.debug("[HotCache] No Redis client — skipping cache write")
        return

    try:
        pipe = redis_client.pipeline(transaction=False)

        # Entities → Hash: entity_name → JSON
        ent_key = _key(profile_id, "entities")
        for ent in entities:
            name = ent.get("name", "").lower().strip()
            if not name:
                continue
            # Merge with existing: append doc_id to doc_ids list
            existing_raw = redis_client.hget(ent_key, name)
            if existing_raw:
                try:
                    existing = json.loads(existing_raw)
                    doc_ids = set(existing.get("doc_ids", []))
                    doc_ids.add(doc_id)
                    existing["doc_ids"] = list(doc_ids)
                    # Update context if new one is more detailed
                    if len(ent.get("context", "")) > len(existing.get("context", "")):
                        existing["context"] = ent["context"]
                    pipe.hset(ent_key, name, json.dumps(existing))
                except (json.JSONDecodeError, TypeError):
                    pipe.hset(ent_key, name, json.dumps({
                        "type": ent.get("type", "unknown"),
                        "context": ent.get("context", ""),
                        "doc_ids": [doc_id],
                    }))
            else:
                pipe.hset(ent_key, name, json.dumps({
                    "type": ent.get("type", "unknown"),
                    "context": ent.get("context", ""),
                    "doc_ids": [doc_id],
                }))

        # Facts → List per document
        facts_key = _key(profile_id, "facts", doc_id)
        if facts:
            pipe.delete(facts_key)
            for fact in facts:
                pipe.rpush(facts_key, json.dumps({
                    "statement": fact.get("statement", ""),
                    "evidence": fact.get("evidence", ""),
                    "confidence": fact.get("confidence", 0.0),
                    "location": fact.get("location", {}),
                }))

        # Claims → List per document
        claims_key = _key(profile_id, "claims", doc_id)
        if claims:
            pipe.delete(claims_key)
            for claim in claims:
                pipe.rpush(claims_key, json.dumps({
                    "claim": claim.get("claim", ""),
                    "evidence": claim.get("evidence", ""),
                    "confidence": claim.get("confidence", 0.0),
                }))

        # Relationships → Sorted set by confidence
        rel_key = _key(profile_id, "relations")
        for rel in relationships:
            member = json.dumps({
                "subject": rel.get("subject", ""),
                "object": rel.get("object", ""),
                "relation": rel.get("relation", ""),
                "evidence": rel.get("evidence", ""),
                "doc_id": doc_id,
            })
            score = float(rel.get("confidence", 0.0))
            pipe.zadd(rel_key, {member: score})

        # Domain → Hash per document
        domain_key = _key(profile_id, "doc_domains")
        pipe.hset(domain_key, doc_id, domain)

        # Document summary
        if summary:
            summary_key = _key(profile_id, "doc_summary", doc_id)
            pipe.set(summary_key, summary)

        pipe.execute()
        logger.info(
            "[HotCache] Cached knowledge for doc=%s profile=%s "
            "(entities=%d facts=%d claims=%d rels=%d)",
            doc_id, profile_id,
            len(entities), len(facts), len(claims), len(relationships),
        )

    except Exception as e:
        logger.warning("[HotCache] Cache write failed for doc=%s: %s", doc_id, e)


def recompute_profile_domain(redis_client, profile_id: str) -> str:
    """Recompute the dominant domain for a profile from document domains.

    Called when documents are added or removed.
    Returns the dominant domain string.
    """
    if not redis_client:
        return "general"

    try:
        domain_key = _key(profile_id, "doc_domains")
        all_domains = redis_client.hvals(domain_key)

        if not all_domains:
            return "general"

        # Count domain frequencies
        counts: Dict[str, int] = {}
        for d in all_domains:
            d_str = d if isinstance(d, str) else d.decode("utf-8")
            counts[d_str] = counts.get(d_str, 0) + 1

        dominant = max(counts, key=counts.get)
        redis_client.set(_key(profile_id, "profile_domain"), dominant)
        logger.info("[HotCache] Profile %s domain: %s", profile_id, dominant)
        return dominant

    except Exception as e:
        logger.warning("[HotCache] Profile domain recompute failed: %s", e)
        return "general"


def flush_document_cache(redis_client, profile_id: str, doc_id: str) -> None:
    """Remove all cached knowledge for a document. Called on re-extraction or deletion."""
    if not redis_client:
        return

    try:
        pipe = redis_client.pipeline(transaction=False)

        # Remove document-specific keys
        pipe.delete(_key(profile_id, "facts", doc_id))
        pipe.delete(_key(profile_id, "claims", doc_id))
        pipe.delete(_key(profile_id, "doc_summary", doc_id))

        # Remove from domain hash
        pipe.hdel(_key(profile_id, "doc_domains"), doc_id)

        # Remove relationships for this doc (scan sorted set)
        rel_key = _key(profile_id, "relations")
        # We need to scan members containing this doc_id
        cursor = 0
        to_remove = []
        while True:
            cursor, members = redis_client.zscan(rel_key, cursor, match=f'*"{doc_id}"*', count=100)
            for member, _score in members:
                to_remove.append(member)
            if cursor == 0:
                break
        for member in to_remove:
            pipe.zrem(rel_key, member)

        # Clean entities: remove doc_id from doc_ids lists
        ent_key = _key(profile_id, "entities")
        all_entities = redis_client.hgetall(ent_key)
        for name, raw in all_entities.items():
            try:
                data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                doc_ids = data.get("doc_ids", [])
                if doc_id in doc_ids:
                    doc_ids.remove(doc_id)
                    if doc_ids:
                        data["doc_ids"] = doc_ids
                        name_str = name if isinstance(name, str) else name.decode("utf-8")
                        pipe.hset(ent_key, name_str, json.dumps(data))
                    else:
                        # No more documents reference this entity — remove it
                        name_str = name if isinstance(name, str) else name.decode("utf-8")
                        pipe.hdel(ent_key, name_str)
            except (json.JSONDecodeError, TypeError):
                continue

        pipe.execute()
        logger.info("[HotCache] Flushed cache for doc=%s profile=%s", doc_id, profile_id)

        # Recompute profile domain
        recompute_profile_domain(redis_client, profile_id)

    except Exception as e:
        logger.warning("[HotCache] Flush failed for doc=%s: %s", doc_id, e)


# ---------------------------------------------------------------------------
# Read operations (called at query time)
# ---------------------------------------------------------------------------

def lookup_entities(
    redis_client,
    profile_id: str,
    entity_names: List[str],
) -> List[Dict[str, Any]]:
    """Look up entities by name from the hot cache.

    Returns list of entity dicts with type, context, doc_ids.
    """
    if not redis_client or not entity_names:
        return []

    results = []
    ent_key = _key(profile_id, "entities")

    try:
        for name in entity_names:
            raw = redis_client.hget(ent_key, name.lower().strip())
            if raw:
                data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                data["name"] = name
                results.append(data)
    except Exception as e:
        logger.warning("[HotCache] Entity lookup failed: %s", e)

    return results


def get_document_facts(
    redis_client,
    profile_id: str,
    doc_id: str,
    max_facts: int = 20,
) -> List[Dict[str, Any]]:
    """Get cached facts for a document."""
    if not redis_client:
        return []

    try:
        facts_key = _key(profile_id, "facts", doc_id)
        raw_list = redis_client.lrange(facts_key, 0, max_facts - 1)
        return [json.loads(r if isinstance(r, str) else r.decode("utf-8")) for r in raw_list]
    except Exception as e:
        logger.warning("[HotCache] Facts read failed for doc=%s: %s", doc_id, e)
        return []


def get_document_claims(
    redis_client,
    profile_id: str,
    doc_id: str,
) -> List[Dict[str, Any]]:
    """Get cached claims for a document."""
    if not redis_client:
        return []

    try:
        claims_key = _key(profile_id, "claims", doc_id)
        raw_list = redis_client.lrange(claims_key, 0, -1)
        return [json.loads(r if isinstance(r, str) else r.decode("utf-8")) for r in raw_list]
    except Exception as e:
        logger.warning("[HotCache] Claims read failed for doc=%s: %s", doc_id, e)
        return []


def get_document_summary(
    redis_client,
    profile_id: str,
    doc_id: str,
) -> str:
    """Get the cached knowledge summary for a document."""
    if not redis_client:
        return ""

    try:
        raw = redis_client.get(_key(profile_id, "doc_summary", doc_id))
        if raw:
            return raw if isinstance(raw, str) else raw.decode("utf-8")
    except Exception as e:
        logger.warning("[HotCache] Summary read failed for doc=%s: %s", doc_id, e)
    return ""


def get_profile_domain(redis_client, profile_id: str) -> str:
    """Get the dominant domain for a profile."""
    if not redis_client:
        return "general"

    try:
        raw = redis_client.get(_key(profile_id, "profile_domain"))
        if raw:
            return raw if isinstance(raw, str) else raw.decode("utf-8")
    except Exception as e:
        logger.warning("[HotCache] Profile domain read failed: %s", e)
    return "general"


def get_top_relationships(
    redis_client,
    profile_id: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Get top relationships by confidence from the hot cache."""
    if not redis_client:
        return []

    try:
        rel_key = _key(profile_id, "relations")
        raw_list = redis_client.zrevrange(rel_key, 0, max_results - 1, withscores=True)
        results = []
        for member, score in raw_list:
            data = json.loads(member if isinstance(member, str) else member.decode("utf-8"))
            data["confidence"] = score
            results.append(data)
        return results
    except Exception as e:
        logger.warning("[HotCache] Relationships read failed: %s", e)
        return []


def get_all_document_ids_in_profile(
    redis_client,
    profile_id: str,
) -> List[str]:
    """Get all document IDs that have cached knowledge in this profile."""
    if not redis_client:
        return []

    try:
        domain_key = _key(profile_id, "doc_domains")
        keys = redis_client.hkeys(domain_key)
        return [k if isinstance(k, str) else k.decode("utf-8") for k in keys]
    except Exception as e:
        logger.warning("[HotCache] Doc ID list failed: %s", e)
        return []


__all__ = [
    "cache_document_knowledge",
    "recompute_profile_domain",
    "flush_document_cache",
    "lookup_entities",
    "get_document_facts",
    "get_document_claims",
    "get_document_summary",
    "get_profile_domain",
    "get_top_relationships",
    "get_all_document_ids_in_profile",
]
