"""Backfill task -- updates Qdrant chunk payloads with KG node IDs after KG completes."""

from src.celery_app import app
from src.api.document_status import append_audit_log, update_stage
import logging
import time

logger = logging.getLogger(__name__)

# Batch size for Qdrant scroll operations
_SCROLL_BATCH = 64


def _build_entity_lookup(neo4j_store, document_id: str,
                         subscription_id: str, profile_id: str) -> dict:
    """Query Neo4j for all Entity nodes linked to this document and build a
    normalized-name -> [entity_id] lookup.

    The graph schema (from kg/ingest.py) stores entities with:
      - entity_id (unique, tenant-prefixed)
      - name, normalized_name
      - subscription_id, profile_id

    Entities are linked to documents via MENTIONS edges:
      (Document)-[:MENTIONS]->(Entity)
    """
    query = (
        "MATCH (d:Document {doc_id: $doc_id, subscription_id: $sub_id, "
        "profile_id: $prof_id})-[:MENTIONS]->(e:Entity) "
        "RETURN e.entity_id AS entity_id, e.name AS name, "
        "e.normalized_name AS normalized_name"
    )
    rows = neo4j_store.run_query(query, {
        "doc_id": str(document_id),
        "sub_id": str(subscription_id),
        "prof_id": str(profile_id),
    })

    lookup: dict = {}  # lower(name) -> [entity_id, ...]
    for row in rows:
        entity_id = row.get("entity_id")
        if not entity_id:
            continue
        name = (row.get("name") or "").lower().strip()
        norm = (row.get("normalized_name") or "").lower().strip()
        if name:
            lookup.setdefault(name, []).append(str(entity_id))
        if norm and norm != name:
            lookup.setdefault(norm, []).append(str(entity_id))

    return lookup


def _scroll_and_update_qdrant(qdrant_client, collection_name: str,
                              document_id: str, entity_lookup: dict) -> int:
    """Scroll through all Qdrant points for this document and attach kg_node_ids."""
    from qdrant_client.http.models import (
        Filter, FieldCondition, MatchValue, SetPayload,
    )

    doc_filter = Filter(must=[
        FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
    ])

    updated = 0
    offset = None

    while True:
        batch, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=doc_filter,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )

        if not batch:
            break

        for point in batch:
            payload = getattr(point, "payload", None) or {}
            chunk_entities = payload.get("entities", [])
            if not chunk_entities:
                continue

            kg_ids = []
            seen = set()
            for ent_text in chunk_entities:
                key = str(ent_text).lower().strip()
                for nid in entity_lookup.get(key, []):
                    if nid not in seen:
                        kg_ids.append(nid)
                        seen.add(nid)

            if kg_ids:
                point_id = getattr(point, "id", None)
                if point_id is not None:
                    qdrant_client.set_payload(
                        collection_name=collection_name,
                        payload={"kg_node_ids": kg_ids},
                        points=[point_id],
                    )
                    updated += 1

        offset = next_offset
        if next_offset is None:
            break

    return updated


def _mark_kg_linked(document_id: str) -> None:
    """Set embedding.kg_linked = true in MongoDB."""
    update_stage(document_id, "embedding", {"kg_linked": True})


@app.task(bind=True, name="src.tasks.backfill.backfill_kg_refs",
          max_retries=2, soft_time_limit=600)
def backfill_kg_refs(self, document_id: str, subscription_id: str,
                     profile_id: str):
    """Backfill KG node IDs into Qdrant chunk payloads.

    Called when KG building completes AFTER embedding was already done.
    Steps:
      1. Query Neo4j for all entities linked to this document
      2. Build entity name -> [entity_id] lookup
      3. Scroll Qdrant chunks for this document
      4. Match chunk entities against KG lookup, set kg_node_ids payload
      5. Mark embedding.kg_linked = true in MongoDB
    """
    try:
        append_audit_log(document_id, "KG_BACKFILL_STARTED",
                        celery_task_id=self.request.id)

        # 1. Connect to Neo4j
        from src.kg.neo4j_store import Neo4jStore
        neo4j_store = Neo4jStore()

        # 2. Build entity lookup from KG
        entity_lookup = _build_entity_lookup(
            neo4j_store, document_id, subscription_id, profile_id,
        )
        logger.info(
            "KG backfill: found %d entity keys for document %s",
            len(entity_lookup), document_id,
        )

        if not entity_lookup:
            append_audit_log(document_id, "KG_BACKFILL_SKIPPED",
                            reason="no_entities_in_kg")
            _mark_kg_linked(document_id)
            neo4j_store.close()
            return

        # 3. Connect to Qdrant and update chunks
        from src.api.dw_newron import get_qdrant_client
        qdrant_client = get_qdrant_client()
        collection_name = str(subscription_id)

        updated_count = _scroll_and_update_qdrant(
            qdrant_client, collection_name, document_id, entity_lookup,
        )

        logger.info(
            "KG backfill: updated %d chunks with kg_node_ids for document %s",
            updated_count, document_id,
        )

        # 4. Mark as linked in MongoDB
        _mark_kg_linked(document_id)

        append_audit_log(
            document_id, "KG_BACKFILL_COMPLETED",
            chunks_updated=updated_count,
            entity_keys=len(entity_lookup),
        )

        neo4j_store.close()

    except Exception as exc:
        logger.error("KG backfill failed for document %s: %s",
                     document_id, exc, exc_info=True)
        append_audit_log(document_id, "KG_BACKFILL_FAILED", error=str(exc))
        self.retry(exc=exc)
