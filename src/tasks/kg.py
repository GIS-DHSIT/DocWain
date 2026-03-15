"""Knowledge graph building Celery task -- runs independently, never blocks pipeline."""

import json
import time

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import update_stage, append_audit_log, get_document_record
from src.api.statuses import STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blob helpers
# ---------------------------------------------------------------------------

def _download_blob_json(blob_path: str) -> dict:
    """Download and parse a JSON blob from Azure Blob storage."""
    from src.api.blob_content_store import get_blob_client

    container = get_blob_client()
    blob_client = container.get_blob_client(blob_path)
    raw = blob_client.download_blob().readall()
    return json.loads(raw)


def _try_download_blob_json(blob_path: str) -> dict | None:
    """Download JSON blob, returning None on any failure."""
    try:
        return _download_blob_json(blob_path)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Extraction-to-graph adapter
# ---------------------------------------------------------------------------

def _extraction_to_graph_payload(
    extraction: dict,
    screening: dict | None,
    subscription_id: str,
    profile_id: str,
    document_id: str,
):
    """Convert extraction JSON into a GraphIngestPayload using existing infra.

    The extraction JSON is produced by ExtractionEngine.to_dict() and contains:
    - entities: list of {text, type, confidence, ...}
    - relationships: list of {entity1, entity2, relation_type, ...}
    - tables: list of table dicts
    - sections / text content
    - metadata (doc_type, languages, etc.)
    """
    from src.kg.ingest import (
        GraphIngestPayload,
        GraphEntity,
        GraphMention,
        GraphField,
    )
    from src.kg.entity_extractor import normalize_entity_name, EntityExtractor

    tenant_prefix = f"{subscription_id}::{profile_id}::"

    # --- Document node ---
    meta = extraction.get("metadata") or {}
    doc_name = (
        meta.get("source_file")
        or meta.get("filename")
        or meta.get("doc_name")
        or document_id
    )
    document = {
        "doc_id": str(document_id),
        "profile_id": str(profile_id),
        "subscription_id": str(subscription_id),
        "doc_name": doc_name,
        "document_category": meta.get("doc_type") or meta.get("document_type") or "generic",
        "detected_language": _first_language(meta.get("languages")),
        "created_at": meta.get("created_at"),
        "graph_version": f"extraction_{document_id}",
    }

    entities: dict[str, GraphEntity] = {}
    mentions: list[GraphMention] = []
    fields: list[GraphField] = []

    # --- Entities from extraction result ---
    raw_entities = extraction.get("entities") or []
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        text = (ent.get("text") or ent.get("name") or "").strip()
        ent_type = ent.get("type") or ent.get("entity_type") or "UNKNOWN"
        confidence = float(ent.get("confidence", 0.5))
        if not text:
            continue

        normalized = normalize_entity_name(text)
        if not normalized:
            continue

        entity_id = f"{tenant_prefix}{ent_type}::{normalized}"
        entities.setdefault(
            entity_id,
            GraphEntity(
                entity_id=entity_id,
                name=text,
                type=ent_type,
                normalized_name=normalized,
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            ),
        )

        # Create a mention linking entity to document
        edge_key = f"extraction::{document_id}::{entity_id}"
        evidence = ent.get("evidence") or ent.get("context") or None
        mentions.append(
            GraphMention(
                doc_id=str(document_id),
                entity_id=entity_id,
                chunk_id=ent.get("chunk_id") or str(document_id),
                evidence_span=evidence[:200] if evidence else None,
                confidence=confidence,
                edge_key=edge_key,
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            )
        )

    # --- Also run EntityExtractor on combined text for additional entities ---
    combined_text = extraction.get("full_text") or extraction.get("text") or ""
    # Use section texts if available
    if not combined_text:
        sections = extraction.get("sections") or []
        section_texts = []
        for sec in sections:
            if isinstance(sec, dict):
                t = sec.get("text") or sec.get("content") or ""
                if t:
                    section_texts.append(str(t))
        combined_text = "\n".join(section_texts)

    if combined_text:
        extractor = EntityExtractor(use_nlp=True)
        extracted = extractor.extract_with_metadata(combined_text[:20000])  # cap for perf
        for ent in extracted:
            entity_id = f"{tenant_prefix}{ent.entity_id}"
            entities.setdefault(
                entity_id,
                GraphEntity(
                    entity_id=entity_id,
                    name=ent.name,
                    type=ent.type,
                    normalized_name=ent.normalized_name,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                ),
            )
            edge_key = f"text_extract::{document_id}::{entity_id}"
            mentions.append(
                GraphMention(
                    doc_id=str(document_id),
                    entity_id=entity_id,
                    chunk_id=str(document_id),
                    evidence_span=None,
                    confidence=ent.confidence,
                    edge_key=edge_key,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                )
            )

    # --- Structured fields from tables ---
    tables = extraction.get("tables") or []
    for tidx, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        # Tables may have headers/rows or key-value pairs
        headers = table.get("headers") or []
        rows = table.get("rows") or []
        for ridx, row in enumerate(rows):
            if isinstance(row, dict):
                items = row.items()
            elif isinstance(row, list) and headers:
                items = zip(headers, row)
            else:
                continue
            for key, value in items:
                if value is None or str(value).strip() == "":
                    continue
                normalized_value = normalize_entity_name(str(value))
                if not normalized_value:
                    continue
                entity_id = f"{tenant_prefix}FIELD::{str(key).lower()}::{normalized_value}"
                entities.setdefault(
                    entity_id,
                    GraphEntity(
                        entity_id=entity_id,
                        name=str(value),
                        type="FIELD",
                        normalized_name=normalized_value,
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                    ),
                )
                edge_key = f"table::{document_id}::{entity_id}::t{tidx}r{ridx}"
                fields.append(
                    GraphField(
                        doc_id=str(document_id),
                        entity_id=entity_id,
                        chunk_id=str(document_id),
                        key=str(key),
                        value=str(value),
                        confidence=0.80,
                        edge_key=edge_key,
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                    )
                )

    # --- Screening-derived entities (flagged items become nodes) ---
    if screening:
        findings = screening.get("findings") or screening.get("results") or []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            finding_type = finding.get("type") or finding.get("plugin") or "SCREENING"
            finding_text = finding.get("text") or finding.get("value") or finding.get("description") or ""
            if not finding_text:
                continue
            normalized = normalize_entity_name(str(finding_text)[:100])
            if not normalized:
                continue
            entity_id = f"{tenant_prefix}SCREENING::{finding_type}::{normalized}"
            entities.setdefault(
                entity_id,
                GraphEntity(
                    entity_id=entity_id,
                    name=str(finding_text)[:200],
                    type=f"SCREENING_{finding_type.upper()}",
                    normalized_name=normalized,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                ),
            )
            edge_key = f"screening::{document_id}::{entity_id}"
            mentions.append(
                GraphMention(
                    doc_id=str(document_id),
                    entity_id=entity_id,
                    chunk_id=str(document_id),
                    evidence_span=finding.get("context"),
                    confidence=float(finding.get("confidence", 0.9)),
                    edge_key=edge_key,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                )
            )

    if not mentions and not fields:
        return None

    # --- Typed relationships from extraction ---
    typed_relationships = []
    raw_rels = extraction.get("relationships") or []
    for rel in raw_rels:
        if isinstance(rel, dict):
            typed_relationships.append(rel)

    return GraphIngestPayload(
        document=document,
        entities=list(entities.values()),
        mentions=mentions,
        fields=fields,
        typed_relationships=typed_relationships,
        temporal_spans=extraction.get("temporal_spans") or [],
    )


def _first_language(languages) -> str | None:
    if isinstance(languages, list) and languages:
        return str(languages[0])
    if isinstance(languages, str):
        return languages
    return None


# ---------------------------------------------------------------------------
# Entity dedup — create SAME_AS edges for matching entities in the profile
# ---------------------------------------------------------------------------

def _dedup_entities_in_profile(store, payload, subscription_id: str,
                               profile_id: str) -> int:
    """Query existing profile graph for entity dedup.

    For each entity in the payload, check if an entity with the same
    normalized_name already exists under this profile. If so, create a
    SAME_AS edge between them.

    Returns the number of SAME_AS edges created.
    """
    if not payload or not payload.entities:
        return 0

    # Collect normalized names from the new payload
    new_entities = {}
    for ent in payload.entities:
        key = (ent.type, ent.normalized_name)
        new_entities[key] = ent.entity_id

    if not new_entities:
        return 0

    # Query Neo4j for existing entities in this profile with matching
    # normalized names but different entity_ids
    normalized_names = list({ent.normalized_name for ent in payload.entities if ent.normalized_name})
    if not normalized_names:
        return 0

    new_entity_ids = set(new_entities.values())

    query = (
        "UNWIND $names AS norm "
        "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof}) "
        "WHERE e.normalized_name = norm "
        "RETURN e.entity_id AS entity_id, e.normalized_name AS normalized_name, e.type AS type"
    )
    try:
        rows = store.run_query(query, {
            "names": normalized_names,
            "sub": str(subscription_id),
            "prof": str(profile_id),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("KG dedup query failed: %s", exc)
        return 0

    created = 0
    for row in rows:
        existing_id = row.get("entity_id")
        existing_norm = row.get("normalized_name")
        existing_type = row.get("type")
        if not existing_id or not existing_norm:
            continue

        # Find matching new entity
        new_id = new_entities.get((existing_type, existing_norm))
        if not new_id:
            # Try type-agnostic match
            for (etype, enorm), eid in new_entities.items():
                if enorm == existing_norm:
                    new_id = eid
                    break

        if new_id and existing_id != new_id and existing_id not in new_entity_ids:
            # Create SAME_AS edge
            same_as_query = (
                "MATCH (e1:Entity {entity_id: $e1_id}) "
                "MATCH (e2:Entity {entity_id: $e2_id}) "
                "MERGE (e1)-[r:SAME_AS]->(e2) "
                "ON CREATE SET r.created_at = timestamp()"
            )
            try:
                store.run_query(same_as_query, {
                    "e1_id": new_id,
                    "e2_id": existing_id,
                })
                created += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("KG SAME_AS creation skipped: %s", exc)

    return created


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------

@app.task(bind=True, name="src.tasks.kg.build_knowledge_graph",
          max_retries=2, soft_time_limit=1500)
def build_knowledge_graph(self, document_id: str, subscription_id: str,
                          profile_id: str):
    """Build knowledge graph from extraction + screening data.

    Runs async after screening. Never blocks the main pipeline.

    Steps:
    1. Load extraction JSON from Azure Blob
    2. Load screening JSON from Azure Blob (if available)
    3. Convert extraction data to GraphIngestPayload
    4. Ingest nodes and edges via Neo4jStore
    5. Dedup entities within the profile (SAME_AS edges)
    6. Update MongoDB with node/edge counts
    """
    start_time = time.time()

    try:
        update_stage(document_id, "knowledge_graph", STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        append_audit_log(document_id, "KG_BUILD_STARTED",
                        celery_task_id=self.request.id)

        # 1. Load extraction JSON from Azure Blob
        extraction_path = f"{subscription_id}/{profile_id}/{document_id}/extraction.json"
        extraction = _download_blob_json(extraction_path)
        if not extraction:
            raise ValueError(f"Empty extraction JSON at {extraction_path}")

        logger.info("KG: loaded extraction JSON for document %s (%d keys)",
                    document_id, len(extraction))

        # 2. Load screening JSON (optional — may not exist yet)
        screening_path = f"{subscription_id}/{profile_id}/{document_id}/screening.json"
        screening = _try_download_blob_json(screening_path)
        if screening:
            logger.info("KG: loaded screening JSON for document %s", document_id)

        # 3. Build GraphIngestPayload from extraction + screening data
        payload = _extraction_to_graph_payload(
            extraction=extraction,
            screening=screening,
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=document_id,
        )

        if payload is None:
            logger.info("KG: no entities/mentions found for document %s — skipping",
                       document_id)
            update_stage(document_id, "knowledge_graph", status=STAGE_COMPLETED,
                        summary={"node_count": 0, "edge_count": 0,
                                 "skipped": True, "reason": "no_entities"})
            append_audit_log(document_id, "KG_BUILD_COMPLETED",
                           node_count=0, edge_count=0, skipped=True)
            return

        # 4. Ingest into Neo4j using existing infrastructure
        from src.kg.neo4j_store import Neo4jStore
        from src.kg.ingest import ingest_graph_payload

        store = Neo4jStore()
        try:
            store.ensure_graph_constraints()
            ingest_graph_payload(store, payload)

            node_count = len(payload.entities)
            edge_count = len(payload.mentions) + len(payload.fields)
            rel_count = len(payload.typed_relationships)

            logger.info(
                "KG: ingested %d entities, %d mentions, %d fields, %d relationships for doc %s",
                node_count, len(payload.mentions), len(payload.fields),
                rel_count, document_id,
            )

            # 5. Entity dedup — create SAME_AS edges for matching entities
            same_as_count = _dedup_entities_in_profile(
                store, payload, subscription_id, profile_id
            )
            if same_as_count:
                logger.info("KG: created %d SAME_AS dedup edges for doc %s",
                           same_as_count, document_id)
                edge_count += same_as_count

        finally:
            store.close()

        # 6. Update MongoDB with counts
        duration_seconds = round(time.time() - start_time, 2)
        summary = {
            "node_count": node_count,
            "edge_count": edge_count,
            "relationship_count": rel_count,
            "same_as_count": same_as_count,
            "duration_seconds": duration_seconds,
            "has_screening": screening is not None,
        }

        update_stage(document_id, "knowledge_graph", status=STAGE_COMPLETED,
                    summary=summary)
        append_audit_log(document_id, "KG_BUILD_COMPLETED", **summary)

        logger.info(
            "KG build completed for %s in %.2fs: %d nodes, %d edges",
            document_id, duration_seconds, node_count, edge_count,
        )

    except SoftTimeLimitExceeded:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": "KG build timed out", "code": "TIMEOUT"}
        update_stage(document_id, "knowledge_graph", STAGE_FAILED, error=error)
        append_audit_log(document_id, "KG_BUILD_FAILED", error="timeout",
                        duration_seconds=duration_seconds)

    except Exception as exc:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": str(exc), "code": "KG_ERROR"}
        update_stage(document_id, "knowledge_graph", STAGE_FAILED, error=error)
        append_audit_log(document_id, "KG_BUILD_FAILED", error=str(exc),
                        duration_seconds=duration_seconds)
        self.retry(exc=exc)
