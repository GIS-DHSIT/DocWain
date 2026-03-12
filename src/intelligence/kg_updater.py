from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Iterable, Optional

from src.api.config import Config
from src.kg.neo4j_store import Neo4jStore

logger = get_logger(__name__)

class KGUpdater:
    def __init__(self, *, enabled: Optional[bool] = None, redis_client: Optional[Any] = None) -> None:
        self.enabled = bool(enabled if enabled is not None else getattr(Config.KnowledgeGraph, "ENABLED", False))
        self.redis_client = redis_client
        self._store: Optional[Neo4jStore] = None
        if self.enabled:
            try:
                self._store = Neo4jStore()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Neo4j unavailable for KGUpdater: %s", exc)
                self._store = None
                self.enabled = False

    def update(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        source_name: str,
        doc_domain: str,
        sections: Iterable[Dict[str, Any]],
        chunk_metadata: Iterable[Dict[str, Any]],
        section_facts: Iterable[Dict[str, Any]],
    ) -> Dict[str, int]:
        if not self.enabled or not self._store:
            return {"sections": 0, "chunks": 0, "entities": 0, "mentions": 0}

        store = self._store
        updated_at = time.time()

        try:
            store.ensure_constraints()
        except Exception:
            pass

        # Document node
        store.run_query(
            "MERGE (d:Document {document_id: $document_id}) "
            "SET d.source_name = $source_name, d.doc_domain = $doc_domain, "
            "d.profile_id = $profile_id, d.subscription_id = $subscription_id, d.updated_at = $updated_at",
            {
                "document_id": document_id,
                "source_name": source_name,
                "doc_domain": doc_domain,
                "profile_id": profile_id,
                "subscription_id": subscription_id,
                "updated_at": updated_at,
            },
        )

        # Sections
        section_rows = []
        for sec in sections:
            section_rows.append(
                {
                    "section_id": sec.get("section_id"),
                    "section_title": sec.get("section_title"),
                    "section_kind": sec.get("section_kind"),
                    "document_id": document_id,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                }
            )
        if section_rows:
            store.run_query(
                "UNWIND $rows AS row "
                "MERGE (s:Section {section_id: row.section_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "SET s.section_title = row.section_title, s.section_kind = row.section_kind, "
                "    s.subscription_id = row.subscription_id, s.profile_id = row.profile_id "
                "WITH row, s "
                "MATCH (d:Document {document_id: row.document_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (d)-[:HAS_SECTION]->(s)",
                {"rows": section_rows},
            )

        # Chunks
        chunk_rows = []
        for meta in chunk_metadata:
            chunk_rows.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "document_id": document_id,
                    "section_id": meta.get("section_id"),
                    "chunk_kind": meta.get("chunk_kind"),
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                }
            )
        if chunk_rows:
            store.run_query(
                "UNWIND $rows AS row "
                "MERGE (c:Chunk {chunk_id: row.chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "SET c.document_id = row.document_id, c.chunk_kind = row.chunk_kind, "
                "    c.subscription_id = row.subscription_id, c.profile_id = row.profile_id "
                "WITH row, c "
                "MATCH (s:Section {section_id: row.section_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (s)-[:HAS_CHUNK]->(c)",
                {"rows": chunk_rows},
            )

        # Entities and mentions
        entity_rows = []
        mention_rows = []
        about_rows = []
        for fact in section_facts:
            section_id = (fact.get("provenance") or {}).get("section_id")
            for ent in fact.get("entities") or []:
                entity_id = f"{subscription_id}::{profile_id}::{ent.get('type')}::{ent.get('normalized')}"
                entity_rows.append(
                    {
                        "entity_id": entity_id,
                        "type": ent.get("type"),
                        "value": ent.get("value"),
                        "normalized": ent.get("normalized"),
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    }
                )
                about_rows.append(
                    {
                        "section_id": section_id,
                        "entity_id": entity_id,
                        "confidence": ent.get("confidence", 0.5),
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    }
                )
            for span in fact.get("evidence_spans") or []:
                ent_val = span.get("value")
                if not ent_val:
                    continue
                entity_id = f"{subscription_id}::{profile_id}::MENTION::{str(ent_val).lower().strip()}"
                entity_rows.append(
                    {
                        "entity_id": entity_id,
                        "type": "MENTION",
                        "value": ent_val,
                        "normalized": str(ent_val).lower().strip(),
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    }
                )
                mention_rows.append(
                    {
                        "chunk_id": span.get("chunk_id"),
                        "entity_id": entity_id,
                        "confidence": 0.6,
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    }
                )

        if entity_rows:
            store.run_query(
                "UNWIND $rows AS row "
                "MERGE (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "SET e.type = row.type, e.value = row.value, e.normalized = row.normalized, "
                "    e.subscription_id = row.subscription_id, e.profile_id = row.profile_id",
                {"rows": entity_rows},
            )

        if mention_rows:
            store.run_query(
                "UNWIND $rows AS row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MATCH (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (c)-[r:MENTIONS]->(e) "
                "SET r.confidence = row.confidence",
                {"rows": mention_rows},
            )

        if about_rows:
            store.run_query(
                "UNWIND $rows AS row "
                "MATCH (s:Section {section_id: row.section_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MATCH (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (s)-[r:ABOUT]->(e) "
                "SET r.confidence = row.confidence",
                {"rows": about_rows},
            )

        return {
            "sections": len(section_rows),
            "chunks": len(chunk_rows),
            "entities": len(entity_rows),
            "mentions": len(mention_rows),
        }

__all__ = ["KGUpdater"]
