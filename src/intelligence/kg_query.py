from __future__ import annotations

from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.intelligence.kg_builder import KGStoreBase, MemoryKGStore, Neo4jKGStore, get_kg_store
from src.kg.entity_extractor import EntityExtractor, normalize_entity_name

logger = get_logger(__name__)

@dataclass
class KGQueryResult:
    doc_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)

class KGQueryService:
    def __init__(
        self,
        *,
        store: Optional[KGStoreBase] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ) -> None:
        self.store = store or get_kg_store()
        self.entity_extractor = entity_extractor or EntityExtractor()

    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        extracted = self.entity_extractor.extract_with_metadata(query or "")
        results = []
        for ent in extracted:
            results.append(
                {
                    "entity_id": ent.entity_id,
                    "type": ent.type,
                    "value": ent.name,
                    "normalized_value": ent.normalized_name,
                    "confidence": ent.confidence,
                }
            )
        return results

    def query(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        domain_hint: Optional[str],
        entities: List[Dict[str, Any]],
        limit: int = 20,
    ) -> KGQueryResult:
        if not entities:
            return KGQueryResult()
        if isinstance(self.store, Neo4jKGStore):
            return self._query_neo4j(
                subscription_id=subscription_id,
                profile_id=profile_id,
                domain_hint=domain_hint,
                entities=entities,
                limit=limit,
            )
        if isinstance(self.store, MemoryKGStore):
            return self._query_memory(
                subscription_id=subscription_id,
                profile_id=profile_id,
                domain_hint=domain_hint,
                entities=entities,
            )
        return KGQueryResult()

    def _query_neo4j(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        domain_hint: Optional[str],
        entities: List[Dict[str, Any]],
        limit: int,
    ) -> KGQueryResult:
        names = [normalize_entity_name(ent.get("value") or ent.get("normalized_value") or "") for ent in entities]
        names = [n for n in names if n]
        if not names:
            return KGQueryResult()
        domain = domain_hint if domain_hint not in {"unknown", "mixed", None, ""} else None
        query = (
            "UNWIND $names AS name "
            "MATCH (d:Document {profile_id: $profile_id, subscription_id: $subscription_id}) "
            "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
            "WHERE (e.normalized_value = name OR e.value CONTAINS name OR name CONTAINS e.normalized_value) "
            "  AND ($domain IS NULL OR d.doc_domain = $domain) "
            "RETURN d.document_id AS document_id, collect(DISTINCT c.chunk_id) AS chunk_ids, "
            "       collect(DISTINCT e.entity_id) AS entity_ids, count(DISTINCT e) AS hits "
            "ORDER BY hits DESC "
            "LIMIT $limit"
        )
        rows = self.store.store.run_query(
            query,
            {
                "names": names,
                "profile_id": str(profile_id),
                "subscription_id": str(subscription_id),
                "domain": domain,
                "limit": int(limit),
            },
        )
        doc_ids: List[str] = []
        chunk_ids: List[str] = []
        entity_ids: List[str] = []
        for row in rows:
            doc_id = row.get("document_id")
            if doc_id:
                doc_ids.append(str(doc_id))
            chunk_ids.extend([str(c) for c in (row.get("chunk_ids") or []) if c])
            entity_ids.extend([str(e) for e in (row.get("entity_ids") or []) if e])
        return KGQueryResult(
            doc_ids=sorted(set(doc_ids)),
            chunk_ids=sorted(set(chunk_ids)),
            entities=[{"entity_id": eid} for eid in sorted(set(entity_ids))],
        )

    def _query_memory(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        domain_hint: Optional[str],
        entities: List[Dict[str, Any]],
    ) -> KGQueryResult:
        name_set = {normalize_entity_name(ent.get("value") or ent.get("normalized_value") or "") for ent in entities}
        name_set = {n for n in name_set if n}
        if not name_set:
            return KGQueryResult()
        doc_ids: List[str] = []
        chunk_ids: List[str] = []
        for chunk_id, entity_ids in self.store.mentions.items():
            chunk = self.store.chunks.get(chunk_id)
            if not chunk:
                continue
            doc = self.store.documents.get(chunk.document_id)
            if not doc:
                continue
            if doc.get("subscription_id") != str(subscription_id):
                continue
            if doc.get("profile_id") != str(profile_id):
                continue
            if domain_hint and domain_hint not in {"mixed", "unknown"}:
                if doc.get("doc_domain") != domain_hint:
                    continue
            for entity_id in entity_ids:
                ent = self.store.entities.get(entity_id)
                if not ent:
                    continue
                if ent.normalized_value in name_set:
                    doc_ids.append(chunk.document_id)
                    chunk_ids.append(chunk.chunk_id)
                    break
        return KGQueryResult(
            doc_ids=sorted(set(doc_ids)),
            chunk_ids=sorted(set(chunk_ids)),
            entities=entities,
        )

__all__ = ["KGQueryService", "KGQueryResult"]
