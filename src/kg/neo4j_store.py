import datetime as dt
from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from neo4j import GraphDatabase
except Exception:  # noqa: BLE001
    GraphDatabase = None

from src.api.config import Config

logger = get_logger(__name__)

@dataclass
class KGState:
    name: str
    last_qdrant_offset: Optional[Any]
    last_sync_at: Optional[str]

class Neo4jStore:
    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        driver=None,
    ):
        if GraphDatabase is None:
            raise ImportError("neo4j driver is not installed. Install it with 'pip install neo4j'.")
        self.uri = uri or Config.Neo4j.URI
        self.user = user or Config.Neo4j.USER
        self.password = password or Config.Neo4j.PASSWORD
        self.database = database or Config.Neo4j.DATABASE
        self.driver = driver or GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def _session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def ensure_constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT kg_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT kg_section_key IF NOT EXISTS FOR (s:Section) REQUIRE s.section_key IS UNIQUE",
            "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT kg_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT kg_state_name IF NOT EXISTS FOR (k:KGState) REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT kg_timeline_id IF NOT EXISTS FOR (t:Timeline) REQUIRE t.timeline_id IS UNIQUE",
        ]
        with self._session() as session:
            for query in queries:
                session.run(query)

    def ensure_graph_constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT kg_graph_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT kg_graph_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        ]
        with self._session() as session:
            for query in queries:
                session.run(query)

    def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self._session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def get_state(self, name: str) -> KGState:
        query = (
            "MERGE (k:KGState {name: $name}) "
            "ON CREATE SET k.last_qdrant_offset = null, k.last_sync_at = null "
            "RETURN k.name AS name, k.last_qdrant_offset AS last_qdrant_offset, k.last_sync_at AS last_sync_at"
        )
        with self._session() as session:
            record = session.run(query, name=name).single()
            if not record:
                return KGState(name=name, last_qdrant_offset=None, last_sync_at=None)
            return KGState(
                name=record.get("name"),
                last_qdrant_offset=record.get("last_qdrant_offset"),
                last_sync_at=record.get("last_sync_at"),
            )

    def update_state(self, name: str, last_qdrant_offset: Optional[Any]) -> KGState:
        last_sync_at = dt.datetime.utcnow().isoformat()
        query = (
            "MERGE (k:KGState {name: $name}) "
            "SET k.last_qdrant_offset = $last_qdrant_offset, k.last_sync_at = $last_sync_at "
            "RETURN k.name AS name, k.last_qdrant_offset AS last_qdrant_offset, k.last_sync_at AS last_sync_at"
        )
        with self._session() as session:
            record = session.run(
                query,
                name=name,
                last_qdrant_offset=last_qdrant_offset,
                last_sync_at=last_sync_at,
            ).single()
            return KGState(
                name=record.get("name"),
                last_qdrant_offset=record.get("last_qdrant_offset"),
                last_sync_at=record.get("last_sync_at"),
            )

    def fetch_existing_hashes(
        self,
        chunk_ids: Iterable[str],
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        ids = [str(cid) for cid in chunk_ids if cid is not None]
        if not ids:
            return {}
        query = (
            "UNWIND $chunk_ids AS id "
            "OPTIONAL MATCH (c:Chunk {chunk_id: id, subscription_id: $subscription_id, profile_id: $profile_id}) "
            "RETURN id AS chunk_id, c.chunk_hash AS chunk_hash"
        )
        with self._session() as session:
            result = session.run(
                query,
                chunk_ids=ids,
                subscription_id=str(subscription_id) if subscription_id else "",
                profile_id=str(profile_id) if profile_id else "",
            )
            return {record["chunk_id"]: record.get("chunk_hash") for record in result}

    def upsert_batch(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        query = (
            "UNWIND $rows AS row "
            "MERGE (d:Document {document_id: row.document_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "SET d.filename = coalesce(row.filename, d.filename), "
            "    d.source_file = coalesce(row.source_file, d.source_file), "
            "    d.doc_type = coalesce(row.doc_type, d.doc_type), "
            "    d.document_type = coalesce(row.document_type, d.document_type), "
            "    d.subscription_id = row.subscription_id, "
            "    d.profile_id = row.profile_id "
            "MERGE (s:Section {section_key: row.section_key, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "SET s.document_id = row.document_id, "
            "    s.section_title = coalesce(row.section_title, s.section_title), "
            "    s.section_path = coalesce(row.section_path, s.section_path), "
            "    s.subscription_id = row.subscription_id, "
            "    s.profile_id = row.profile_id "
            "MERGE (d)-[:HAS_SECTION]->(s) "
            "MERGE (c:Chunk {chunk_id: row.chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "ON CREATE SET c.chunk_hash = row.chunk_hash "
            "SET c.document_id = row.document_id, "
            "    c.chunk_index = coalesce(row.chunk_index, c.chunk_index), "
            "    c.page_start = coalesce(row.page_start, c.page_start), "
            "    c.page_end = coalesce(row.page_end, c.page_end), "
            "    c.chunk_hash = coalesce(row.chunk_hash, c.chunk_hash), "
            "    c.chunk_char_len = coalesce(row.chunk_char_len, c.chunk_char_len), "
            "    c.subscription_id = row.subscription_id, "
            "    c.profile_id = row.profile_id "
            "MERGE (s)-[:HAS_CHUNK]->(c) "
            "WITH row, c "
            "OPTIONAL MATCH (c)-[r:MENTIONS]->() "
            "DELETE r "
            "WITH row, c "
            "UNWIND coalesce(row.entities, []) AS ent "
            "MERGE (e:Entity {entity_id: ent.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "SET e.name = ent.name, e.type = ent.type, "
            "    e.subscription_id = row.subscription_id, "
            "    e.profile_id = row.profile_id "
            "MERGE (c)-[:MENTIONS]->(e)"
        )
        with self._session() as session:
            session.run(query, rows=rows)

    def get_status(self, name: str) -> Dict[str, Any]:
        state = self.get_state(name)
        counts_query = (
            "RETURN "
            "size([(d:Document) | d]) AS documents, "
            "size([(s:Section) | s]) AS sections, "
            "size([(c:Chunk) | c]) AS chunks, "
            "size([(e:Entity) | e]) AS entities"
        )
        with self._session() as session:
            record = session.run(counts_query).single()
        return {
            "state_name": state.name,
            "last_qdrant_offset": state.last_qdrant_offset,
            "last_sync_at": state.last_sync_at,
            "counts": {
                "documents": record.get("documents") if record else 0,
                "sections": record.get("sections") if record else 0,
                "chunks": record.get("chunks") if record else 0,
                "entities": record.get("entities") if record else 0,
            },
        }

    def replace_mentions(
        self,
        chunk_ids: List[str],
        mentions: List[Dict[str, Any]],
        *,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> None:
        if chunk_ids:
            delete_query = (
                "UNWIND $chunk_ids AS chunk_id "
                "MATCH (c:Chunk {chunk_id: chunk_id, subscription_id: $subscription_id, profile_id: $profile_id})-[r:MENTIONS]->() "
                "DELETE r"
            )
            with self._session() as session:
                session.run(
                    delete_query,
                    chunk_ids=chunk_ids,
                    subscription_id=str(subscription_id) if subscription_id else "",
                    profile_id=str(profile_id) if profile_id else "",
                )

        if not mentions:
            return

        insert_query = (
            "UNWIND $mentions AS row "
            "MERGE (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "SET e.name = row.name, e.type = row.type, "
            "    e.subscription_id = row.subscription_id, e.profile_id = row.profile_id "
            "WITH row "
            "MATCH (c:Chunk {chunk_id: row.chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MERGE (c)-[:MENTIONS]->(e)"
        )
        with self._session() as session:
            session.run(insert_query, mentions=mentions)

    def upsert_next_links(self, links: List[Dict[str, Any]]) -> None:
        if not links:
            return
        query = (
            "UNWIND $links AS row "
            "MATCH (c1:Chunk {chunk_id: row.from_chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MATCH (c2:Chunk {chunk_id: row.to_chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MERGE (c1)-[:NEXT]->(c2)"
        )
        with self._session() as session:
            session.run(query, links=links)

    def probe_entities(
        self,
        *,
        entity_ids: List[str],
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        limit: int = 20,
        timeout_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        query = (
            "UNWIND $entity_ids AS eid "
            "MATCH (e:Entity {entity_id: eid, subscription_id: $subscription_id, profile_id: $profile_id})<-[:MENTIONS]-(c:Chunk) "
            "<-[:HAS_CHUNK]-(s:Section)<-[:HAS_SECTION]-(d:Document {subscription_id: $subscription_id, profile_id: $profile_id}) "
            "RETURN d.document_id AS document_id, s.section_path AS section_path, count(*) AS hits "
            "ORDER BY hits DESC "
            "LIMIT $limit"
        )
        params = {
            "entity_ids": entity_ids,
            "limit": int(limit),
            "subscription_id": str(subscription_id) if subscription_id else "",
            "profile_id": str(profile_id) if profile_id else "",
        }
        with self._session() as session:
            try:
                result = session.run(query, **params, timeout=(timeout_ms / 1000.0) if timeout_ms else None)
            except TypeError:
                result = session.run(query, **params)
            return [dict(record) for record in result]

    def create_entity_relationship(
        self,
        entity1_id: str,
        entity2_id: str,
        relation_type: str = "RELATED_TO",
        frequency: int = 1,
    ) -> None:
        """Create or update a relationship between two entities."""
        query = (
            "MATCH (e1:Entity {entity_id: $e1_id}) "
            "MATCH (e2:Entity {entity_id: $e2_id}) "
            "MERGE (e1)-[r:RELATED_TO]->(e2) "
            "ON CREATE SET r.frequency = $freq, r.relation_type = $rel_type, r.created_at = timestamp() "
            "ON MATCH SET r.frequency = r.frequency + $freq"
        )
        with self._session() as session:
            session.run(
                query,
                e1_id=entity1_id,
                e2_id=entity2_id,
                freq=frequency,
                rel_type=relation_type,
            )

    def create_timeline_node(
        self,
        document_id: str,
        start_date: str,
        end_date: str,
        description: str,
        entity_ids: Optional[List[str]] = None,
    ) -> None:
        """Create a Timeline node linked to a document and related entities."""
        import uuid

        timeline_id = f"tl_{document_id}_{uuid.uuid4().hex[:8]}"
        query = (
            "MERGE (t:Timeline {timeline_id: $tid}) "
            "SET t.document_id = $doc_id, t.start_date = $start, t.end_date = $end, "
            "    t.description = $desc, t.updated_at = timestamp() "
            "WITH t "
            "MATCH (d:Document {document_id: $doc_id}) "
            "MERGE (d)-[:HAS_TIMELINE]->(t)"
        )
        with self._session() as session:
            session.run(
                query,
                tid=timeline_id,
                doc_id=document_id,
                start=start_date,
                end=end_date,
                desc=description,
            )

        if entity_ids:
            for eid in entity_ids[:10]:  # Cap at 10 entity links
                link_query = (
                    "MATCH (t:Timeline {timeline_id: $tid}) "
                    "MATCH (e:Entity {entity_id: $eid}) "
                    "MERGE (t)-[:INVOLVES]->(e)"
                )
                with self._session() as session:
                    session.run(link_query, tid=timeline_id, eid=eid)

    def create_document_similarity(
        self,
        doc1_id: str,
        doc2_id: str,
        similarity: float,
    ) -> None:
        """Create a similarity relationship between two documents."""
        if similarity < 0.85:
            return
        query = (
            "MATCH (d1:Document {document_id: $d1_id}) "
            "MATCH (d2:Document {document_id: $d2_id}) "
            "MERGE (d1)-[r:SIMILAR_TO]->(d2) "
            "SET r.similarity = $sim, r.updated_at = timestamp()"
        )
        with self._session() as session:
            session.run(query, d1_id=doc1_id, d2_id=doc2_id, sim=similarity)
