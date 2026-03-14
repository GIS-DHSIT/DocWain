from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.api.config import Config
from src.kg.entity_extractor import EntityExtractor
from src.kg.neo4j_store import Neo4jStore

logger = get_logger(__name__)

@dataclass
class KGChunk:
    chunk_id: str
    document_id: str
    section_title: Optional[str]
    page: Optional[int]
    chunk_kind: Optional[str]
    text: str
    subscription_id: str
    profile_id: str

@dataclass
class KGEntity:
    entity_id: str
    type: str
    value: str
    normalized_value: str
    subscription_id: str
    profile_id: str

@dataclass
class KGMention:
    chunk_id: str
    entity_id: str
    confidence: float
    span: Optional[str]
    subscription_id: str
    profile_id: str

class KGStoreBase:
    def ingest(
        self,
        *,
        document: Dict[str, Any],
        chunks: List[KGChunk],
        entities: List[KGEntity],
        mentions: List[KGMention],
        about: List[Dict[str, Any]],
    ) -> None:
        raise NotImplementedError

class Neo4jKGStore(KGStoreBase):
    def __init__(self) -> None:
        self.store = Neo4jStore()
        try:
            self.store.ensure_constraints()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Neo4j constraints skipped: %s", exc)

    def ingest(
        self,
        *,
        document: Dict[str, Any],
        chunks: List[KGChunk],
        entities: List[KGEntity],
        mentions: List[KGMention],
        about: List[Dict[str, Any]],
    ) -> None:
        doc_query = (
            "MERGE (d:Document {document_id: $document_id}) "
            "SET d.source_name = $source_name, "
            "    d.doc_domain = $doc_domain, "
            "    d.profile_id = $profile_id, "
            "    d.subscription_id = $subscription_id, "
            "    d.updated_at = $updated_at"
        )
        self.store.run_query(doc_query, document)

        if entities:
            entity_query = (
                "UNWIND $entities AS ent "
                "MERGE (e:Entity {entity_id: ent.entity_id, subscription_id: ent.subscription_id, profile_id: ent.profile_id}) "
                "SET e.type = ent.type, "
                "    e.value = ent.value, "
                "    e.normalized_value = ent.normalized_value, "
                "    e.subscription_id = ent.subscription_id, "
                "    e.profile_id = ent.profile_id"
            )
            self.store.run_query(entity_query, {"entities": [e.__dict__ for e in entities]})

        if chunks:
            chunk_rows = [c.__dict__ for c in chunks]
            chunk_query = (
                "UNWIND $chunks AS row "
                "MERGE (c:Chunk {chunk_id: row.chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "SET c.document_id = row.document_id, "
                "    c.section_title = row.section_title, "
                "    c.page = row.page, "
                "    c.chunk_kind = row.chunk_kind, "
                "    c.subscription_id = row.subscription_id, "
                "    c.profile_id = row.profile_id "
                "WITH row, c "
                "MATCH (d:Document {document_id: row.document_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (d)-[:HAS_CHUNK]->(c)"
            )
            self.store.run_query(chunk_query, {"chunks": chunk_rows})

        if mentions:
            mention_rows = [m.__dict__ for m in mentions]
            mention_query = (
                "UNWIND $mentions AS row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MATCH (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (c)-[r:MENTIONS {chunk_id: row.chunk_id, entity_id: row.entity_id}]->(e) "
                "SET r.confidence = row.confidence, "
                "    r.span = row.span"
            )
            self.store.run_query(mention_query, {"mentions": mention_rows})

        if about:
            about_query = (
                "UNWIND $about AS row "
                "MATCH (d:Document {document_id: row.document_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MATCH (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
                "MERGE (d)-[r:ABOUT]->(e) "
                "SET r.confidence = row.confidence"
            )
            self.store.run_query(about_query, {"about": about})

class MemoryKGStore(KGStoreBase):
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, KGChunk] = {}
        self.entities: Dict[str, KGEntity] = {}
        self.mentions: Dict[str, List[str]] = {}

    def ingest(
        self,
        *,
        document: Dict[str, Any],
        chunks: List[KGChunk],
        entities: List[KGEntity],
        mentions: List[KGMention],
        about: List[Dict[str, Any]],
    ) -> None:
        self.documents[document["document_id"]] = document
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        for ent in entities:
            self.entities[ent.entity_id] = ent
        for mention in mentions:
            self.mentions.setdefault(mention.chunk_id, [])
            if mention.entity_id not in self.mentions[mention.chunk_id]:
                self.mentions[mention.chunk_id].append(mention.entity_id)

_MEMORY_STORE: Optional[MemoryKGStore] = None

def get_kg_store() -> KGStoreBase:
    global _MEMORY_STORE
    if getattr(Config.KnowledgeGraph, "ENABLED", False):
        try:
            return Neo4jKGStore()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Neo4j unavailable; falling back to memory KG store: %s", exc)
    if _MEMORY_STORE is None:
        _MEMORY_STORE = MemoryKGStore()
    return _MEMORY_STORE

class KGBuilder:
    def __init__(
        self,
        *,
        store: Optional[KGStoreBase] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ) -> None:
        self.store = store or get_kg_store()
        self.entity_extractor = entity_extractor or EntityExtractor()

    def ingest_document(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        source_name: str,
        doc_domain: Optional[str],
        chunks: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chunk_list: List[KGChunk] = []
        entity_map: Dict[str, KGEntity] = {}
        mentions: List[KGMention] = []
        about_counts: Dict[str, int] = {}

        for idx, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "")
            if not text.strip():
                continue
            chunk_id = str(chunk.get("chunk_id") or f"{document_id}::chunk::{idx}")
            chunk_list.append(
                KGChunk(
                    chunk_id=chunk_id,
                    document_id=str(document_id),
                    section_title=chunk.get("section_title"),
                    page=chunk.get("page"),
                    chunk_kind=chunk.get("chunk_kind"),
                    text=text,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                )
            )
            extracted = self.entity_extractor.extract_with_metadata(text)
            for ent in extracted:
                entity_id = f"{subscription_id}::{profile_id}::{ent.entity_id}"
                entity_map.setdefault(
                    entity_id,
                    KGEntity(
                        entity_id=entity_id,
                        type=ent.type,
                        value=ent.name,
                        normalized_value=ent.normalized_name,
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                    ),
                )
                span = _extract_span(text, ent.name)
                mentions.append(
                    KGMention(
                        chunk_id=chunk_id,
                        entity_id=entity_id,
                        confidence=ent.confidence,
                        span=span,
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                    )
                )
                about_counts[entity_id] = about_counts.get(entity_id, 0) + 1

        about_edges = [
            {
                "document_id": str(document_id),
                "entity_id": entity_id,
                "confidence": float(count) / max(len(chunk_list) or 1, 1),
                "subscription_id": str(subscription_id),
                "profile_id": str(profile_id),
            }
            for entity_id, count in about_counts.items()
        ]

        document = {
            "document_id": str(document_id),
            "source_name": source_name,
            "doc_domain": doc_domain or "unknown",
            "profile_id": str(profile_id),
            "subscription_id": str(subscription_id),
            "updated_at": time.time(),
        }

        self.store.ingest(
            document=document,
            chunks=chunk_list,
            entities=list(entity_map.values()),
            mentions=mentions,
            about=about_edges,
        )

        return {
            "chunks_ingested": len(chunk_list),
            "entities_ingested": len(entity_map),
            "mentions_ingested": len(mentions),
        }

def _extract_span(text: str, name: str, window: int = 40) -> Optional[str]:
    if not text or not name:
        return None
    lowered = text.lower()
    needle = name.lower()
    idx = lowered.find(needle)
    if idx == -1:
        return None
    start = max(0, idx - window)
    end = min(len(text), idx + len(needle) + window)
    return text[start:end].strip()

__all__ = ["KGBuilder", "get_kg_store", "KGStoreBase", "Neo4jKGStore", "MemoryKGStore"]
