from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.kg.entity_extractor import EntityExtractor
from src.api.vector_store import build_qdrant_filter
from src.kg.qdrant_reader import QdrantChunk, QdrantKGReader
from src.kg.neo4j_store import Neo4jStore

logger = get_logger(__name__)

@dataclass
class KGSyncStats:
    processed_points: int = 0
    skipped_points: int = 0
    chunks_upserted: int = 0
    entities_seen: int = 0
    mentions_created: int = 0
    last_qdrant_offset: Optional[Any] = None

class KGSyncService:
    def __init__(
        self,
        *,
        qdrant_reader: QdrantKGReader,
        neo4j_store: Neo4jStore,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        self.qdrant_reader = qdrant_reader
        self.neo4j_store = neo4j_store
        self.entity_extractor = entity_extractor or EntityExtractor()

    @staticmethod
    def _normalize_section_key(document_id: str, section_path: Optional[str], section_title: Optional[str]) -> str:
        raw = section_path or section_title or "document"
        raw = str(raw).strip().lower() or "document"
        raw = re.sub(r"\s+", " ", raw)
        return f"{document_id}::{raw}"

    @staticmethod
    def _should_skip(existing_hash: Optional[str], incoming_hash: Optional[str]) -> bool:
        if not existing_hash or not incoming_hash:
            return False
        return existing_hash == incoming_hash

    def _build_rows(self, chunks: List[QdrantChunk]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        links: Set[Tuple[str, str, str, str]] = set()

        for chunk in chunks:
            if not chunk.subscription_id or not chunk.profile_id:
                continue
            section_key = self._normalize_section_key(chunk.document_id, chunk.section_path, chunk.section_title)
            rows.append(
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "source_file": chunk.source_file,
                    "doc_type": chunk.doc_type,
                    "document_type": chunk.document_type,
                    "section_key": section_key,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_hash": chunk.chunk_hash or f"nohash:{chunk.chunk_id}",
                    "chunk_char_len": chunk.chunk_char_len,
                    "subscription_id": chunk.subscription_id,
                    "profile_id": chunk.profile_id,
                }
            )

            if chunk.prev_chunk_id:
                links.add((str(chunk.prev_chunk_id), chunk.chunk_id, str(chunk.subscription_id), str(chunk.profile_id)))
            if chunk.next_chunk_id:
                links.add((chunk.chunk_id, str(chunk.next_chunk_id), str(chunk.subscription_id), str(chunk.profile_id)))

            if chunk.text:
                extracted = self.entity_extractor.extract(chunk.text)
            else:
                extracted = []

            row_entities = []
            for entity in extracted:
                tenant_entity_id = f"{chunk.subscription_id}::{chunk.profile_id}::{entity.entity_id}"
                row_entities.append(
                    {
                        "entity_id": tenant_entity_id,
                        "name": entity.name,
                        "type": entity.type,
                    }
                )
            rows[-1]["entities"] = row_entities

        link_rows = [
            {
                "from_chunk_id": a,
                "to_chunk_id": b,
                "subscription_id": sub,
                "profile_id": prof,
            }
            for a, b, sub, prof in sorted(links)
        ]
        return rows, link_rows

    def run(
        self,
        *,
        batch_size: int,
        max_points: int,
        state_name: str,
        subscription_id: str,
        profile_id: str,
    ) -> KGSyncStats:
        self.neo4j_store.ensure_constraints()
        state = self.neo4j_store.get_state(state_name)
        offset = state.last_qdrant_offset

        stats = KGSyncStats()
        current_offset = offset

        scroll_filter = build_qdrant_filter(
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
        )
        for batch in self.qdrant_reader.scroll_batches(
            batch_size=batch_size,
            offset=current_offset,
            max_points=max_points,
            scroll_filter=scroll_filter,
        ):
            stats.last_qdrant_offset = batch.next_offset
            current_offset = batch.next_offset
            if not batch.points:
                if batch.next_offset is None:
                    break
                continue

            chunk_ids = [chunk.chunk_id for chunk in batch.points]
            existing_hashes = self.neo4j_store.fetch_existing_hashes(
                chunk_ids,
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            )

            to_process: List[QdrantChunk] = []
            for chunk in batch.points:
                stats.processed_points += 1
                if not chunk.chunk_id or not chunk.document_id:
                    stats.skipped_points += 1
                    continue
                incoming_hash = chunk.chunk_hash or f"nohash:{chunk.chunk_id}"
                if self._should_skip(existing_hashes.get(chunk.chunk_id), incoming_hash):
                    stats.skipped_points += 1
                    continue
                to_process.append(chunk)

            if not to_process:
                if stats.processed_points >= max_points:
                    break
                if batch.next_offset is None:
                    break
                continue

            rows, links = self._build_rows(to_process)
            self.neo4j_store.upsert_batch(rows)
            self.neo4j_store.upsert_next_links(links)

            stats.chunks_upserted += len(to_process)
            all_entities = {ent["entity_id"] for row in rows for ent in (row.get("entities") or [])}
            stats.entities_seen += len(all_entities)
            stats.mentions_created += sum(len(row.get("entities") or []) for row in rows)

            if stats.processed_points >= max_points:
                break
            if batch.next_offset is None:
                break

        updated_state = self.neo4j_store.update_state(state_name, stats.last_qdrant_offset)
        stats.last_qdrant_offset = updated_state.last_qdrant_offset
        return stats
