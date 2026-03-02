import datetime as dt
from typing import Any, Dict, Iterable, List, Optional

from src.kg.entity_extractor import EntityExtractor
from src.kg.neo4j_store import KGState
from src.kg.qdrant_reader import QdrantBatch, QdrantChunk
from src.kg.sync_service import KGSyncService


def make_chunk(
    *,
    chunk_id: str,
    chunk_hash: Optional[str],
    text: str = "Sample text",
    document_id: str = "doc-1",
    chunk_index: int = 0,
    prev_chunk_id: Optional[str] = None,
    next_chunk_id: Optional[str] = None,
) -> QdrantChunk:
    return QdrantChunk(
        chunk_id=chunk_id,
        chunk_hash=chunk_hash,
        text=text,
        document_id=document_id,
        filename="file.pdf",
        source_file="file.pdf",
        section_title="Intro",
        section_path="Intro",
        chunk_index=chunk_index,
        page_start=1,
        page_end=1,
        doc_type="doc",
        document_type="doc",
        chunk_char_len=len(text),
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
        subscription_id="sub",
        profile_id="profile",
    )


class FakeQdrantReader:
    def __init__(self, batches: List[QdrantBatch]):
        self.batches = batches
        self.offset_map: Dict[Any, int] = {None: 0}
        for idx, batch in enumerate(batches):
            if batch.next_offset is not None:
                self.offset_map[batch.next_offset] = idx + 1

    def scroll_batches(self, *, batch_size: int, offset: Any = None, max_points: Optional[int] = None, scroll_filter=None):
        start_idx = self.offset_map.get(offset, 0)
        remaining = max_points if max_points is not None else None
        for batch in self.batches[start_idx:]:
            if remaining is not None and remaining <= 0:
                break
            points = batch.points
            if remaining is not None and len(points) > remaining:
                points = points[:remaining]
            yield QdrantBatch(points=points, next_offset=batch.next_offset)
            if remaining is not None:
                remaining -= len(points)


class FakeNeo4jStore:
    def __init__(self):
        self.chunks: Dict[str, Optional[str]] = {}
        self.mentions: set[tuple[str, str]] = set()
        self.next_links: set[tuple[str, str]] = set()
        self.state: Dict[str, KGState] = {}

    def ensure_constraints(self) -> None:
        return None

    def get_state(self, name: str) -> KGState:
        return self.state.get(name) or KGState(name=name, last_qdrant_offset=None, last_sync_at=None)

    def update_state(self, name: str, last_qdrant_offset: Optional[Any]) -> KGState:
        state = KGState(
            name=name,
            last_qdrant_offset=last_qdrant_offset,
            last_sync_at=dt.datetime.utcnow().isoformat(),
        )
        self.state[name] = state
        return state

    def fetch_existing_hashes(self, chunk_ids: Iterable[str], *, subscription_id: Optional[str] = None, profile_id: Optional[str] = None) -> Dict[str, Optional[str]]:
        return {chunk_id: self.chunks.get(chunk_id) for chunk_id in chunk_ids}

    def upsert_batch(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.chunks[row["chunk_id"]] = row.get("chunk_hash")
            entities = row.get("entities") or []
            # replace mentions for this chunk
            self.mentions = {m for m in self.mentions if m[0] != row["chunk_id"]}
            for ent in entities:
                self.mentions.add((row["chunk_id"], ent["entity_id"]))

    def replace_mentions(self, chunk_ids: List[str], mentions: List[Dict[str, Any]]) -> None:
        for chunk_id in chunk_ids:
            self.mentions = {m for m in self.mentions if m[0] != chunk_id}
        for mention in mentions:
            self.mentions.add((mention["chunk_id"], mention["entity_id"]))

    def upsert_next_links(self, links: List[Dict[str, Any]]) -> None:
        for link in links:
            self.next_links.add((link["from_chunk_id"], link["to_chunk_id"]))


def test_entity_extractor_patterns():
    extractor = EntityExtractor(skills=["Python", "FastAPI"])
    text = "Contact: jane.doe@example.com, visit https://example.com, call +1 (415) 555-1234. 5 years with Python and FastAPI."
    entities = extractor.extract(text)
    ids = {e.entity_id for e in entities}
    assert "email::jane.doe@example.com" in ids
    assert "url::https://example.com" in ids
    assert "phone::14155551234" in ids
    assert "duration_years::5 years" in ids
    assert "skill::python" in ids
    assert "skill::fastapi" in ids


def test_sync_skips_unchanged_chunks():
    chunk1 = make_chunk(chunk_id="c1", chunk_hash="hash1", text="alpha")
    chunk2 = make_chunk(chunk_id="c2", chunk_hash="hash2", text="beta")
    reader = FakeQdrantReader([QdrantBatch(points=[chunk1, chunk2], next_offset=1)])
    store = FakeNeo4jStore()
    store.chunks["c1"] = "hash1"
    service = KGSyncService(qdrant_reader=reader, neo4j_store=store)

    stats = service.run(batch_size=2, max_points=10, state_name="default", subscription_id="sub", profile_id="profile")
    assert stats.chunks_upserted == 1
    assert stats.skipped_points == 1
    assert store.chunks["c1"] == "hash1"
    assert store.chunks["c2"] == "hash2"


def test_missing_hash_defaults_to_nohash():
    chunk = make_chunk(chunk_id="c9", chunk_hash=None, text="epsilon")
    reader = FakeQdrantReader([QdrantBatch(points=[chunk], next_offset=None)])
    store = FakeNeo4jStore()
    service = KGSyncService(qdrant_reader=reader, neo4j_store=store)

    stats = service.run(batch_size=1, max_points=10, state_name="default", subscription_id="sub", profile_id="profile")
    assert stats.chunks_upserted == 1
    assert store.chunks["c9"] == "nohash:c9"


def test_idempotent_sync():
    chunk = make_chunk(chunk_id="c3", chunk_hash="hash3", text="gamma")
    reader1 = FakeQdrantReader([QdrantBatch(points=[chunk], next_offset=1)])
    store = FakeNeo4jStore()
    service1 = KGSyncService(qdrant_reader=reader1, neo4j_store=store)
    stats1 = service1.run(batch_size=1, max_points=10, state_name="default", subscription_id="sub", profile_id="profile")

    store.state["default"] = KGState(name="default", last_qdrant_offset=None, last_sync_at=dt.datetime.utcnow().isoformat())

    reader2 = FakeQdrantReader([QdrantBatch(points=[chunk], next_offset=1)])
    service2 = KGSyncService(qdrant_reader=reader2, neo4j_store=store)
    stats2 = service2.run(batch_size=1, max_points=10, state_name="default", subscription_id="sub", profile_id="profile")

    assert stats1.chunks_upserted == 1
    assert stats2.chunks_upserted == 0
    assert stats2.skipped_points == 1
    assert len(store.mentions) > 0


def test_state_persistence():
    chunk = make_chunk(chunk_id="c4", chunk_hash="hash4", text="delta")
    reader = FakeQdrantReader([QdrantBatch(points=[chunk], next_offset=99)])
    store = FakeNeo4jStore()
    service = KGSyncService(qdrant_reader=reader, neo4j_store=store)

    stats = service.run(batch_size=1, max_points=10, state_name="default", subscription_id="sub", profile_id="profile")
    state = store.state["default"]

    assert stats.last_qdrant_offset == 99
    assert state.last_qdrant_offset == 99
    assert state.last_sync_at is not None


def test_scroll_pagination_respects_max_points():
    batch1 = QdrantBatch(points=[make_chunk(chunk_id="c5", chunk_hash="h5"), make_chunk(chunk_id="c6", chunk_hash="h6")], next_offset=10)
    batch2 = QdrantBatch(points=[make_chunk(chunk_id="c7", chunk_hash="h7"), make_chunk(chunk_id="c8", chunk_hash="h8")], next_offset=20)
    reader = FakeQdrantReader([batch1, batch2])
    store = FakeNeo4jStore()
    service = KGSyncService(qdrant_reader=reader, neo4j_store=store)

    stats = service.run(batch_size=2, max_points=3, state_name="default", subscription_id="sub", profile_id="profile")
    assert stats.processed_points == 3
    assert stats.last_qdrant_offset == 20
