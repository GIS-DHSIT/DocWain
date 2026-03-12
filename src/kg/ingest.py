import hashlib
import json
from src.utils.logging_utils import get_logger
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.api.config import Config
from src.kg.entity_extractor import EntityExtractor, normalize_entity_name
from src.kg.neo4j_store import Neo4jStore

logger = get_logger(__name__)

DEFAULT_QUEUE_KEY = "kg:ingest:queue"
DEFAULT_DEAD_KEY = "kg:ingest:dead"

@dataclass
class GraphEntity:
    entity_id: str
    name: str
    type: str
    normalized_name: str
    subscription_id: str = ""
    profile_id: str = ""

@dataclass
class GraphMention:
    doc_id: str
    entity_id: str
    chunk_id: str
    evidence_span: Optional[str]
    confidence: float
    edge_key: str
    subscription_id: str = ""
    profile_id: str = ""

@dataclass
class GraphField:
    doc_id: str
    entity_id: str
    chunk_id: str
    key: str
    value: str
    confidence: float
    edge_key: str
    subscription_id: str = ""
    profile_id: str = ""

@dataclass
class GraphIngestPayload:
    document: Dict[str, Any]
    entities: List[GraphEntity]
    mentions: List[GraphMention]
    fields: List[GraphField]
    attempts: int = 0
    typed_relationships: List[Dict[str, Any]] = field(default_factory=list)
    temporal_spans: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "entities": [entity.__dict__ for entity in self.entities],
            "mentions": [mention.__dict__ for mention in self.mentions],
            "fields": [field.__dict__ for field in self.fields],
            "attempts": self.attempts,
            "typed_relationships": self.typed_relationships,
            "temporal_spans": self.temporal_spans,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "GraphIngestPayload":
        document = payload.get("document") or {}
        sub = document.get("subscription_id") or ""
        prof = document.get("profile_id") or ""

        def _coerce_entity(entity: Dict[str, Any]) -> GraphEntity:
            entity.setdefault("subscription_id", sub)
            entity.setdefault("profile_id", prof)
            return GraphEntity(**entity)

        def _coerce_mention(mention: Dict[str, Any]) -> GraphMention:
            mention.setdefault("subscription_id", sub)
            mention.setdefault("profile_id", prof)
            return GraphMention(**mention)

        def _coerce_field(field: Dict[str, Any]) -> GraphField:
            field.setdefault("subscription_id", sub)
            field.setdefault("profile_id", prof)
            return GraphField(**field)

        return GraphIngestPayload(
            document=document,
            entities=[_coerce_entity(entity) for entity in payload.get("entities", [])],
            mentions=[_coerce_mention(mention) for mention in payload.get("mentions", [])],
            fields=[_coerce_field(field) for field in payload.get("fields", [])],
            attempts=int(payload.get("attempts", 0)),
            typed_relationships=payload.get("typed_relationships") or [],
            temporal_spans=payload.get("temporal_spans") or [],
        )

class GraphIngestQueue:
    def __init__(
        self,
        *,
        redis_client: Optional[Any] = None,
        queue_key: str = DEFAULT_QUEUE_KEY,
        dead_key: str = DEFAULT_DEAD_KEY,
        enabled: Optional[bool] = None,
        max_retries: int = 3,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.redis_client = redis_client
        self.queue_key = queue_key
        self.dead_key = dead_key
        self.enabled = enabled if enabled is not None else bool(getattr(Config.KnowledgeGraph, "ENABLED", False))
        self.max_retries = max_retries
        self.poll_interval_s = poll_interval_s
        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_worker, name="kg-ingest-worker", daemon=True)
            self._worker.start()

    def enqueue(self, payload: GraphIngestPayload) -> None:
        if not self.enabled:
            return
        self.start()
        serialized = json.dumps(payload.to_dict())
        if self.redis_client is not None:
            try:
                self.redis_client.lpush(self.queue_key, serialized)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to enqueue KG payload in Redis: %s", exc)
        self._queue.put(serialized)

    def _pop_payload(self) -> Optional[str]:
        if self.redis_client is not None:
            try:
                item = self.redis_client.brpop(self.queue_key, timeout=1)
                if item:
                    return item[1]
            except Exception as exc:  # noqa: BLE001
                logger.warning("KG Redis pop failed: %s", exc)
        try:
            return self._queue.get(timeout=1)
        except queue.Empty:
            return None

    def _requeue(self, payload: GraphIngestPayload) -> None:
        payload.attempts += 1
        serialized = json.dumps(payload.to_dict())
        if payload.attempts > self.max_retries:
            if self.redis_client is not None:
                try:
                    self.redis_client.lpush(self.dead_key, serialized)
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to push KG payload to dead letter queue: %s", exc)
            logger.error("KG payload dropped after %s attempts", payload.attempts)
            return
        if self.redis_client is not None:
            try:
                self.redis_client.lpush(self.queue_key, serialized)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to requeue KG payload in Redis: %s", exc)
        self._queue.put(serialized)

    def _run_worker(self) -> None:
        if not self.enabled:
            return
        try:
            store = Neo4jStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("KG worker disabled: %s", exc)
            return
        try:
            store.ensure_graph_constraints()
        except Exception as exc:  # noqa: BLE001
            logger.warning("KG constraints setup failed: %s", exc)
        while True:
            payload_raw = self._pop_payload()
            if not payload_raw:
                time.sleep(self.poll_interval_s)
                continue
            try:
                payload = GraphIngestPayload.from_dict(json.loads(payload_raw))
                ingest_graph_payload(store, payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("KG ingest failed: %s", exc, exc_info=True)
                try:
                    payload = GraphIngestPayload.from_dict(json.loads(payload_raw))
                    self._requeue(payload)
                except Exception:  # noqa: BLE001
                    logger.error("Failed to requeue KG payload after error")

_graph_ingest_queue: Optional[GraphIngestQueue] = None

def get_graph_ingest_queue(redis_client: Optional[Any] = None) -> GraphIngestQueue:
    global _graph_ingest_queue
    if _graph_ingest_queue is None:
        _graph_ingest_queue = GraphIngestQueue(redis_client=redis_client)
    return _graph_ingest_queue

def _deep_entities_to_graph_entities(deep_entities: list) -> list:
    """Convert deep analysis EntityMention dicts to GraphEntity-compatible dicts.

    Each incoming dict is expected to have at least ``text`` and ``type`` keys.
    Returns a list of dicts with ``text``, ``type``, ``confidence``, and
    optional ``page`` — ready for conversion to :class:`GraphEntity` instances
    during graph payload construction.
    """
    result: List[Dict[str, Any]] = []
    for ent in (deep_entities or []):
        if not isinstance(ent, dict):
            continue
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        result.append({
            "text": text,
            "type": ent.get("type", "UNKNOWN"),
            "confidence": float(ent.get("confidence", 0.5)),
            "page": ent.get("page"),
        })
    return result

def build_graph_payload(
    *,
    embeddings_payload: Dict[str, Any],
    subscription_id: str,
    profile_id: str,
    document_id: str,
    doc_name: str,
    doc_metadata: Optional[Dict[str, Any]] = None,
    deep_entities: Optional[List[Dict[str, Any]]] = None,
    typed_relationships: Optional[List[Dict[str, Any]]] = None,
) -> Optional[GraphIngestPayload]:
    if not embeddings_payload:
        return None
    if not getattr(Config.KnowledgeGraph, "ENABLED", False):
        return None

    texts = embeddings_payload.get("texts") or []
    chunk_metadata = embeddings_payload.get("chunk_metadata") or []
    if not texts:
        return None

    doc_metadata = doc_metadata or embeddings_payload.get("doc_metadata") or {}
    detected_language = _coerce_language(doc_metadata.get("languages"))
    document_category = doc_metadata.get("document_type") or doc_metadata.get("doc_type") or "generic"

    graph_version = _graph_version(document_id, chunk_metadata)
    tenant_prefix = f"{subscription_id}::{profile_id}::"

    document = {
        "doc_id": str(document_id),
        "profile_id": str(profile_id),
        "subscription_id": str(subscription_id),
        "doc_name": doc_name,
        "document_category": document_category,
        "detected_language": detected_language,
        "created_at": doc_metadata.get("created_at"),
        "graph_version": graph_version,
    }

    extractor = EntityExtractor()
    entities: Dict[str, GraphEntity] = {}
    mentions: List[GraphMention] = []
    fields: List[GraphField] = []

    for idx, text in enumerate(texts):
        meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
        chunk_id = meta.get("chunk_id")
        if not chunk_id:
            continue
        extracted = extractor.extract_with_metadata(text)
        for ent in extracted:
            entities.setdefault(
                f"{tenant_prefix}{ent.entity_id}",
                GraphEntity(
                    entity_id=f"{tenant_prefix}{ent.entity_id}",
                    name=ent.name,
                    type=ent.type,
                    normalized_name=ent.normalized_name,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                ),
            )
            evidence_span = _extract_evidence_span(text, ent.name)
            edge_key = f"mentions::{document_id}::{tenant_prefix}{ent.entity_id}::{chunk_id}"
            mentions.append(
                GraphMention(
                    doc_id=str(document_id),
                    entity_id=f"{tenant_prefix}{ent.entity_id}",
                    chunk_id=str(chunk_id),
                    evidence_span=evidence_span,
                    confidence=ent.confidence,
                    edge_key=edge_key,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                )
            )

        field_map = _extract_structured_fields(meta)
        if field_map:
            for key, value in field_map:
                normalized_value = normalize_entity_name(str(value))
                entity_id = f"{tenant_prefix}FIELD::{key.lower()}::{normalized_value}"
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
                edge_key = f"field::{document_id}::{entity_id}::{chunk_id}"
                fields.append(
                    GraphField(
                        doc_id=str(document_id),
                        entity_id=entity_id,
                        chunk_id=str(chunk_id),
                        key=key,
                        value=str(value),
                        confidence=0.85,
                        edge_key=edge_key,
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                    )
                )

    # ── Deep analysis entities (from document understanding pipeline) ──
    deep_ents = _deep_entities_to_graph_entities(deep_entities)
    for dent in deep_ents:
        ent_type = dent["type"]
        ent_name = dent["text"]
        confidence = dent["confidence"]
        normalized = normalize_entity_name(ent_name)
        if not normalized:
            continue
        entity_id = f"{tenant_prefix}{ent_type}::{normalized}"
        entities.setdefault(
            entity_id,
            GraphEntity(
                entity_id=entity_id,
                name=ent_name,
                type=ent_type,
                normalized_name=normalized,
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            ),
        )
        # Deep entities are document-level — link to the document via a
        # synthetic mention using the first available chunk_id (or doc_id).
        first_chunk_id = None
        if chunk_metadata:
            first_chunk_id = chunk_metadata[0].get("chunk_id")
        ref_id = first_chunk_id or str(document_id)
        edge_key = f"deep::{document_id}::{entity_id}"
        mentions.append(
            GraphMention(
                doc_id=str(document_id),
                entity_id=entity_id,
                chunk_id=ref_id,
                evidence_span=None,
                confidence=confidence,
                edge_key=edge_key,
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            )
        )

    if not mentions and not fields:
        return None

    return GraphIngestPayload(
        document=document,
        entities=list(entities.values()),
        mentions=mentions,
        fields=fields,
        typed_relationships=list(typed_relationships) if typed_relationships else [],
        temporal_spans=[],  # Populated by callers that pass deep_result.temporal_spans
    )

def ingest_graph_payload(store: Neo4jStore, payload: GraphIngestPayload) -> None:
    if not payload or not payload.document:
        return

    documents = [payload.document]
    doc_sub = payload.document.get("subscription_id")
    doc_prof = payload.document.get("profile_id")

    entities = [entity.__dict__ for entity in payload.entities]
    mentions = [
        mention.__dict__
        for mention in payload.mentions
        if mention.subscription_id == doc_sub and mention.profile_id == doc_prof
    ]
    fields = [
        field.__dict__
        for field in payload.fields
        if field.subscription_id == doc_sub and field.profile_id == doc_prof
    ]
    if payload.mentions and not mentions:
        logger.warning("KG Guard: dropped mentions with cross-profile scope for doc_id=%s", payload.document.get("doc_id"))
    if payload.fields and not fields:
        logger.warning("KG Guard: dropped fields with cross-profile scope for doc_id=%s", payload.document.get("doc_id"))

    store.ensure_graph_constraints()

    doc_query = (
        "UNWIND $docs AS doc "
        "MERGE (d:Document {doc_id: doc.doc_id}) "
        "SET d.profile_id = doc.profile_id, "
        "    d.subscription_id = doc.subscription_id, "
        "    d.doc_name = doc.doc_name, "
        "    d.document_category = doc.document_category, "
        "    d.detected_language = doc.detected_language, "
        "    d.graph_version = doc.graph_version, "
        "    d.created_at = coalesce(d.created_at, doc.created_at, datetime())"
    )
    store.run_query(doc_query, {"docs": documents})

    if entities:
        entity_query = (
            "UNWIND $entities AS ent "
            "MERGE (e:Entity {entity_id: ent.entity_id}) "
            "SET e.name = ent.name, e.type = ent.type, e.normalized_name = ent.normalized_name, "
            "    e.subscription_id = ent.subscription_id, e.profile_id = ent.profile_id "
            "WITH e, ent "
            "FOREACH (_ IN CASE WHEN ent.type = 'PERSON' THEN [1] ELSE [] END | SET e:Person) "
            "FOREACH (_ IN CASE WHEN ent.type = 'ORGANIZATION' THEN [1] ELSE [] END | SET e:Organization) "
            "FOREACH (_ IN CASE WHEN ent.type = 'SKILL' THEN [1] ELSE [] END | SET e:Skill) "
            "FOREACH (_ IN CASE WHEN ent.type = 'LOCATION' THEN [1] ELSE [] END | SET e:Location) "
            "FOREACH (_ IN CASE WHEN ent.type = 'DATE' THEN [1] ELSE [] END | SET e:Date) "
            "FOREACH (_ IN CASE WHEN ent.type = 'AMOUNT' THEN [1] ELSE [] END | SET e:Amount) "
            "FOREACH (_ IN CASE WHEN ent.type IN ['CLAUSE','TERM'] THEN [1] ELSE [] END | SET e:Clause) "
            "FOREACH (_ IN CASE WHEN ent.type = 'PRODUCT' THEN [1] ELSE [] END | SET e:Product) "
            "FOREACH (_ IN CASE WHEN ent.type = 'LAW' THEN [1] ELSE [] END | SET e:Law) "
            "FOREACH (_ IN CASE WHEN ent.type = 'EVENT' THEN [1] ELSE [] END | SET e:Event) "
            "FOREACH (_ IN CASE WHEN ent.type = 'MEDICAL_TERM' THEN [1] ELSE [] END | SET e:MedicalTerm) "
            "FOREACH (_ IN CASE WHEN ent.type = 'FINANCIAL_INSTRUMENT' THEN [1] ELSE [] END | SET e:FinancialInstrument)"
        )
        store.run_query(entity_query, {"entities": entities})

    if mentions:
        mention_query = (
            "UNWIND $mentions AS row "
            "MATCH (d:Document {doc_id: row.doc_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MATCH (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MERGE (d)-[r:MENTIONS {edge_key: row.edge_key}]->(e) "
            "SET r.chunk_id = row.chunk_id, "
            "    r.evidence_span = row.evidence_span, "
            "    r.confidence = row.confidence"
        )
        store.run_query(mention_query, {"mentions": mentions})

    if fields:
        field_query = (
            "UNWIND $fields AS row "
            "MATCH (d:Document {doc_id: row.doc_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MATCH (e:Entity {entity_id: row.entity_id, subscription_id: row.subscription_id, profile_id: row.profile_id}) "
            "MERGE (d)-[r:HAS_FIELD {edge_key: row.edge_key}]->(e) "
            "SET r.key = row.key, "
            "    r.value = row.value, "
            "    r.confidence = row.confidence, "
            "    r.chunk_id = row.chunk_id"
        )
        store.run_query(field_query, {"fields": fields})

    # ── Typed relationships from deep analysis ──
    # Build a lookup of entity normalized name → entity_id from the just-ingested entities.
    typed_rels = payload.typed_relationships or []
    if typed_rels:
        # Map normalized entity names to entity_ids for lookup
        entity_name_to_id: Dict[str, str] = {}
        for ent in payload.entities:
            key = ent.normalized_name.lower() if ent.normalized_name else ent.name.lower()
            entity_name_to_id[key] = ent.entity_id
            # Also map by raw name in case normalized differs
            entity_name_to_id[ent.name.lower()] = ent.entity_id

        created_rels = 0
        for rel in typed_rels:
            e1_text = (rel.get("entity1") or "").lower()
            e2_text = (rel.get("entity2") or "").lower()
            relation_type = rel.get("relation_type", "RELATED_TO")

            e1_id = entity_name_to_id.get(e1_text)
            e2_id = entity_name_to_id.get(e2_text)

            if e1_id and e2_id and e1_id != e2_id:
                try:
                    store.create_entity_relationship(
                        entity1_id=e1_id,
                        entity2_id=e2_id,
                        relation_type=relation_type,
                        frequency=1,
                    )
                    created_rels += 1
                except Exception as _rel_exc:  # noqa: BLE001
                    logger.debug("KG typed relationship creation skipped: %s", _rel_exc)

        if created_rels:
            logger.info(
                "KG: created %d typed relationship(s) for doc_id=%s",
                created_rels,
                payload.document.get("doc_id"),
            )

    # ── Timeline nodes from temporal spans ──
    temporal_spans = payload.temporal_spans or []
    doc_id_for_tl = payload.document.get("doc_id", "")
    if temporal_spans and doc_id_for_tl:
        created_tl = 0
        for span in temporal_spans:
            start = span.get("start_date") or span.get("raw_text") or ""
            end = span.get("end_date") or span.get("raw_text") or start
            description = span.get("description") or span.get("raw_text") or ""
            if not start or not description:
                continue
            try:
                store.create_timeline_node(
                    document_id=doc_id_for_tl,
                    start_date=str(start),
                    end_date=str(end),
                    description=str(description),
                    entity_ids=None,
                )
                created_tl += 1
            except Exception as _tl_exc:  # noqa: BLE001
                logger.debug("KG timeline node creation skipped: %s", _tl_exc)

        if created_tl:
            logger.info(
                "KG: created %d timeline node(s) for doc_id=%s",
                created_tl,
                doc_id_for_tl,
            )

def _extract_evidence_span(text: str, name: str, window: int = 48) -> Optional[str]:
    if not text or not name:
        return None
    lowered = text.lower()
    needle = name.lower()
    idx = lowered.find(needle)
    if idx == -1:
        return None
    start = max(idx - window, 0)
    end = min(idx + len(needle) + window, len(text))
    return text[start:end].strip()

def _extract_structured_fields(meta: Dict[str, Any]) -> List[tuple[str, Any]]:
    candidates = []
    for key in ("fields", "structured_fields", "extracted_fields"):
        data = meta.get(key)
        if isinstance(data, dict):
            for field_key, value in data.items():
                if value is None:
                    continue
                candidates.append((str(field_key), value))
    return candidates

def _graph_version(document_id: str, chunk_metadata: Iterable[Dict[str, Any]]) -> str:
    hashes = []
    for meta in chunk_metadata:
        chunk_hash = meta.get("chunk_hash") or meta.get("chunk_id")
        if chunk_hash:
            hashes.append(str(chunk_hash))
    seed = "|".join(sorted(hashes)) or str(document_id)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()

def _coerce_language(languages: Any) -> Optional[str]:
    if isinstance(languages, list) and languages:
        return str(languages[0])
    if isinstance(languages, str):
        return languages
    return None

__all__ = [
    "GraphIngestPayload",
    "GraphIngestQueue",
    "build_graph_payload",
    "get_graph_ingest_queue",
    "ingest_graph_payload",
]
