import hashlib
from src.utils.logging_utils import get_logger
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    MatchAny,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.api.config import Config
from src.api.pipeline_models import ChunkRecord
from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS, ensure_payload_indexes, list_payload_indexes

logger = get_logger(__name__)

# Per-collection creation locks to prevent parallel creation attempts
_collection_locks: Dict[str, threading.Lock] = {}
_collection_locks_guard = threading.Lock()

def _get_collection_lock(collection_name: str) -> threading.Lock:
    with _collection_locks_guard:
        if collection_name not in _collection_locks:
            _collection_locks[collection_name] = threading.Lock()
        return _collection_locks[collection_name]

class QdrantCollectionSchemaMismatch(ValueError):
    def __init__(self, details: Dict[str, Any]):
        message = (
            "Qdrant collection schema mismatch: "
            f"collection={details.get('collection')} "
            f"expected_size={details.get('expected_size')} actual_size={details.get('actual_size')} "
            f"expected_distance={details.get('expected_distance')} actual_distance={details.get('actual_distance')}"
        )
        super().__init__(message)
        self.code = "qdrant_schema_mismatch"
        self.details = details

PAYLOAD_INDEX_FIELDS = list(REQUIRED_PAYLOAD_INDEX_FIELDS)

def build_collection_name(subscription_id: str, profile_id: Optional[str] = None) -> str:
    """Collection name scoped only by subscription; profile isolation is enforced via payload filters."""
    if not subscription_id:
        raise ValueError("subscription_id is required to build collection name")

    safe_sub = str(subscription_id).strip().replace(" ", "_")
    return f"{safe_sub}"

def compute_chunk_id(
    subscription_id: str,
    profile_id: str,
    document_id: str,
    source_file: str,
    chunk_index: int,
    chunk_text: str,
    prefix: str = "chunk",
    text_prefix_len: int = 128,
) -> str:
    """
    Deterministic chunk id to prevent duplicates across retraining.
    sha1(subscription|profile|document|file|index|text_prefix)
    """
    base = "|".join(
        [
            str(subscription_id),
            str(profile_id),
            str(document_id),
            str(source_file),
            str(chunk_index),
            (chunk_text or "")[:text_prefix_len],
        ]
    )
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"

def build_qdrant_filter(
    subscription_id: str,
    profile_id: str,
    document_id: Optional[object] = None,
    doc_domain: Optional[object] = None,
    section_kind: Optional[object] = None,
    chunk_kind: Optional[object] = None,
) -> Filter:
    """
    Build a strict Qdrant filter that always scopes by subscription_id + profile_id.

    Optional filters:
    - document_id: str or list[str]
    - domain: str or list[str] (doc_domain)
    - section_kind: str or list[str] (section_kind / section.kind)
    """
    if not profile_id or not str(profile_id).strip():
        raise ValueError("profile_id is required for retrieval to enforce isolation")
    if not subscription_id or not str(subscription_id).strip():
        raise ValueError("subscription_id is required for retrieval to enforce isolation")

    must: List[object] = [
        FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
        FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
    ]

    logger = get_logger(__name__)
    logger.debug(
        "build_qdrant_filter: subscription_id=%s profile_id=%s document_id=%s",
        subscription_id, profile_id, document_id,
    )

    def _coerce_values(value: object) -> List[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if v is not None and str(v).strip()]
        return [str(value)] if value is not None and str(value).strip() else []

    doc_values = _coerce_values(document_id)
    if doc_values:
        if len(doc_values) == 1:
            must.append(FieldCondition(key="document_id", match=MatchValue(value=doc_values[0])))
        else:
            must.append(FieldCondition(key="document_id", match=MatchAny(any=doc_values)))

    domain_values = _coerce_values(doc_domain)
    if domain_values:
        if len(domain_values) == 1:
            must.append(FieldCondition(key="doc_domain", match=MatchValue(value=domain_values[0])))
        else:
            must.append(FieldCondition(key="doc_domain", match=MatchAny(any=domain_values)))

    section_values = _coerce_values(section_kind)
    if section_values:
        if len(section_values) == 1:
            must.append(FieldCondition(key="section_kind", match=MatchValue(value=section_values[0])))
        else:
            must.append(FieldCondition(key="section_kind", match=MatchAny(any=section_values)))

    chunk_values = _coerce_values(chunk_kind)
    if chunk_values:
        if len(chunk_values) == 1:
            must.append(FieldCondition(key="chunk_kind", match=MatchValue(value=chunk_values[0])))
        else:
            must.append(FieldCondition(key="chunk_kind", match=MatchAny(any=chunk_values)))

    logger.debug(
        "Qdrant filter built",
        extra={
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "document_id": doc_values,
            "doc_domain": domain_values,
            "section_kind": section_values,
            "chunk_kind": chunk_values,
        },
    )
    return Filter(must=must)

class QdrantVectorStore:
    """Encapsulates Qdrant collection lifecycle and scoped operations."""

    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or QdrantClient(
            url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120
        )
        self.collection_dims: dict[str, int] = {}

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create or validate a collection with dense + sparse vectors and payload indexes."""
        if not vector_size or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")

        def _normalize_distance(value: Any) -> Optional[str]:
            if value is None:
                return None
            if hasattr(value, "name"):
                return str(value.name).lower()
            text = str(value)
            return text.split(".")[-1].lower() if text else None

        def _existing_vector_config(info):
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            if hasattr(vectors, "size"):
                return vectors.size, getattr(vectors, "distance", None)
            if isinstance(vectors, dict):
                if "size" in vectors:
                    return vectors["size"], vectors.get("distance")
                if "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    vec = vectors["content_vector"]
                    return vec.get("size"), vec.get("distance")
            return None, None

        def _validate_schema(info):
            dim, dist = _existing_vector_config(info)
            if dim is not None and int(dim) != int(vector_size):
                details = {
                    "collection": collection_name,
                    "expected_size": int(vector_size),
                    "actual_size": int(dim),
                    "expected_distance": _normalize_distance(Distance.COSINE),
                    "actual_distance": _normalize_distance(dist),
                }
                raise QdrantCollectionSchemaMismatch(details)
            expected_dist = _normalize_distance(Distance.COSINE)
            actual_dist = _normalize_distance(dist)
            if actual_dist and expected_dist and actual_dist != expected_dist:
                details = {
                    "collection": collection_name,
                    "expected_size": int(vector_size),
                    "actual_size": int(dim) if dim is not None else None,
                    "expected_distance": expected_dist,
                    "actual_distance": actual_dist,
                }
                raise QdrantCollectionSchemaMismatch(details)

        exists = False
        try:
            collections = self.client.get_collections().collections
            existing = {col.name for col in collections}
            exists = collection_name in existing
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not list collections; will check directly: %s", exc)

        if exists:
            info = self.client.get_collection(collection_name)
            _validate_schema(info)
        else:
            # Use per-collection lock to prevent parallel creation races
            lock = _get_collection_lock(collection_name)
            with lock:
                # Re-check after acquiring lock (another thread may have created it)
                try:
                    info = self.client.get_collection(collection_name)
                    _validate_schema(info)
                    exists = True
                except Exception as exc:  # noqa: BLE001
                    logger.info("Creating Qdrant collection %s", collection_name)
                    try:
                        self.client.create_collection(
                            collection_name=collection_name,
                            vectors_config={"content_vector": VectorParams(size=vector_size, distance=Distance.COSINE)},
                            sparse_vectors_config={"keywords_vector": SparseVectorParams()},
                        )
                    except UnexpectedResponse as create_exc:
                        if create_exc.status_code == 409:
                            logger.info("Qdrant collection %s already exists; validating schema", collection_name)
                            info = self.client.get_collection(collection_name)
                            _validate_schema(info)
                            exists = True
                        else:
                            raise
                    except Exception as create_exc:  # noqa: BLE001
                        raise create_exc from exc
        self.collection_dims[collection_name] = int(vector_size)

        self.ensure_payload_indexes(collection_name, PAYLOAD_INDEX_FIELDS, create_missing=True)

    def get_payload_indexes(self, collection_name: str) -> List[str]:
        try:
            return list_payload_indexes(self.client, collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch collection info for %s: %s", collection_name, exc)
            return []

    def ensure_payload_indexes(
        self,
        collection_name: str,
        fields: Iterable[str],
        *,
        create_missing: bool = True,
    ) -> Dict[str, Any]:
        return ensure_payload_indexes(
            client=self.client,
            collection_name=collection_name,
            required_fields=list(fields),
            create_missing=create_missing,
        )

    def _get_collection_dim(self, collection_name: str) -> Optional[int]:
        if collection_name in self.collection_dims:
            return self.collection_dims[collection_name]
        try:
            info = self.client.get_collection(collection_name)
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            dim = None
            if hasattr(vectors, "size"):
                dim = vectors.size
            elif isinstance(vectors, dict):
                if "size" in vectors:
                    dim = vectors["size"]
                elif "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    dim = vectors["content_vector"].get("size")
            if dim:
                self.collection_dims[collection_name] = int(dim)
            return int(dim) if dim else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch collection dim for %s: %s", collection_name, exc)
            return None

    def _compute_point_id(self, payload: dict, fallback_chunk_id: str) -> uuid.UUID:
        subscription_id = payload.get("subscription_id")
        profile_id = payload.get("profile_id")
        document_id = payload.get("document_id")
        section_id = (payload.get("section") or {}).get("id") or payload.get("section_id")
        chunk_index = (payload.get("chunk") or {}).get("index") or payload.get("chunk_index")
        chunk_hash = (payload.get("chunk") or {}).get("hash") or payload.get("chunk_hash")
        if all([subscription_id, profile_id, document_id, section_id, chunk_index is not None, chunk_hash]):
            base = "|".join(
                [
                    str(subscription_id),
                    str(profile_id),
                    str(document_id),
                    str(section_id),
                    str(chunk_index),
                    str(chunk_hash),
                ]
            )
            return uuid.uuid5(uuid.NAMESPACE_URL, base)
        return uuid.uuid5(uuid.NAMESPACE_URL, fallback_chunk_id)

    def upsert_records(
        self, collection_name: str, records: Iterable[ChunkRecord], batch_size: int = 100
    ) -> int:
        """Idempotent upsert using deterministic point ids."""
        expected_dim = self._get_collection_dim(collection_name)
        all_points: List[PointStruct] = []
        for record in records:
            if expected_dim and len(record.dense_vector) != expected_dim:
                raise ValueError(
                    f"Vector dim mismatch for collection {collection_name}: "
                    f"{len(record.dense_vector)} != {expected_dim}"
                )
            sparse_vec = record.sparse_vector
            if sparse_vec and not isinstance(sparse_vec, SparseVector):
                sparse_vec = SparseVector(
                    indices=[int(i) for i in sparse_vec.get("indices", [])],
                    values=[float(v) for v in sparse_vec.get("values", [])],
                )

            vector_payload = {"content_vector": record.dense_vector}
            if sparse_vec is not None:
                vector_payload["keywords_vector"] = sparse_vec

            # Qdrant accepts UUID or integer IDs; derive deterministic UUID from payload fields
            point_id = self._compute_point_id(record.payload or {}, record.chunk_id)
            all_points.append(PointStruct(id=point_id, vector=vector_payload, payload=record.payload))

        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            result = self.client.upsert(collection_name=collection_name, points=batch, wait=True)
            status = getattr(result, "status", None)
            if status and status not in {"completed", "acknowledged"}:
                raise ValueError(f"Qdrant upsert failed with status={status}")
        return len(all_points)

    def delete_document(self, subscription_id: str, profile_id: str, document_id: str) -> dict:
        """Delete all embeddings for a document scoped to subscription collection and profile filter."""
        collection_name = build_collection_name(subscription_id)

        # Guard: check if collection exists before attempting delete to avoid 404 errors
        try:
            collections = self.client.get_collections().collections
            if collection_name not in {col.name for col in collections}:
                logger.debug(
                    "[DELETE] Collection %s does not exist — nothing to delete",
                    collection_name,
                )
                return {
                    "status": "ok",
                    "message": "collection does not exist",
                    "document_id": str(document_id),
                }
        except Exception:
            pass  # If we can't check, proceed with delete and let it fail naturally

        delete_filter = build_qdrant_filter(
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            document_id=str(document_id),
        )
        self.client.delete(collection_name=collection_name, points_selector=delete_filter, wait=True)
        return {"status": "success", "collection": collection_name, "document_id": str(document_id)}
