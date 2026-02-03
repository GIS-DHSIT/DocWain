import hashlib
import logging
import uuid
from typing import Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.api.config import Config
from src.api.pipeline_models import ChunkRecord

logger = logging.getLogger(__name__)

PAYLOAD_INDEX_FIELDS = [
    "subscription_id",
    "profile_id",
    "profile_name",
    "profileId",
    "document_id",
    "source_file",
    "page",
    "section_title",
    "document_type",
    "chunk_kind",
    "chunk_id",
    "chunk_type",
]


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

        def _existing_dim(info):
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            if hasattr(vectors, "size"):
                return vectors.size
            if isinstance(vectors, dict):
                if "size" in vectors:
                    return vectors["size"]
                if "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    return vectors["content_vector"].get("size")
            return None

        exists = False
        try:
            collections = self.client.get_collections().collections
            existing = {col.name for col in collections}
            exists = collection_name in existing
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not list collections; will check directly: %s", exc)

        if exists:
            info = self.client.get_collection(collection_name)
            dim = _existing_dim(info)
            if dim is not None and int(dim) != int(vector_size):
                raise ValueError(f"Collection '{collection_name}' dimension mismatch: {dim} vs {vector_size}")
        else:
            try:
                info = self.client.get_collection(collection_name)
                dim = _existing_dim(info)
                if dim is not None and int(dim) != int(vector_size):
                    raise ValueError(f"Collection '{collection_name}' dimension mismatch: {dim} vs {vector_size}")
                exists = True
            except Exception:
                logger.info("Creating Qdrant collection %s", collection_name)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"content_vector": VectorParams(size=vector_size, distance=Distance.COSINE)},
                    sparse_vectors_config={"keywords_vector": SparseVectorParams()},
                )
        self.collection_dims[collection_name] = int(vector_size)

        for field in PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name, field_name=field, field_schema="keyword"
                )
            except Exception as idx_exc:  # noqa: BLE001
                logger.debug("Payload index for %s exists or failed softly: %s", field, idx_exc)

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

    def upsert_records(
        self, collection_name: str, records: Iterable[ChunkRecord], batch_size: int = 100
    ) -> int:
        """Idempotent upsert using deterministic chunk ids."""
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

            # Qdrant accepts UUID or integer IDs; derive a deterministic UUID from chunk_id
            point_id = uuid.uuid5(uuid.NAMESPACE_URL, record.chunk_id)
            all_points.append(PointStruct(id=point_id, vector=vector_payload, payload=record.payload))

        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch, wait=True)
        return len(all_points)

    def delete_document(self, subscription_id: str, profile_id: str, document_id: str) -> dict:
        """Delete all embeddings for a document scoped to subscription collection and profile filter."""
        if not profile_id:
            raise ValueError("profile_id is required for deletion")
        collection_name = build_collection_name(subscription_id)
        delete_filter = Filter(
            must=[
                FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
                FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
                FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            ]
        )
        self.client.delete(collection_name=collection_name, points_selector=delete_filter, wait=True)
        return {"status": "success", "collection": collection_name, "document_id": str(document_id)}
