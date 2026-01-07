import hashlib
import logging
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
    "document_id",
    "source_file",
    "page",
    "section_title",
    "chunk_id",
    "chunk_type",
]


def build_collection_name(subscription_id: str, profile_id: str) -> str:
    """Per-subscription-per-profile collection naming for strict isolation."""
    if not subscription_id:
        raise ValueError("subscription_id is required to build collection name")
    if not profile_id:
        raise ValueError("profile_id is required to build collection name")

    safe_sub = str(subscription_id).strip().replace(" ", "_")
    safe_profile = str(profile_id).strip().replace(" ", "_")
    return f"{safe_sub}__{safe_profile}"


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

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create or validate a collection with dense + sparse vectors and payload indexes."""
        if not vector_size or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")

        collections = self.client.get_collections().collections
        existing = {col.name for col in collections}

        if collection_name not in existing:
            logger.info("Creating Qdrant collection %s", collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"content_vector": VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={"keywords_vector": SparseVectorParams()},
            )
        else:
            info = self.client.get_collection(collection_name)
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            existing_dim = None
            if hasattr(vectors, "size"):
                existing_dim = vectors.size
            elif isinstance(vectors, dict):
                if "size" in vectors:
                    existing_dim = vectors["size"]
                elif "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    existing_dim = vectors["content_vector"].get("size")

            if existing_dim is not None and int(existing_dim) != int(vector_size):
                raise ValueError(
                    f"Collection '{collection_name}' dimension mismatch: {existing_dim} vs {vector_size}"
                )

        for field in PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name, field_name=field, field_schema="keyword"
                )
            except Exception as idx_exc:  # noqa: BLE001
                logger.debug("Payload index for %s exists or failed softly: %s", field, idx_exc)

    def upsert_records(
        self, collection_name: str, records: Iterable[ChunkRecord], batch_size: int = 100
    ) -> int:
        """Idempotent upsert using deterministic chunk ids."""
        all_points: List[PointStruct] = []
        for record in records:
            sparse_vec = record.sparse_vector
            if sparse_vec and not isinstance(sparse_vec, SparseVector):
                sparse_vec = SparseVector(
                    indices=[int(i) for i in sparse_vec.get("indices", [])],
                    values=[float(v) for v in sparse_vec.get("values", [])],
                )

            vector_payload = {"content_vector": record.dense_vector}
            if sparse_vec is not None:
                vector_payload["keywords_vector"] = sparse_vec

            all_points.append(PointStruct(id=record.chunk_id, vector=vector_payload, payload=record.payload))

        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch, wait=True)
        return len(all_points)

    def delete_document(self, subscription_id: str, profile_id: str, document_id: str) -> dict:
        """Delete all embeddings for a document scoped to subscription+profile."""
        if not profile_id:
            raise ValueError("profile_id is required for deletion")
        collection_name = build_collection_name(subscription_id, profile_id)
        delete_filter = Filter(
            must=[
                FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
                FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
                FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            ]
        )
        self.client.delete(collection_name=collection_name, points_selector=delete_filter, wait=True)
        return {"status": "success", "collection": collection_name, "document_id": str(document_id)}
