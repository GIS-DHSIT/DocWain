import argparse
import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, SparseVector

from src.api.config import Config
from src.api.vector_store import QdrantVectorStore, build_collection_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_vectors(point) -> Dict[str, Any]:
    vectors = getattr(point, "vector", {}) or {}
    if isinstance(vectors, dict) and "content_vector" in vectors:
        dense_vec = vectors.get("content_vector")
        sparse_vec = vectors.get("keywords_vector")
    else:
        dense_vec = vectors
        sparse_vec = None
    vector_payload = {"content_vector": dense_vec}
    if sparse_vec:
        if isinstance(sparse_vec, dict) and not isinstance(sparse_vec, SparseVector):
            sparse_vec = SparseVector(
                indices=[int(i) for i in sparse_vec.get("indices", [])],
                values=[float(v) for v in sparse_vec.get("values", [])],
            )
        vector_payload["keywords_vector"] = sparse_vec
    return vector_payload


def migrate_subscription(collection_name: str, client: QdrantClient):
    """Copy points from subscription-level collection into per-profile collections."""
    store = QdrantVectorStore(client)
    profiles = set()
    vector_dim = None

    scroll = client.scroll(collection_name=collection_name, with_payload=True, with_vectors=True, limit=64)
    points = scroll[0] if isinstance(scroll, tuple) else getattr(scroll, "points", [])
    if not points:
        logger.info("Collection %s is empty; nothing to migrate", collection_name)
        return

    for pt in points:
        payload = pt.payload or {}
        profile_id = payload.get("profile_id")
        if profile_id:
            profiles.add(str(profile_id))
        vectors = _extract_vectors(pt)
        dense_vec = vectors.get("content_vector")
        if dense_vec is not None:
            vector_dim = vector_dim or len(dense_vec)

    if not profiles:
        logger.warning("No profile_id payloads found in %s; skipping", collection_name)
        return

    logger.info("Found profiles in %s: %s", collection_name, profiles)

    target_collection = build_collection_name(collection_name)
    store.ensure_collection(target_collection, vector_dim or Config.Model.EMBEDDING_DIM)

    offset = None
    total = 0
    while True:
        scroll_result = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=128,
            offset=offset,
        )
        if isinstance(scroll_result, tuple):
            batch = scroll_result[0] or []
            offset = scroll_result[1]
        else:
            batch = getattr(scroll_result, "points", []) or []
            offset = getattr(scroll_result, "next_page_offset", None)

        if not batch:
            break

        points_to_upsert: List[PointStruct] = []
        for pt in batch:
            vectors = _extract_vectors(pt)
            points_to_upsert.append(
                PointStruct(
                    id=pt.id,
                    vector=vectors,
                    payload=pt.payload,
                )
            )
        client.upsert(collection_name=target_collection, points=points_to_upsert, wait=True)
        total += len(points_to_upsert)
        if not offset:
            break
    logger.info("Migrated %s points to %s", total, target_collection)


def main():
    parser = argparse.ArgumentParser(description="Migrate subscription collections to per-profile collections.")
    parser.add_argument(
        "--subscription",
        required=True,
        help="Subscription collection name to migrate (existing subscription-level collection).",
    )
    args = parser.parse_args()

    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    migrate_subscription(args.subscription, client)


if __name__ == "__main__":
    main()
