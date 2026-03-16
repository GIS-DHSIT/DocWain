"""Qdrant collection setup -- ensures payload indexes exist for enriched payloads."""

import logging
from qdrant_client import QdrantClient
from src.api.config import Config

logger = logging.getLogger(__name__)


def ensure_payload_indexes(collection_name: str, client: QdrantClient = None):
    """Create payload indexes for efficient filtered search."""
    if client is None:
        client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)

    indexes = [
        ("profile_id", "keyword"),
        ("document_id", "keyword"),
        ("domain_tags", "keyword"),
        ("doc_category", "keyword"),
        ("chunk.type", "keyword"),
        ("quality_grade", "keyword"),
        ("entities", "keyword"),
    ]

    for field_name, field_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info("Created index %s on %s", field_name, collection_name)
        except Exception as e:
            # Index may already exist
            logger.debug("Index %s on %s: %s", field_name, collection_name, e)
