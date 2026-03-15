"""Builds enriched Qdrant payloads from extraction + screening + KG data."""

import logging

logger = logging.getLogger(__name__)


def build_enriched_payload(
    chunk: dict,
    chunk_index: int,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    extraction_data: dict,
    screening_summary: dict,
    kg_node_ids: list = None,
    quality_grade: str = "C"
) -> dict:
    """Build enriched Qdrant point payload for a single chunk.

    Args:
        chunk: Chunk dict with text, section info, provenance
        chunk_index: Position in document
        document_id: Document identifier
        subscription_id: Org identifier
        profile_id: Department/domain identifier
        extraction_data: Full extraction from Azure Blob
        screening_summary: Screening summary from MongoDB
        kg_node_ids: KG node IDs linked to entities in this chunk
        quality_grade: Chunk quality grade (A-F)

    Returns:
        Qdrant payload dict
    """
    chunk_text = chunk.get("text", "")
    chunk_entities = []
    chunk_entity_types = []
    chunk_importance = 0.0

    entity_scores = screening_summary.get("entity_scores", {})
    for entity in extraction_data.get("entities", []):
        entity_text = entity.get("text", "") if isinstance(entity, dict) else ""
        if entity_text and entity_text.lower() in chunk_text.lower():
            chunk_entities.append(entity_text)
            entity_type = entity.get("type", "UNKNOWN") if isinstance(entity, dict) else "UNKNOWN"
            chunk_entity_types.append(entity_type)
            score = entity_scores.get(entity_text, 0.0)
            chunk_importance = max(chunk_importance, score)

    return {
        # Identity
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": document_id,

        # Chunk metadata
        "chunk": {
            "id": f"{document_id}_chunk_{chunk_index}",
            "index": chunk_index,
            "type": chunk.get("type", "text"),
            "hash": chunk.get("hash", ""),
            "token_count": chunk.get("token_count", 0),
        },

        # Section context
        "section": chunk.get("section", {
            "id": "", "title": "", "path": [], "level": 0
        }),

        # Provenance
        "provenance": chunk.get("provenance", {
            "page_start": 0, "page_end": 0
        }),

        # Enrichment from screening
        "entities": chunk_entities,
        "entity_types": list(set(chunk_entity_types)),
        "domain_tags": screening_summary.get("domain_tags", []),
        "doc_category": screening_summary.get("doc_category", "unknown"),
        "importance_score": chunk_importance,

        # KG linkage
        "kg_node_ids": kg_node_ids or [],

        # Quality
        "quality_grade": quality_grade,

        # Source text
        "text": chunk_text
    }
