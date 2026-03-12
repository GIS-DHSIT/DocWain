"""Populate the knowledge graph from extraction results."""
from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
from collections import Counter
from typing import List

from .graph_adapter import CypherGraphAdapter, GraphEdge, GraphNode
from .models import ExtractionResult, StructuredDocument

logger = get_logger(__name__)

def populate_graph(
    *,
    adapter: CypherGraphAdapter,
    extraction: ExtractionResult,
    structured_doc: StructuredDocument,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    fingerprint_tags: List[str],
) -> None:
    """Write extraction results into the graph database.

    Creates Document, Entity, and Chunk nodes plus the edges that connect
    them (APPEARS_IN, BELONGS_TO, RELATES_TO, MENTIONED_IN).
    """

    # -- Document node -------------------------------------------------------
    adapter.upsert_node(GraphNode(
        node_id=document_id,
        node_type="Document",
        properties={
            "filename": document_id,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "fingerprint_tags": fingerprint_tags,
        },
    ))

    # -- Entity nodes --------------------------------------------------------
    entity_ids: set[str] = set()
    for ent in extraction.entities:
        entity_ids.add(ent.entity_id)
        adapter.upsert_node(GraphNode(
            node_id=ent.entity_id,
            node_type="Entity",
            properties={
                "text": ent.text,
                "label": ent.label,
                "normalized": ent.normalized,
                "confidence": ent.confidence,
                "source": ent.source,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
            },
        ))

    # -- Chunk nodes ---------------------------------------------------------
    unit_ids: set[str] = set()
    for unit in structured_doc.units:
        unit_ids.add(unit.unit_id)
        text_hash = hashlib.sha1(unit.text.encode("utf-8")).hexdigest()[:12]
        adapter.upsert_node(GraphNode(
            node_id=unit.unit_id,
            node_type="Chunk",
            properties={
                "unit_type": unit.unit_type.value if hasattr(unit.unit_type, "value") else str(unit.unit_type),
                "text_hash": text_hash,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
            },
        ))

    # -- APPEARS_IN edges (Entity -> Chunk) ----------------------------------
    for ent in extraction.entities:
        if ent.unit_id in unit_ids:
            adapter.upsert_edge(GraphEdge(
                source_id=ent.entity_id,
                target_id=ent.unit_id,
                edge_type="APPEARS_IN",
                properties={
                    "char_start": ent.char_start,
                    "char_end": ent.char_end,
                    "confidence": ent.confidence,
                },
            ))

    # -- BELONGS_TO edges (Chunk -> Document) --------------------------------
    for unit in structured_doc.units:
        adapter.upsert_edge(GraphEdge(
            source_id=unit.unit_id,
            target_id=document_id,
            edge_type="BELONGS_TO",
            properties={},
        ))

    # -- RELATES_TO edges (Entity -> Entity from FactTriples) ----------------
    for fact in extraction.facts:
        if fact.subject_id in entity_ids and fact.object_id and fact.object_id in entity_ids:
            adapter.upsert_edge(GraphEdge(
                source_id=fact.subject_id,
                target_id=fact.object_id,
                edge_type="RELATES_TO",
                properties={
                    "predicate": fact.predicate,
                    "fact_id": fact.fact_id,
                    "confidence": fact.confidence,
                    "raw_text": fact.raw_text,
                },
            ))

    # -- MENTIONED_IN edges (Entity -> Document, with frequency) -------------
    freq: Counter[str] = Counter()
    for ent in extraction.entities:
        freq[ent.entity_id] += 1

    for eid, count in freq.items():
        adapter.upsert_edge(GraphEdge(
            source_id=eid,
            target_id=document_id,
            edge_type="MENTIONED_IN",
            properties={"frequency": count},
        ))

    logger.info(
        "Graph populated: doc=%s, %d entities, %d chunks, %d facts",
        document_id, len(extraction.entities), len(structured_doc.units),
        len(extraction.facts),
    )
