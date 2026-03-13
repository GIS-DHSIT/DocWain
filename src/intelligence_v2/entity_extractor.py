"""KG writer for entities, facts, and relationships extracted by the summarizer."""

from __future__ import annotations

from typing import Any, Dict

from src.intelligence_v2.summarizer import AnalysisResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _entity_id(
    subscription_id: str, profile_id: str, entity_type: str, value: str
) -> str:
    """Deterministic entity identifier scoped to subscription + profile."""
    normalized = value.strip().lower()
    return f"{subscription_id}::{profile_id}::{entity_type}::{normalized}"


class IntelligenceKGWriter:
    """Persist analysis entities, facts, and relationships into Neo4j."""

    def __init__(self, neo4j_store: Any) -> None:
        self.store = neo4j_store

    def write(
        self,
        analysis: AnalysisResult,
        document_id: str,
        subscription_id: str,
        profile_id: str,
    ) -> Dict[str, int]:
        """Write analysis results to the knowledge graph.

        Returns a dict with counts of entities, facts, and relationships written.
        """
        # 1. MERGE Document node
        self.store.run_query(
            "MERGE (d:Document {document_id: $document_id, "
            "subscription_id: $subscription_id, profile_id: $profile_id}) "
            "SET d.doc_type = $doc_type",
            {
                "document_id": document_id,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "doc_type": analysis.document_type,
            },
        )

        entity_count = 0
        fact_count = 0
        rel_count = 0

        # 2. Entities
        if analysis.entities:
            entities_param = []
            for ent in analysis.entities:
                etype = ent.get("type", "UNKNOWN")
                value = ent.get("value", "")
                role = ent.get("role", "")
                eid = _entity_id(subscription_id, profile_id, etype, value)
                entities_param.append(
                    {
                        "entity_id": eid,
                        "type": etype,
                        "value": value,
                        "role": role,
                    }
                )

            self.store.run_query(
                "UNWIND $entities AS ent "
                "MERGE (e:Entity {entity_id: ent.entity_id, "
                "subscription_id: $subscription_id, profile_id: $profile_id}) "
                "SET e.type = ent.type, e.value = ent.value, "
                "    e.subscription_id = $subscription_id, "
                "    e.profile_id = $profile_id "
                "WITH ent, e "
                "MATCH (d:Document {document_id: $document_id, "
                "subscription_id: $subscription_id, profile_id: $profile_id}) "
                "MERGE (d)-[r:HAS_ENTITY]->(e) "
                "SET r.role = ent.role",
                {
                    "entities": entities_param,
                    "document_id": document_id,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                },
            )
            entity_count = len(analysis.entities)

        # 3. Facts
        if analysis.facts:
            facts_param = []
            for idx, fact in enumerate(analysis.facts):
                facts_param.append(
                    {
                        "fact_id": f"{document_id}::fact::{idx}",
                        "claim": fact.get("claim", ""),
                        "evidence": fact.get("evidence", ""),
                    }
                )

            self.store.run_query(
                "UNWIND $facts AS f "
                "MERGE (fact:Fact {fact_id: f.fact_id, "
                "subscription_id: $subscription_id, profile_id: $profile_id}) "
                "SET fact.claim = f.claim, fact.evidence = f.evidence, "
                "    fact.subscription_id = $subscription_id, "
                "    fact.profile_id = $profile_id "
                "WITH f, fact "
                "MATCH (d:Document {document_id: $document_id, "
                "subscription_id: $subscription_id, profile_id: $profile_id}) "
                "MERGE (d)-[:HAS_FACT]->(fact)",
                {
                    "facts": facts_param,
                    "document_id": document_id,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                },
            )
            fact_count = len(analysis.facts)

        # 4. Relationships
        if analysis.relationships:
            for rel in analysis.relationships:
                source = rel.get("source", {})
                target = rel.get("target", {})
                src_id = _entity_id(
                    subscription_id,
                    profile_id,
                    source.get("type", "UNKNOWN"),
                    source.get("value", ""),
                )
                tgt_id = _entity_id(
                    subscription_id,
                    profile_id,
                    target.get("type", "UNKNOWN"),
                    target.get("value", ""),
                )

                self.store.run_query(
                    "MATCH (e1:Entity {entity_id: $src_id, "
                    "subscription_id: $subscription_id, profile_id: $profile_id}) "
                    "MATCH (e2:Entity {entity_id: $tgt_id, "
                    "subscription_id: $subscription_id, profile_id: $profile_id}) "
                    "MERGE (e1)-[r:RELATED_TO]->(e2) "
                    "SET r.relation_type = $relation_type, "
                    "    r.context = $context, "
                    "    r.document_id = $document_id",
                    {
                        "src_id": src_id,
                        "tgt_id": tgt_id,
                        "relation_type": rel.get("relation_type", "RELATED"),
                        "context": rel.get("context", ""),
                        "document_id": document_id,
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    },
                )
            rel_count = len(analysis.relationships)

        logger.info(
            "KG write complete for doc=%s: entities=%d, facts=%d, relationships=%d",
            document_id,
            entity_count,
            fact_count,
            rel_count,
        )

        return {
            "entities": entity_count,
            "facts": fact_count,
            "relationships": rel_count,
        }
