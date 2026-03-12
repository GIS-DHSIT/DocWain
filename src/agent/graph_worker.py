from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.kg.neo4j_store import Neo4jStore
from src.kg.retrieval import GraphAugmenter, GraphHints

logger = get_logger(__name__)

@dataclass
class GraphWorkerResult:
    graph_hints: GraphHints
    candidate_persons: List[Dict[str, Any]] = field(default_factory=list)
    candidate_doc_ids: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

class GraphWorker:
    def __init__(self, augmenter: GraphAugmenter, neo4j_store: Neo4jStore) -> None:
        self.augmenter = augmenter
        self.neo4j_store = neo4j_store
        self.max_persons = int(getattr(Config.KnowledgeGraph, "MAX_GRAPH_RESULTS", 200))
        self.max_returned = 50

    @classmethod
    def from_config(cls) -> Optional["GraphWorker"]:
        if not getattr(Config.KnowledgeGraph, "ENABLED", False):
            return None
        try:
            store = Neo4jStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphWorker unavailable: %s", exc)
            return None
        augmenter = GraphAugmenter(neo4j_store=store)
        return cls(augmenter, store)

    def run(self, query: str, subscription_id: str, profile_id: str) -> GraphWorkerResult:
        graph_hints = self.augmenter.augment(
            query,
            subscription_id,
            profile_id,
            agent_mode=True,
        )
        if not graph_hints.entities_in_query:
            return GraphWorkerResult(graph_hints=graph_hints)

        skill_names = [
            ent.name
            for ent in (graph_hints.entities_in_query + graph_hints.related_entities)
            if ent.type == "SKILL"
        ]
        org_names = [
            ent.name
            for ent in (graph_hints.entities_in_query + graph_hints.related_entities)
            if ent.type == "ORGANIZATION"
        ]
        candidate_persons: List[Dict[str, Any]] = []

        if skill_names:
            candidate_persons.extend(
                self._find_people_with_skills(skill_names, subscription_id, profile_id)
            )
        if org_names:
            candidate_persons.extend(
                self._find_people_by_org(org_names, subscription_id, profile_id)
            )

        candidate_persons = self._dedupe_people(candidate_persons)[: self.max_returned]
        candidate_doc_ids = sorted(
            {person.get("doc_id") for person in candidate_persons if person.get("doc_id")}
        )
        constraints = []
        if skill_names:
            constraints.append(f"skills: {', '.join(skill_names[:6])}")
        if org_names:
            constraints.append(f"organizations: {', '.join(org_names[:6])}")

        return GraphWorkerResult(
            graph_hints=graph_hints,
            candidate_persons=candidate_persons,
            candidate_doc_ids=candidate_doc_ids,
            constraints=constraints,
        )

    def _find_people_with_skills(
        self, skill_names: List[str], subscription_id: str, profile_id: str
    ) -> List[Dict[str, Any]]:
        query = (
            "UNWIND $skills AS skill_name "
            "MATCH (d:Document {profile_id: $profile_id, subscription_id: $subscription_id})"
            "-[:MENTIONS|HAS_FIELD]->(p:Person) "
            "MATCH (p)-[r:HAS_SKILL]->(s:Skill) "
            "WHERE s.name CONTAINS skill_name "
            "RETURN p.name AS person, d.doc_id AS doc_id, r.chunk_id AS chunk_id, r.confidence AS confidence "
            "LIMIT $limit"
        )
        rows = self.neo4j_store.run_query(
            query,
            {
                "skills": skill_names,
                "profile_id": str(profile_id),
                "subscription_id": str(subscription_id),
                "limit": self.max_persons,
            },
        )
        return [
            {
                "name": row.get("person"),
                "doc_id": row.get("doc_id"),
                "chunk_id": row.get("chunk_id"),
                "confidence": row.get("confidence") or 0.6,
            }
            for row in rows
            if row.get("person")
        ]

    def _find_people_by_org(
        self, org_names: List[str], subscription_id: str, profile_id: str
    ) -> List[Dict[str, Any]]:
        query = (
            "UNWIND $orgs AS org_name "
            "MATCH (d:Document {profile_id: $profile_id, subscription_id: $subscription_id})"
            "-[:MENTIONS|HAS_FIELD]->(p:Person) "
            "MATCH (p)-[r:WORKED_AT]->(o:Organization) "
            "WHERE o.name CONTAINS org_name "
            "RETURN p.name AS person, d.doc_id AS doc_id, r.chunk_id AS chunk_id, r.confidence AS confidence "
            "LIMIT $limit"
        )
        rows = self.neo4j_store.run_query(
            query,
            {
                "orgs": org_names,
                "profile_id": str(profile_id),
                "subscription_id": str(subscription_id),
                "limit": self.max_persons,
            },
        )
        return [
            {
                "name": row.get("person"),
                "doc_id": row.get("doc_id"),
                "chunk_id": row.get("chunk_id"),
                "confidence": row.get("confidence") or 0.6,
            }
            for row in rows
            if row.get("person")
        ]

    @staticmethod
    def _dedupe_people(people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for person in people:
            key = (person.get("name"), person.get("doc_id"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(person)
        return unique

__all__ = ["GraphWorker", "GraphWorkerResult"]
