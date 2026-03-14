from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.api.config import Config
from src.kg.entity_extractor import EntityExtractor, ExtractedEntity, normalize_entity_name
from src.kg.neo4j_store import Neo4jStore

logger = get_logger(__name__)

@dataclass
class GraphEntityHint:
    name: str
    type: str
    node_id: str
    confidence: float

@dataclass
class GraphSnippet:
    text: str
    doc_id: str
    doc_name: Optional[str]
    chunk_id: str
    relation: str

@dataclass
class GraphHints:
    entities_in_query: List[GraphEntityHint] = field(default_factory=list)
    related_entities: List[GraphEntityHint] = field(default_factory=list)
    candidate_filters: Dict[str, Any] = field(default_factory=dict)
    graph_snippets: List[GraphSnippet] = field(default_factory=list)
    query_expansion_terms: List[str] = field(default_factory=list)
    evidence_chunk_ids: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)

class GraphAugmenter:
    def __init__(
        self,
        *,
        neo4j_store: Optional[Neo4jStore] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        max_expansion_entities: Optional[int] = None,
        max_graph_snippets: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.neo4j_store = neo4j_store
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.max_expansion_entities = max_expansion_entities or getattr(
            Config.KnowledgeGraph, "MAX_EXPANSION_ENTITIES", 8
        )
        self.max_graph_snippets = max_graph_snippets or getattr(
            Config.KnowledgeGraph, "MAX_GRAPH_SNIPPETS", 10
        )
        self.enabled = enabled if enabled is not None else bool(getattr(Config.KnowledgeGraph, "ENABLED", False))

    def augment(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        *,
        agent_mode: bool = False,
    ) -> GraphHints:
        if not self.enabled or not self.neo4j_store:
            return GraphHints()
        if not query:
            return GraphHints()

        extracted = self.entity_extractor.extract_with_metadata(query)
        if not extracted:
            return GraphHints()

        try:
            matched_entities, entity_doc_map = self._match_entities(
                extracted, subscription_id, profile_id
            )
            if not matched_entities:
                return GraphHints()
            entity_ids = [ent.node_id for ent in matched_entities]
            related_entities, snippets = self._expand_entities(
                entity_ids,
                subscription_id,
                profile_id,
                max_hops=2 if agent_mode else 1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("KG augment failed: %s", exc)
            return GraphHints()

        doc_ids = self._select_candidate_docs(entity_doc_map)
        snippets = [s for s in snippets if s.chunk_id]
        snippets = snippets[: self.max_graph_snippets]
        related_entities = self._dedupe_entities(related_entities)

        expansion_terms = self._build_expansion_terms(matched_entities, related_entities)
        evidence_chunk_ids = [snippet.chunk_id for snippet in snippets if snippet.chunk_id]

        return GraphHints(
            entities_in_query=matched_entities,
            related_entities=related_entities,
            candidate_filters={"document_ids": doc_ids},
            graph_snippets=snippets,
            query_expansion_terms=expansion_terms,
            evidence_chunk_ids=evidence_chunk_ids,
            doc_ids=doc_ids,
        )

    def _match_entities(
        self,
        extracted: Sequence[ExtractedEntity],
        subscription_id: str,
        profile_id: str,
    ) -> Tuple[List[GraphEntityHint], Dict[str, List[str]]]:
        names = [ent.normalized_name for ent in extracted]
        query = (
            "UNWIND $names AS name "
            "MATCH (d:Document {profile_id: $profile_id, subscription_id: $subscription_id})"
            "-[:MENTIONS|HAS_FIELD]->(e:Entity) "
            "WHERE e.subscription_id = $subscription_id AND e.profile_id = $profile_id "
            "  AND (e.normalized_name = name "
            "   OR e.normalized_name CONTAINS name "
            "   OR name CONTAINS e.normalized_name) "
            "RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type, "
            "       collect(DISTINCT d.doc_id) AS doc_ids"
        )
        rows = self.neo4j_store.run_query(
            query,
            {"names": names, "profile_id": str(profile_id), "subscription_id": str(subscription_id)},
        )
        entity_doc_map: Dict[str, List[str]] = {}
        hints: List[GraphEntityHint] = []
        for row in rows:
            entity_id = row.get("entity_id")
            if not entity_id:
                continue
            entity_doc_map[entity_id] = [str(d) for d in (row.get("doc_ids") or []) if d]
            hints.append(
                GraphEntityHint(
                    name=str(row.get("name") or ""),
                    type=str(row.get("type") or ""),
                    node_id=str(entity_id),
                    confidence=0.7,
                )
            )
        return self._dedupe_entities(hints), entity_doc_map

    def _expand_entities(
        self,
        entity_ids: List[str],
        subscription_id: str,
        profile_id: str,
        *,
        max_hops: int = 1,
    ) -> Tuple[List[GraphEntityHint], List[GraphSnippet]]:
        if not entity_ids:
            return [], []

        related: List[GraphEntityHint] = []
        snippets: List[GraphSnippet] = []

        one_hop_query = (
            "MATCH (d:Document {profile_id: $profile_id, subscription_id: $subscription_id})"
            "-[r:MENTIONS|HAS_FIELD]->(e:Entity) "
            "WHERE e.entity_id IN $entity_ids AND e.subscription_id = $subscription_id AND e.profile_id = $profile_id "
            "MATCH (d)-[r2:MENTIONS|HAS_FIELD]->(rel:Entity) "
            "RETURN rel.entity_id AS entity_id, rel.name AS name, rel.type AS type, "
            "       r2.chunk_id AS chunk_id, d.doc_id AS doc_id, d.doc_name AS doc_name, type(r2) AS relation "
            "LIMIT 200"
        )
        rows = self.neo4j_store.run_query(
            one_hop_query,
            {"entity_ids": entity_ids, "profile_id": str(profile_id), "subscription_id": str(subscription_id)},
        )
        for row in rows:
            related.append(
                GraphEntityHint(
                    name=str(row.get("name") or ""),
                    type=str(row.get("type") or ""),
                    node_id=str(row.get("entity_id") or ""),
                    confidence=0.6,
                )
            )
            snippet = self._snippet_from_row(row)
            if snippet:
                snippets.append(snippet)

        if max_hops > 1:
            two_hop_query = (
                "MATCH (d1:Document {profile_id: $profile_id, subscription_id: $subscription_id})"
                "-[:MENTIONS|HAS_FIELD]->(e:Entity) "
                "WHERE e.entity_id IN $entity_ids AND e.subscription_id = $subscription_id AND e.profile_id = $profile_id "
                "MATCH (d1)-[:MENTIONS|HAS_FIELD]->(mid:Entity) "
                "MATCH (d2:Document {profile_id: $profile_id, subscription_id: $subscription_id})"
                "-[:MENTIONS|HAS_FIELD]->(mid) "
                "MATCH (d2)-[:MENTIONS|HAS_FIELD]->(rel:Entity) "
                "RETURN rel.entity_id AS entity_id, rel.name AS name, rel.type AS type, "
                "       d2.doc_id AS doc_id, d2.doc_name AS doc_name "
                "LIMIT 200"
            )
            rows = self.neo4j_store.run_query(
                two_hop_query,
                {"entity_ids": entity_ids, "profile_id": str(profile_id), "subscription_id": str(subscription_id)},
            )
            for row in rows:
                related.append(
                    GraphEntityHint(
                        name=str(row.get("name") or ""),
                        type=str(row.get("type") or ""),
                        node_id=str(row.get("entity_id") or ""),
                        confidence=0.55,
                    )
                )

        return self._dedupe_entities(related), snippets

    def _select_candidate_docs(self, entity_doc_map: Dict[str, List[str]]) -> List[str]:
        if not entity_doc_map:
            return []
        doc_sets = [set(docs) for docs in entity_doc_map.values() if docs]
        if not doc_sets:
            return []
        intersected = set.intersection(*doc_sets) if len(doc_sets) > 1 else doc_sets[0]
        if intersected:
            return sorted(intersected)
        merged = set().union(*doc_sets)
        return sorted(merged)

    def _build_expansion_terms(
        self,
        query_entities: Sequence[GraphEntityHint],
        related_entities: Sequence[GraphEntityHint],
    ) -> List[str]:
        excluded_types = {"KEYWORD", "ID", "EMAIL", "PHONE", "URL"}
        query_names = {normalize_entity_name(ent.name) for ent in query_entities}
        terms: List[str] = []
        for ent in related_entities:
            if ent.type in excluded_types:
                continue
            normalized = normalize_entity_name(ent.name)
            if not normalized or normalized in query_names:
                continue
            terms.append(ent.name)
            if len(terms) >= self.max_expansion_entities:
                break
        return terms

    @staticmethod
    def _snippet_from_row(row: Dict[str, Any]) -> Optional[GraphSnippet]:
        chunk_id = row.get("chunk_id")
        doc_id = row.get("doc_id")
        name = row.get("name")
        rel_type = row.get("relation")
        if not chunk_id or not doc_id or not name:
            return None
        doc_name = row.get("doc_name")
        text = f"{doc_name or doc_id} {rel_type.lower()} {name}"
        return GraphSnippet(
            text=text,
            doc_id=str(doc_id),
            doc_name=str(doc_name) if doc_name else None,
            chunk_id=str(chunk_id),
            relation=str(rel_type),
        )

    @staticmethod
    def _dedupe_entities(entities: Sequence[GraphEntityHint]) -> List[GraphEntityHint]:
        seen = set()
        deduped = []
        for ent in entities:
            key = (ent.node_id, ent.name, ent.type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ent)
        return deduped

__all__ = ["GraphAugmenter", "GraphHints", "GraphSnippet", "GraphEntityHint"]
