from __future__ import annotations

from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.api.config import Config
from src.kg.neo4j_store import Neo4jStore

logger = get_logger(__name__)

@dataclass(frozen=True)
class EvidencePointer:
    file_name: str
    document_id: str
    section_id: str
    page: Optional[int]
    snippet: str
    snippet_sha: str

@dataclass
class KGEntity:
    entity_norm: str
    entity_type: str
    surface_forms: List[str]

class InMemoryKGStore:
    def __init__(self) -> None:
        self.entities: Dict[Tuple[str, str, str, str], KGEntity] = {}
        self.sections: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.documents: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.mentions: Dict[Tuple[str, str, str, str], List[EvidencePointer]] = {}
        self.co_occurs: Dict[Tuple[str, str, str, str, str, str], List[EvidencePointer]] = {}

    def upsert_document(
        self,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        file_name: str,
        doc_domain: Optional[str] = None,
    ) -> None:
        key = (str(subscription_id), str(profile_id))
        self.documents[key] = {"document_id": document_id, "file_name": file_name, "doc_domain": doc_domain}

    def upsert_section(self, subscription_id: str, profile_id: str, document_id: str, section_id: str) -> None:
        key = (str(subscription_id), str(profile_id), str(section_id))
        self.sections[key] = {"document_id": document_id, "section_id": section_id}

    def upsert_entity(self, subscription_id: str, profile_id: str, entity_norm: str, entity_type: str, surface_form: str) -> None:
        key = (str(subscription_id), str(profile_id), str(entity_norm), str(entity_type))
        entry = self.entities.get(key)
        if entry is None:
            entry = KGEntity(entity_norm=entity_norm, entity_type=entity_type, surface_forms=[surface_form])
            self.entities[key] = entry
            return
        if surface_form and surface_form not in entry.surface_forms:
            entry.surface_forms.append(surface_form)

    def add_mention(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norm: str,
        entity_type: str,
        section_id: str,
        evidence: EvidencePointer,
    ) -> None:
        key = (str(subscription_id), str(profile_id), str(entity_norm), str(entity_type))
        self.mentions.setdefault(key, [])
        if evidence not in self.mentions[key]:
            self.mentions[key].append(evidence)

    def add_cooccurrence(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norm_a: str,
        entity_type_a: str,
        entity_norm_b: str,
        entity_type_b: str,
        evidence: EvidencePointer,
    ) -> None:
        key = (
            str(subscription_id),
            str(profile_id),
            str(entity_norm_a),
            str(entity_type_a),
            str(entity_norm_b),
            str(entity_type_b),
        )
        self.co_occurs.setdefault(key, [])
        if evidence not in self.co_occurs[key]:
            self.co_occurs[key].append(evidence)

    def find_sections_for_entities(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norms: Iterable[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entity_norm in entity_norms:
            for key, evidences in self.mentions.items():
                sub, prof, norm, _etype = key
                if sub != str(subscription_id) or prof != str(profile_id):
                    continue
                if norm != str(entity_norm):
                    continue
                for ev in evidences:
                    results.append(
                        {
                            "document_id": ev.document_id,
                            "section_id": ev.section_id,
                            "file_name": ev.file_name,
                            "page": ev.page,
                        }
                    )
                    if len(results) >= limit:
                        return results
        return results

class KGStore:
    def __init__(self, neo4j_store: Optional[Neo4jStore] = None) -> None:
        self.enabled = bool(getattr(Config.KnowledgeGraph, "ENABLED", False))
        self.neo4j_store = None
        self.memory_store = InMemoryKGStore()
        if not self.enabled:
            return
        if neo4j_store is not None:
            self.neo4j_store = neo4j_store
        else:
            try:
                self.neo4j_store = Neo4jStore()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neo4j unavailable, using in-memory KG store: %s", exc)
                self.neo4j_store = None
        if self.neo4j_store:
            try:
                self._ensure_constraints()
            except Exception as exc:  # noqa: BLE001
                logger.warning("KG constraints setup failed: %s", exc)

    def _ensure_constraints(self) -> None:
        if not self.neo4j_store:
            return
        queries = [
            "CREATE CONSTRAINT kg_entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
            "CREATE CONSTRAINT kg_section_key IF NOT EXISTS FOR (s:Section) REQUIRE s.section_key IS UNIQUE",
            "CREATE CONSTRAINT kg_document_key IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_key IS UNIQUE",
        ]
        for query in queries:
            self.neo4j_store.run_query(query)

    def upsert_document(
        self,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        file_name: str,
        doc_domain: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        if self.neo4j_store:
            doc_key = _doc_key(subscription_id, profile_id, document_id)
            query = (
                "MERGE (d:Document {document_id: $document_id}) "
                "SET d.doc_key = $doc_key, d.file_name = $file_name, "
                "d.doc_domain = coalesce($doc_domain, d.doc_domain), "
                "d.subscription_id = $subscription_id, d.profile_id = $profile_id"
            )
            self.neo4j_store.run_query(
                query,
                {
                    "doc_key": doc_key,
                    "document_id": str(document_id),
                    "file_name": file_name,
                    "doc_domain": doc_domain,
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                },
            )
        else:
            self.memory_store.upsert_document(subscription_id, profile_id, document_id, file_name, doc_domain=doc_domain)

    def upsert_section(self, subscription_id: str, profile_id: str, document_id: str, section_id: str) -> None:
        if not self.enabled:
            return
        if self.neo4j_store:
            doc_key = _doc_key(subscription_id, profile_id, document_id)
            section_key = _section_key(subscription_id, profile_id, section_id)
            query = (
                "MERGE (s:Section {section_key: $section_key}) "
                "SET s.section_id = $section_id, s.document_id = $document_id, "
                "s.subscription_id = $subscription_id, s.profile_id = $profile_id "
                "WITH s "
                "MERGE (d:Document {document_id: $document_id}) "
                "SET d.doc_key = $doc_key, d.subscription_id = $subscription_id, d.profile_id = $profile_id "
                "MERGE (d)-[:HAS_SECTION]->(s)"
            )
            self.neo4j_store.run_query(
                query,
                {
                    "doc_key": doc_key,
                    "section_key": section_key,
                    "section_id": str(section_id),
                    "document_id": str(document_id),
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                },
            )
        else:
            self.memory_store.upsert_section(subscription_id, profile_id, document_id, section_id)

    def upsert_entity(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norm: str,
        entity_type: str,
        surface_form: str,
    ) -> None:
        if not self.enabled:
            return
        if self.neo4j_store:
            entity_key = _entity_key(subscription_id, profile_id, entity_norm, entity_type)
            query = (
                "MERGE (e:Entity {entity_key: $entity_key}) "
                "SET e.entity_norm = $entity_norm, e.type = $entity_type, "
                "e.subscription_id = $subscription_id, e.profile_id = $profile_id "
                "WITH e "
                "SET e.surface_forms = CASE "
                "WHEN $surface_form IS NULL OR $surface_form = '' THEN coalesce(e.surface_forms, []) "
                "WHEN e.surface_forms IS NULL THEN [$surface_form] "
                "WHEN NOT $surface_form IN e.surface_forms THEN e.surface_forms + $surface_form "
                "ELSE e.surface_forms END"
            )
            self.neo4j_store.run_query(
                query,
                {
                    "entity_key": entity_key,
                    "entity_norm": entity_norm,
                    "entity_type": entity_type,
                    "surface_form": surface_form,
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                },
            )
        else:
            self.memory_store.upsert_entity(subscription_id, profile_id, entity_norm, entity_type, surface_form)

    def add_mention(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norm: str,
        entity_type: str,
        section_id: str,
        document_id: str,
        evidence: EvidencePointer,
        doc_domain: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        if self.neo4j_store:
            entity_key = _entity_key(subscription_id, profile_id, entity_norm, entity_type)
            section_key = _section_key(subscription_id, profile_id, section_id)
            doc_key = _doc_key(subscription_id, profile_id, document_id)
            query = (
                "MERGE (e:Entity {entity_key: $entity_key}) "
                "SET e.entity_norm = $entity_norm, e.type = $entity_type, "
                "e.subscription_id = $subscription_id, e.profile_id = $profile_id "
                "MERGE (s:Section {section_key: $section_key}) "
                "SET s.section_id = $section_id, s.document_id = $document_id, "
                "s.subscription_id = $subscription_id, s.profile_id = $profile_id "
                "MERGE (d:Document {document_id: $document_id}) "
                "SET d.document_id = $document_id, d.file_name = $file_name, "
                "d.doc_domain = coalesce($doc_domain, d.doc_domain), "
                "d.subscription_id = $subscription_id, d.profile_id = $profile_id "
                "MERGE (s)-[:IN_DOCUMENT]->(d) "
                "MERGE (e)-[m:MENTIONED_IN {section_key: $section_key, entity_key: $entity_key}]->(s) "
                "SET m.file_name = $file_name, m.document_id = $document_id, m.section_id = $section_id, "
                "m.doc_domain = coalesce($doc_domain, m.doc_domain), "
                "m.page = $page, m.snippet = $snippet, m.snippet_sha = $snippet_sha"
            )
            self.neo4j_store.run_query(
                query,
                {
                    "entity_key": entity_key,
                    "entity_norm": entity_norm,
                    "entity_type": entity_type,
                    "section_key": section_key,
                    "section_id": str(section_id),
                    "doc_key": doc_key,
                    "document_id": str(document_id),
                    "file_name": evidence.file_name,
                    "page": evidence.page,
                    "snippet": evidence.snippet,
                    "snippet_sha": evidence.snippet_sha,
                    "doc_domain": doc_domain,
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                },
            )
        else:
            self.memory_store.add_mention(subscription_id, profile_id, entity_norm, entity_type, section_id, evidence)

    def add_cooccurrence(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norm_a: str,
        entity_type_a: str,
        entity_norm_b: str,
        entity_type_b: str,
        section_id: str,
        document_id: str,
        evidence: EvidencePointer,
        doc_domain: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        if self.neo4j_store:
            entity_key_a = _entity_key(subscription_id, profile_id, entity_norm_a, entity_type_a)
            entity_key_b = _entity_key(subscription_id, profile_id, entity_norm_b, entity_type_b)
            section_key = _section_key(subscription_id, profile_id, section_id)
            doc_key = _doc_key(subscription_id, profile_id, document_id)
            query = (
                "MERGE (a:Entity {entity_key: $entity_key_a}) "
                "SET a.entity_norm = $entity_norm_a, a.type = $entity_type_a, a.subscription_id = $subscription_id, a.profile_id = $profile_id "
                "MERGE (b:Entity {entity_key: $entity_key_b}) "
                "SET b.entity_norm = $entity_norm_b, b.type = $entity_type_b, b.subscription_id = $subscription_id, b.profile_id = $profile_id "
                "MERGE (s:Section {section_key: $section_key}) "
                "SET s.section_id = $section_id, s.document_id = $document_id, s.subscription_id = $subscription_id, s.profile_id = $profile_id "
                "MERGE (d:Document {document_id: $document_id}) "
                "SET d.document_id = $document_id, d.file_name = $file_name, d.doc_domain = coalesce($doc_domain, d.doc_domain), "
                "d.subscription_id = $subscription_id, d.profile_id = $profile_id "
                "MERGE (s)-[:IN_DOCUMENT]->(d) "
                "MERGE (a)-[c:CO_OCCURS_WITH {section_key: $section_key, entity_key_a: $entity_key_a, entity_key_b: $entity_key_b}]->(b) "
                "SET c.file_name = $file_name, c.document_id = $document_id, c.section_id = $section_id, "
                "c.doc_domain = coalesce($doc_domain, c.doc_domain), "
                "c.page = $page, c.snippet = $snippet, c.snippet_sha = $snippet_sha"
            )
            self.neo4j_store.run_query(
                query,
                {
                    "entity_key_a": entity_key_a,
                    "entity_norm_a": entity_norm_a,
                    "entity_type_a": entity_type_a,
                    "entity_key_b": entity_key_b,
                    "entity_norm_b": entity_norm_b,
                    "entity_type_b": entity_type_b,
                    "section_key": section_key,
                    "section_id": str(section_id),
                    "doc_key": doc_key,
                    "document_id": str(document_id),
                    "file_name": evidence.file_name,
                    "page": evidence.page,
                    "snippet": evidence.snippet,
                    "snippet_sha": evidence.snippet_sha,
                    "doc_domain": doc_domain,
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                },
            )
        else:
            self.memory_store.add_cooccurrence(
                subscription_id,
                profile_id,
                entity_norm_a,
                entity_type_a,
                entity_norm_b,
                entity_type_b,
                evidence,
            )

    def find_sections_for_entities(
        self,
        subscription_id: str,
        profile_id: str,
        entity_norms: Iterable[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        if not entity_norms:
            return []
        if self.neo4j_store:
            query = (
                "UNWIND $entity_norms AS norm "
                "MATCH (e:Entity {entity_norm: norm, subscription_id: $subscription_id, profile_id: $profile_id}) "
                "MATCH (e)-[m:MENTIONED_IN]->(s:Section) "
                "RETURN m.document_id AS document_id, m.section_id AS section_id, m.file_name AS file_name, m.page AS page "
                "LIMIT $limit"
            )
            rows = self.neo4j_store.run_query(
                query,
                {
                    "entity_norms": [str(n) for n in entity_norms],
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                    "limit": int(limit),
                },
            )
            results: List[Dict[str, Any]] = []
            for row in rows:
                results.append(
                    {
                        "document_id": row.get("document_id"),
                        "section_id": row.get("section_id"),
                        "file_name": row.get("file_name"),
                        "page": row.get("page"),
                    }
                )
            return results
        return self.memory_store.find_sections_for_entities(subscription_id, profile_id, entity_norms, limit=limit)

def _entity_key(subscription_id: str, profile_id: str, entity_norm: str, entity_type: str) -> str:
    return f"{subscription_id}:{profile_id}:{entity_type}:{entity_norm}"

def _section_key(subscription_id: str, profile_id: str, section_id: str) -> str:
    return f"{subscription_id}:{profile_id}:{section_id}"

def _doc_key(subscription_id: str, profile_id: str, document_id: str) -> str:
    return f"{subscription_id}:{profile_id}:{document_id}"

__all__ = ["KGStore", "EvidencePointer"]
