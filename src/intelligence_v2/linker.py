"""Document linker — discovers and creates RELATED_DOC edges between documents
that share KG entities within the same subscription + profile scope."""

from typing import Dict, List

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cypher: find documents sharing entities with a given document
# ---------------------------------------------------------------------------

_FIND_RELATED_CYPHER = (
    "MATCH (d:Document {document_id: $document_id, "
    "       subscription_id: $subscription_id, profile_id: $profile_id})"
    "-[:HAS_ENTITY]->(e:Entity)"
    "<-[:HAS_ENTITY]-(other:Document {subscription_id: $subscription_id, "
    "       profile_id: $profile_id}) "
    "WHERE other.document_id <> $document_id "
    "WITH other.document_id AS related_doc_id, "
    "     count(DISTINCT e) AS shared_entities, "
    "     collect(DISTINCT e.value)[..5] AS entity_names "
    "WHERE shared_entities >= $min_shared "
    "RETURN related_doc_id, shared_entities, entity_names "
    "ORDER BY shared_entities DESC "
    "LIMIT $limit"
)

# ---------------------------------------------------------------------------
# Cypher: batch-create RELATED_DOC edges via UNWIND + MERGE
# ---------------------------------------------------------------------------

_MERGE_RELATED_CYPHER = (
    "UNWIND $links AS link "
    "MATCH (d1:Document {document_id: link.source_id, "
    "       subscription_id: $subscription_id, profile_id: $profile_id}) "
    "MATCH (d2:Document {document_id: link.target_id, "
    "       subscription_id: $subscription_id, profile_id: $profile_id}) "
    "MERGE (d1)-[r:RELATED_DOC]->(d2) "
    "SET r.shared_entities = link.shared_entities, r.updated_at = timestamp()"
)


class DocumentLinker:
    """Discovers and persists inter-document relationships based on shared KG
    entities.  All operations are scoped to a single subscription + profile to
    enforce strict tenant isolation."""

    def __init__(self, neo4j_store) -> None:
        self._store = neo4j_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_related(
        self,
        document_id: str,
        subscription_id: str,
        profile_id: str,
        min_shared: int = 1,
        limit: int = 10,
    ) -> List[Dict]:
        """Return documents sharing at least *min_shared* entities with
        *document_id*, sorted by shared entity count descending."""

        params = {
            "document_id": document_id,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "min_shared": min_shared,
            "limit": limit,
        }

        rows = self._store.run_query(_FIND_RELATED_CYPHER, params)

        results: List[Dict] = []
        for row in rows:
            results.append(
                {
                    "document_id": row["related_doc_id"],
                    "shared_entities": row["shared_entities"],
                    "entity_names": row.get("entity_names", []),
                }
            )

        logger.info(
            "find_related doc=%s sub=%s prof=%s => %d related docs",
            document_id,
            subscription_id,
            profile_id,
            len(results),
        )
        return results

    def link(
        self,
        document_id: str,
        subscription_id: str,
        profile_id: str,
    ) -> int:
        """Discover related documents and persist RELATED_DOC edges.

        Returns the number of links created/updated.
        """

        related = self.find_related(document_id, subscription_id, profile_id)
        if not related:
            return 0

        links = [
            {
                "source_id": document_id,
                "target_id": r["document_id"],
                "shared_entities": r["shared_entities"],
            }
            for r in related
        ]

        self._store.run_query(
            _MERGE_RELATED_CYPHER,
            {
                "links": links,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
            },
        )

        logger.info(
            "link doc=%s => %d RELATED_DOC edges created/updated",
            document_id,
            len(links),
        )
        return len(links)
