from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    import neo4j
except ImportError:
    neo4j = None  # type: ignore[assignment]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    """A node in the knowledge graph."""
    node_id: str
    node_type: str = Field(
        ..., description="Entity, Document, or Chunk"
    )
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphEdge(BaseModel):
    """An edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: str = Field(
        ...,
        description="APPEARS_IN, BELONGS_TO, RELATES_TO, or MENTIONED_IN",
    )
    properties: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
# CypherGraphAdapter — works with Neo4j *or* Memgraph (both speak Cypher)
# ---------------------------------------------------------------------------

class CypherGraphAdapter:
    """Thin adapter that executes Cypher queries against a bolt driver."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    # -- low-level -----------------------------------------------------------

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return a list of record dicts."""
        with self._driver.session() as session:
            result = session.run(query, parameters=params or {})
            return [record.data() for record in result]

    # -- schema --------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create indexes used by the adapter (idempotent)."""
        for label in ("Entity", "Document", "Chunk"):
            self._run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.node_id)"
            )
            self._run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.subscription_id, n.profile_id)"
            )

    # -- node / edge upsert --------------------------------------------------

    def upsert_node(self, node: GraphNode) -> None:
        query = (
            f"MERGE (n:{node.node_type} {{node_id: $node_id}}) "
            "SET n += $props"
        )
        self._run(query, {"node_id": node.node_id, "props": node.properties})

    def upsert_edge(self, edge: GraphEdge) -> None:
        query = (
            "MATCH (a {node_id: $src}), (b {node_id: $tgt}) "
            f"MERGE (a)-[r:{edge.edge_type}]->(b) "
            "SET r += $props"
        )
        self._run(query, {
            "src": edge.source_id,
            "tgt": edge.target_id,
            "props": edge.properties,
        })

    # -- read helpers --------------------------------------------------------

    def get_entity_facts(
        self,
        entity_text: str,
        subscription_id: str,
        profile_id: str,
    ) -> List[Dict[str, Any]]:
        """Return facts (RELATES_TO targets) for an entity, scoped to profile."""
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[r:RELATES_TO]->(t) "
            "WHERE toLower(e.text) = toLower($entity_text) "
            "AND t.subscription_id = $sub AND t.profile_id = $prof "
            "RETURN e.node_id AS entity_id, r AS rel, "
            "properties(t) AS target_props, labels(t) AS target_labels"
        )
        return self._run(query, {
            "entity_text": entity_text,
            "sub": subscription_id,
            "prof": profile_id,
        })

    def get_related_entities(
        self,
        entity_text: str,
        subscription_id: str,
        profile_id: str,
    ) -> List[Dict[str, Any]]:
        """Return entities connected to the given entity within the profile."""
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[r]-(other:Entity) "
            "WHERE toLower(e.text) = toLower($entity_text) "
            "AND other.subscription_id = $sub AND other.profile_id = $prof "
            "RETURN other.node_id AS node_id, other.text AS text, "
            "type(r) AS rel_type, properties(other) AS props"
        )
        return self._run(query, {
            "entity_text": entity_text,
            "sub": subscription_id,
            "prof": profile_id,
        })

    def get_document_entities(
        self,
        document_id: str,
        subscription_id: str,
        profile_id: str,
    ) -> List[Dict[str, Any]]:
        """Return entities mentioned in a document."""
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[:MENTIONED_IN]->"
            "(d:Document {node_id: $doc_id, subscription_id: $sub, profile_id: $prof}) "
            "RETURN e.node_id AS node_id, e.text AS text, "
            "properties(e) AS props"
        )
        return self._run(query, {
            "doc_id": document_id,
            "sub": subscription_id,
            "prof": profile_id,
        })

    def get_all_entities(
        self,
        subscription_id: str,
        profile_id: str,
    ) -> List[Dict[str, Any]]:
        """Return every entity in the profile."""
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof}) "
            "RETURN e.node_id AS node_id, e.text AS text, "
            "properties(e) AS props"
        )
        return self._run(query, {"sub": subscription_id, "prof": profile_id})

    def count_entities(
        self,
        subscription_id: str,
        profile_id: str,
    ) -> int:
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof}) "
            "RETURN count(e) AS cnt"
        )
        rows = self._run(query, {"sub": subscription_id, "prof": profile_id})
        return rows[0]["cnt"] if rows else 0

    def count_facts(
        self,
        subscription_id: str,
        profile_id: str,
    ) -> int:
        query = (
            "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof})"
            "-[r:RELATES_TO]->(t) "
            "WHERE t.subscription_id = $sub AND t.profile_id = $prof "
            "RETURN count(r) AS cnt"
        )
        rows = self._run(query, {"sub": subscription_id, "prof": profile_id})
        return rows[0]["cnt"] if rows else 0

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _try_connect_neo4j() -> Any | None:
    """Attempt to connect to Neo4j. Returns a driver or None."""
    if neo4j is None:
        return None
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "neo4j")
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
        return driver
    except Exception:
        logger.debug("Neo4j not available at %s", uri)
        return None

def _try_connect_memgraph() -> Any | None:
    """Attempt to connect to Memgraph. Returns a driver or None."""
    if neo4j is None:
        return None
    uri = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7688")
    try:
        driver = neo4j.GraphDatabase.driver(uri)
        driver.verify_connectivity()
        logger.info("Connected to Memgraph at %s", uri)
        return driver
    except Exception:
        logger.debug("Memgraph not available at %s", uri)
        return None

# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_adapter_instance: Optional[CypherGraphAdapter] = None
_adapter_lock = threading.Lock()

def get_graph_adapter() -> Optional[CypherGraphAdapter]:
    """Return a singleton CypherGraphAdapter (Neo4j preferred, Memgraph fallback).

    Returns None if neither database is reachable (graceful degradation).
    Uses double-check locking for thread safety.
    """
    global _adapter_instance
    if _adapter_instance is not None:
        return _adapter_instance

    with _adapter_lock:
        # Double-check inside lock
        if _adapter_instance is not None:
            return _adapter_instance

        driver = _try_connect_neo4j()
        if driver is None:
            driver = _try_connect_memgraph()
        if driver is None:
            logger.warning("No graph database available; graph features disabled.")
            return None

        _adapter_instance = CypherGraphAdapter(driver)
        return _adapter_instance
