"""Three-layer unified retriever — Qdrant + Neo4j + MongoDB."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedRetriever:
    """Orchestrates retrieval across three data layers in parallel.

    Layer 1: Qdrant — dense + sparse hybrid search with metadata pre-filtering
    Layer 2: Neo4j — KG entity matching and relationship traversal (1-2 hops)
    Layer 3: MongoDB — document-level metadata and screening summaries
    """

    def __init__(self, qdrant_client=None, neo4j_driver=None, mongo_db=None):
        self._qdrant = qdrant_client
        self._neo4j = neo4j_driver
        self._mongo_db = mongo_db

    def retrieve(self, query: str, profile_id: str, subscription_id: str,
                 query_understanding: dict = None, top_k: int = 20) -> dict:
        """Execute three-layer retrieval in parallel.

        Args:
            query: User's search query
            profile_id: Profile scope
            subscription_id: Subscription scope (collection name)
            query_understanding: Dict with intent, entities, suggested_format, sub_queries
            top_k: Number of results to return

        Returns:
            dict with: chunks, kg_context, doc_metadata, merged_context
        """
        query_understanding = query_understanding or {}
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._qdrant_search, query, profile_id, subscription_id,
                    query_understanding, top_k
                ): "qdrant",
                executor.submit(
                    self._kg_search, query, profile_id, subscription_id,
                    query_understanding
                ): "neo4j",
                executor.submit(
                    self._metadata_search, profile_id, subscription_id,
                    query_understanding
                ): "mongodb",
            }

            for future in as_completed(futures):
                layer = futures[future]
                try:
                    results[layer] = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"{layer} retrieval failed: {e}")
                    results[layer] = {}

        # Merge and assemble context
        merged = self._assemble_context(results, query, top_k)
        return merged

    def _qdrant_search(self, query: str, profile_id: str,
                       subscription_id: str, query_understanding: dict,
                       top_k: int) -> dict:
        """Layer 1: Qdrant dense + sparse hybrid search with metadata filtering."""
        try:
            qdrant_client = self._get_qdrant_client()
            collection_name = subscription_id

            # Build payload filter
            must_filters = [
                {"key": "profile_id", "match": {"value": profile_id}}
            ]

            # Add metadata pre-filters from query understanding
            if query_understanding.get("domain_tags"):
                must_filters.append({
                    "key": "domain_tags",
                    "match": {"any": query_understanding["domain_tags"]}
                })
            if query_understanding.get("doc_category"):
                must_filters.append({
                    "key": "doc_category",
                    "match": {"value": query_understanding["doc_category"]}
                })

            # Generate query embedding
            query_vector = self._embed_query(query)

            # Dense search
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

            filter_conditions = []
            filter_conditions.append(
                FieldCondition(key="profile_id", match=MatchValue(value=profile_id))
            )

            search_filter = Filter(must=filter_conditions)

            dense_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=("dense", query_vector),
                query_filter=search_filter,
                limit=top_k * 2,
                with_payload=True
            )

            # TODO: Add sparse search and RRF merge
            # For now, return dense results only

            chunks = []
            for hit in dense_results:
                chunks.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "payload": hit.payload
                })

            return {"chunks": chunks, "total_hits": len(chunks)}

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return {"chunks": [], "total_hits": 0}

    def _kg_search(self, query: str, profile_id: str,
                   subscription_id: str, query_understanding: dict) -> dict:
        """Layer 2: Neo4j KG entity matching and relationship traversal."""
        try:
            driver = self._get_neo4j_driver()
            if not driver:
                return {"entities": [], "relationships": [], "expanded_context": []}

            query_entities = query_understanding.get("entities", [])
            if not query_entities:
                # Try basic entity extraction from query
                # TODO: Use NER or LLM for entity extraction
                return {"entities": [], "relationships": [], "expanded_context": []}

            with driver.session() as session:
                # Match query entities to graph nodes (1-2 hops)
                cypher = """
                MATCH (e:Entity)
                WHERE e.subscription_id = $sub_id
                  AND e.profile_id = $prof_id
                  AND toLower(e.name) IN $entity_names
                OPTIONAL MATCH (e)-[r]->(related:Entity)
                WHERE related.subscription_id = $sub_id
                  AND related.profile_id = $prof_id
                RETURN e.name as entity_name, e.type as entity_type,
                       type(r) as rel_type, r.predicate as predicate,
                       related.name as related_name, related.type as related_type
                LIMIT 50
                """
                result = session.run(cypher, {
                    "sub_id": subscription_id,
                    "prof_id": profile_id,
                    "entity_names": [e.lower() for e in query_entities]
                })

                entities = []
                relationships = []
                for record in result:
                    entities.append({
                        "name": record["entity_name"],
                        "type": record["entity_type"]
                    })
                    if record["related_name"]:
                        relationships.append({
                            "subject": record["entity_name"],
                            "predicate": record["predicate"] or record["rel_type"],
                            "object": record["related_name"],
                            "object_type": record["related_type"]
                        })

                return {
                    "entities": entities,
                    "relationships": relationships,
                    "expanded_context": []  # TODO: pull sections for matched entities
                }

        except Exception as e:
            logger.error(f"Neo4j search failed: {e}")
            return {"entities": [], "relationships": [], "expanded_context": []}

    def _metadata_search(self, profile_id: str, subscription_id: str,
                         query_understanding: dict) -> dict:
        """Layer 3: MongoDB document-level metadata."""
        try:
            db = self._get_mongo_db()
            if not db:
                return {"documents": []}

            # Fetch document metadata for the profile
            docs = list(db["documents"].find(
                {
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "pipeline_status": "TRAINING_COMPLETED"
                },
                {
                    "document_id": 1, "source_file": 1,
                    "screening.summary": 1, "file_type": 1,
                    "_id": 0
                }
            ).limit(100))

            return {"documents": docs}

        except Exception as e:
            logger.error(f"MongoDB metadata search failed: {e}")
            return {"documents": []}

    def _assemble_context(self, results: dict, query: str, top_k: int) -> dict:
        """Merge results from all three layers into unified context."""
        qdrant_results = results.get("qdrant", {})
        kg_results = results.get("neo4j", {})
        mongo_results = results.get("mongodb", {})

        chunks = qdrant_results.get("chunks", [])

        # TODO: Cross-encoder rerank (BAAI/bge-reranker-base)
        # For now, trust Qdrant's scoring

        # Deduplicate overlapping chunks
        seen_texts = set()
        deduped_chunks = []
        for chunk in chunks:
            text_key = chunk.get("text", "")[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduped_chunks.append(chunk)

        # Trim to top_k
        deduped_chunks = deduped_chunks[:top_k]

        # Build document metadata lookup
        doc_meta = {}
        for doc in mongo_results.get("documents", []):
            doc_meta[doc.get("document_id")] = doc

        # Attach document metadata to chunks
        for chunk in deduped_chunks:
            doc_id = chunk.get("payload", {}).get("document_id")
            if doc_id and doc_id in doc_meta:
                chunk["document_meta"] = doc_meta[doc_id]

        return {
            "chunks": deduped_chunks,
            "kg_context": kg_results,
            "doc_metadata": doc_meta,
            "retrieval_stats": {
                "qdrant_hits": qdrant_results.get("total_hits", 0),
                "kg_entities": len(kg_results.get("entities", [])),
                "kg_relationships": len(kg_results.get("relationships", [])),
                "documents_in_profile": len(mongo_results.get("documents", []))
            }
        }

    def _get_qdrant_client(self):
        """Get or create Qdrant client."""
        if self._qdrant:
            return self._qdrant
        from qdrant_client import QdrantClient
        from src.api.config import Config
        self._qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
        return self._qdrant

    def _get_neo4j_driver(self):
        """Get or create Neo4j driver."""
        if self._neo4j:
            return self._neo4j
        try:
            from neo4j import GraphDatabase
            from src.api.config import Config
            self._neo4j = GraphDatabase.driver(
                Config.Neo4j.URI,
                auth=(Config.Neo4j.USER, Config.Neo4j.PASSWORD)
            )
            return self._neo4j
        except Exception:
            return None

    def _get_mongo_db(self):
        """Get or create MongoDB connection."""
        if self._mongo_db:
            return self._mongo_db
        try:
            from pymongo import MongoClient
            from src.api.config import Config
            client = MongoClient(Config.MongoDB.URI)
            self._mongo_db = client[Config.MongoDB.DB]
            return self._mongo_db
        except Exception:
            return None

    def _embed_query(self, query: str) -> list:
        """Generate embedding for query text."""
        try:
            from src.embedding.model_loader import encode_with_fallback
            return encode_with_fallback(query)
        except Exception:
            # Fallback: try sentence-transformers directly
            from sentence_transformers import SentenceTransformer
            from src.api.config import Config
            model = SentenceTransformer(Config.Model.EMBEDDING_MODEL)
            return model.encode(query).tolist()
