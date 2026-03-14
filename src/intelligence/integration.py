"""
Integration module for DocWain Intelligence Layer.

This module provides integration points between the intelligence layer
and the existing extraction/embedding pipelines:
- Document intelligence processing during extraction
- Q&A generation and caching after embedding
- Response formatting for RAG queries
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.intelligence.document_intelligence import (
    DocumentIntelligence,
    DocumentMetadata,
    ExtractedEntities,
    StructuredDocument,
    DocumentDomain,
)
from src.intelligence.qa_generator import (
    GeneratedQA,
    QACacheManager,
    QAGenerator,
    QAGenerationResult,
)
from src.intelligence.response_formatter import (
    FormattedResponse,
    ResponseFormatter,
    format_acknowledged_response,
)

logger = get_logger(__name__)

@dataclass
class IntelligenceResult:
    """Result of intelligence processing for a document."""

    document_id: str
    metadata: Optional[DocumentMetadata] = None
    structured_doc: Optional[StructuredDocument] = None
    entities: Optional[ExtractedEntities] = None
    qa_pairs: List[GeneratedQA] = field(default_factory=list)
    domain: str = "generic"
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "structured_doc": self.structured_doc.to_dict() if self.structured_doc else None,
            "entities": self.entities.to_dict() if self.entities else None,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "domain": self.domain,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
        }

class DocumentIntelligenceProcessor:
    """
    Processes documents through the intelligence layer.

    Coordinates document intelligence, Q&A generation, and caching.
    """

    def __init__(
        self,
        redis_client=None,
        enable_qa_generation: bool = True,
        enable_entity_extraction: bool = True,
        qa_cache_ttl: int = 86400 * 30,  # 30 days
        max_qa_per_document: int = 20,
    ):
        """
        Initialize the processor.

        Args:
            redis_client: Optional Redis client for Q&A caching.
            enable_qa_generation: Whether to generate Q&A pairs.
            enable_entity_extraction: Whether to extract entities.
            qa_cache_ttl: TTL for cached Q&A pairs in seconds.
            max_qa_per_document: Maximum Q&A pairs to generate per document.
        """
        self.document_intelligence = DocumentIntelligence(
            enable_deep_ner=enable_entity_extraction,
            extract_nouns=True,
        )
        self.qa_generator = QAGenerator(
            max_qa_per_document=max_qa_per_document,
            min_confidence=0.5,
            enable_domain_specific=True,
        )
        self.qa_cache = QACacheManager(redis_client=redis_client)
        self.response_formatter = ResponseFormatter()
        self.enable_qa_generation = enable_qa_generation
        self.enable_entity_extraction = enable_entity_extraction
        self._redis_client = redis_client
        self._qa_cache_ttl = qa_cache_ttl

    def process_document(
        self,
        document_id: str,
        content: Union[str, bytes],
        filename: str,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        file_size: Optional[int] = None,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceResult:
        """
        Process a document through the intelligence layer.

        Args:
            document_id: Unique document identifier.
            content: Document content (text or bytes).
            filename: Original filename.
            subscription_id: Optional subscription ID.
            profile_id: Optional profile ID.
            file_size: Optional file size in bytes.
            raw_metadata: Optional raw metadata from extraction.

        Returns:
            IntelligenceResult with all extracted intelligence.
        """
        start_time = time.time()
        errors: List[str] = []

        # Convert bytes to string if needed
        text_content = content
        if isinstance(content, bytes):
            try:
                text_content = content.decode("utf-8", errors="replace")
            except Exception as e:
                errors.append(f"Content decode error: {e}")
                text_content = ""

        # Process document through intelligence layer
        structured_doc: Optional[StructuredDocument] = None
        try:
            structured_doc = self.document_intelligence.process_document(
                document_id=document_id,
                text=text_content,
                filename=filename,
                file_size=file_size or 0,
                raw_metadata=raw_metadata,
            )
        except Exception as e:
            logger.error("Document intelligence processing failed: %s", e)
            errors.append(f"Intelligence processing failed: {e}")

        # Extract metadata and entities from structured document
        metadata = structured_doc.metadata if structured_doc else None
        entities = structured_doc.entities if structured_doc else None
        domain = structured_doc.domain.value if structured_doc else "generic"

        # Generate Q&A pairs if enabled
        qa_pairs: List[GeneratedQA] = []
        if self.enable_qa_generation and structured_doc:
            try:
                qa_result = self.qa_generator.generate(structured_doc)
                qa_pairs = qa_result.qa_pairs
                logger.info(
                    "Generated %d Q&A pairs for document %s",
                    len(qa_pairs), document_id
                )

                # Cache Q&A pairs in Redis if available
                if self._redis_client and qa_pairs:
                    cached_count = self.qa_cache.cache_qa_pairs(
                        document_id=document_id,
                        qa_pairs=qa_pairs,
                        ttl_seconds=self._qa_cache_ttl,
                    )
                    logger.info("Cached %d Q&A pairs for document %s", cached_count, document_id)

            except Exception as e:
                logger.error("Q&A generation failed: %s", e)
                errors.append(f"Q&A generation failed: {e}")

        processing_time_ms = (time.time() - start_time) * 1000

        return IntelligenceResult(
            document_id=document_id,
            metadata=metadata,
            structured_doc=structured_doc,
            entities=entities,
            qa_pairs=qa_pairs,
            domain=domain,
            processing_time_ms=processing_time_ms,
            errors=errors,
        )

    def get_cached_qa_pairs(
        self,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve cached Q&A pairs for a document.

        Args:
            document_id: The document ID.

        Returns:
            List of cached Q&A pairs as dictionaries.
        """
        return self.qa_cache.get_qa_pairs(document_id)

    def find_matching_qa(
        self,
        document_id: str,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find cached Q&A pairs matching a query.

        Args:
            document_id: The document ID.
            query: User query to match.
            max_results: Maximum results to return.

        Returns:
            List of matching Q&A pairs with scores.
        """
        return self.qa_cache.find_matching_qa(
            document_id=document_id,
            query=query,
            max_results=max_results,
        )

    def format_rag_response(
        self,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> FormattedResponse:
        """
        Format a RAG response with acknowledgement.

        Args:
            query: The user's query.
            answer: The generated answer.
            sources: Optional list of source documents.
            confidence: Optional confidence score.

        Returns:
            FormattedResponse with acknowledgement.
        """
        return self.response_formatter.format_response(
            query=query,
            content=answer,
            sources=sources,
            confidence=confidence,
        )

class KnowledgeGraphBuilder:
    """
    Builds knowledge graph relationships for documents.

    Creates hierarchical relationships:
    subscription_id → profile_id → document_ids
    """

    def __init__(self, neo4j_client=None, mongodb_client=None):
        """
        Initialize the knowledge graph builder.

        Args:
            neo4j_client: Optional Neo4j client for graph storage.
            mongodb_client: Optional MongoDB client for metadata.
        """
        self._neo4j = neo4j_client
        self._mongodb = mongodb_client

    def build_document_graph(
        self,
        document_id: str,
        intelligence_result: IntelligenceResult,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph for a document.

        Creates nodes and relationships for:
        - Document with metadata
        - Extracted entities
        - Hierarchical ownership (subscription → profile → document)

        Args:
            document_id: The document ID.
            intelligence_result: Intelligence processing result.
            subscription_id: Optional subscription ID.
            profile_id: Optional profile ID.

        Returns:
            Graph construction result with node/edge counts.
        """
        nodes_created = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # Build document node
            doc_node = self._create_document_node(
                document_id,
                intelligence_result,
                subscription_id,
                profile_id,
            )
            nodes_created += 1

            # Build entity nodes
            if intelligence_result.entities:
                entity_nodes = self._create_entity_nodes(
                    document_id,
                    intelligence_result.entities,
                )
                nodes_created += len(entity_nodes)
                relationships_created += len(entity_nodes)  # Doc -> Entity

            # Build hierarchy relationships
            if subscription_id:
                self._create_subscription_relationship(
                    subscription_id, profile_id, document_id
                )
                relationships_created += 2 if profile_id else 1

        except Exception as e:
            logger.error("Knowledge graph build failed: %s", e)
            errors.append(str(e))

        return {
            "document_id": document_id,
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "errors": errors,
        }

    def _create_document_node(
        self,
        document_id: str,
        result: IntelligenceResult,
        subscription_id: Optional[str],
        profile_id: Optional[str],
    ) -> Dict[str, Any]:
        """Create a document node in the knowledge graph."""
        node = {
            "id": document_id,
            "type": "Document",
            "domain": result.domain,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "created_at": time.time(),
        }

        if result.metadata:
            node.update({
                "filename": result.metadata.filename,
                "extension": result.metadata.extension,
                "language": result.metadata.language,
                "page_count": result.metadata.page_count,
            })

        # Store in MongoDB if available
        if self._mongodb:
            try:
                self._mongodb.document_nodes.update_one(
                    {"id": document_id},
                    {"$set": node},
                    upsert=True,
                )
            except Exception as e:
                logger.warning("Failed to store document node: %s", e)

        return node

    def _create_entity_nodes(
        self,
        document_id: str,
        entities: ExtractedEntities,
    ) -> List[Dict[str, Any]]:
        """Create entity nodes linked to the document."""
        entity_nodes = []

        for entity_type, entity_list in [
            ("Person", entities.persons),
            ("Organization", entities.organizations),
            ("Location", entities.locations),
            ("Date", entities.dates),
            ("Money", entities.monetary_values),
            ("Product", entities.products),
        ]:
            for entity_value in entity_list:
                node = {
                    "type": entity_type,
                    "value": entity_value,
                    "document_id": document_id,
                    "created_at": time.time(),
                }
                entity_nodes.append(node)

        # Bulk insert if MongoDB available
        if self._mongodb and entity_nodes:
            try:
                self._mongodb.entity_nodes.insert_many(entity_nodes)
            except Exception as e:
                logger.warning("Failed to store entity nodes: %s", e)

        return entity_nodes

    def _create_subscription_relationship(
        self,
        subscription_id: str,
        profile_id: Optional[str],
        document_id: str,
    ) -> None:
        """Create hierarchical ownership relationships."""
        if self._mongodb:
            try:
                # Subscription -> Document relationship
                self._mongodb.relationships.update_one(
                    {
                        "from_id": subscription_id,
                        "from_type": "Subscription",
                        "to_id": document_id,
                        "to_type": "Document",
                    },
                    {
                        "$set": {
                            "relationship": "OWNS",
                            "updated_at": time.time(),
                        }
                    },
                    upsert=True,
                )

                # Profile -> Document relationship if profile exists
                if profile_id:
                    self._mongodb.relationships.update_one(
                        {
                            "from_id": profile_id,
                            "from_type": "Profile",
                            "to_id": document_id,
                            "to_type": "Document",
                        },
                        {
                            "$set": {
                                "relationship": "CONTAINS",
                                "subscription_id": subscription_id,
                                "updated_at": time.time(),
                            }
                        },
                        upsert=True,
                    )
            except Exception as e:
                logger.warning("Failed to create relationships: %s", e)

def process_document_intelligence(
    document_id: str,
    content: Union[str, bytes],
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    file_size: Optional[int] = None,
    raw_metadata: Optional[Dict[str, Any]] = None,
    redis_client=None,
) -> IntelligenceResult:
    """
    Convenience function to process document intelligence.

    Args:
        document_id: Document ID.
        content: Document content.
        filename: Original filename.
        subscription_id: Optional subscription ID.
        profile_id: Optional profile ID.
        file_size: Optional file size in bytes.
        raw_metadata: Optional raw metadata from extraction.
        redis_client: Optional Redis client for Q&A caching.

    Returns:
        IntelligenceResult with extracted intelligence.
    """
    processor = DocumentIntelligenceProcessor(
        redis_client=redis_client,
    )
    return processor.process_document(
        document_id=document_id,
        content=content,
        filename=filename,
        subscription_id=subscription_id,
        profile_id=profile_id,
        file_size=file_size,
        raw_metadata=raw_metadata,
    )

__all__ = [
    "IntelligenceResult",
    "DocumentIntelligenceProcessor",
    "KnowledgeGraphBuilder",
    "process_document_intelligence",
]
