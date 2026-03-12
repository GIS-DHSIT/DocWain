"""
Practical integration example for enhanced embedding system.

This module demonstrates how to integrate the enhanced embedding components
into DocWain's existing RAG pipeline.
"""

from __future__ import annotations

import logging

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

# Import enhanced components
from src.embedding.orchestrator import EnhancedEmbeddingOrchestrator
from src.embedding.reranker import RerankingConfig
from src.embedding.threshold_tuner import ThresholdConfig

logger = get_logger(__name__)

class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with caching, reranking, and adaptive thresholds.
    
    This is a reference implementation showing how to integrate the enhanced
    embedding system with DocWain's existing retrieve-augment-generate pipeline.
    """

    def __init__(
        self,
        qdrant_client: Any,
        embedder: Any,
        llm_generator: Any,
        redis_client: Optional[Any] = None,
        cross_encoder_model: Optional[Any] = None,
    ):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.llm_generator = llm_generator

        # Initialize enhanced embedding orchestrator
        self.orchestrator = EnhancedEmbeddingOrchestrator(
            redis_client=redis_client,
            cross_encoder_model=cross_encoder_model,
            reranking_config=RerankingConfig(
                use_cross_encoder=cross_encoder_model is not None,
                alpha_dense=0.6,
                alpha_sparse=0.2,
                alpha_semantic=0.2,
                enable_diversity_boost=True,
            ),
            threshold_config=ThresholdConfig(
                base_threshold=0.20,
                min_results_target=5,
            ),
        )

    def retrieve(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        document_id: Optional[str] = None,
        top_k: int = 50,
        query_intent: Optional[str] = None,
    ) -> List[Any]:
        """
        Enhanced retrieval with caching, adaptive thresholds, and reranking.
        
        Args:
            query: User query
            subscription_id: Subscription identifier
            profile_id: Profile identifier
            document_id: Optional document filter
            top_k: Number of results to retrieve
            query_intent: Optional query intent (skills, experience, etc.)
            
        Returns:
            List of ranked chunks
        """
        # Step 1: Embed query with cache
        logger.info("Step 1: Embedding query (with cache)")
        query_vector = self.orchestrator.embed_query_efficient(
            query,
            self.embedder,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )

        # Step 2: Compute adaptive threshold
        logger.info("Step 2: Computing adaptive threshold")
        threshold = self.orchestrator.compute_adaptive_threshold(
            query,
            document_count=100,  # Replace with actual count
            query_intent=query_intent,
        )

        # Step 3: Retrieve from vector database
        logger.info(f"Step 3: Retrieving from Qdrant (threshold={threshold:.3f})")
        from src.api.vector_store import build_collection_name, build_qdrant_filter

        collection = build_collection_name(subscription_id)
        q_filter = build_qdrant_filter(
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=document_id,
        )

        # Initial retrieval with adaptive threshold
        results = self.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=q_filter,
            limit=top_k,
            score_threshold=threshold,  # Use adaptive threshold
            with_payload=True,
        )

        chunks = self._points_to_chunks(results)

        # Step 4: Adjust threshold if insufficient results
        logger.info(f"Step 4: Coverage check ({len(chunks)} results)")
        if len(chunks) < 5:
            new_threshold = self.orchestrator.adjust_threshold_for_coverage(
                threshold,
                len(chunks),
            )

            if new_threshold < threshold:
                logger.info(
                    f"Lowering threshold for coverage: {threshold:.3f} -> {new_threshold:.3f}"
                )
                results = self.qdrant_client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=top_k,
                    score_threshold=new_threshold,
                    with_payload=True,
                )
                chunks = self._points_to_chunks(results)

        # Step 5: Rerank for better relevance
        logger.info(f"Step 5: Reranking {len(chunks)} results")
        chunks = self.orchestrator.rerank_retrieved_chunks(
            query,
            chunks,
            query_intent=query_intent,
        )

        # Step 6: Filter to threshold
        logger.info("Step 6: Final filtering")
        final_chunks = [c for c in chunks if c.score >= threshold]

        logger.info(
            f"Final result: {len(final_chunks)} chunks from {len(results)} retrieved"
        )
        return final_chunks

    def generate_response(
        self,
        query: str,
        chunks: List[Any],
    ) -> str:
        """
        Generate response from retrieved chunks.
        
        Args:
            query: Original user query
            chunks: Retrieved and reranked chunks
            
        Returns:
            Generated response
        """
        logger.info("Generating response from chunks")

        # Build context from chunks
        context_parts = []
        for chunk in chunks[:5]:  # Use top 5 chunks
            text = getattr(chunk, "text", "")
            if text:
                context_parts.append(text)

        context = "\n\n".join(context_parts)

        # Generate response using LLM
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm_generator.generate(prompt)
        return response

    def answer(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        document_id: Optional[str] = None,
        top_k: int = 50,
        query_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve -> rerank -> generate.
        
        Args:
            query: User query
            subscription_id: Subscription identifier
            profile_id: Profile identifier
            document_id: Optional document filter
            top_k: Number of results to retrieve
            query_intent: Optional query intent
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"RAG pipeline started for query: {query[:50]}...")

        # Retrieve with enhancements
        chunks = self.retrieve(
            query,
            subscription_id,
            profile_id,
            document_id=document_id,
            top_k=top_k,
            query_intent=query_intent,
        )

        if not chunks:
            return {
                "response": "I could not find relevant information in the documents.",
                "chunks": [],
                "metrics": self.orchestrator.get_orchestrator_metrics(),
            }

        # Generate response
        response = self.generate_response(query, chunks)

        return {
            "response": response,
            "chunks": [
                {
                    "id": getattr(c, "chunk_id", ""),
                    "text": getattr(c, "text", "")[:200],
                    "score": getattr(c, "score", 0.0),
                    "source": (getattr(c, "meta", {}) or {}).get("source_name"),
                }
                for c in chunks[:5]
            ],
            "metrics": self.orchestrator.get_orchestrator_metrics(),
        }

    def _points_to_chunks(self, points_response: Any) -> List[Any]:
        """Convert Qdrant points to chunk objects."""
        chunks = []
        points = getattr(points_response, "points", []) or []

        for point in points:
            if not point:
                continue

            # Simple chunk object
            class Chunk:
                def __init__(self, point):
                    self.chunk_id = point.id
                    self.score = point.score
                    self.text = (point.payload or {}).get("canonical_text", "")
                    self.meta = point.payload or {}

            chunks.append(Chunk(point))

        return chunks

# Example usage function
def example_usage():
    """Demonstrates how to use the enhanced RAG pipeline."""
    import os

    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Initialize components
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # Mock LLM generator
    class MockLLMGenerator:
        def generate(self, prompt: str) -> str:
            return "This is a generated response."

    llm = MockLLMGenerator()

    # Create enhanced pipeline
    pipeline = EnhancedRAGPipeline(
        qdrant_client=qdrant_client,
        embedder=embedder,
        llm_generator=llm,
        redis_client=None,  # Add Redis client for caching
        cross_encoder_model=None,  # Add cross-encoder model if available
    )

    # Answer a question
    result = pipeline.answer(
        query="What are your recent projects?",
        subscription_id="sub-123",
        profile_id="prof-456",
        query_intent="experience",
    )

    print(f"Response: {result['response']}")
    print(f"Sources: {len(result['chunks'])} chunks")
    print(f"Cache hit rate: {result['metrics']['cache']['hit_rate']:.1%}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()

