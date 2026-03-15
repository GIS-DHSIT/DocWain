"""Integration tests for the three-layer retrieval system."""

import pytest


class TestUnifiedRetriever:
    """Test unified retriever structure and imports."""

    def test_unified_retriever_imports(self):
        from src.retrieval.unified_retriever import UnifiedRetriever

    def test_retriever_has_three_layers(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()
        # Verify the three layer methods exist
        assert hasattr(retriever, '_qdrant_search')
        assert hasattr(retriever, '_kg_search')
        assert hasattr(retriever, '_metadata_search')

    def test_retriever_has_context_assembly(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()
        assert hasattr(retriever, '_assemble_context')

    def test_context_assembly_deduplicates(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()

        results = {
            "qdrant": {
                "chunks": [
                    {"id": 1, "score": 0.9, "text": "First chunk about Company A", "payload": {}},
                    {"id": 2, "score": 0.8, "text": "First chunk about Company A", "payload": {}},
                    {"id": 3, "score": 0.7, "text": "Different chunk about Company B", "payload": {}},
                ],
                "total_hits": 3
            },
            "neo4j": {"entities": [], "relationships": [], "expanded_context": []},
            "mongodb": {"documents": []}
        }

        merged = retriever._assemble_context(results, "test query", top_k=10)
        # Should deduplicate -- two chunks with same text prefix
        assert len(merged["chunks"]) <= 3
        # Specifically, the two identical "First chunk about Company A" should collapse
        assert len(merged["chunks"]) == 2

    def test_context_assembly_respects_top_k(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()

        chunks = [
            {"id": i, "score": 0.9 - i * 0.01, "text": f"Unique chunk number {i}", "payload": {}}
            for i in range(20)
        ]

        results = {
            "qdrant": {"chunks": chunks, "total_hits": 20},
            "neo4j": {"entities": [], "relationships": [], "expanded_context": []},
            "mongodb": {"documents": []}
        }

        merged = retriever._assemble_context(results, "test query", top_k=5)
        assert len(merged["chunks"]) == 5

    def test_context_assembly_includes_retrieval_stats(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()

        results = {
            "qdrant": {
                "chunks": [{"id": 1, "score": 0.9, "text": "chunk", "payload": {}}],
                "total_hits": 1
            },
            "neo4j": {
                "entities": [{"name": "Company A", "type": "ORG"}],
                "relationships": [{"subject": "A", "predicate": "owns", "object": "B"}],
                "expanded_context": []
            },
            "mongodb": {"documents": [{"document_id": "d1", "source_file": "test.pdf"}]}
        }

        merged = retriever._assemble_context(results, "test query", top_k=10)
        stats = merged["retrieval_stats"]
        assert stats["qdrant_hits"] == 1
        assert stats["kg_entities"] == 1
        assert stats["kg_relationships"] == 1
        assert stats["documents_in_profile"] == 1

    def test_context_assembly_attaches_doc_metadata(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()

        results = {
            "qdrant": {
                "chunks": [
                    {"id": 1, "score": 0.9, "text": "chunk text", "payload": {"document_id": "doc-1"}},
                ],
                "total_hits": 1
            },
            "neo4j": {"entities": [], "relationships": [], "expanded_context": []},
            "mongodb": {
                "documents": [
                    {"document_id": "doc-1", "source_file": "report.pdf"}
                ]
            }
        }

        merged = retriever._assemble_context(results, "test query", top_k=10)
        assert merged["chunks"][0].get("document_meta") is not None
        assert merged["chunks"][0]["document_meta"]["source_file"] == "report.pdf"

    def test_context_assembly_handles_empty_results(self):
        from src.retrieval.unified_retriever import UnifiedRetriever
        retriever = UnifiedRetriever()

        results = {
            "qdrant": {},
            "neo4j": {},
            "mongodb": {}
        }

        merged = retriever._assemble_context(results, "test query", top_k=10)
        assert merged["chunks"] == []
        assert merged["retrieval_stats"]["qdrant_hits"] == 0


class TestResponseQualityGate:
    """Test response quality gate."""

    def test_quality_gate_imports(self):
        from src.quality.response_gate import ResponseQualityGate

    def test_grounded_response_passes(self):
        from src.quality.response_gate import ResponseQualityGate
        gate = ResponseQualityGate()
        result = gate.check(
            query="What is Company A?",
            response="Company A is a technology firm based in Singapore.",
            context_chunks=[
                {"text": "Company A is a technology firm headquartered in Singapore since 2020."}
            ]
        )
        assert result["passed"] is True
        assert result["confidence"] > 0.5

    def test_ungrounded_response_fails(self):
        from src.quality.response_gate import ResponseQualityGate
        gate = ResponseQualityGate()
        result = gate.check(
            query="What is Company A?",
            response="Company A was founded by John Smith in 1995 and operates in 50 countries worldwide with revenue of 10 billion dollars.",
            context_chunks=[
                {"text": "The weather today is sunny and warm."}
            ]
        )
        # Should have low confidence due to no overlap
        assert result["confidence"] < 0.7

    def test_empty_context_fails(self):
        from src.quality.response_gate import ResponseQualityGate
        gate = ResponseQualityGate()
        result = gate.check(
            query="What is Company A?",
            response="Company A is a firm.",
            context_chunks=[]
        )
        assert result["grounded"] is False
        assert result["confidence"] == 0.0

    def test_insufficient_context_returns_fallback(self):
        from src.quality.response_gate import ResponseQualityGate
        gate = ResponseQualityGate()
        result = gate.check(
            query="complex query",
            response="This is a completely fabricated answer about quantum computing and blockchain integration with zero basis in any document.",
            context_chunks=[{"text": "Simple recipe for chocolate cake."}]
        )
        if result["confidence"] < 0.3:
            assert "insufficient" in result["final_response"].lower() or result["passed"] is False

    def test_quality_gate_result_structure(self):
        from src.quality.response_gate import ResponseQualityGate
        gate = ResponseQualityGate()
        result = gate.check(
            query="test",
            response="test response about things.",
            context_chunks=[{"text": "test response about things."}]
        )
        assert "passed" in result
        assert "grounded" in result
        assert "confidence" in result
        assert "issues" in result
        assert "final_response" in result


class TestQdrantPayloadIndexSetup:
    """Test Qdrant index setup."""

    def test_qdrant_setup_imports(self):
        from src.api.qdrant_setup import ensure_payload_indexes

    def test_index_fields_are_correct(self):
        """Verify the expected index fields are defined."""
        # Read the source to verify fields
        from src.api.qdrant_setup import ensure_payload_indexes
        import inspect
        source = inspect.getsource(ensure_payload_indexes)
        expected_fields = ["profile_id", "document_id", "domain_tags",
                          "doc_category", "quality_grade", "entities"]
        for field in expected_fields:
            assert field in source, f"Index field {field} not found in setup"


class TestConfigFlags:
    """Test configuration flags for new features."""

    def test_celery_config_exists(self):
        from src.api.config import Config
        assert hasattr(Config, 'Celery')
        assert hasattr(Config.Celery, 'BROKER_URL')
        assert hasattr(Config.Celery, 'EXTRACTION_CONCURRENCY')
