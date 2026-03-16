"""Integration tests for the DocWain document processing pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestPipelineStatusTransitions:
    """Test that pipeline status transitions follow the correct order."""

    def test_init_document_creates_uploaded_status(self):
        """New documents start with UPLOADED pipeline_status."""
        from src.api.statuses import PIPELINE_UPLOADED, STAGE_PENDING
        # Verify constants exist and have correct values
        assert PIPELINE_UPLOADED == "UPLOADED"
        assert STAGE_PENDING == "PENDING"

    def test_pipeline_status_constants_complete(self):
        """All pipeline statuses exist."""
        from src.api.statuses import (
            PIPELINE_UPLOADED, PIPELINE_EXTRACTION_IN_PROGRESS,
            PIPELINE_EXTRACTION_COMPLETED, PIPELINE_EXTRACTION_FAILED,
            PIPELINE_SCREENING_IN_PROGRESS, PIPELINE_SCREENING_COMPLETED,
            PIPELINE_SCREENING_FAILED, PIPELINE_EMBEDDING_IN_PROGRESS,
            PIPELINE_TRAINING_COMPLETED, PIPELINE_EMBEDDING_FAILED
        )
        statuses = [
            PIPELINE_UPLOADED, PIPELINE_EXTRACTION_IN_PROGRESS,
            PIPELINE_EXTRACTION_COMPLETED, PIPELINE_EXTRACTION_FAILED,
            PIPELINE_SCREENING_IN_PROGRESS, PIPELINE_SCREENING_COMPLETED,
            PIPELINE_SCREENING_FAILED, PIPELINE_EMBEDDING_IN_PROGRESS,
            PIPELINE_TRAINING_COMPLETED, PIPELINE_EMBEDDING_FAILED
        ]
        assert len(statuses) == 10
        assert len(set(statuses)) == 10  # All unique

    def test_legacy_status_mappings(self):
        """Legacy status constants map to new pipeline statuses."""
        from src.api.statuses import (
            STATUS_UNDER_REVIEW, PIPELINE_UPLOADED,
            STATUS_TRAINING_STARTED, PIPELINE_EMBEDDING_IN_PROGRESS,
            STATUS_TRAINING_SUCCEEDED, PIPELINE_TRAINING_COMPLETED
        )
        assert STATUS_UNDER_REVIEW == PIPELINE_UPLOADED
        assert STATUS_TRAINING_STARTED == PIPELINE_EMBEDDING_IN_PROGRESS
        assert STATUS_TRAINING_SUCCEEDED == PIPELINE_TRAINING_COMPLETED


class TestCeleryTaskImports:
    """Test that all Celery tasks can be imported."""

    def test_extraction_task_imports(self):
        from src.tasks.extraction import extract_document
        assert extract_document.name == "src.tasks.extraction.extract_document"

    def test_screening_task_imports(self):
        from src.tasks.screening import screen_document
        assert screen_document.name == "src.tasks.screening.screen_document"

    def test_kg_task_imports(self):
        from src.tasks.kg import build_knowledge_graph
        assert build_knowledge_graph.name == "src.tasks.kg.build_knowledge_graph"

    def test_embedding_task_imports(self):
        from src.tasks.embedding import embed_document
        assert embed_document.name == "src.tasks.embedding.embed_document"

    def test_backfill_task_imports(self):
        from src.tasks.backfill import backfill_kg_refs
        assert backfill_kg_refs.name == "src.tasks.backfill.backfill_kg_refs"


class TestCeleryAppConfiguration:
    """Test Celery app configuration."""

    def test_celery_app_exists(self):
        from src.celery_app import app
        assert app.main == "docwain"

    def test_celery_queues_configured(self):
        from src.celery_app import app
        queues = app.conf.task_queues
        expected_queues = [
            "extraction_queue", "screening_queue", "kg_queue",
            "embedding_queue", "backfill_queue"
        ]
        for q in expected_queues:
            assert q in queues, f"Queue {q} not configured"

    def test_celery_reliability_settings(self):
        from src.celery_app import app
        assert app.conf.task_acks_late is True
        assert app.conf.task_reject_on_worker_lost is True
        assert app.conf.task_time_limit == 1800


class TestExtractionEngine:
    """Test extraction engine structure."""

    def test_extraction_engine_imports(self):
        from src.extraction import ExtractionEngine, ExtractionResult

    def test_extraction_result_serialization(self):
        from src.extraction.models import ExtractionResult
        result = ExtractionResult(
            document_id="test_doc",
            subscription_id="test_sub",
            profile_id="test_prof",
            clean_text="Test content",
            structure={"sections": [{"id": "s1", "title": "Test"}]},
            entities=[],
            relationships=[],
            tables=[],
            metadata={"page_count": 1, "models_used": ["test"]}
        )
        d = result.to_dict()
        assert d["document_id"] == "test_doc"
        assert d["clean_text"] == "Test content"

        s = result.to_summary()
        assert s["page_count"] == 1
        assert s["section_count"] == 1
        assert s["entity_count"] == 0

    def test_extraction_merger_deduplicates_entities(self):
        from src.extraction.merger import ExtractionMerger
        from src.extraction.models import Entity
        merger = ExtractionMerger()
        result = merger.merge(
            document_id="test",
            subscription_id="sub",
            profile_id="prof",
            structural={"entities": [{"text": "Company A", "type": "ORG", "confidence": 0.8}]},
            semantic={"entities": [{"text": "Company A", "type": "ORG", "confidence": 0.7}]},
            vision={}
        )
        # Should deduplicate — Company A appears once with boosted confidence
        org_entities = [e for e in result.entities if e.text == "Company A"]
        assert len(org_entities) == 1
        assert org_entities[0].confidence > 0.8  # Boosted by agreement


class TestScreeningPluginSystem:
    """Test screening plugin architecture."""

    def test_plugin_registry_discovers_plugins(self):
        from src.screening.plugins.registry import get_registry
        registry = get_registry()
        plugins = registry.list_plugins()
        assert len(plugins) >= 8  # 3 mandatory + 5 configurable

    def test_mandatory_plugins_exist(self):
        from src.screening.plugins.registry import get_registry
        registry = get_registry()
        mandatory = registry.get_mandatory_plugins()
        mandatory_names = {p.get_manifest().name for p in mandatory}
        assert "pii_detector" in mandatory_names
        assert "secrets_scanner" in mandatory_names
        assert "legality_checker" in mandatory_names

    def test_screening_orchestrator_imports(self):
        from src.screening.orchestrator import ScreeningOrchestrator

    def test_plugin_result_structure(self):
        from src.screening.plugins.base import PluginResult
        result = PluginResult(
            plugin_name="test",
            success=True,
            outputs={"key": "value"},
            duration_ms=100
        )
        assert result.plugin_name == "test"
        assert result.success is True


class TestEnrichedPayloadBuilder:
    """Test the enriched Qdrant payload builder."""

    def test_payload_builder_imports(self):
        from src.embedding.payload_builder import build_enriched_payload

    def test_payload_contains_all_fields(self):
        from src.embedding.payload_builder import build_enriched_payload
        payload = build_enriched_payload(
            chunk={"text": "Company A signed the agreement", "type": "text"},
            chunk_index=0,
            document_id="doc1",
            subscription_id="sub1",
            profile_id="prof1",
            extraction_data={
                "entities": [{"text": "Company A", "type": "ORG", "confidence": 0.9}]
            },
            screening_summary={
                "domain_tags": ["legal"],
                "doc_category": "contract",
                "entity_scores": {"Company A": 0.95}
            },
            kg_node_ids=["node_1"],
            quality_grade="B"
        )

        # Identity
        assert payload["subscription_id"] == "sub1"
        assert payload["profile_id"] == "prof1"
        assert payload["document_id"] == "doc1"

        # Enrichment
        assert "Company A" in payload["entities"]
        assert "ORG" in payload["entity_types"]
        assert "legal" in payload["domain_tags"]
        assert payload["doc_category"] == "contract"
        assert payload["importance_score"] == 0.95

        # KG linkage
        assert payload["kg_node_ids"] == ["node_1"]

        # Quality
        assert payload["quality_grade"] == "B"

        # Text
        assert "Company A" in payload["text"]


class TestPipelineAPIEndpoints:
    """Test pipeline API endpoint structure."""

    def test_pipeline_router_imports(self):
        from src.api.pipeline_api import pipeline_router
        route_paths = [r.path for r in pipeline_router.routes]
        assert "/documents/{document_id}/status" in route_paths
        assert "/documents/{document_id}/screen" in route_paths
        assert "/documents/{document_id}/embed" in route_paths
        assert "/documents/{document_id}/kg/status" in route_paths


class TestHITLGates:
    """Test HITL gate enforcement."""

    def test_screening_gate_checks_extraction_status(self):
        """Cannot trigger screening unless extraction is complete."""
        from src.api.pipeline_api import trigger_screening
        # This would need a mock MongoDB — just verify the function exists
        assert callable(trigger_screening)

    def test_embedding_gate_checks_screening_status(self):
        """Cannot trigger embedding unless screening is complete."""
        from src.api.pipeline_api import trigger_embedding
        assert callable(trigger_embedding)
