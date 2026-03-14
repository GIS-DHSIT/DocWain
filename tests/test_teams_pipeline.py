"""Tests for the Teams document storage layer."""

import unittest
from unittest.mock import MagicMock, patch

from src.teams.teams_storage import (
    STATUS_EMBEDDING,
    STATUS_EMBEDDING_COMPLETED,
    STATUS_EMBEDDING_FAILED,
    STATUS_EXTRACTING,
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
    STATUS_SCREENING_PENDING_CONSENT,
    STATUS_SCREENING_REJECTED,
    TeamsDocumentStorage,
)


class TestTeamsDocumentStorage(unittest.TestCase):
    """Unit tests for TeamsDocumentStorage with a mocked MongoDB collection."""

    def setUp(self):
        with patch("src.teams.teams_storage._get_teams_collection", return_value=MagicMock()):
            self.storage = TeamsDocumentStorage()
        # Ensure the collection is a fresh mock for each test
        self.storage._collection = MagicMock()

    # ── create ──────────────────────────────────────────────────────

    def test_create_document_record(self):
        result = self.storage.create_document(
            document_id="doc-1",
            filename="report.pdf",
            subscription_id="sub-1",
            profile_id="prof-1",
        )
        self.assertEqual(result, "doc-1")
        self.storage._collection.insert_one.assert_called_once()

        record = self.storage._collection.insert_one.call_args[0][0]
        self.assertEqual(record["document_id"], "doc-1")
        self.assertEqual(record["filename"], "report.pdf")
        self.assertEqual(record["subscription_id"], "sub-1")
        self.assertEqual(record["profile_id"], "prof-1")
        self.assertEqual(record["status"], STATUS_EXTRACTING)
        self.assertEqual(record["source"], "teams")
        self.assertEqual(record["extraction"]["status"], "IN_PROGRESS")
        self.assertEqual(record["screening"]["status"], "PENDING")
        self.assertEqual(record["embedding"]["status"], "PENDING")

    # ── update_status ───────────────────────────────────────────────

    def test_update_status(self):
        self.storage.update_status("doc-1", STATUS_EMBEDDING)
        self.storage._collection.update_one.assert_called_once()

        args = self.storage._collection.update_one.call_args
        self.assertEqual(args[0][0], {"document_id": "doc-1"})
        self.assertEqual(args[0][1]["$set"]["status"], STATUS_EMBEDDING)

    # ── extraction ──────────────────────────────────────────────────

    def test_store_extraction_result(self):
        self.storage.store_extraction_result(
            document_id="doc-1",
            doc_type="invoice",
            summary="An invoice for services.",
            key_entities=["Acme Corp"],
            key_facts=["Total: $500"],
            intent_tags=["billing"],
            confidence=0.95,
        )
        self.storage._collection.update_one.assert_called_once()

        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_EXTRACTION_COMPLETED)
        self.assertEqual(set_fields["extraction.doc_type"], "invoice")
        self.assertEqual(set_fields["extraction.summary"], "An invoice for services.")
        self.assertEqual(set_fields["extraction.key_entities"], ["Acme Corp"])
        self.assertEqual(set_fields["extraction.key_facts"], ["Total: $500"])
        self.assertEqual(set_fields["extraction.intent_tags"], ["billing"])
        self.assertEqual(set_fields["extraction.confidence"], 0.95)

    # ── screening ───────────────────────────────────────────────────

    def test_store_screening_result_low_risk(self):
        self.storage.store_screening_result(
            document_id="doc-1",
            risk_level="LOW",
            risk_score=0.1,
            findings=[],
        )
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_SCREENING_COMPLETED)

    def test_store_screening_result_high_risk(self):
        self.storage.store_screening_result(
            document_id="doc-1",
            risk_level="HIGH",
            risk_score=0.85,
            findings=["PII detected"],
        )
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_SCREENING_PENDING_CONSENT)

    # ── consent ─────────────────────────────────────────────────────

    def test_consent_proceed(self):
        self.storage.consent_proceed("doc-1")
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_SCREENING_COMPLETED)

    def test_consent_reject(self):
        self.storage.consent_reject("doc-1")
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_SCREENING_REJECTED)

    # ── embedding ───────────────────────────────────────────────────

    def test_mark_embedding_started(self):
        self.storage.mark_embedding_started("doc-1")
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_EMBEDDING)

    def test_mark_embedding_completed(self):
        self.storage.mark_embedding_completed("doc-1", chunks_count=42, quality_grade="A")
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_EMBEDDING_COMPLETED)
        self.assertEqual(set_fields["embedding.chunks_count"], 42)
        self.assertEqual(set_fields["embedding.quality_grade"], "A")

    def test_mark_embedding_failed(self):
        self.storage.mark_embedding_failed("doc-1", error="timeout")
        set_fields = self.storage._collection.update_one.call_args[0][1]["$set"]
        self.assertEqual(set_fields["status"], STATUS_EMBEDDING_FAILED)
        self.assertEqual(set_fields["embedding.error"], "timeout")

    # ── read ────────────────────────────────────────────────────────

    def test_get_document(self):
        self.storage._collection.find_one.return_value = {"document_id": "doc-1", "status": "extracting"}
        result = self.storage.get_document("doc-1")
        self.storage._collection.find_one.assert_called_once_with({"document_id": "doc-1"})
        self.assertEqual(result["document_id"], "doc-1")

    # ── collection unavailable ──────────────────────────────────────

    def test_create_document_no_collection(self):
        self.storage._collection = None
        result = self.storage.create_document("doc-1", "f.pdf", "sub", "prof")
        self.assertEqual(result, "doc-1")

    def test_get_document_no_collection(self):
        self.storage._collection = None
        result = self.storage.get_document("doc-1")
        self.assertIsNone(result)


class TestPipelineCards(unittest.TestCase):
    """Tests for pipeline-specific Adaptive Card templates."""

    def test_identification_card_renders(self):
        import json
        from src.teams.cards import build_card

        card = build_card(
            "identification_card",
            doc_type="Invoice",
            confidence="95%",
            summary="Monthly invoice for cloud services.",
            entities_text="- Acme Corp\n- $5,000",
            intent_text="billing",
            action1_title="View line items",
            action1_query="What are the line items?",
            action2_title="Check total",
            action2_query="What is the total amount?",
        )
        rendered = json.dumps(card)
        assert "Invoice" in rendered
        assert "95%" in rendered
        assert "Monthly invoice for cloud services." in rendered
        assert "Acme Corp" in rendered
        assert "billing" in rendered
        assert "Document Identified" in rendered
        assert "domain_query" in rendered

    def test_screening_consent_card_renders(self):
        import json
        from src.teams.cards import build_card

        card = build_card(
            "screening_consent_card",
            risk_level="HIGH",
            risk_score="72/100",
            findings_text="- PII detected\n- Sensitive financial data",
            document_id="doc-123",
        )
        rendered = json.dumps(card)
        assert "HIGH" in rendered
        assert "72/100" in rendered
        assert "Proceed Anyway" in rendered
        assert "Cancel Embedding" in rendered
        assert "doc-123" in rendered
        assert "pipeline_consent_proceed" in rendered
        assert "pipeline_consent_reject" in rendered
        assert "ATTENTION" in rendered

    def test_screening_passed_card_renders(self):
        import json
        from src.teams.cards import build_card

        card = build_card(
            "screening_passed_card",
            risk_level="LOW",
            risk_score="8/100",
        )
        rendered = json.dumps(card)
        assert "PASSED" in rendered
        assert "LOW" in rendered
        assert "No significant findings" in rendered

    def test_embedding_complete_card_renders(self):
        import json
        from src.teams.cards import build_card

        card = build_card(
            "embedding_complete_card",
            filename="report.pdf",
            chunks_count="42",
            quality_text="A (Excellent)",
            question1="What is the main topic?",
            question2="Summarize the key findings",
            question3="What are the recommendations?",
            action1_title="Ask a question",
            action1_query="What is this document about?",
        )
        rendered = json.dumps(card)
        assert "report.pdf" in rendered
        assert "42" in rendered
        assert "A (Excellent)" in rendered
        assert "What is the main topic?" in rendered
        assert "Summarize the key findings" in rendered
        assert "What are the recommendations?" in rendered
        assert "Document Ready" in rendered
        assert "Summarize" in rendered
        assert "summarize_recent" in rendered


import asyncio
from dataclasses import dataclass, field
from typing import List
from src.teams.state import TeamsStateStore


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestTeamsDocumentPipeline(unittest.TestCase):
    """Unit tests for the 3-stage TeamsDocumentPipeline."""

    def _make_pipeline(self):
        from src.teams.pipeline import TeamsDocumentPipeline

        with patch("src.teams.teams_storage._get_teams_collection", return_value=MagicMock()):
            storage = TeamsDocumentStorage()
        storage._collection = MagicMock()
        storage._collection.insert_one = MagicMock()
        storage._collection.update_one = MagicMock()
        state_store = TeamsStateStore()
        state_store.client = None
        return TeamsDocumentPipeline(storage=storage, state_store=state_store)

    def _make_context(self):
        ctx = MagicMock()
        ctx.subscription_id = "sub-test"
        ctx.profile_id = "prof-test"
        return ctx

    # ── Stage 1: Identify ────────────────────────────────────────────

    @patch("src.teams.pipeline.get_logger")
    @patch("src.doc_understanding.understand.understand_document")
    @patch("src.doc_understanding.content_map.build_content_map")
    @patch("src.doc_understanding.identify.identify_document")
    @patch("src.api.dataHandler.fileProcessor")
    def test_stage1_identify_returns_result(
        self, mock_fp, mock_identify, mock_cm, mock_understand, mock_log
    ):
        mock_fp.return_value = {"doc1.txt": "This is test content for a document."}
        mock_identify.return_value = {"document_type": "invoice", "confidence": 0.95}
        mock_cm.return_value = {}
        mock_understand.return_value = {
            "document_summary": "An invoice document.",
            "key_entities": ["Acme Corp"],
            "key_facts": ["Total: $500"],
            "intent_tags": ["billing"],
        }

        pipeline = self._make_pipeline()
        context = self._make_context()

        with patch("src.teams.insights.generate_proactive_insights") as mock_insights:
            mock_insights_result = MagicMock()
            mock_insights_result.suggested_questions = ["Q1", "Q2", "Q3"]
            mock_insights.return_value = mock_insights_result

            result = _run(pipeline.stage_identify(
                file_bytes=b"fake pdf bytes",
                filename="invoice.pdf",
                content_type="application/pdf",
                context=context,
                correlation_id="test-001",
            ))

        self.assertIsNotNone(result)
        self.assertIn("document_id", result)
        self.assertEqual(result["doc_type"], "invoice")
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertEqual(result["summary"], "An invoice document.")
        self.assertEqual(result["key_entities"], ["Acme Corp"])
        self.assertEqual(result["key_facts"], ["Total: $500"])
        self.assertEqual(result["intent_tags"], ["billing"])
        self.assertIn("card", result)
        self.assertIn("extracted_text", result)

    @patch("src.teams.pipeline.get_logger")
    @patch("src.api.dataHandler.fileProcessor")
    def test_stage1_identify_no_content_returns_none(self, mock_fp, mock_log):
        mock_fp.return_value = {}

        pipeline = self._make_pipeline()
        context = self._make_context()

        result = _run(pipeline.stage_identify(
            file_bytes=b"empty",
            filename="empty.pdf",
            content_type="application/pdf",
            context=context,
            correlation_id="test-002",
        ))

        self.assertIsNone(result)

    # ── Stage 2: Screen ──────────────────────────────────────────────

    @patch("src.teams.pipeline.get_logger")
    @patch("src.teams.attachments._run_security_screening")
    def test_stage2_screen_low_risk_auto_proceeds(self, mock_screening, mock_log):
        @dataclass
        class MockScreening:
            risk_level: str = "LOW"
            overall_score: float = 5.0
            top_findings: List[str] = field(default_factory=lambda: ["No sensitive data detected"])
            tools_run: int = 3

        mock_screening.return_value = MockScreening()

        pipeline = self._make_pipeline()

        result = _run(pipeline.stage_screen(
            document_id="doc-001",
            extracted_text="Some safe text content.",
            filename="safe.pdf",
            correlation_id="test-003",
        ))

        self.assertFalse(result["needs_consent"])
        self.assertEqual(result["risk_level"], "LOW")
        self.assertIn("card", result)

    @patch("src.teams.pipeline.get_logger")
    @patch("src.teams.attachments._run_security_screening")
    def test_stage2_screen_high_risk_needs_consent(self, mock_screening, mock_log):
        @dataclass
        class MockScreening:
            risk_level: str = "HIGH"
            overall_score: float = 72.0
            top_findings: List[str] = field(default_factory=lambda: ["PII detected", "Sensitive financial data"])
            tools_run: int = 3

        mock_screening.return_value = MockScreening()

        pipeline = self._make_pipeline()

        result = _run(pipeline.stage_screen(
            document_id="doc-002",
            extracted_text="SSN: 123-45-6789, Credit card: 4111...",
            filename="sensitive.pdf",
            correlation_id="test-004",
        ))

        self.assertTrue(result["needs_consent"])
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertIn("card", result)

    @patch("src.teams.pipeline.get_logger")
    @patch("src.teams.attachments._run_security_screening")
    def test_stage2_screen_failure_defaults_low(self, mock_screening, mock_log):
        mock_screening.return_value = None  # Screening failed

        pipeline = self._make_pipeline()

        result = _run(pipeline.stage_screen(
            document_id="doc-003",
            extracted_text="Some text.",
            filename="unknown.pdf",
            correlation_id="test-005",
        ))

        self.assertFalse(result["needs_consent"])
        self.assertEqual(result["risk_level"], "LOW")
        self.assertIn("card", result)


class TestTeamsCollectionName(unittest.TestCase):
    """Tests for the teams collection name builder."""

    def test_format(self):
        from src.teams.pipeline import _build_teams_collection_name

        name = _build_teams_collection_name("sub-123", "prof-456")
        self.assertTrue(name.startswith("teams_"))
        self.assertIn("sub-123", name)
        self.assertIn("prof-456", name)

    def test_sanitization(self):
        from src.teams.pipeline import _build_teams_collection_name

        name = _build_teams_collection_name("sub:with:colons", "prof with spaces")
        self.assertNotIn(":", name)
        self.assertNotIn(" ", name)

    def test_truncation(self):
        from src.teams.pipeline import _build_teams_collection_name

        long_sub = "a" * 100
        long_prof = "b" * 100
        name = _build_teams_collection_name(long_sub, long_prof)
        # Each part is truncated to 32 chars
        self.assertTrue(name.startswith("teams_"))
        parts = name.split("_", 2)
        self.assertLessEqual(len(parts[1]), 32)
        self.assertLessEqual(len(parts[2]), 32)


if __name__ == "__main__":
    unittest.main()
