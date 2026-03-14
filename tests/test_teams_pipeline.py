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


if __name__ == "__main__":
    unittest.main()
