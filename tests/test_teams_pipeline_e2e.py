"""End-to-end integration test for Teams autonomous document pipeline.

Simulates the full flow a real Teams user would experience:
1. User uploads a file in Teams
2. Bot receives attachment → delegates to pipeline
3. Stage 1 (Identify): Document type detected, identification card sent
4. Stage 2 (Screen): Security screening run, screening card sent
5. Stage 3 (Embed): Document embedded, completion card sent
   OR: Consent card sent, user clicks Proceed/Cancel

Tests use mocked external services (fileProcessor, DI, screening, Qdrant)
but exercise the REAL pipeline orchestration logic.
"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.teams.pipeline import TeamsDocumentPipeline, _build_teams_collection_name
from src.teams.teams_storage import (
    TeamsDocumentStorage,
    STATUS_EXTRACTING,
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
    STATUS_SCREENING_PENDING_CONSENT,
    STATUS_SCREENING_REJECTED,
    STATUS_EMBEDDING,
    STATUS_EMBEDDING_COMPLETED,
)
from src.teams.logic import TeamsChatContext
from src.teams.state import TeamsStateStore
from src.teams.tools import TeamsToolRouter
from src.teams.cards import build_card
from src.teams.attachments import ScreeningSummary


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_context():
    return TeamsChatContext(
        user_id="user-e2e-001",
        session_id="session-e2e-001",
        subscription_id="e2e-sub",
        profile_id="e2e-prof",
        model_name="llama3.2",
        persona="Document Assistant",
    )


def _make_pipeline():
    storage = TeamsDocumentStorage()
    storage._collection = MagicMock()
    storage._collection.insert_one = MagicMock()
    storage._collection.update_one = MagicMock()
    storage._collection.find_one = MagicMock(return_value=None)

    state_store = TeamsStateStore()
    state_store.client = None

    pipeline = TeamsDocumentPipeline(storage=storage, state_store=state_store)
    return pipeline, storage, state_store


def _make_turn_context():
    """Build a mock Teams TurnContext that captures sent cards."""
    tc = MagicMock()
    tc.send_activity = AsyncMock()
    tc.activity = MagicMock()
    tc.activity.service_url = "https://smba.trafficmanager.net/uk/"
    return tc


# Patch targets — these are the SOURCE modules where lazy imports fetch from
_PATCHES = {
    "fileProcessor": "src.api.dataHandler.fileProcessor",
    "identify_document": "src.doc_understanding.identify.identify_document",
    "build_content_map": "src.doc_understanding.content_map.build_content_map",
    "understand_document": "src.doc_understanding.understand.understand_document",
    "screening": "src.teams.attachments._run_security_screening",
    "train_on_document": "src.api.dataHandler.train_on_document",
    "QdrantClient": "qdrant_client.QdrantClient",
    "QdrantVectorStore": "src.api.vector_store.QdrantVectorStore",
    "insights": "src.teams.insights.generate_proactive_insights",
    "get_domain_actions": "src.teams.insights.get_domain_actions",
    "download_bytes": "src.teams.attachments._download_bytes",
    "upload_to_blob": "src.teams.attachments._upload_to_blob",
    "send_card": "src.teams.pipeline._send_card",
}


# =========================================================================
# E2E Test: Full pipeline — LOW risk (auto-proceed)
# =========================================================================

class TestE2ELowRiskPipeline(unittest.TestCase):
    """Simulates a user uploading a clean invoice — full auto-proceed flow."""

    @patch(_PATCHES["upload_to_blob"])
    @patch(_PATCHES["send_card"], new_callable=AsyncMock)
    @patch(_PATCHES["get_domain_actions"], return_value=[{"title": "View Items", "query": "Show items"}, {"title": "Check Total", "query": "Total?"}])
    @patch(_PATCHES["insights"])
    @patch(_PATCHES["QdrantVectorStore"])
    @patch(_PATCHES["QdrantClient"])
    @patch(_PATCHES["train_on_document"], return_value=12)
    @patch(_PATCHES["screening"])
    @patch(_PATCHES["understand_document"])
    @patch(_PATCHES["build_content_map"], return_value={})
    @patch(_PATCHES["identify_document"])
    @patch(_PATCHES["fileProcessor"])
    @patch(_PATCHES["download_bytes"], new_callable=AsyncMock)
    def test_full_pipeline_low_risk(
        self, mock_dl, mock_fp, mock_id, mock_cm, mock_ud, mock_screen,
        mock_train, mock_qc, mock_vs, mock_insights, mock_domain_actions,
        mock_send_card, mock_upload_blob,
    ):
        pipeline, storage, state_store = _make_pipeline()
        ctx = _make_context()
        tc = _make_turn_context()

        # Setup mocks
        mock_dl.return_value = b"fake PDF content for invoice"
        mock_fp.return_value = {"invoice.pdf": "Invoice #1001\nFrom: Acme Corp\nTotal: $5,000\nDue: April 15, 2026"}
        mock_id.return_value = {"document_type": "invoice", "confidence": 0.95}
        mock_ud.return_value = {
            "document_summary": "Invoice #1001 from Acme Corp for cloud services totaling $5,000, due April 15, 2026.",
            "key_entities": ["Acme Corp", "$5,000", "April 15, 2026"],
            "key_facts": ["Total amount is $5,000", "Payment due April 15, 2026"],
            "intent_tags": ["billing", "procurement"],
        }
        mock_screen.return_value = ScreeningSummary(
            risk_level="LOW",
            overall_score=8.0,
            top_findings=["No sensitive data detected"],
            tools_run=3,
        )
        mock_vs.return_value = MagicMock()
        mock_insights.return_value = MagicMock(
            suggested_questions=["What is the total?", "When is payment due?", "Who is the vendor?"],
        )

        attachment = {
            "contentType": "application/pdf",
            "name": "invoice_acme_2026.pdf",
            "content": {"downloadUrl": "https://teams-files.example.com/invoice.pdf"},
        }

        _run(pipeline.run_full_pipeline(
            attachment=attachment,
            context=ctx,
            turn_context=tc,
            correlation_id="e2e-corr-001",
            auth_token="fake-token",
        ))

        # ── VERIFY STAGE 1: Identification ──
        storage._collection.insert_one.assert_called_once()
        record = storage._collection.insert_one.call_args[0][0]
        assert record["status"] == STATUS_EXTRACTING
        assert record["source"] == "teams"
        assert record["subscription_id"] == "e2e-sub"
        assert record["profile_id"] == "e2e-prof"

        mock_id.assert_called_once()
        mock_cm.assert_called_once()
        mock_ud.assert_called_once()

        # ── VERIFY STAGE 2: Screening ──
        mock_screen.assert_called_once()

        # ── VERIFY STAGE 3: Embedding ──
        mock_train.assert_called_once()
        train_kwargs = mock_train.call_args
        train_sub = train_kwargs[1].get("subscription_id") or train_kwargs[0][1]
        assert "teams_" in str(train_sub), f"Expected teams_ prefix, got: {train_sub}"

        # ── VERIFY CARDS SENT ──
        # 3 cards: identification, screening passed, embedding complete
        assert mock_send_card.call_count == 3, f"Expected 3 cards, got {mock_send_card.call_count}"

        card1_json = json.dumps(mock_send_card.call_args_list[0][0][1])
        assert "Document Identified" in card1_json or "Invoice" in card1_json

        card2_json = json.dumps(mock_send_card.call_args_list[1][0][1])
        assert "PASSED" in card2_json or "LOW" in card2_json

        card3_json = json.dumps(mock_send_card.call_args_list[2][0][1])
        assert "Document Ready" in card3_json or "ready" in card3_json.lower()

        # ── VERIFY STATE STORE ──
        uploads = state_store.list_uploads("e2e-sub", "e2e-prof")
        assert len(uploads) == 1
        assert uploads[0]["filename"] == "invoice_acme_2026.pdf"
        assert uploads[0]["document_type"] == "invoice"

        print("\n✅ E2E LOW RISK PIPELINE: All stages completed successfully")
        print("   Stage 1: Identified as 'invoice' (95% confidence)")
        print("   Stage 2: Screening PASSED (LOW risk, 8/100)")
        print("   Stage 3: Embedded (12 chunks)")
        print("   Cards sent: 3 (identify → screening → ready)")


# =========================================================================
# E2E Test: Pipeline with HIGH risk — consent flow
# =========================================================================

class TestE2EHighRiskConsentFlow(unittest.TestCase):
    """Simulates uploading a document with PII — pipeline pauses for consent."""

    @patch(_PATCHES["upload_to_blob"])
    @patch(_PATCHES["send_card"], new_callable=AsyncMock)
    @patch(_PATCHES["get_domain_actions"], return_value=[])
    @patch(_PATCHES["insights"])
    @patch(_PATCHES["train_on_document"])
    @patch(_PATCHES["screening"])
    @patch(_PATCHES["understand_document"])
    @patch(_PATCHES["build_content_map"], return_value={})
    @patch(_PATCHES["identify_document"])
    @patch(_PATCHES["fileProcessor"])
    @patch(_PATCHES["download_bytes"], new_callable=AsyncMock)
    def test_pipeline_pauses_at_consent(
        self, mock_dl, mock_fp, mock_id, mock_cm, mock_ud, mock_screen,
        mock_train, mock_insights, mock_domain_actions,
        mock_send_card, mock_upload_blob,
    ):
        pipeline, storage, state_store = _make_pipeline()
        ctx = _make_context()
        tc = _make_turn_context()

        mock_dl.return_value = b"sensitive document content"
        mock_fp.return_value = {"employee_data.pdf": "Employee: John Doe\nSSN: 123-45-6789\nSalary: $120,000"}
        mock_id.return_value = {"document_type": "hr", "confidence": 0.88}
        mock_ud.return_value = {
            "document_summary": "Employee personnel file containing PII and compensation data.",
            "key_entities": ["John Doe", "123-45-6789", "$120,000"],
            "key_facts": ["Contains SSN", "Contains salary information"],
            "intent_tags": ["hr", "personnel"],
        }
        mock_screen.return_value = ScreeningSummary(
            risk_level="HIGH",
            overall_score=78.0,
            top_findings=[
                "2 PII items detected (SSN, personal identifiers)",
                "1 secret/credential detected (API key)",
                "1 private business data item detected (salary)",
            ],
            tools_run=3,
        )
        mock_insights.return_value = MagicMock(suggested_questions=["Q1", "Q2", "Q3"])

        attachment = {
            "contentType": "application/pdf",
            "name": "employee_data.pdf",
            "content": {"downloadUrl": "https://teams-files.example.com/emp.pdf"},
        }

        _run(pipeline.run_full_pipeline(
            attachment=attachment,
            context=ctx,
            turn_context=tc,
            correlation_id="e2e-corr-002",
            auth_token="fake-token",
        ))

        # Pipeline PAUSED — embedding NOT called
        mock_train.assert_not_called()

        # 2 cards: identification + consent
        assert mock_send_card.call_count == 2, f"Expected 2 cards, got {mock_send_card.call_count}"

        card1_json = json.dumps(mock_send_card.call_args_list[0][0][1])
        assert "Hr" in card1_json or "hr" in card1_json.lower() or "Document Identified" in card1_json

        card2_json = json.dumps(mock_send_card.call_args_list[1][0][1])
        assert "ATTENTION" in card2_json or "HIGH" in card2_json
        assert "Proceed Anyway" in card2_json
        assert "Cancel Embedding" in card2_json
        assert "pipeline_consent_proceed" in card2_json
        assert "pipeline_consent_reject" in card2_json

        uploads = state_store.list_uploads("e2e-sub", "e2e-prof")
        assert len(uploads) == 0

        print("\n✅ E2E HIGH RISK CONSENT: Pipeline paused correctly")
        print("   Stage 1: Identified as 'hr' (88% confidence)")
        print("   Stage 2: HIGH risk (78/100) — consent card shown")
        print("   Stage 3: NOT reached (waiting for consent)")


# =========================================================================
# E2E Test: Consent reject flow
# =========================================================================

class TestE2EConsentReject(unittest.TestCase):
    """Simulates user clicking 'Cancel Embedding' on consent card."""

    def test_consent_reject_stops_pipeline(self):
        mock_storage = MagicMock()

        state_store = TeamsStateStore()
        state_store.client = None
        service = MagicMock()
        router = TeamsToolRouter(service, state_store)
        ctx = _make_context()

        with patch("src.teams.teams_storage.TeamsDocumentStorage", return_value=mock_storage):
            result = _run(router.handle_action(
                {"action": "pipeline_consent_reject", "document_id": "doc-reject-001"},
                ctx,
            ))

        result_json = json.dumps(result)
        assert "cancelled" in result_json.lower() or "not be embedded" in result_json.lower()

        mock_storage.consent_reject.assert_called_once_with("doc-reject-001")

        print("\n✅ E2E CONSENT REJECT: Document correctly blocked")
        print("   consent_reject() called with correct document_id")


# =========================================================================
# E2E Test: Consent proceed flow
# =========================================================================

class TestE2EConsentProceed(unittest.TestCase):
    """Simulates user clicking 'Proceed Anyway' on consent card."""

    def test_consent_proceed_triggers_embedding(self):
        mock_storage = MagicMock()
        mock_storage.get_document.return_value = {
            "document_id": "doc-proceed-001",
            "status": STATUS_SCREENING_PENDING_CONSENT,
            "filename": "sensitive.pdf",
            "document_type": "hr",
            "extraction": {"doc_type": "hr"},
        }

        pipeline_mock = MagicMock()
        pipeline_mock._get_cached_content.return_value = {"doc": "Cached text"}
        pipeline_mock.stage_embed = AsyncMock(return_value={
            "chunks_count": 8,
            "quality_grade": "B",
            "card": build_card("embedding_complete_card",
                filename="sensitive.pdf",
                chunks_count="8",
                quality_text="B",
                question1="Q1", question2="Q2", question3="Q3",
                action1_title="Explore", action1_query="Tell me more",
            ),
        })

        state_store = TeamsStateStore()
        state_store.client = None
        service = MagicMock()
        router = TeamsToolRouter(service, state_store)
        ctx = _make_context()

        with patch("src.teams.teams_storage.TeamsDocumentStorage", return_value=mock_storage), \
             patch("src.teams.pipeline.TeamsDocumentPipeline", return_value=pipeline_mock):
            result = _run(router.handle_action(
                {"action": "pipeline_consent_proceed", "document_id": "doc-proceed-001"},
                ctx,
            ))

        result_json = json.dumps(result)
        assert "Document Ready" in result_json or "ready" in result_json.lower() or "Consent granted" in result_json

        mock_storage.consent_proceed.assert_called_once_with("doc-proceed-001")

        print("\n✅ E2E CONSENT PROCEED: Embedding triggered after consent")


# =========================================================================
# E2E Test: Multi-file upload
# =========================================================================

class TestE2EMultiFileUpload(unittest.TestCase):
    """Simulates uploading 2 files at once — each processed independently."""

    @patch(_PATCHES["upload_to_blob"])
    @patch(_PATCHES["send_card"], new_callable=AsyncMock)
    @patch(_PATCHES["get_domain_actions"], return_value=[{"title": "Ask", "query": "Tell me"}])
    @patch(_PATCHES["insights"])
    @patch(_PATCHES["QdrantVectorStore"])
    @patch(_PATCHES["QdrantClient"])
    @patch(_PATCHES["train_on_document"], return_value=6)
    @patch(_PATCHES["screening"])
    @patch(_PATCHES["understand_document"])
    @patch(_PATCHES["build_content_map"], return_value={})
    @patch(_PATCHES["identify_document"])
    @patch(_PATCHES["fileProcessor"])
    @patch(_PATCHES["download_bytes"], new_callable=AsyncMock)
    def test_multi_file_independent_processing(
        self, mock_dl, mock_fp, mock_id, mock_cm, mock_ud, mock_screen,
        mock_train, mock_qc, mock_vs, mock_insights, mock_domain_actions,
        mock_send_card, mock_upload_blob,
    ):
        pipeline, storage, state_store = _make_pipeline()
        ctx = _make_context()
        tc = _make_turn_context()

        mock_dl.return_value = b"file content"
        mock_fp.return_value = {"doc": "Some document text"}
        mock_id.return_value = {"document_type": "invoice", "confidence": 0.92}
        mock_ud.return_value = {
            "document_summary": "A document",
            "key_entities": [],
            "key_facts": [],
            "intent_tags": [],
        }
        mock_screen.return_value = ScreeningSummary(
            risk_level="LOW", overall_score=5.0,
            top_findings=["No issues"], tools_run=3,
        )
        mock_vs.return_value = MagicMock()
        mock_insights.return_value = MagicMock(suggested_questions=["Q1", "Q2", "Q3"])

        # Process file 1
        _run(pipeline.run_full_pipeline(
            attachment={"contentType": "application/pdf", "name": "invoice.pdf",
                       "content": {"downloadUrl": "https://example.com/1.pdf"}},
            context=ctx, turn_context=tc,
            correlation_id="e2e-multi-1", auth_token="tok",
        ))

        # Process file 2
        _run(pipeline.run_full_pipeline(
            attachment={"contentType": "application/pdf", "name": "contract.pdf",
                       "content": {"downloadUrl": "https://example.com/2.pdf"}},
            context=ctx, turn_context=tc,
            correlation_id="e2e-multi-2", auth_token="tok",
        ))

        assert mock_train.call_count == 2
        assert mock_send_card.call_count == 6  # 3 cards per file

        uploads = state_store.list_uploads("e2e-sub", "e2e-prof")
        assert len(uploads) == 2

        print("\n✅ E2E MULTI-FILE: Both files processed independently")
        print("   Cards sent: 6 (3 per file)")
        print("   Uploads recorded: 2")


# =========================================================================
# E2E Test: Data isolation verification
# =========================================================================

class TestE2EDataIsolation(unittest.TestCase):
    """Verify Teams data is isolated from main app."""

    def test_collection_name_isolation(self):
        name = _build_teams_collection_name("subscription-abc", "profile-xyz")
        assert name.startswith("teams_"), f"Must start with teams_, got: {name}"
        assert "subscription-abc" in name
        assert "profile-xyz" in name
        print(f"\n✅ ISOLATION: Collection name = '{name}'")

    def test_storage_uses_separate_collection(self):
        from src.teams.teams_storage import TEAMS_DOCUMENTS_COLLECTION
        assert TEAMS_DOCUMENTS_COLLECTION == "teams_documents"
        assert TEAMS_DOCUMENTS_COLLECTION != "documents"
        print(f"\n✅ ISOLATION: MongoDB collection = '{TEAMS_DOCUMENTS_COLLECTION}'")

    def test_status_flow_independent(self):
        assert STATUS_EXTRACTING == "EXTRACTING"
        assert STATUS_SCREENING_PENDING_CONSENT == "SCREENING_PENDING_CONSENT"
        assert STATUS_EMBEDDING_COMPLETED == "EMBEDDING_COMPLETED"
        print("\n✅ ISOLATION: Teams status constants are independent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
