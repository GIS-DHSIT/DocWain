"""Verify pipeline processes all extracted documents through DI."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_stage_identify_processes_all_docs():
    """DI pipeline must understand ALL extracted documents, not just the first."""
    from src.teams.pipeline import TeamsDocumentPipeline

    storage = MagicMock()
    storage.create_document = MagicMock()
    storage.store_extraction_result = MagicMock()

    pipeline = TeamsDocumentPipeline(
        storage=storage,
        state_store=MagicMock(),
    )

    understood_texts = []

    def mock_identify(*, extracted, filename, llm_client):
        return {"document_type": "invoice", "confidence": 0.9}

    def mock_content_map(extracted):
        return {}

    def mock_understand(*, extracted, doc_type, llm_client):
        text = extracted if isinstance(extracted, str) else getattr(extracted, "full_text", "")
        understood_texts.append(text[:50] if text else "empty")
        return {
            "document_summary": f"Summary of {text[:20]}",
            "key_entities": [{"type": "AMOUNT", "text": "$100"}],
            "key_facts": [{"fact": "test fact", "evidence_pointer": "p1"}],
            "intent_tags": ["billing"],
        }

    context = MagicMock()
    context.subscription_id = "test-sub"
    context.profile_id = "test-profile"

    extracted_docs = {
        "doc1.pdf": "First document content about billing",
        "doc2.pdf": "Second document content about shipping",
        "doc3.pdf": "Third document content about returns",
    }

    def mock_file_processor(file_bytes, filename):
        return extracted_docs

    with patch("src.doc_understanding.identify.identify_document", mock_identify), \
         patch("src.doc_understanding.content_map.build_content_map", mock_content_map), \
         patch("src.doc_understanding.understand.understand_document", mock_understand), \
         patch("src.llm.gateway.get_llm_gateway", return_value=MagicMock()), \
         patch("src.api.dataHandler.fileProcessor", mock_file_processor):

        result = await pipeline.stage_identify(
            file_bytes=b"fake",
            filename="archive.zip",
            content_type="application/zip",
            context=context,
            correlation_id="test-123",
        )

    assert result is not None
    assert len(understood_texts) >= 3, (
        f"Only {len(understood_texts)} docs understood, expected 3. "
        "Pipeline is only processing first extracted document."
    )
