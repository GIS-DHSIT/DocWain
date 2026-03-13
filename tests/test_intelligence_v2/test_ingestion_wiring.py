"""Test that document_understanding_service uses the new DocumentAnalyzer."""
import json
from unittest.mock import MagicMock, patch


def test_run_document_understanding_calls_new_analyzer():
    """After wiring, run_document_understanding should use DocumentAnalyzer."""
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance.analyze.return_value = {
        "document_id": "doc_123",
        "intelligence": {
            "document_type": "invoice",
            "summary": "test summary",
            "entities": [],
            "facts": [],
            "answerable_topics": [],
        },
        "kg_stats": {"entities": 0, "facts": 0, "relationships": 0, "links": 0},
    }

    mock_neo4j_instance = MagicMock()

    from dataclasses import dataclass
    @dataclass
    class FakeIdent:
        document_type: str = "invoice"
        doc_name: str = "file.pdf"
        confidence: float = 0.9
        file_format: str = "pdf"
        page_count: int = 1
        language: str = None
        created_date: str = None

    with patch("src.api.document_understanding_service.load_extracted_pickle") as mock_pickle, \
         patch("src.api.document_understanding_service.update_stage"), \
         patch("src.api.document_understanding_service.update_document_fields"), \
         patch("src.api.document_understanding_service.resolve_profile_name", return_value="Finance"), \
         patch("src.api.document_understanding_service.identify_document", return_value=FakeIdent()), \
         patch("src.api.document_understanding_service.build_content_map", return_value={}), \
         patch("src.api.document_understanding_service._update_metadata"), \
         patch("src.intelligence_v2.analyzer.DocumentAnalyzer", return_value=mock_analyzer_instance), \
         patch("src.llm.gateway.get_llm_gateway", return_value=MagicMock()), \
         patch("pymongo.MongoClient") as mock_mongo, \
         patch("src.kg.neo4j_store.Neo4jStore", return_value=mock_neo4j_instance), \
         patch("src.api.document_understanding_service.embed_documents", return_value=None):

        mock_pickle.return_value = {"file.pdf": {"full_text": "test content", "sections": []}}
        mock_db = MagicMock()
        mock_mongo.return_value.__getitem__.return_value.__getitem__.return_value = mock_db

        from src.api.document_understanding_service import run_document_understanding
        result = run_document_understanding(
            document_id="doc_123",
            subscription_id="sub_1",
            profile_id="prof_1",
        )

        assert mock_analyzer_instance.analyze.called
        call_kwargs = mock_analyzer_instance.analyze.call_args[1]
        assert call_kwargs["document_id"] == "doc_123"
        assert result["understanding"]["summary"] == "test summary"
