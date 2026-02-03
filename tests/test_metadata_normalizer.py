from src.metadata.normalizer import normalize_ingestion_metadata


def test_metadata_conflict_doc_type_local():
    normalized = normalize_ingestion_metadata(
        {
            "document_type": "report",
            "doc_type": "LOCAL",
            "document_id": "doc-1",
            "subscription_id": "sub-1",
            "profile_id": "prof-1",
            "document_name": "report.pdf",
        }
    )

    assert normalized["document_type"] == "report"
    assert normalized["ingestion_source"] == "LOCAL"
    assert normalized["source_type"] == "LOCAL"
    assert normalized["doc_type"] == "LOCAL"
    assert "metadata_warnings" in normalized
