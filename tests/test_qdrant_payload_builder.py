from src.embedding.pipeline.payload_normalizer import build_qdrant_payload
from src.embedding.pipeline.schema_normalizer import EMBED_PIPELINE_VERSION


def test_build_qdrant_payload_normalizes_fields():
    raw = {
        "subscription_id": "67fde0754e36c00b14cea7f5",
        "profile_id": "6982be3d2743a92f89fb6951",
        "document_id": "6982c08f2743a92f89fb6a08",
        "text": "JPMorganChase&Co.—VicePresident(April2018-Present)\n"
        "● Managingateamof15Developersacrossdifferentgloballocations.",
        "text_data": {
            "clean": "JPMorganChase&Co.—VicePresident(April2018-Present)\n"
            "● Managingateamof15Developersacrossdifferentgloballocations."
        },
        "source": {"name": "SuryaGarisetti_Resume.pdf"},
        "document": {"type": "report", "ingestion_source": "LOCAL"},
        "section": {"id": "9c8776671818", "title": "PROJECTS", "path": ["PROJECTS"]},
        "chunk": {
            "id": "chunk_31c083a6d5c0f2e7e8d9b35a97ceb64c77906fc00749fedd0086c99bd4282fe1",
            "index": 20,
            "count": 24,
            "type": "fact",
            "role": "structured_field",
            "hash": "5dab651c65be633db1a5f7381dad86267b94ba7a3b0f6951ca00eaa207fb9038",
        },
        "section_title": "PROJECTS",
        "section_path": "PROJECTS",
        "page_start": 2,
        "page_end": 2,
        "page": 2,
    }

    payload = build_qdrant_payload(raw)

    assert payload["subscription_id"] == "67fde0754e36c00b14cea7f5"
    assert payload["profile_id"] == "6982be3d2743a92f89fb6951"
    assert payload["document_id"] == "6982c08f2743a92f89fb6a08"
    assert payload["source_name"] == "SuryaGarisetti_Resume.pdf"
    assert payload["source"]["name"] == "SuryaGarisetti_Resume.pdf"
    assert payload["connector_type"] == "LOCAL"
    assert payload["file_type"] == "pdf"
    assert payload["doc_domain"] == "unknown"
    assert payload["section_id"] == "9c8776671818"
    assert payload["section"]["id"] == "9c8776671818"
    assert payload["section_title"] == "PROJECTS"
    assert payload["section_path"] == ["PROJECTS"]
    assert payload["section_kind"] == "misc"
    assert payload["page"] == 2
    assert payload["chunk_id"] == "chunk_31c083a6d5c0f2e7e8d9b35a97ceb64c77906fc00749fedd0086c99bd4282fe1"
    assert payload["chunk"]["id"] == "chunk_31c083a6d5c0f2e7e8d9b35a97ceb64c77906fc00749fedd0086c99bd4282fe1"
    assert payload["chunk_index"] == 20
    assert payload["chunk_count"] == 24
    assert payload["chunk_kind"] == "fact"
    assert payload.get("chunking_mode") is None
    assert payload["hash"] == "5dab651c65be633db1a5f7381dad86267b94ba7a3b0f6951ca00eaa207fb9038"
    assert payload["content"].startswith("JPMorgan Chase & Co. — Vice President (April 2018 - Present)")
    assert payload["canonical_text"] == payload["content"]
    assert payload["canonical_text_len"] == len(payload["canonical_text"])
    assert payload["canonical_token_count"] > 0
    assert payload["embedding_text"]
    assert payload["embedding_text"] != payload["content"]
    assert payload["doc_version_hash"]
    assert len(payload["doc_version_hash"]) == 12
    assert payload["embed_pipeline_version"] == EMBED_PIPELINE_VERSION
