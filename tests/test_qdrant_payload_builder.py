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

    # Core identity fields
    assert payload["subscription_id"] == "67fde0754e36c00b14cea7f5"
    assert payload["profile_id"] == "6982be3d2743a92f89fb6951"
    assert payload["document_id"] == "6982c08f2743a92f89fb6a08"

    # Source (flat only — nested objects removed in embedding rebuild)
    assert payload["source_name"] == "SuryaGarisetti_Resume.pdf"
    assert "source" not in payload  # nested source dict removed

    # Domain & section — classifier enriches generic values
    assert payload["doc_domain"] == "resume"  # inferred from filename "SuryaGarisetti_Resume.pdf"
    assert payload["section_id"] == "9c8776671818"
    assert "section" not in payload  # nested section dict removed
    assert payload["section_kind"] == "experience"  # "PROJECTS" title → experience
    assert payload["section_title"] == "PROJECTS"

    # Location
    assert payload["page"] == 2
    assert payload["chunk_index"] == 20

    # Chunk identity (flat only — nested objects removed in embedding rebuild)
    assert payload["chunk_id"] == "chunk_31c083a6d5c0f2e7e8d9b35a97ceb64c77906fc00749fedd0086c99bd4282fe1"
    assert "chunk" not in payload  # nested chunk dict removed
    assert payload["chunk_kind"] == "fact"

    # Integrity
    assert payload["hash"] == "5dab651c65be633db1a5f7381dad86267b94ba7a3b0f6951ca00eaa207fb9038"

    # Text fields — canonical_text stays clean, embedding_text gets section prefix
    assert "JPMorganChase" in payload["canonical_text"]  # camelCase preserved (no forced split)
    assert "2018-Present" in payload["canonical_text"]
    assert not payload["canonical_text"].startswith("[")  # canonical stays clean
    assert payload["embedding_text"]
    assert payload["embedding_text"].startswith("[Experience] PROJECTS:")
    assert payload["embedding_text"] != payload["canonical_text"]

    # Pipeline version
    assert payload["embed_pipeline_version"] == EMBED_PIPELINE_VERSION

    # Slim payload should NOT contain bloated fields
    assert "content" not in payload
    assert "connector_type" not in payload
    assert "file_type" not in payload
    assert "canonical_text_len" not in payload
    assert "canonical_token_count" not in payload
    assert "provenance" not in payload
    assert "document" not in payload
    assert "detected_language" not in payload
