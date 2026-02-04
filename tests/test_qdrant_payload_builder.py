from src.embedding.pipeline.payload_normalizer import build_qdrant_payload


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

    expected = {
        "subscription_id": "67fde0754e36c00b14cea7f5",
        "profile_id": "6982be3d2743a92f89fb6951",
        "document_id": "6982c08f2743a92f89fb6a08",
        "source_name": "SuryaGarisetti_Resume.pdf",
        "document_type": "report",
        "ingestion_source": "LOCAL",
        "section_title": "PROJECTS",
        "section_path": ["PROJECTS"],
        "page": 2,
        "chunk_id": "chunk_31c083a6d5c0f2e7e8d9b35a97ceb64c77906fc00749fedd0086c99bd4282fe1",
        "chunk_index": 20,
        "chunk_count": 24,
        "chunk_role": "structured_field",
        "chunk_kind": "fact",
        "hash": "5dab651c65be633db1a5f7381dad86267b94ba7a3b0f6951ca00eaa207fb9038",
        "content": "JPMorgan Chase & Co. — Vice President (April 2018 - Present)\n"
        "● Managing a team of 15 Developers across different global locations.",
    }

    assert build_qdrant_payload(raw) == expected
