from src.api import embedding_service


def test_embedding_integrity_report_matches(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "get_document_record",
        lambda *_args, **_kwargs: {
            "status": "SCREENING_COMPLETED",
            "subscription_id": "sub-1",
            "profile_id": "prof-1",
        },
    )
    monkeypatch.setattr(
        embedding_service,
        "_load_extracted_for_doc",
        lambda *_args, **_kwargs: ({"texts": ["a", "b", "c"]}, {"source": "blob"}),
    )
    monkeypatch.setattr(embedding_service, "_normalize_extracted_docs", lambda payload: {"document": payload})
    monkeypatch.setattr(embedding_service, "_screen_payload", lambda *_args, **_kwargs: (3, [1.0]))
    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda *_args, **_kwargs: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda *_args, **_kwargs: "prof-1")
    monkeypatch.setattr(embedding_service, "_count_qdrant_points", lambda *_args, **_kwargs: 3)

    report = embedding_service.embedding_integrity_report(document_id="doc-1")
    assert report["summary"]["matched"] == 1
    entry = report["documents"][0]
    assert entry["expected_chunks"] == 3
    assert entry["qdrant_count"] == 3
    assert entry["matches"] is True


def test_embedding_integrity_report_stage_fallback(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "get_document_record",
        lambda *_args, **_kwargs: {
            "status": "TRAINING_COMPLETED",
            "subscription_id": "sub-1",
            "profile_id": "prof-1",
            "embedding": {"chunking": {"chunks": 2, "coverage_ratio": 0.9}},
        },
    )
    monkeypatch.setattr(
        embedding_service,
        "_load_extracted_for_doc",
        lambda *_args, **_kwargs: (None, {"source": "missing"}),
    )
    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda *_args, **_kwargs: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda *_args, **_kwargs: "prof-1")
    monkeypatch.setattr(embedding_service, "_count_qdrant_points", lambda *_args, **_kwargs: 1)

    report = embedding_service.embedding_integrity_report(document_id="doc-2")
    entry = report["documents"][0]
    assert entry["extracted"]["source"] == "stage"
    assert entry["expected_chunks"] == 2
    assert entry["qdrant_count"] == 1
    assert entry["matches"] is False
    assert report["summary"]["mismatched"] == 1
