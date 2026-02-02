from src.api import dataHandler


def test_screening_does_not_trigger_embed(monkeypatch):
    called = {"train": 0}

    def _fake_file_processor(file_bytes, filename):
        return {filename: "Sample text."}

    def _fake_save_extracted_pickle(document_id, extracted_doc):
        return {"path": f"/tmp/{document_id}.pkl", "blob_name": f"{document_id}.pkl", "sha256": "abc"}

    def _fake_update_extraction_metadata(document_id, subscription_id, path, sha256):
        return None

    def _fake_run_security_screening(document_id):
        return {"overall_risk_level": "LOW"}

    def _fake_update_security_screening(document_id, report, status):
        return None

    def _fake_get_subscription_pii_setting(subscription_id):
        return False

    def _fake_update_pii_stats(document_id, count, high_conf, items):
        return None

    def _fake_train_on_document(*args, **kwargs):
        called["train"] += 1
        return {"chunks": 1, "points_saved": 1}

    monkeypatch.setattr(dataHandler, "fileProcessor", _fake_file_processor)
    monkeypatch.setattr(dataHandler, "save_extracted_pickle", _fake_save_extracted_pickle)
    monkeypatch.setattr(dataHandler, "update_extraction_metadata", _fake_update_extraction_metadata)
    monkeypatch.setattr(dataHandler, "run_security_screening", _fake_run_security_screening)
    monkeypatch.setattr(dataHandler, "update_security_screening", _fake_update_security_screening)
    monkeypatch.setattr(dataHandler, "get_subscription_pii_setting", _fake_get_subscription_pii_setting)
    monkeypatch.setattr(dataHandler, "update_pii_stats", _fake_update_pii_stats)
    monkeypatch.setattr(dataHandler, "resolve_subscription_id", lambda doc_id, sub=None: sub or "sub-1")
    monkeypatch.setattr(dataHandler, "resolve_profile_id", lambda doc_id, prof=None: prof or "profile-1")
    monkeypatch.setattr(dataHandler, "train_on_document", _fake_train_on_document)

    result = dataHandler.process_document_pipeline(
        document_id="doc-1",
        file_bytes=b"content",
        filename="doc.txt",
        subscription_id="sub-1",
        profile_id="profile-1",
        embed_after=False,
    )

    assert result["embedding"]["status"] == "pending"
    assert called["train"] == 0
