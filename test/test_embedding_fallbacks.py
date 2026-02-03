import pickle

from src.api import embedding_service
from src.api.blob_store import BlobInfo
from src.api.statuses import STATUS_SCREENING_COMPLETED


class FakeBlobStore:
    def __init__(self, payload):
        self.prefix = ""
        self.payload = payload

    def try_acquire_lease(self, *_args, **_kwargs):
        return "lease-id"

    def download_blob(self, *_args, **_kwargs):
        return self.payload

    def delete_blob(self, *_args, **_kwargs):
        return False

    def release_lease(self, *_args, **_kwargs):
        return True


def _noop(*_args, **_kwargs):
    return None


def test_prepare_extracted_docs_incomplete_triggers_fallback(monkeypatch):
    calls = {"fallback": 0}

    monkeypatch.setattr(embedding_service, "_normalize_extracted_docs", lambda payload: {"doc": payload})

    state = {"first": True}

    def _assess(_docs):
        if state["first"]:
            state["first"] = False
            return {"has_data": True, "incomplete": True, "total_chars": 500, "coverage_values": [0.9]}
        return {"has_data": True, "incomplete": False, "total_chars": 800, "coverage_values": [1.0]}

    monkeypatch.setattr(embedding_service, "_assess_extracted_docs", _assess)

    def _fallback(**_kwargs):
        calls["fallback"] += 1
        return {"doc": {"texts": ["a", "b"]}}

    monkeypatch.setattr(embedding_service, "_reextract_from_source", _fallback)
    monkeypatch.setattr(embedding_service, "_screen_payload", lambda *_args, **_kwargs: (2, [1.0]))

    docs, expected, coverage, err = embedding_service._prepare_extracted_docs(
        document_id="doc-1",
        extracted={"texts": ["x"]},
        record={"name": "file.pdf"},
        subscription_id="sub-1",
    )

    assert err is None
    assert expected == 2
    assert coverage == [1.0]
    assert calls["fallback"] == 1
    assert isinstance(docs, dict)


def test_process_blob_no_data_present(monkeypatch):
    payload = pickle.dumps({"texts": []}, protocol=pickle.HIGHEST_PROTOCOL)
    store = FakeBlobStore(payload)
    blob = BlobInfo(name="doc-2.pkl", metadata={"document_id": "doc-2"})

    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda *_args, **_kwargs: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda *_args, **_kwargs: "prof-1")
    monkeypatch.setattr(
        embedding_service,
        "get_document_record",
        lambda *_args, **_kwargs: {"status": STATUS_SCREENING_COMPLETED},
    )
    monkeypatch.setattr(embedding_service, "update_stage", _noop)
    monkeypatch.setattr(embedding_service, "update_document_fields", _noop)
    monkeypatch.setattr(embedding_service, "_count_qdrant_points", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        embedding_service,
        "_verify_post_upsert_count",
        lambda **kwargs: (int(kwargs.get("expected_chunks", 0)), True),
    )
    monkeypatch.setattr(
        embedding_service,
        "_prepare_extracted_docs",
        lambda **_kwargs: (None, None, [], "no_data_present"),
    )

    result = embedding_service._process_blob(store=store, blob=blob, subscription_id=None, profile_id=None, doc_type=None)

    assert result["status"] == "FAILED"
    assert result["error"] == "no_data_present"


def test_process_blob_meta_error_retries_cpu(monkeypatch):
    payload = pickle.dumps({"texts": ["chunk-1"]}, protocol=pickle.HIGHEST_PROTOCOL)
    store = FakeBlobStore(payload)
    blob = BlobInfo(name="doc-3.pkl", metadata={"document_id": "doc-3"})

    monkeypatch.setattr(embedding_service, "resolve_subscription_id", lambda *_args, **_kwargs: "sub-1")
    monkeypatch.setattr(embedding_service, "resolve_profile_id", lambda *_args, **_kwargs: "prof-1")
    monkeypatch.setattr(
        embedding_service,
        "get_document_record",
        lambda *_args, **_kwargs: {"status": STATUS_SCREENING_COMPLETED},
    )
    monkeypatch.setattr(embedding_service, "update_stage", _noop)
    monkeypatch.setattr(embedding_service, "update_document_fields", _noop)
    monkeypatch.setattr(embedding_service, "_count_qdrant_points", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        embedding_service,
        "_verify_post_upsert_count",
        lambda **kwargs: (int(kwargs.get("expected_chunks", 0)), True),
    )
    monkeypatch.setattr(
        embedding_service,
        "_prepare_extracted_docs",
        lambda **_kwargs: ({"file.pdf": {"texts": ["a", "b"]}}, 2, [1.0], None),
    )

    calls = {"train": 0, "cpu_reload": 0}

    def _train(*_args, **_kwargs):
        calls["train"] += 1
        if calls["train"] == 1:
            raise RuntimeError("Cannot copy out of meta tensor; no data!")
        return {"chunks": 2, "points_saved": 2, "dropped_chunks": 0}

    def _get_model(*_args, **kwargs):
        if kwargs.get("reload") and kwargs.get("device") == "cpu":
            calls["cpu_reload"] += 1
        return object()

    monkeypatch.setattr(embedding_service, "train_on_document", _train)
    monkeypatch.setattr(embedding_service, "get_model", _get_model)

    result = embedding_service._process_blob(store=store, blob=blob, subscription_id=None, profile_id=None, doc_type=None)

    assert result["status"] == "COMPLETED"
    assert calls["cpu_reload"] == 1
    assert calls["train"] >= 2
