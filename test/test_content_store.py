from pathlib import Path

from src.api.content_store import delete_extracted_pickle, load_extracted_pickle, save_extracted_pickle


def test_pickle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_CONTENT_DIR", str(tmp_path))
    payload = {"text": "hello", "items": [1, 2, 3]}

    info = save_extracted_pickle("doc-1", payload)
    assert Path(info["path"]).exists()
    assert info["sha256"]

    loaded = load_extracted_pickle("doc-1")
    assert loaded == payload

    assert delete_extracted_pickle("doc-1") is True
    assert delete_extracted_pickle("doc-1") is False
