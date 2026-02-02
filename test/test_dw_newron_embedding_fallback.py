from src.api import dw_newron


def test_load_model_candidates_retries_cpu_without_kwargs(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, _name, device=None, model_kwargs=None, local_files_only=False):
            calls.append({"device": device, "model_kwargs": model_kwargs, "local_files_only": local_files_only})
            if device == "cuda":
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            if device == "cpu" and model_kwargs:
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            self._target_device = device

        def get_sentence_embedding_dimension(self):
            return 768

    monkeypatch.setattr(dw_newron, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(dw_newron, "_embedding_device", lambda: "cuda")
    monkeypatch.setattr(dw_newron, "_model_kwargs_for_device", lambda _device: {"torch_dtype": object()})
    monkeypatch.setattr(dw_newron, "_configure_hf_env", lambda: None)
    monkeypatch.setattr(
        dw_newron.Config.Model,
        "SENTENCE_TRANSFORMERS_CANDIDATES",
        ["fake-model"],
        raising=False,
    )
    monkeypatch.setattr(dw_newron.Config.Model, "DISABLE_HF", False, raising=False)

    model = dw_newron._load_model_candidates()

    assert model._target_device == "cpu"
    assert calls[0]["device"] == "cuda"
    assert calls[0]["model_kwargs"] is not None
    assert calls[1]["device"] == "cpu"
    assert calls[1]["model_kwargs"] is not None
    assert calls[2]["device"] == "cpu"
    assert calls[2]["model_kwargs"] is None
