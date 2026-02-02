from src.api import dataHandler


def test_encode_with_fallback_recovers_from_meta_tensor(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, device: str):
            self.device = device

        def encode(self, texts, **kwargs):
            calls.append({"device": self.device, "batch_size": kwargs.get("batch_size")})
            if self.device != "cpu":
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            return [f"vec:{t}" for t in texts]

    state = {"device": "cuda"}

    def fake_get_model(*, reload: bool = False, device=None):
        if reload and device:
            state["device"] = str(device)
        target = str(device or state["device"])
        state["device"] = target
        return FakeModel(target)

    monkeypatch.setattr(dataHandler, "get_model", fake_get_model)
    monkeypatch.setattr(dataHandler, "_MODEL_DEVICE", "cuda")

    out = dataHandler.encode_with_fallback(["a", "b"], normalize_embeddings=True)

    assert out == ["vec:a", "vec:b"]
    assert calls[0]["device"] == "cuda"
    assert calls[1]["device"] == "cpu"
    assert calls[1]["batch_size"] <= calls[0]["batch_size"]


def test_load_sentence_transformer_retries_without_model_kwargs(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name, device=None, model_kwargs=None):
            calls.append({"device": device, "model_kwargs": model_kwargs})
            if device == "cuda":
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            if device == "cpu" and model_kwargs:
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            self._target_device = device

    monkeypatch.setattr(dataHandler, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(dataHandler, "_model_kwargs_for_device", lambda _device: {"torch_dtype": object()})

    model = dataHandler._load_sentence_transformer("fake-model", "cuda")

    assert model._target_device == "cpu"
    assert calls[0]["device"] == "cuda"
    assert calls[0]["model_kwargs"] is not None
    assert calls[1]["device"] == "cpu"
    assert calls[1]["model_kwargs"] is not None
    assert calls[2]["device"] == "cpu"
    assert calls[2]["model_kwargs"] is None
