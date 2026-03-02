import pytest

from src.embedding import model_loader


class FakeSentenceTransformer:
    load_calls = 0

    def __init__(self, name, device="cpu", model_kwargs=None, **_kwargs):
        type(self).load_calls += 1
        if name == "bad-model":
            raise RuntimeError("cannot copy out of meta tensor")
        self.name = name
        self.device = device
        self.encode_calls = 0

    def to(self, device):
        self.device = device
        return self

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False, batch_size=1, show_progress_bar=None):
        self.encode_calls += 1
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 3


def test_model_loader_meta_tensor_fallback(monkeypatch):
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(model_loader.Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", ["bad-model"])
    model_loader._MODEL = None
    model_loader._MODEL_NAME = None
    model_loader._MODEL_DEVICE = None
    model_loader._MODEL_DIM = None
    model_loader._MODEL_CACHE = {}
    model_loader._FALLBACK_USED = False

    model, _dim = model_loader.get_embedding_model(reload=True, device="cpu")
    name, dim, device = model_loader.get_model_info()

    assert name == "sentence-transformers/all-mpnet-base-v2"
    assert dim == 3
    assert device == "cpu"
    assert model.encode_calls == 1


def test_model_loader_rejects_meta_tensors(monkeypatch):
    class MetaParam:
        is_meta = True

    class MetaSentenceTransformer(FakeSentenceTransformer):
        def parameters(self):
            return [MetaParam()]

    MetaSentenceTransformer.load_calls = 0
    monkeypatch.setattr(model_loader, "SentenceTransformer", MetaSentenceTransformer)
    monkeypatch.setattr(model_loader.Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", ["meta-model"])
    model_loader._MODEL = None
    model_loader._MODEL_NAME = None
    model_loader._MODEL_DEVICE = None
    model_loader._MODEL_DIM = None
    model_loader._MODEL_CACHE = {}
    model_loader._FALLBACK_USED = True

    with pytest.raises(RuntimeError):
        model_loader.get_embedding_model(reload=True, device="cpu")

    assert MetaSentenceTransformer.load_calls == 1
    assert model_loader._MODEL_CACHE == {}

    with pytest.raises(RuntimeError):
        model_loader.get_embedding_model(reload=True, device="cpu")

    assert MetaSentenceTransformer.load_calls == 2
