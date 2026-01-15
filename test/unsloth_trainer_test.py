import sys
import types
from pathlib import Path

import torch
import pytest

from src.api.config import Config
from src.finetune.models import FinetuneRequest
from src.finetune.unsloth_trainer import UnslothFinetuneManager


class _DummyModel(torch.nn.Module):
    def __init__(self, *, meta: bool = False):
        super().__init__()
        device = torch.device("meta") if meta else torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.zeros(1, device=device))


class _DummyTokenizer:
    pass


def _install_unsloth_stub(monkeypatch, *, fail_with_meta: bool = False, return_meta_first: bool = False):
    attempts = []
    fail_flag = fail_with_meta
    meta_first_flag = return_meta_first

    class _StubFastLanguageModel:
        fail_with_meta = fail_flag
        return_meta_first = meta_first_flag

        @staticmethod
        def from_pretrained(*args, **kwargs):
            attempts.append(kwargs)
            if _StubFastLanguageModel.fail_with_meta:
                _StubFastLanguageModel.fail_with_meta = False
                raise NotImplementedError("Cannot copy out of meta tensor; no data!")
            if _StubFastLanguageModel.return_meta_first:
                _StubFastLanguageModel.return_meta_first = False
                return _DummyModel(meta=True), _DummyTokenizer()
            return _DummyModel(), _DummyTokenizer()

        @staticmethod
        def get_peft_model(model, *args, **kwargs):
            model.peft_ready = True
            return model

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=_StubFastLanguageModel))
    return attempts


def _manager(tmp_path, monkeypatch, *, cuda_available: bool = True):
    monkeypatch.setattr(Config.Path, "APP_HOME", Path(tmp_path), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    if cuda_available:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 1024**3, 12 * 1024**3))
    return UnslothFinetuneManager()


def test_load_model_retries_on_meta_error(monkeypatch, tmp_path):
    attempts = _install_unsloth_stub(monkeypatch, fail_with_meta=True)
    manager = _manager(tmp_path, monkeypatch)
    load_notes = {}
    request = FinetuneRequest(profile_id="p1", output_dir=str(tmp_path))

    model, tokenizer = manager._load_model("dummy-model", request, load_notes)

    assert isinstance(model, _DummyModel)
    assert isinstance(tokenizer, _DummyTokenizer)
    assert load_notes["meta_retry"] is True
    # Initial attempt fails early, so only the safe retry is recorded in notes
    assert len(load_notes["attempts"]) == 1
    assert load_notes["attempts"][0]["mode"] == "safe"
    assert not manager._has_meta_tensors(model)
    assert len(attempts) == 2


def test_load_model_retries_when_meta_weights_detected(monkeypatch, tmp_path):
    attempts = _install_unsloth_stub(monkeypatch, return_meta_first=True)
    manager = _manager(tmp_path, monkeypatch)
    load_notes = {}
    request = FinetuneRequest(profile_id="p2", output_dir=str(tmp_path))

    model, _ = manager._load_model("dummy-model", request, load_notes)

    assert isinstance(model, _DummyModel)
    assert load_notes["meta_retry"] is True
    assert len(load_notes["attempts"]) == 2
    assert load_notes["attempts"][0]["mode"] == "default"
    assert load_notes["attempts"][1]["mode"] == "safe"
    assert not manager._has_meta_tensors(model)
    # First call returned meta tensors, second returned a materialized model
    assert len(attempts) == 2


def test_save_prefers_merged_method(monkeypatch, tmp_path):
    manager = _manager(tmp_path, monkeypatch)
    called = {}

    class _Model:
        def save_pretrained_merged(self, path, tokenizer=None, save_method=None):
            called["path"] = path
            called["tokenizer"] = tokenizer
            called["save_method"] = save_method

    class _Tok:
        pass

    manager._save_model(_Model(), _Tok(), Path(tmp_path) / "merged")
    assert called["path"].endswith("merged")
    assert isinstance(called["tokenizer"], _Tok)
    assert called["save_method"] == "merged_16bit"


def test_save_falls_back_to_class_helper(monkeypatch, tmp_path):
    manager = _manager(tmp_path, monkeypatch)
    called = {}

    class _FLM:
        @staticmethod
        def save_pretrained(model, tokenizer, save_directory=None, save_method=None):
            called["model"] = model
            called["tokenizer"] = tokenizer
            called["save_directory"] = save_directory
            called["save_method"] = save_method

    class _Model:
        pass

    class _Tok:
        pass

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=_FLM))

    manager._save_model(_Model(), _Tok(), Path(tmp_path) / "merged_class")
    assert called["save_directory"].endswith("merged_class")
    assert isinstance(called["model"], _Model)
    assert isinstance(called["tokenizer"], _Tok)
    assert called["save_method"] == "merged_16bit"


def test_save_final_fallback_hf(monkeypatch, tmp_path):
    manager = _manager(tmp_path, monkeypatch)
    model_called = {}
    tok_called = {}

    class _Model:
        def save_pretrained(self, path):
            model_called["path"] = path

    class _Tok:
        def save_pretrained(self, path):
            tok_called["path"] = path

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=object))

    manager._save_model(_Model(), _Tok(), Path(tmp_path) / "merged_hf")
    assert model_called["path"].endswith("merged_hf")
    assert tok_called["path"].endswith("merged_hf")


def test_load_raises_clean_error_on_cpu_only(monkeypatch, tmp_path):
    manager = _manager(tmp_path, monkeypatch, cuda_available=False)
    request = FinetuneRequest(profile_id="cpu", output_dir=str(tmp_path))
    with pytest.raises(RuntimeError) as err:
        manager._load_model("dummy", request, {})
    assert "CUDA is required" in str(err.value)


def test_load_retries_with_offload_on_bnb_dispatch_error(monkeypatch, tmp_path):
    call_kwargs = []

    class _FLM:
        @staticmethod
        def from_pretrained(**kwargs):
            if not call_kwargs:
                call_kwargs.append(kwargs)
                raise ValueError("Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM.")
            call_kwargs.append(kwargs)
            return _DummyModel(), _DummyTokenizer()

        @staticmethod
        def get_peft_model(model, *args, **kwargs):
            return model

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=_FLM))
    manager = _manager(tmp_path, monkeypatch)
    request = FinetuneRequest(profile_id="p3", output_dir=str(tmp_path))

    model, tok = manager._load_model("dummy-model", request, {})

    assert isinstance(model, _DummyModel)
    assert isinstance(tok, _DummyTokenizer)
    assert len(call_kwargs) == 2
    retry_kwargs = call_kwargs[1]
    assert retry_kwargs.get("llm_int8_enable_fp32_cpu_offload") is True
    assert "device_map" in retry_kwargs


def test_load_does_not_retry_on_other_value_errors(monkeypatch, tmp_path):
    class _FLM:
        @staticmethod
        def from_pretrained(**kwargs):
            raise ValueError("some other failure")

        @staticmethod
        def get_peft_model(model, *args, **kwargs):
            return model

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=_FLM))
    manager = _manager(tmp_path, monkeypatch)
    request = FinetuneRequest(profile_id="p4", output_dir=str(tmp_path))

    with pytest.raises(ValueError):
        manager._load_model("dummy", request, {})
