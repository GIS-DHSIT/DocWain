"""Tests for OpenAICompatibleClient per-call options passthrough."""

import json
import io
from unittest.mock import patch, MagicMock

import pytest


def _make_mock_config():
    """Return a mock Config object with LLM defaults."""
    cfg = MagicMock()
    cfg.LLM.TEMPERATURE = 0.0
    cfg.LLM.MAX_TOKENS = 2048
    return cfg


def _build_response(text="hello world"):
    """Build a fake HTTP response body matching OpenAI chat format."""
    body = json.dumps({
        "choices": [{"message": {"content": text}}],
    }).encode("utf-8")
    resp = io.BytesIO(body)
    resp.read = resp.read  # already works
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    return resp


@pytest.fixture()
def _patch_deps():
    """Patch external dependencies so OpenAICompatibleClient can be imported."""
    with patch("src.llm.clients._get_config", return_value=_make_mock_config()), \
         patch("src.llm.clients._get_metrics_store") as mock_metrics:
        mock_metrics.return_value.available = False
        yield


@pytest.fixture()
def client(_patch_deps):
    from src.llm.clients import OpenAICompatibleClient
    return OpenAICompatibleClient(
        model_name="test-model",
        endpoint="http://localhost:9999/v1/chat/completions",
    )


# ── Tests ──────────────────────────────────────────────────────────


class TestGenerateKwargs:
    """generate() honours per-call temperature / max_tokens / top_p."""

    def test_defaults_used_when_no_kwargs(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response()) as mock_open:
            client.generate("hi")
            req = mock_open.call_args[0][0]
            payload = json.loads(req.data.decode())
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] == 2048
            assert "top_p" not in payload

    def test_temperature_override(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response()) as mock_open:
            client.generate("hi", temperature=0.7)
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 2048  # default kept

    def test_max_tokens_override(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response()) as mock_open:
            client.generate("hi", max_tokens=512)
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["max_tokens"] == 512
            assert payload["temperature"] == 0.0  # default kept

    def test_top_p_added_when_provided(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response()) as mock_open:
            client.generate("hi", top_p=0.9)
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["top_p"] == 0.9

    def test_all_overrides_together(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response()) as mock_open:
            client.generate("hi", temperature=0.05, max_tokens=512, top_p=0.95)
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["temperature"] == 0.05
            assert payload["max_tokens"] == 512
            assert payload["top_p"] == 0.95


class TestGenerateWithMetadataOptions:
    """generate_with_metadata() extracts options and forwards them."""

    def test_options_override_payload(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response("answer")) as mock_open:
            text, meta = client.generate_with_metadata(
                "question",
                options={"temperature": 0.05, "max_tokens": 512},
            )
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["temperature"] == 0.05
            assert payload["max_tokens"] == 512
            assert text == "answer"
            assert meta["model"] == "test-model"
            assert meta["backend"] == "vllm"

    def test_defaults_when_no_options(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response("ok")) as mock_open:
            text, meta = client.generate_with_metadata("question")
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] == 2048
            assert text == "ok"

    def test_partial_options_only_override_specified(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response("partial")) as mock_open:
            text, _ = client.generate_with_metadata(
                "question",
                options={"temperature": 0.2},
            )
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["temperature"] == 0.2
            assert payload["max_tokens"] == 2048  # default preserved

    def test_num_predict_alias_for_max_tokens(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response("alias")) as mock_open:
            client.generate_with_metadata(
                "question",
                options={"num_predict": 1024},
            )
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["max_tokens"] == 1024

    def test_top_p_forwarded_via_options(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response("top")) as mock_open:
            client.generate_with_metadata(
                "question",
                options={"top_p": 0.9},
            )
            payload = json.loads(mock_open.call_args[0][0].data.decode())
            assert payload["top_p"] == 0.9

    def test_return_tuple_format(self, client):
        with patch("src.llm.clients.request.urlopen", return_value=_build_response("result")):
            result = client.generate_with_metadata("question")
            assert isinstance(result, tuple)
            assert len(result) == 2
            text, meta = result
            assert isinstance(text, str)
            assert isinstance(meta, dict)
            assert "response" in meta
            assert "model" in meta
            assert "backend" in meta
