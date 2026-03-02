from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api import dw_newron
from src.execution.common import chunk_text_stream


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):  # noqa: ANN001
        return self.store.get(key)

    def setex(self, key, ttl, value):  # noqa: ANN001, ARG002
        self.store[key] = value


class Fake429(Exception):
    def __init__(self, retry_after="1"):
        super().__init__("429 Too Many Requests")
        self.code = 429
        self.status = 429
        self.response = SimpleNamespace(headers={"Retry-After": retry_after})


class FakeOllama:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):  # noqa: ANN001, ANN002
        self.calls += 1
        return "local-fallback-response"

    def generate_with_metadata(self, *args, **kwargs):  # noqa: ANN001, ANN002
        self.calls += 1
        return "local-fallback-response", {"response": "local-fallback-response"}


class DummyMetrics:
    available = False

    def record(self, *args, **kwargs):  # noqa: ANN001, ANN002
        return None


def test_gemini_429_retries_and_fallback(monkeypatch):
    calls = {"count": 0}
    sleeps = []
    fake_redis = FakeRedis()

    def fake_generate_text(*args, **kwargs):  # noqa: ANN001, ANN002
        calls["count"] += 1
        raise Fake429("1")

    def fake_sleep(duration):  # noqa: ANN001
        sleeps.append(duration)

    monkeypatch.setattr(dw_newron, "generate_text", fake_generate_text)
    monkeypatch.setattr(dw_newron, "configure_gemini", lambda: "test-key")
    monkeypatch.setattr(dw_newron, "get_redis_client", lambda: fake_redis)
    monkeypatch.setattr(dw_newron, "get_metrics_store", lambda: DummyMetrics())
    monkeypatch.setattr(dw_newron.time, "sleep", fake_sleep)

    gemini = dw_newron.GeminiClient(model_name="gemini-2.5-flash")
    fallback = FakeOllama()
    client = dw_newron.ResilientLLMClient(gemini, fallback)

    result = client.generate("hello", max_retries=2, backoff=0.1)
    assert result == "local-fallback-response"
    assert fallback.calls == 1
    assert calls["count"] >= 1
    assert sleeps


def test_gemini_cooldown_skips_primary(monkeypatch):
    calls = {"count": 0}
    fake_redis = FakeRedis()

    def fake_generate_text(*args, **kwargs):  # noqa: ANN001, ANN002
        calls["count"] += 1
        raise Fake429("0")

    monkeypatch.setattr(dw_newron, "generate_text", fake_generate_text)
    monkeypatch.setattr(dw_newron, "configure_gemini", lambda: "test-key")
    monkeypatch.setattr(dw_newron, "get_redis_client", lambda: fake_redis)
    monkeypatch.setattr(dw_newron, "get_metrics_store", lambda: DummyMetrics())

    gemini = dw_newron.GeminiClient(model_name="gemini-2.5-flash")
    fallback = FakeOllama()
    client = dw_newron.ResilientLLMClient(gemini, fallback)

    with pytest.raises(Exception):
        gemini.generate("first", max_retries=1, backoff=0.0)
    with pytest.raises(Exception):
        gemini.generate("second", max_retries=1, backoff=0.0)

    assert gemini.in_cooldown()
    cooldown_key = gemini._cooldown_key
    assert fake_redis.get(cooldown_key)

    result = client.generate("third", max_retries=1, backoff=0.0)
    assert result == "local-fallback-response"
    assert fallback.calls >= 1
    assert calls["count"] >= 2


def test_streaming_uses_fallback_on_rate_limit(monkeypatch):
    def fake_generate_text(*args, **kwargs):  # noqa: ANN001, ANN002
        raise Fake429("0")

    monkeypatch.setattr(dw_newron, "generate_text", fake_generate_text)
    monkeypatch.setattr(dw_newron, "configure_gemini", lambda: "test-key")
    monkeypatch.setattr(dw_newron, "get_redis_client", lambda: FakeRedis())
    monkeypatch.setattr(dw_newron, "get_metrics_store", lambda: DummyMetrics())

    gemini = dw_newron.GeminiClient(model_name="gemini-2.5-flash")
    fallback = FakeOllama()
    client = dw_newron.ResilientLLMClient(gemini, fallback)

    text = client.generate("stream this", max_retries=1, backoff=0.0)
    streamed = "".join(chunk_text_stream(text, chunk_size=5))
    assert streamed == "local-fallback-response"
