"""Tests for the correlation ID middleware with latency tracking."""

import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.middleware.correlation import CorrelationIdMiddleware


@pytest.fixture
def app():
    """Create a FastAPI app with the correlation middleware."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ok")
    async def ok_endpoint():
        return {"status": "ok"}

    @app.get("/error")
    async def error_endpoint():
        raise RuntimeError("boom")

    return app


@pytest.fixture
def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_response_has_correlation_id_header(client):
    """Response should include an X-Correlation-Id header."""
    response = await client.get("/ok")
    assert response.status_code == 200
    assert "x-correlation-id" in response.headers
    # Should be a non-empty string (UUID format)
    assert len(response.headers["x-correlation-id"]) > 0


@pytest.mark.asyncio
async def test_response_has_response_time_header(client):
    """Response should include an X-Response-Time-Ms header with a valid float."""
    response = await client.get("/ok")
    assert response.status_code == 200
    assert "x-response-time-ms" in response.headers
    duration = float(response.headers["x-response-time-ms"])
    assert duration >= 0


@pytest.mark.asyncio
async def test_completion_log_includes_latency(client, caplog):
    """The completion log message should include latency in ms."""
    with caplog.at_level(logging.INFO, logger="src.middleware.correlation"):
        response = await client.get("/ok")
        assert response.status_code == 200

    completion_logs = [
        r for r in caplog.records if "Request completed" in r.message
    ]
    assert len(completion_logs) >= 1
    log_msg = completion_logs[0].message
    assert "ms)" in log_msg


@pytest.mark.asyncio
async def test_error_log_includes_latency(client, caplog):
    """The error log message should include latency when the request fails."""
    with caplog.at_level(logging.ERROR, logger="src.middleware.correlation"):
        with pytest.raises(RuntimeError):
            await client.get("/error")

    error_logs = [
        r for r in caplog.records if "Request failed" in r.message
    ]
    assert len(error_logs) >= 1
    log_msg = error_logs[0].message
    assert "ms)" in log_msg


@pytest.mark.asyncio
async def test_provided_correlation_id_is_preserved(client):
    """When the client sends X-Correlation-Id, it should be echoed back."""
    custom_id = "test-trace-12345"
    response = await client.get(
        "/ok", headers={"X-Correlation-Id": custom_id}
    )
    assert response.status_code == 200
    assert response.headers["x-correlation-id"] == custom_id
