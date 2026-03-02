"""Integration tests for the gateway HTTP endpoints.

Uses FastAPI TestClient against the mounted app.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.main import app

client = TestClient(app)


class TestScreenEndpoint:
    """Integration tests for POST /api/gateway/screen."""

    def test_screen_integrity_without_doc_ids(self):
        """Screening without doc_ids returns missing_doc_ids error."""
        response = client.post("/api/gateway/screen", json={
            "category": ["integrity"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "missing_doc_ids"

    def test_screen_batch_without_profile_ids(self):
        """Batch screening without profile_ids returns missing_profile_ids error."""
        response = client.post("/api/gateway/screen", json={
            "category": ["run"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "missing_profile_ids"

    def test_screen_has_correlation_id(self):
        """Response includes custom correlation_id from header."""
        response = client.post(
            "/api/gateway/screen",
            json={"category": ["integrity"]},
            headers={"x-correlation-id": "test-corr-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] == "test-corr-123"

    def test_screen_has_timestamp(self):
        """Response includes ISO 8601 timestamp."""
        response = client.post("/api/gateway/screen", json={"category": ["integrity"]})
        data = response.json()
        assert data["timestamp"]
        assert "T" in data["timestamp"]

    def test_screen_has_duration_ms(self):
        """Response includes duration_ms."""
        response = client.post("/api/gateway/screen", json={"category": ["integrity"]})
        data = response.json()
        assert isinstance(data["duration_ms"], int)
        assert data["duration_ms"] >= 0

    def test_screen_action_field_in_response(self):
        """Response action matches the screening category."""
        response = client.post("/api/gateway/screen", json={
            "category": ["integrity"],
        })
        data = response.json()
        assert data["action"] == "screen:integrity"

    @patch("src.gateway.api._resolve_session_context")
    def test_session_id_resolves_profile(self, mock_resolve):
        """x-session-id header is used to resolve profile_ids for batch screening."""
        mock_resolve.return_value = {
            "profile_id": "resolved-profile-123",
            "subscription_id": "sub-1",
        }

        # This will still fail because _run_parallel_screening needs real DB,
        # but we can verify the resolution was attempted
        response = client.post(
            "/api/gateway/screen",
            json={"category": ["run"]},
            headers={
                "x-session-id": "test-session-abc",
                "x-subscription-id": "sub-1",
            },
        )
        mock_resolve.assert_called_once_with("test-session-abc", "sub-1")


class TestOldEndpointsRemoved:
    """Verify old unified gateway endpoints return 404."""

    def test_old_execute_returns_404(self):
        """POST /api/gateway/execute now returns 404/405."""
        response = client.post("/api/gateway/execute", json={"action": "discover:all"})
        assert response.status_code in (404, 405)

    def test_old_actions_returns_404(self):
        """GET /api/gateway/actions now returns 404/405."""
        response = client.get("/api/gateway/actions")
        assert response.status_code in (404, 405)

    def test_tools_run_is_live(self):
        """POST /api/tools/run should be reachable (tools router is mounted)."""
        response = client.post("/api/tools/run", json={"tool_name": "translator"})
        # Endpoint is live — may return 200 or 500 depending on backend, but not 404
        assert response.status_code != 404

    def test_old_screening_health_404(self):
        """GET /api/screening/health should return 404."""
        response = client.get("/api/screening/health")
        assert response.status_code in (404, 405)
