import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api import docwain_api


class TeamsMessagesRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(docwain_api.app)

    def test_empty_body_returns_bot_message(self):
        with patch(
            "src.api.docwain_api.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
        ) as mock_handle:
            response = self.client.post("/teams/messages", data=b"")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("type"), "message")
        mock_handle.assert_not_called()

    def test_invalid_json_returns_bot_message(self):
        with patch(
            "src.api.docwain_api.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
        ) as mock_handle:
            response = self.client.post(
                "/teams/messages",
                data="not-json",
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("type"), "message")
        mock_handle.assert_not_called()

    def test_text_plain_fallback_routes(self):
        expected = {"type": "message", "text": "ok"}
        with patch(
            "src.api.docwain_api.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_handle:
            response = self.client.post(
                "/teams/messages",
                data="hello",
                headers={"Content-Type": "text/plain;charset=UTF-8"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        mock_handle.assert_awaited()
        activity_arg = mock_handle.await_args.args[0]
        self.assertEqual(activity_arg.get("text"), "hello")
        self.assertEqual(activity_arg.get("conversation", {}).get("id"), "teams-text-fallback")

    def test_valid_activity_is_forwarded(self):
        expected = {"type": "message", "text": "ok"}
        with patch(
            "src.api.docwain_api.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_handle:
            response = self.client.post("/teams/messages", json={"text": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        mock_handle.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
