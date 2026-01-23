import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import importlib.machinery
import sys
import types

# Stub heavy optional deps so tests can import the API module without extra installs.
datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = None
datasets_stub.load_dataset = lambda *args, **kwargs: None
datasets_stub.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
sys.modules["datasets"] = datasets_stub

from fastapi.testclient import TestClient

from src import main as docwain_api
from src.teams.attachments import ingest_attachments
from src.teams.logic import TeamsChatContext, TeamsChatService
from src.teams.tools import TeamsToolRouter


class TeamsMessagesRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(docwain_api.app)
        self.path = "/api/teams/messages"

    def test_empty_body_returns_bot_message(self):
        with patch(
            "src.main.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
        ) as mock_handle:
            response = self.client.post(self.path, data=b"")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("type"), "message")
        mock_handle.assert_not_called()

    def test_invalid_json_returns_bot_message(self):
        with patch(
            "src.main.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
        ) as mock_handle:
            response = self.client.post(
                self.path,
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
            "src.main.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_handle:
            response = self.client.post(
                self.path,
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
            "src.main.teams_adapter.handle_teams_activity",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_handle:
            response = self.client.post(self.path, json={"text": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        mock_handle.assert_awaited_once()


class TeamsIngestionTests(unittest.TestCase):
    def setUp(self):
        self.context = TeamsChatContext(
            user_id="user",
            session_id="session",
            subscription_id="session",
            profile_id="user",
            model_name="llama",
            persona="Document Assistant",
        )

    def test_ingest_attachments_trains_documents(self):
        activity = {
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.teams.file.download.info",
                    "content": {"downloadUrl": "https://example", "fileName": "doc.pdf", "uniqueId": "abc"},
                }
            ]
        }
        with patch("src.teams.attachments._download_bytes", new_callable=AsyncMock, return_value=b"data") as mock_dl, \
                patch("src.teams.attachments.fileProcessor", return_value={"doc.pdf": "content"}) as mock_fp, \
                patch("src.teams.attachments.train_on_document", new_callable=MagicMock) as mock_train:
            result = asyncio.run(
                ingest_attachments(
                    activity,
                    turn_context=None,
                    context=self.context,
                    correlation_id="corr",
                    state_store=None,
                    connector_token="token",
                )
            )

        self.assertEqual(result.documents_created, 1)
        self.assertIn("doc.pdf", result.filenames[0])
        mock_dl.assert_awaited()
        mock_fp.assert_called_once()
        mock_train.assert_called_once()


class DummyStateStore:
    def __init__(self, uploads):
        self.uploads = uploads

    def list_uploads(self, *args, **kwargs):
        return self.uploads

    def get_preferences(self, *args, **kwargs):
        return {}

    def set_preferences(self, *args, **kwargs):
        return {}


class TeamsToolRouterTests(unittest.TestCase):
    def setUp(self):
        uploads = [
            {"filename": "doc.pdf", "doc_tag": "abc", "documents_created": 1},
            {"filename": "other.docx", "doc_tag": "def", "documents_created": 1},
        ]
        self.store = DummyStateStore(uploads)
        self.router = TeamsToolRouter(TeamsChatService(), self.store)
        self.context = TeamsChatService.build_context(user_id="user", session_id="session")

    def test_summarize_routes_to_docwain(self):
        class StubRouter(TeamsToolRouter):
            async def _ask_docwain(self, question, context):
                return {"response": "summary ready", "sources": []}

        router = StubRouter(TeamsChatService(), self.store)
        payload = asyncio.run(router.handle_action({"action": "summarize_recent"}, self.context))
        self.assertIn("attachments", payload)
        content = payload["attachments"][0]["content"]
        self.assertIn("Summary", json.dumps(content))

    def test_list_docs_renders_card(self):
        payload = asyncio.run(self.router.handle_action({"action": "list_docs"}, self.context))
        content = payload["attachments"][0]["content"]
        text = content["body"][1]["text"]
        self.assertIn("doc.pdf", text)
        self.assertIn("abc", json.dumps(payload))

    def test_unknown_action_shows_help(self):
        payload = asyncio.run(self.router.handle_action({"action": "unknown"}, self.context))
        content = payload["attachments"][0]["content"]
        self.assertEqual(content.get("type"), "AdaptiveCard")

if __name__ == "__main__":
    unittest.main()
