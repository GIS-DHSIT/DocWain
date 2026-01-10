import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.teams import adapter as teams_adapter
from src.teams import attachments as teams_attachments
from src.api.config import Config


class HandleAttachmentActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_downloads_and_trains_from_attachment(self):
        activity = {
            "conversation": {"id": "conversation-id"},
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.teams.file.download.info",
                    "content": {"downloadUrl": "http://example.com/file.txt"},
                }
            ],
        }
        with patch("src.teams.attachments._download_bytes", return_value=b"dummy content") as mock_download, \
                patch("src.teams.attachments.fileProcessor", return_value={"sample.txt": "document content"}), \
                patch("src.teams.attachments.train_on_document") as mock_train:
            response = await teams_adapter.handle_attachment_activity(activity, correlation_id="corr-1")

        self.assertEqual(response["type"], "message")
        self.assertIn("Successfully processed", response["text"])
        mock_download.assert_awaited_once()
        mock_train.assert_called_once()

    async def test_download_url_snake_case(self):
        activity = {
            "conversation": {"id": "conversation-id"},
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.teams.file.download.info",
                    "content": {"download_url": "http://example.com/file.txt"},
                }
            ],
        }
        with patch("src.teams.attachments._download_bytes", return_value=b"dummy content"), \
                patch("src.teams.attachments.fileProcessor", return_value={"sample.txt": "document content"}), \
                patch("src.teams.attachments.train_on_document"):
            response = await teams_adapter.handle_attachment_activity(activity, correlation_id="corr-2")

        self.assertEqual(response["type"], "message")
        self.assertIn("Successfully processed", response["text"])

    async def test_missing_download_url(self):
        activity = {
            "conversation": {"id": "conversation-id"},
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.teams.file.download.info",
                    "content": {},
                }
            ],
        }
        response = await teams_adapter.handle_attachment_activity(activity, correlation_id="corr-3")
        self.assertEqual(response["type"], "message")
        self.assertIn("could not be processed", response["text"])

    async def test_oversized_file_rejected(self):
        activity = {
            "conversation": {"id": "conversation-id"},
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.teams.file.download.info",
                    "content": {"downloadUrl": "http://example.com/file.txt"},
                }
            ],
        }
        with patch("src.teams.attachments._download_bytes", side_effect=ValueError("Attachment exceeds size limit")):
            response = await teams_adapter.handle_attachment_activity(activity, correlation_id="corr-4")
        self.assertEqual(response["type"], "message")
        self.assertIn("failed", response["text"].lower())

    async def test_bypasses_non_file_attachments(self):
        activity = {
            "conversation": {"id": "conversation-id"},
            "attachments": [{"contentType": "image/png", "content": {}}],
        }
        response = await teams_adapter.handle_attachment_activity(activity, correlation_id="corr-5")
        self.assertEqual(response["type"], "message")
        self.assertIn("upload the image", response["text"])

    async def test_http_client_timeout_used(self):
        async def _fake_stream(*args, **kwargs):
            raise httpx.RequestError("boom", request=None)  # type: ignore[arg-type]

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                return _FakeStream()

        class _FakeStream:
            async def __aenter__(self):
                raise httpx.RequestError("boom", request=None)  # type: ignore[arg-type]
            async def __aexit__(self, exc_type, exc, tb):
                return False

        with patch("src.teams.attachments.httpx.AsyncClient", side_effect=_FakeClient) as mock_client:
            with self.assertRaises(httpx.RequestError):
                await teams_attachments._download_bytes(
                    "http://example.com/file.txt",
                    headers=None,
                    timeout=1.5,
                    retries=0,
                    max_bytes=10,
                )
        self.assertTrue(mock_client.called)
        self.assertEqual(mock_client.call_args.kwargs.get("timeout"), 1.5)


class HandleTeamsActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_attachment_short_circuits_question_flow(self):
        activity = {
            "attachments": [
                {"contentType": "application/vnd.microsoft.teams.file.download.info", "content": {}}
            ]
        }
        expected = {"type": "message", "text": "from attachment handler"}

        with patch(
            "src.teams.adapter.handle_attachment_activity",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_handle, patch("src.teams.adapter.TEAMS_CHAT_SERVICE.answer_question") as mock_answer:
            response = await teams_adapter.handle_teams_activity(activity)

        self.assertEqual(response, expected)
        mock_handle.assert_awaited_once_with(activity)
        mock_answer.assert_not_called()

    async def test_question_path_handles_answer_errors(self):
        activity = {"text": "Hello", "conversation": {"id": "abc"}}
        with patch(
            "src.teams.adapter.TEAMS_CHAT_SERVICE.answer_question",
            side_effect=teams_adapter.TeamsChatError("boom"),
        ):
            response = await teams_adapter.handle_teams_activity(activity)

        self.assertEqual(response["type"], "message")
        self.assertIn("snag", response["text"])

    async def test_question_path_persists_history_and_uses_session(self):
        activity = {
            "text": "Hi there",
            "conversation": {"id": "conv-1"},
            "from": {"id": "user-1"},
        }
        answer_payload = {"response": "hello", "sources": []}
        with patch(
            "src.teams.adapter.TEAMS_CHAT_SERVICE.answer_question",
            return_value=teams_adapter.TeamsAnswerResult(
                answer=answer_payload,
                subscription_id="conv-1",
                profile_id="user-1",
                fallback_used=False,
            ),
        ) as mock_answer, \
                patch("src.teams.adapter.add_message_to_history") as mock_history:
            response = await teams_adapter.handle_teams_activity(activity)

        self.assertEqual(response["type"], "message")
        self.assertIn("hello", response["text"])
        mock_answer.assert_called_once()
        mock_history.assert_called_once()


class VerifySharedSecretTests(unittest.TestCase):
    def test_verify_shared_secret_uses_constant_time_compare(self):
        headers = {"x-teams-shared-secret": "secret"}
        with patch.object(Config.Teams, "SHARED_SECRET", "secret"), \
                patch("src.teams.adapter.hmac.compare_digest", return_value=True) as mock_compare:
            teams_adapter.verify_shared_secret(headers, raw_body=b"")
        mock_compare.assert_called()

    def test_verify_shared_secret_missing_returns_401(self):
        with patch.object(Config.Teams, "SHARED_SECRET", "secret"):
            with self.assertRaises(teams_adapter.TeamsAuthError):
                teams_adapter.verify_shared_secret({})


if __name__ == "__main__":
    unittest.main()
