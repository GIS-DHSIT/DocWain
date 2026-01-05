import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.api import teams_adapter
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

        fake_download = MagicMock()
        fake_download.download_url = "http://example.com/file.txt"
        fake_download.name = "sample.txt"
        fake_download.unique_id = "file-123"

        with tempfile.TemporaryDirectory() as tmpdir, \
                patch.dict(os.environ, {"TEAMS_UPLOAD_DIR": tmpdir}, clear=False), \
                patch("src.api.teams_adapter.FileDownloadInfo.deserialize", return_value=fake_download), \
                patch("src.api.teams_adapter.requests.get") as mock_get, \
                patch("src.api.teams_adapter.fileProcessor", return_value={"sample.txt": "document content"}), \
                patch("src.api.teams_adapter.train_on_document") as mock_train:

            mock_response = MagicMock()
            mock_response.content = b"dummy content"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            response = await teams_adapter.handle_attachment_activity(activity)
            temp_path = os.path.join(tmpdir, "sample.txt")

            self.assertEqual(response["type"], "message")
            self.assertIn("File processed successfully", response["text"])
            mock_get.assert_called_once_with("http://example.com/file.txt")
            mock_train.assert_called_once_with(
                "document content",
                subscription_id="conversation-id",
                profile_tag=Config.Teams.DEFAULT_PROFILE,
                doc_tag="file-123",
                doc_name="sample.txt",
            )
            self.assertFalse(os.path.exists(temp_path))

    async def test_bypasses_non_file_attachments(self):
        activity = {
            "conversation": {"id": "conversation-id"},
            "attachments": [{"contentType": "image/png", "content": {}}],
        }

        response = await teams_adapter.handle_attachment_activity(activity)
        self.assertEqual(response["type"], "message")
        self.assertIn("Unable to process the attachment", response["text"])


class HandleTeamsActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_attachment_short_circuits_question_flow(self):
        activity = {
            "attachments": [
                {"contentType": "application/vnd.microsoft.teams.file.download.info", "content": {}}
            ]
        }
        expected = {"type": "message", "text": "from attachment handler"}

        with patch(
            "src.api.teams_adapter.handle_attachment_activity",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_handle, patch("src.api.teams_adapter.answer_question") as mock_answer:
            response = await teams_adapter.handle_teams_activity(activity)

        self.assertEqual(response, expected)
        mock_handle.assert_awaited_once_with(activity)
        mock_answer.assert_not_called()

    async def test_question_path_handles_answer_errors(self):
        activity = {"text": "Hello", "conversation": {"id": "abc"}}
        with patch("src.api.teams_adapter.answer_question", side_effect=RuntimeError("boom")):
            response = await teams_adapter.handle_teams_activity(activity)

        self.assertEqual(response["type"], "message")
        self.assertIn("snag", response["text"])


if __name__ == "__main__":
    unittest.main()
