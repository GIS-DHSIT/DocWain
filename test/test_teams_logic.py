import unittest
from unittest.mock import MagicMock, patch

from src.teams.logic import TeamsChatService, TeamsChatError


class TeamsChatServiceTests(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.service = TeamsChatService(qdrant_client=self.mock_client)

    def test_build_context_applies_defaults(self):
        ctx = TeamsChatService.build_context(
            user_id="user-1",
            session_id="session-1",
            model_name="custom-model",
            persona="custom-persona",
        )

        self.assertEqual(ctx.subscription_id, "session-1")
        self.assertEqual(ctx.profile_id, "user-1")
        self.assertEqual(ctx.model_name, "custom-model")
        self.assertEqual(ctx.persona, "custom-persona")

    def test_build_context_sanitizes_colons_in_ids(self):
        ctx = TeamsChatService.build_context(
            user_id="a:12345:xyz",
            session_id="conv:abc/def\\ghi",
        )
        self.assertNotIn(":", ctx.subscription_id)
        self.assertNotIn("/", ctx.subscription_id)
        self.assertNotIn("\\", ctx.subscription_id)
        self.assertNotIn(":", ctx.profile_id)

    def test_build_context_truncates_long_ids(self):
        ctx = TeamsChatService.build_context(
            user_id="x" * 200,
            session_id="y" * 200,
        )
        self.assertLessEqual(len(ctx.subscription_id), 64)
        self.assertLessEqual(len(ctx.profile_id), 64)

    def test_answer_question_prefers_session_scope_when_available(self):
        ctx = self.service.build_context(user_id="user-1", session_id="session-1")

        with patch.object(self.service, "ensure_collection"), \
                patch("src.teams.logic.dw_newron.answer_question", return_value={"response": "ok", "context_found": True}) as mock_answer:
            result = self.service.answer_question("hi", ctx)

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.subscription_id, ctx.subscription_id)
        self.assertEqual(result.profile_id, ctx.profile_id)
        mock_answer.assert_called_once_with(
            query="hi",
            user_id="user-1",
            profile_id=ctx.profile_id,
            subscription_id=ctx.subscription_id,
            model_name=ctx.model_name,
            persona=ctx.persona,
            session_id=ctx.session_id,
        )

    def test_answer_question_uses_internet_fallback_when_no_context(self):
        ctx = self.service.build_context(user_id="user-1", session_id="session-1")
        with patch.object(self.service, "ensure_collection"), \
                patch("src.teams.logic.dw_newron.answer_question", return_value={"response": "", "context_found": False}), \
                patch.object(self.service, "_answer_with_internet_mode", return_value={"response": "web", "context_found": False}) as mock_web:
            result = self.service.answer_question("hello", ctx)

        self.assertTrue(result.fallback_used)
        self.assertTrue(result.internet_mode)
        self.assertEqual(result.answer["response"], "web")
        mock_web.assert_called_once()

    def test_delete_all_documents_calls_qdrant(self):
        self.mock_client.count.return_value = MagicMock(count=5)
        deleted = self.service.delete_all_documents("sub-1", "prof-1")
        self.assertEqual(deleted, 5)
        self.mock_client.count.assert_called_once()
        self.mock_client.delete.assert_called_once()

    def test_delete_all_documents_returns_zero_when_empty(self):
        self.mock_client.count.return_value = MagicMock(count=0)
        deleted = self.service.delete_all_documents("sub-1", "prof-1")
        self.assertEqual(deleted, 0)
        self.mock_client.delete.assert_not_called()


class TeamsStateStoreTests(unittest.TestCase):
    def test_clear_uploads_removes_records(self):
        from src.teams.state import TeamsStateStore
        store = TeamsStateStore()
        store.client = None  # Force in-memory mode
        store.record_upload("sub-1", "prof-1", "test.pdf", "tag-1", 2)
        store.record_upload("sub-1", "prof-1", "test2.pdf", "tag-2", 1)
        self.assertEqual(len(store.list_uploads("sub-1", "prof-1")), 2)
        removed = store.clear_uploads("sub-1", "prof-1")
        self.assertEqual(removed, 2)
        self.assertEqual(len(store.list_uploads("sub-1", "prof-1")), 0)

    def test_clear_uploads_returns_zero_when_empty(self):
        from src.teams.state import TeamsStateStore
        store = TeamsStateStore()
        store.client = None
        removed = store.clear_uploads("sub-1", "prof-1")
        self.assertEqual(removed, 0)


class TeamsIsolationTests(unittest.TestCase):
    """Ensure Teams operations never touch core DocWain scoped data."""

    def test_teams_collection_name_uses_session_id(self):
        ctx = TeamsChatService.build_context(user_id="u1", session_id="teams-conv-123")
        # Teams sessions get their own collection name based on session_id
        self.assertEqual(ctx.subscription_id, "teams-conv-123")
        self.assertNotEqual(ctx.subscription_id, "15e0c724-4de0-492e-9861-9e637b3f9076")

    def test_teams_profile_uses_user_id(self):
        ctx = TeamsChatService.build_context(user_id="aad-user-001", session_id="s1")
        self.assertEqual(ctx.profile_id, "aad-user-001")
        self.assertNotEqual(ctx.profile_id, "default")


if __name__ == "__main__":
    unittest.main()
