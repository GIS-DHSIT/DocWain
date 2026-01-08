import unittest
from unittest.mock import MagicMock, patch

from src.teams.logic import TeamsChatService


class TeamsChatServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = TeamsChatService(qdrant_client=MagicMock())

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


if __name__ == "__main__":
    unittest.main()
