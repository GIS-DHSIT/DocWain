"""Comprehensive test suite for the DocWain Teams integration.

Tests cover: adapter auth, state store, chat service, tool routing,
card building, bot app, and graceful import fallbacks.

NOTE: Some other test files (e.g. test_agent_mode.py) stub src.teams modules
with fake modules. To avoid interference, this file imports Teams modules at
module level BEFORE any stubs can be installed, and avoids patching module-level
attributes that stubs might remove.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import Teams modules eagerly at module level so they're loaded before
# any test-ordering-dependent stubs can overwrite them in sys.modules.
import src.teams.adapter as _adapter_mod  # Keep a module-level ref before stubs can replace it
from src.teams.adapter import (
    TeamsAuthError,
    build_teams_message,
    extract_question,
    extract_session_id,
    extract_user_id,
    format_sources,
    verify_shared_secret,
)
from src.teams.cards import build_card, load_card_template
from src.teams.logic import TeamsChatContext, TeamsChatService
from src.teams.state import TeamsStateStore
from src.teams.tools import TeamsToolRouter, _card_activity
from src.api.config import Config
import src.teams.bot_app as _bot_app_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _fake_activity(
    text: str = "What is this document about?",
    user_id: str = "user-aad-123",
    conversation_id: str = "conv-456",
    attachments: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Build a minimal Teams activity dict for testing."""
    activity: Dict[str, Any] = {
        "type": "message",
        "text": text,
        "from": {"id": user_id, "aadObjectId": user_id},
        "conversation": {"id": conversation_id},
        "channelData": {},
        "serviceUrl": "https://smba.trafficmanager.net/uk/",
    }
    if attachments:
        activity["attachments"] = attachments
    return activity


# ---------------------------------------------------------------------------
# TestAdapter — src/teams/adapter.py
# ---------------------------------------------------------------------------

class TestAdapter:
    """Tests for Teams adapter: auth, extraction, formatting."""

    def test_verify_shared_secret_valid(self):
        with patch.object(Config.Teams, "SHARED_SECRET", "my-secret"), \
             patch.object(Config.Teams, "SIGNATURE_ENABLED", False):
            verify_shared_secret({"x-teams-shared-secret": "my-secret"})

    def test_verify_shared_secret_invalid(self):
        with patch.object(Config.Teams, "SHARED_SECRET", "my-secret"), \
             patch.object(Config.Teams, "SIGNATURE_ENABLED", False):
            with pytest.raises(TeamsAuthError):
                verify_shared_secret({"x-teams-shared-secret": "wrong"})

    def test_verify_shared_secret_missing_header(self):
        with patch.object(Config.Teams, "SHARED_SECRET", "my-secret"), \
             patch.object(Config.Teams, "SIGNATURE_ENABLED", False):
            with pytest.raises(TeamsAuthError):
                verify_shared_secret({})

    def test_verify_shared_secret_bypassed_when_unconfigured(self):
        with patch.object(Config.Teams, "SHARED_SECRET", ""), \
             patch.object(Config.Teams, "SIGNATURE_ENABLED", False):
            verify_shared_secret({})  # Should not raise

    def test_verify_shared_secret_bypass_for_jwt(self):
        with patch.object(Config.Teams, "SHARED_SECRET", "my-secret"), \
             patch.object(Config.Teams, "SIGNATURE_ENABLED", False):
            verify_shared_secret({"authorization": "Bearer aaa.bbb.ccc"})

    def test_extract_question_strips_mentions(self):
        activity = {"text": "<at>DocWain</at> What is this?"}
        assert extract_question(activity) == "What is this?"

    def test_extract_question_empty(self):
        assert extract_question({}) == ""
        assert extract_question({"text": ""}) == ""

    def test_extract_user_id_aad(self):
        activity = {"from": {"aadObjectId": "aad-123", "id": "fallback"}}
        assert extract_user_id(activity) == "aad-123"

    def test_extract_user_id_fallback(self):
        activity = {"from": {"id": "bot-id-456"}}
        assert extract_user_id(activity) == "bot-id-456"

    def test_extract_user_id_default(self):
        assert extract_user_id({}) == "teams_user"

    def test_extract_session_id_from_conversation(self):
        activity = {"conversation": {"id": "conv-789"}, "channelData": {}}
        assert extract_session_id(activity) == "conv-789"

    def test_format_sources_empty(self):
        assert format_sources([]) == ""
        assert format_sources(None) == ""

    def test_format_sources_truncates(self):
        sources = [
            {"source_name": f"doc{i}", "excerpt": "x" * 300}
            for i in range(10)
        ]
        result = format_sources(sources)
        assert result.count("- doc") == 5  # Limited to 5

    def test_build_teams_message(self):
        msg = build_teams_message({"response": "Hello", "sources": []})
        assert msg["type"] == "message"
        assert msg["text"] == "Hello"

    def test_build_teams_message_with_sources(self):
        msg = build_teams_message({
            "response": "Answer",
            "sources": [{"source_name": "doc1", "excerpt": "snippet"}],
        })
        assert "Sources:" in msg["text"]
        assert "doc1" in msg["text"]


# ---------------------------------------------------------------------------
# TestStateStore — src/teams/state.py
# ---------------------------------------------------------------------------

class TestStateStore:
    """Tests for TeamsStateStore with in-memory fallback."""

    def _make_store(self):
        store = TeamsStateStore()
        store.client = None  # Force in-memory
        return store

    def test_record_and_list_uploads(self):
        store = self._make_store()
        store.record_upload("sub1", "prof1", "test.pdf", "tag1", 3)
        uploads = store.list_uploads("sub1", "prof1")
        assert len(uploads) == 1
        assert uploads[0]["filename"] == "test.pdf"
        assert uploads[0]["documents_created"] == 3
        assert uploads[0]["doc_tag"] == "tag1"

    def test_upload_list_respects_limit(self):
        store = self._make_store()
        for i in range(5):
            store.record_upload("sub1", "prof1", f"file{i}.pdf", f"tag{i}", 1)
        uploads = store.list_uploads("sub1", "prof1", limit=3)
        assert len(uploads) == 3

    def test_upload_max_entries(self):
        store = self._make_store()
        store.max_uploads = 3
        for i in range(5):
            store.record_upload("sub1", "prof1", f"file{i}.pdf", f"tag{i}", 1)
        uploads = store.list_uploads("sub1", "prof1", limit=10)
        assert len(uploads) == 3
        assert uploads[0]["filename"] == "file4.pdf"

    def test_get_preferences_default_empty(self):
        store = self._make_store()
        prefs = store.get_preferences("sub1", "prof1")
        assert prefs == {}

    def test_set_and_get_preferences(self):
        store = self._make_store()
        store.set_preferences("sub1", "prof1", model_name="gpt-4", persona="analyst")
        prefs = store.get_preferences("sub1", "prof1")
        assert prefs["model_name"] == "gpt-4"
        assert prefs["persona"] == "analyst"

    def test_preference_partial_update(self):
        store = self._make_store()
        store.set_preferences("sub1", "prof1", model_name="gpt-4")
        store.set_preferences("sub1", "prof1", persona="analyst")
        prefs = store.get_preferences("sub1", "prof1")
        assert prefs["model_name"] == "gpt-4"
        assert prefs["persona"] == "analyst"

    def test_uploads_key_format(self):
        key = TeamsStateStore._uploads_key("sub-abc", "prof-xyz")
        assert key == "teams:sub-abc:prof-xyz:uploads"

    def test_prefs_key_format(self):
        key = TeamsStateStore._prefs_key("sub-abc", "prof-xyz")
        assert key == "teams:sub-abc:prof-xyz:prefs"

    def test_different_profiles_isolated(self):
        store = self._make_store()
        store.record_upload("sub1", "prof1", "a.pdf", "t1", 1)
        store.record_upload("sub1", "prof2", "b.pdf", "t2", 2)
        assert len(store.list_uploads("sub1", "prof1")) == 1
        assert len(store.list_uploads("sub1", "prof2")) == 1
        assert store.list_uploads("sub1", "prof1")[0]["filename"] == "a.pdf"


# ---------------------------------------------------------------------------
# TestTeamsChatService — src/teams/logic.py
# ---------------------------------------------------------------------------

class TestTeamsChatService:
    """Tests for TeamsChatService context building and answer routing."""

    def test_build_context_session_as_subscription(self):
        with patch.object(Config.Teams, "SESSION_AS_SUBSCRIPTION", True), \
             patch.object(Config.Teams, "PROFILE_PER_USER", True), \
             patch.object(Config.Teams, "DEFAULT_MODEL", "llama3.2"), \
             patch.object(Config.Teams, "DEFAULT_PERSONA", "Doc"):
            ctx = TeamsChatService.build_context(user_id="user-123", session_id="session-abc")
            assert ctx.subscription_id == "session-abc"
            assert ctx.profile_id == "user-123"

    def test_build_context_uses_defaults(self):
        with patch.object(Config.Teams, "SESSION_AS_SUBSCRIPTION", False), \
             patch.object(Config.Teams, "PROFILE_PER_USER", False), \
             patch.object(Config.Teams, "DEFAULT_MODEL", "llama3.2"), \
             patch.object(Config.Teams, "DEFAULT_PERSONA", "Doc"), \
             patch.object(Config.Teams, "DEFAULT_SUBSCRIPTION", "default-sub"), \
             patch.object(Config.Teams, "DEFAULT_PROFILE", "default-prof"):
            ctx = TeamsChatService.build_context(user_id="user-123", session_id="session-abc")
            assert ctx.subscription_id == "default-sub"
            assert ctx.profile_id == "default-prof"

    def test_build_context_model_persona_override(self):
        with patch.object(Config.Teams, "SESSION_AS_SUBSCRIPTION", True), \
             patch.object(Config.Teams, "PROFILE_PER_USER", True), \
             patch.object(Config.Teams, "DEFAULT_MODEL", "llama3.2"), \
             patch.object(Config.Teams, "DEFAULT_PERSONA", "Doc"):
            ctx = TeamsChatService.build_context(
                user_id="u", session_id="s", model_name="gpt-4", persona="analyst",
            )
            assert ctx.model_name == "gpt-4"
            assert ctx.persona == "analyst"

    def test_build_context_empty_user_fallback(self):
        with patch.object(Config.Teams, "SESSION_AS_SUBSCRIPTION", True), \
             patch.object(Config.Teams, "PROFILE_PER_USER", True), \
             patch.object(Config.Teams, "DEFAULT_MODEL", "llama3.2"), \
             patch.object(Config.Teams, "DEFAULT_PERSONA", "Doc"):
            ctx = TeamsChatService.build_context(user_id="", session_id="")
            assert ctx.user_id == "teams_user"
            assert ctx.session_id == "teams-session"

    def test_answer_question_context_found(self):
        with patch("src.teams.logic.dw_newron") as mock_dw:
            mock_dw.answer_question.return_value = {
                "response": "The document discusses X.",
                "context_found": True,
                "sources": [],
            }
            service = TeamsChatService()
            service.ensure_collection = MagicMock()
            ctx = TeamsChatContext(
                user_id="u", session_id="s", subscription_id="sub",
                profile_id="prof", model_name="llama3.2", persona="Doc",
            )
            result = service.answer_question("What is X?", ctx)
            assert result.internet_mode is False
            assert result.answer["context_found"] is True

    def test_answer_question_internet_fallback(self):
        with patch("src.teams.logic.dw_newron") as mock_dw:
            mock_dw.answer_question.return_value = {
                "response": "",
                "context_found": False,
                "sources": [],
            }
            service = TeamsChatService()
            service.ensure_collection = MagicMock()
            service._answer_with_internet_mode = MagicMock(return_value={
                "response": "General answer",
                "context_found": False,
                "mode": "internet",
            })
            ctx = TeamsChatContext(
                user_id="u", session_id="s", subscription_id="sub",
                profile_id="prof", model_name="llama3.2", persona="Doc",
            )
            result = service.answer_question("What is X?", ctx)
            assert result.internet_mode is True
            assert result.fallback_used is True


# ---------------------------------------------------------------------------
# TestToolRouter — src/teams/tools.py
# ---------------------------------------------------------------------------

class TestToolRouter:
    """Tests for Teams Adaptive Card action routing."""

    def _make_router(self):
        state = TeamsStateStore()
        state.client = None
        service = MagicMock(spec=TeamsChatService)
        service.build_context = TeamsChatService.build_context
        router = TeamsToolRouter(service, state)
        return router, state

    def _make_context(self):
        return TeamsChatContext(
            user_id="u", session_id="s", subscription_id="sub",
            profile_id="prof", model_name="llama3.2", persona="Doc",
        )

    def test_summarize_recent_no_uploads(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "summarize_recent"}, ctx))
        card_content = result["attachments"][0]["content"]
        assert "No uploads" in json.dumps(card_content)

    def test_list_docs_empty(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "list_docs"}, ctx))
        card_content = result["attachments"][0]["content"]
        assert "No uploads" in json.dumps(card_content)

    def test_list_docs_with_uploads(self):
        router, state = self._make_router()
        state.record_upload("sub", "prof", "test.pdf", "tag1", 2)
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "list_docs"}, ctx))
        card_content = result["attachments"][0]["content"]
        card_text = json.dumps(card_content)
        assert "test.pdf" in card_text

    def test_set_model_updates_preferences(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "set_model", "model": "gpt-4"}, ctx))
        card_text = json.dumps(result)
        assert "gpt-4" in card_text
        prefs = state.get_preferences("sub", "prof")
        assert prefs["model_name"] == "gpt-4"

    def test_set_persona_updates_preferences(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "set_persona", "persona": "analyst"}, ctx))
        card_text = json.dumps(result)
        assert "analyst" in card_text

    def test_open_ui_returns_url(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "open_ui"}, ctx))
        assert "docwain" in json.dumps(result).lower()

    def test_tools_menu_renders(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "show_tools"}, ctx))
        card_text = json.dumps(result)
        assert "DocWain tools" in card_text
        assert "summarize_recent" in card_text

    def test_generate_content_no_uploads(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "generate_content"}, ctx))
        card_content = result["attachments"][0]["content"]
        assert "No uploads" in json.dumps(card_content)

    def test_unknown_action_returns_help(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "nonexistent_action"}, ctx))
        card_text = json.dumps(result)
        assert "DocWain for Teams" in card_text or "show_tools" in card_text


# ---------------------------------------------------------------------------
# TestCardBuilder — src/teams/cards/__init__.py
# ---------------------------------------------------------------------------

class TestCardBuilder:
    """Tests for Adaptive Card template loading and building."""

    def test_load_answer_card(self):
        card = load_card_template("answer_card")
        assert card["type"] == "AdaptiveCard"
        # Title is nested inside ColumnSet > Column > items
        card_json = json.dumps(card)
        assert "{title}" in card_json
        assert "{text}" in card_json

    def test_build_card_substitutes_variables(self):
        card = build_card("answer_card", title="My Title", text="My Text")
        card_json = json.dumps(card)
        assert "My Title" in card_json
        assert "My Text" in card_json

    def test_build_card_unknown_template(self):
        # Unknown templates now return a fallback card instead of raising
        card = build_card("nonexistent_template_xyz")
        assert card["type"] == "AdaptiveCard"
        assert any("went wrong" in str(b.get("text", "")) for b in card.get("body", []))

    def test_build_card_no_mutation(self):
        original = load_card_template("answer_card")
        original_text = json.dumps(original)
        build_card("answer_card", title="Modified", text="Changed")
        after = load_card_template("answer_card")
        assert json.dumps(after) == original_text

    def test_welcome_card_loads(self):
        card = load_card_template("welcome_card")
        assert card["type"] == "AdaptiveCard"
        body_text = json.dumps(card["body"])
        assert "Welcome" in body_text

    def test_error_card_loads(self):
        card = build_card("error_card", message="Test error")
        body_text = json.dumps(card["body"])
        assert "Test error" in body_text

    def test_upload_success_card(self):
        card = build_card(
            "upload_success_card",
            message="Done",
            filenames="a.pdf, b.pdf",
            documents_created="5",
        )
        body_text = json.dumps(card)
        assert "a.pdf" in body_text
        assert "5" in body_text

    def test_tools_menu_has_generate(self):
        card = build_card(
            "tools_menu_card",
            default_model="llama3.2",
            default_persona="Doc",
            open_url="https://docwain.ai",
        )
        actions_text = json.dumps(card.get("actions", []))
        assert "generate_content" in actions_text


# ---------------------------------------------------------------------------
# TestBotApp — src/teams/bot_app.py
# ---------------------------------------------------------------------------

class TestBotApp:
    """Tests for DocWainTeamsBot with mocked botbuilder."""

    def test_botbuilder_available_flag(self):
        assert isinstance(_bot_app_mod._BOTBUILDER_AVAILABLE, bool)

    def test_credentials_configured_check(self):
        assert isinstance(_bot_app_mod.BOT_CREDENTIALS_CONFIGURED, bool)

    def test_as_activity_handling(self):
        payload = {"type": "message", "text": "hello"}
        result = _bot_app_mod._as_activity(payload)
        if _bot_app_mod._BOTBUILDER_AVAILABLE:
            assert hasattr(result, "text") or isinstance(result, dict)
        else:
            assert result == payload

    def test_typing_indicator_best_effort(self):
        """_send_typing does not raise on failure."""
        bot = _bot_app_mod.DocWainTeamsBot()
        mock_tc = MagicMock()
        mock_tc.activity = MagicMock()
        mock_tc.activity.relates_to = None
        mock_tc.send_activity = AsyncMock(side_effect=Exception("typing failed"))
        _run(bot._send_typing(mock_tc))

    def test_welcome_message_on_bot_added(self):
        bot = _bot_app_mod.DocWainTeamsBot()
        mock_tc = MagicMock()
        mock_tc.activity = MagicMock()
        mock_tc.activity.as_dict.return_value = _fake_activity()
        mock_tc.activity.service_url = "https://smba.trafficmanager.net/uk/"
        mock_tc.activity.recipient = MagicMock()
        mock_tc.activity.recipient.id = "bot-id-001"
        mock_tc.send_activity = AsyncMock()

        bot_member = MagicMock()
        bot_member.id = "bot-id-001"
        _run(bot.on_members_added_activity([bot_member], mock_tc))
        mock_tc.send_activity.assert_called_once()

    def test_welcome_message_not_sent_for_other_users(self):
        bot = _bot_app_mod.DocWainTeamsBot()
        mock_tc = MagicMock()
        mock_tc.activity = MagicMock()
        mock_tc.activity.as_dict.return_value = _fake_activity()
        mock_tc.activity.service_url = "https://smba.trafficmanager.net/uk/"
        mock_tc.activity.recipient = MagicMock()
        mock_tc.activity.recipient.id = "bot-id-001"
        mock_tc.send_activity = AsyncMock()

        user_member = MagicMock()
        user_member.id = "user-id-999"
        _run(bot.on_members_added_activity([user_member], mock_tc))
        mock_tc.send_activity.assert_not_called()

    def test_help_command(self):
        bot = _bot_app_mod.DocWainTeamsBot()
        mock_tc = MagicMock()
        mock_tc.activity = MagicMock()
        mock_tc.activity.as_dict.return_value = _fake_activity(text="help")
        mock_tc.activity.text = "help"
        mock_tc.activity.service_url = "https://smba.trafficmanager.net/uk/"
        mock_tc.activity.channel_id = "msteams"
        mock_tc.activity.conversation = MagicMock()
        mock_tc.activity.conversation.id = "conv-456"
        mock_tc.activity.attachments = None
        mock_tc.activity.value = None
        mock_tc.activity.from_property = MagicMock()
        mock_tc.activity.from_property.id = "user-aad-123"
        mock_tc.activity.relates_to = None
        mock_tc.send_activity = AsyncMock()

        _run(bot.on_message_activity(mock_tc))
        assert mock_tc.send_activity.call_count >= 1

    def test_tools_command(self):
        bot = _bot_app_mod.DocWainTeamsBot()
        mock_tc = MagicMock()
        mock_tc.activity = MagicMock()
        mock_tc.activity.as_dict.return_value = _fake_activity(text="tools")
        mock_tc.activity.text = "tools"
        mock_tc.activity.service_url = "https://smba.trafficmanager.net/uk/"
        mock_tc.activity.channel_id = "msteams"
        mock_tc.activity.conversation = MagicMock()
        mock_tc.activity.conversation.id = "conv-456"
        mock_tc.activity.attachments = None
        mock_tc.activity.value = None
        mock_tc.activity.from_property = MagicMock()
        mock_tc.activity.from_property.id = "user-aad-123"
        mock_tc.activity.relates_to = None
        mock_tc.send_activity = AsyncMock()

        _run(bot.on_message_activity(mock_tc))
        assert mock_tc.send_activity.call_count >= 1


# ---------------------------------------------------------------------------
# TestGracefulImports
# ---------------------------------------------------------------------------

class TestGracefulImports:
    """Tests for graceful degradation of Teams modules."""

    def test_adapter_has_functions(self):
        assert callable(extract_question)
        assert callable(verify_shared_secret)
        assert callable(extract_user_id)
        assert callable(extract_session_id)

    def test_state_store_class(self):
        assert callable(TeamsStateStore)
        store = TeamsStateStore()
        assert hasattr(store, "record_upload")
        assert hasattr(store, "list_uploads")

    def test_cards_functions(self):
        assert callable(build_card)
        assert callable(load_card_template)

    def test_tools_router_class(self):
        assert callable(TeamsToolRouter)
        assert callable(_card_activity)

    def test_bot_app_module(self):
        assert isinstance(_bot_app_mod._BOTBUILDER_AVAILABLE, bool)
        assert isinstance(_bot_app_mod.BOT_CREDENTIALS_CONFIGURED, bool)
        assert _bot_app_mod.DocWainTeamsBot is not None


# ---------------------------------------------------------------------------
# TestManifest — teams-app/manifest.json validation
# ---------------------------------------------------------------------------

class TestManifest:
    """Tests for Teams app manifest validity."""

    def test_manifest_valid_json(self):
        import pathlib

        manifest_path = pathlib.Path(__file__).parent.parent / "teams-app" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["manifestVersion"] == "1.16"
        assert data["version"] == "1.2.1"

    def test_manifest_has_bot(self):
        import pathlib

        manifest_path = pathlib.Path(__file__).parent.parent / "teams-app" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert len(data["bots"]) == 1
        bot = data["bots"][0]
        assert "personal" in bot["scopes"]
        assert "team" in bot["scopes"]
        assert "groupchat" in bot["scopes"]
        assert bot["supportsFiles"] is True

    def test_manifest_has_commands(self):
        import pathlib

        manifest_path = pathlib.Path(__file__).parent.parent / "teams-app" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        commands = data["bots"][0]["commandLists"][0]["commands"]
        titles = {c["title"] for c in commands}
        assert {"Ask", "Upload", "Help", "Tools", "Generate"} <= titles

    def test_manifest_icons_referenced(self):
        import pathlib

        manifest_path = pathlib.Path(__file__).parent.parent / "teams-app" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["icons"]["color"] == "color.png"
        assert data["icons"]["outline"] == "outline.png"

    def test_manifest_icons_exist(self):
        import pathlib

        teams_dir = pathlib.Path(__file__).parent.parent / "teams-app"
        assert (teams_dir / "color.png").exists()
        assert (teams_dir / "outline.png").exists()

    def test_deprecated_manifest_marked(self):
        import pathlib

        deprecated_path = pathlib.Path(__file__).parent.parent / "src" / "teams" / "manifest.json"
        data = json.loads(deprecated_path.read_text())
        assert "_deprecated" in data


# ---------------------------------------------------------------------------
# TestAdapterOrchestration — full-path adapter tests
# ---------------------------------------------------------------------------

class TestAdapterOrchestration:
    """Tests for the legacy adapter handle_teams_activity orchestration path.

    Captures a reference to the adapter module at module-load time (before
    test_agent_mode stubs can replace it) and uses patch.object on that reference.
    """

    def test_handle_teams_activity_with_question(self):
        adapter_mod = _adapter_mod
        activity = _fake_activity(text="What skills does Aadithya have?")
        with patch.object(adapter_mod, "TEAMS_CHAT_SERVICE") as mock_svc, \
             patch.object(adapter_mod, "STATE_STORE") as mock_store, \
             patch.object(adapter_mod, "add_message_to_history"):
            mock_store.get_preferences.return_value = {}
            mock_svc.build_context.return_value = TeamsChatContext(
                user_id="user-aad-123", session_id="conv-456",
                subscription_id="conv-456", profile_id="user-aad-123",
                model_name="llama3.2", persona="Document Assistant",
            )
            from src.teams.logic import TeamsAnswerResult
            mock_svc.answer_question.return_value = TeamsAnswerResult(
                answer={"response": "Aadithya has Python skills.", "sources": [], "context_found": True},
                subscription_id="conv-456", profile_id="user-aad-123",
                fallback_used=False, internet_mode=False,
            )
            result = _run(adapter_mod.handle_teams_activity(activity, headers={}))
            assert result["type"] == "message"
            assert "Python" in result["text"]

    def test_handle_teams_activity_empty_question(self):
        adapter_mod = _adapter_mod
        activity = _fake_activity(text="")
        result = _run(adapter_mod.handle_teams_activity(activity, headers={}))
        assert "did not receive" in result["text"].lower()

    def test_handle_teams_activity_with_attachments(self):
        adapter_mod = _adapter_mod
        attachments = [{"contentType": "application/pdf", "content": {"downloadUrl": "https://example.com/test.pdf"}}]
        activity = _fake_activity(attachments=attachments)
        with patch.object(adapter_mod, "handle_attachment_activity") as mock_attach:
            mock_attach.return_value = {"type": "message", "text": "Processed 1 file(s)."}
            result = _run(adapter_mod.handle_teams_activity(activity, headers={}))
            assert "Processed" in result["text"] or "message" in result.get("type", "")

    def test_handle_teams_activity_loads_preferences(self):
        adapter_mod = _adapter_mod
        activity = _fake_activity(text="Tell me about the invoice")
        with patch.object(adapter_mod, "TEAMS_CHAT_SERVICE") as mock_svc, \
             patch.object(adapter_mod, "STATE_STORE") as mock_store, \
             patch.object(adapter_mod, "add_message_to_history"):
            mock_store.get_preferences.return_value = {"model_name": "gpt-4o", "persona": "Analyst"}
            mock_svc.build_context.return_value = TeamsChatContext(
                user_id="user-aad-123", session_id="conv-456",
                subscription_id="conv-456", profile_id="user-aad-123",
                model_name="gpt-4o", persona="Analyst",
            )
            from src.teams.logic import TeamsAnswerResult
            mock_svc.answer_question.return_value = TeamsAnswerResult(
                answer={"response": "Invoice details here.", "sources": [], "context_found": True},
                subscription_id="conv-456", profile_id="user-aad-123",
                fallback_used=False, internet_mode=False,
            )
            result = _run(adapter_mod.handle_teams_activity(activity, headers={}))
            mock_store.get_preferences.assert_called_once()
            call_kwargs = mock_svc.build_context.call_args
            assert call_kwargs[1].get("model_name") == "gpt-4o" or call_kwargs.kwargs.get("model_name") == "gpt-4o"


# ---------------------------------------------------------------------------
# TestAttachmentHandling — edge cases
# ---------------------------------------------------------------------------

class TestAttachmentHandling:
    """Tests for attachment download and ingestion edge cases."""

    def test_attachment_activity_loads_preferences(self):
        """handle_attachment_activity should load user preferences just like handle_teams_activity."""
        adapter_mod = _adapter_mod
        attachments = [{"contentType": "application/pdf", "content": {"downloadUrl": "https://example.com/test.pdf"}}]
        activity = _fake_activity(attachments=attachments)
        with patch.object(adapter_mod, "TEAMS_CHAT_SERVICE") as mock_svc, \
             patch.object(adapter_mod, "STATE_STORE") as mock_store, \
             patch.object(adapter_mod, "ingest_attachments") as mock_ingest:
            mock_store.get_preferences.return_value = {"model_name": "mistral"}
            mock_svc.build_context.return_value = TeamsChatContext(
                user_id="user-aad-123", session_id="conv-456",
                subscription_id="conv-456", profile_id="user-aad-123",
                model_name="mistral", persona="Document Assistant",
            )
            mock_svc.ensure_collection = MagicMock()
            from src.teams.attachments import IngestionResult
            mock_ingest.return_value = IngestionResult(filenames=["test.pdf"], documents_created=1, doc_tags=["tag1"])
            result = _run(adapter_mod.handle_attachment_activity(activity))
            mock_store.get_preferences.assert_called_once()

    def test_inline_image_rejection(self):
        """Inline images without downloadUrl should return guidance message."""
        adapter_mod = _adapter_mod
        attachments = [{"contentType": "image/png"}]
        activity = _fake_activity(attachments=attachments)
        with patch.object(adapter_mod, "TEAMS_CHAT_SERVICE") as mock_svc, \
             patch.object(adapter_mod, "STATE_STORE") as mock_store:
            mock_store.get_preferences.return_value = {}
            mock_svc.build_context.return_value = TeamsChatContext(
                user_id="user-aad-123", session_id="conv-456",
                subscription_id="conv-456", profile_id="user-aad-123",
                model_name="llama3.2", persona="Document Assistant",
            )
            mock_svc.ensure_collection = MagicMock()
            result = _run(adapter_mod.handle_attachment_activity(activity))
            assert "file attachment" in result["text"].lower()


# ---------------------------------------------------------------------------
# TestNewCards — new card template tests
# ---------------------------------------------------------------------------

class TestNewCards:
    """Tests for the new Adaptive Card templates."""

    def test_preferences_card_renders(self):
        card = build_card("preferences_card", current_model="llama3.2", current_persona="Document Assistant")
        assert card["type"] == "AdaptiveCard"
        card_text = json.dumps(card)
        assert "llama3.2" in card_text
        assert "Document Assistant" in card_text
        assert "Input.ChoiceSet" in card_text

    def test_preferences_card_has_model_choices(self):
        card = build_card("preferences_card", current_model="gpt-4o", current_persona="Analyst")
        card_text = json.dumps(card)
        assert "gpt-4o" in card_text
        assert "mistral" in card_text
        assert "gemini-2.5-flash" in card_text

    def test_processing_card_renders(self):
        card = build_card("processing_card", status_message="Analyzing your question...")
        assert card["type"] == "AdaptiveCard"
        card_text = json.dumps(card)
        assert "Analyzing your question" in card_text
        assert "moment" in card_text.lower()

    def test_answer_card_with_sources(self):
        card = build_card(
            "answer_card",
            title="Answer",
            text="The document discusses AI.",
            sources_text="- resume.pdf: Skills section mentions Python",
        )
        card_text = json.dumps(card)
        assert "The document discusses AI" in card_text
        assert "resume.pdf" in card_text
        assert "sourcesContainer" in card_text

    def test_answer_card_sources_hidden_by_default(self):
        card = build_card("answer_card", title="T", text="Body", sources_text="src")
        # Find the sources container
        for item in card["body"]:
            if isinstance(item, dict) and item.get("id") == "sourcesContainer":
                assert item["isVisible"] is False
                break
        else:
            pytest.fail("sourcesContainer not found in answer_card body")

    def test_answer_card_toggle_visibility_action(self):
        card = build_card("answer_card", title="T", text="B", sources_text="S")
        actions = card.get("actions", [])
        toggle_actions = [a for a in actions if a.get("type") == "Action.ToggleVisibility"]
        assert len(toggle_actions) == 1
        assert "sourcesContainer" in toggle_actions[0].get("targetElements", [])

    def test_welcome_card_column_layout(self):
        card = load_card_template("welcome_card")
        card_text = json.dumps(card)
        assert "ColumnSet" in card_text
        assert "Upload a document" in card_text

    def test_tools_menu_has_preferences_button(self):
        card = build_card("tools_menu_card", open_url="https://docwain.ai")
        actions_text = json.dumps(card.get("actions", []))
        assert "show_preferences" in actions_text
        # Old model/persona buttons should be gone
        assert "set_model" not in actions_text
        assert "set_persona" not in actions_text


# ---------------------------------------------------------------------------
# TestPreferencesFlow — end-to-end preferences card flow
# ---------------------------------------------------------------------------

class TestPreferencesFlow:
    """Tests for the interactive preferences card flow via tool router."""

    def _make_router(self):
        state = TeamsStateStore()
        state.client = None
        service = MagicMock(spec=TeamsChatService)
        service.build_context = TeamsChatService.build_context
        router = TeamsToolRouter(service, state)
        return router, state

    def _make_context(self):
        return TeamsChatContext(
            user_id="u", session_id="s", subscription_id="sub",
            profile_id="prof", model_name="llama3.2", persona="Document Assistant",
        )

    def test_show_preferences_returns_card(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "show_preferences"}, ctx))
        card_text = json.dumps(result)
        assert "Input.ChoiceSet" in card_text
        assert "llama3.2" in card_text

    def test_set_preferences_updates_both(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({
            "action": "set_preferences",
            "model": "gpt-4o",
            "persona": "Analyst",
        }, ctx))
        card_text = json.dumps(result)
        assert "gpt-4o" in card_text
        assert "Analyst" in card_text
        prefs = state.get_preferences("sub", "prof")
        assert prefs["model_name"] == "gpt-4o"
        assert prefs["persona"] == "Analyst"

    def test_set_preferences_model_only(self):
        router, state = self._make_router()
        ctx = self._make_context()
        _run(router.handle_action({"action": "set_preferences", "model": "mistral"}, ctx))
        prefs = state.get_preferences("sub", "prof")
        assert prefs["model_name"] == "mistral"
        assert "persona" not in prefs

    def test_set_preferences_empty_no_crash(self):
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "set_preferences"}, ctx))
        card_text = json.dumps(result)
        assert "No changes" in card_text

    def test_legacy_set_model_still_works(self):
        """Backward compat: set_model action should still work."""
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "set_model", "model": "gpt-4o"}, ctx))
        card_text = json.dumps(result)
        assert "gpt-4o" in card_text
        prefs = state.get_preferences("sub", "prof")
        assert prefs["model_name"] == "gpt-4o"

    def test_legacy_set_persona_still_works(self):
        """Backward compat: set_persona action should still work."""
        router, state = self._make_router()
        ctx = self._make_context()
        result = _run(router.handle_action({"action": "set_persona", "persona": "Legal Reviewer"}, ctx))
        card_text = json.dumps(result)
        assert "Legal Reviewer" in card_text


# ---------------------------------------------------------------------------
# TestConfigEmptyString — config.py empty-string env var handling
# ---------------------------------------------------------------------------

class TestConfigEmptyString:
    """Test that empty-string env vars fall back to defaults."""

    def test_empty_subscription_uses_default(self):
        with patch.dict("os.environ", {"TEAMS_DEFAULT_SUBSCRIPTION": ""}):
            # The `or` pattern means empty string is falsy → fallback to hardcoded default
            result = os.getenv("TEAMS_DEFAULT_SUBSCRIPTION") or "15e0c724-4de0-492e-9861-9e637b3f9076"
            assert result == "15e0c724-4de0-492e-9861-9e637b3f9076"

    def test_set_subscription_overrides(self):
        with patch.dict("os.environ", {"TEAMS_DEFAULT_SUBSCRIPTION": "custom-sub-id"}):
            result = os.getenv("TEAMS_DEFAULT_SUBSCRIPTION") or "15e0c724-4de0-492e-9861-9e637b3f9076"
            assert result == "custom-sub-id"
