import asyncio
import importlib.machinery
from types import SimpleNamespace

import pytest
from botbuilder.schema import Activity, ActivityTypes, ChannelAccount, ConversationAccount

# Stub heavy optional deps so imports succeed without extra packages.
import sys
import types

datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = None
datasets_stub.load_dataset = lambda *args, **kwargs: None
datasets_stub.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
sys.modules["datasets"] = datasets_stub

from src import main as docwain_api
from src.teams import bot_app


class FakeTurnContext:
    def __init__(self, text: str = "hi"):
        self.sent = []
        self.activity = Activity(
            type=ActivityTypes.message,
            text=text,
            service_url="https://smba.trafficmanager.net/apac/",
            channel_id="msteams",
            conversation=ConversationAccount(id="conv"),
            from_property=ChannelAccount(id="user"),
        )

    async def send_activity(self, activity):
        self.sent.append(activity)
        return activity


def test_on_message_trusts_service_url(monkeypatch):
    trusted = {}
    monkeypatch.setattr(
        bot_app.MicrosoftAppCredentials,
        "trust_service_url",
        lambda url: trusted.setdefault("url", url),
    )
    ctx = FakeTurnContext()
    bot = bot_app.DocWainTeamsBot()
    asyncio.run(bot.on_message_activity(ctx))
    assert trusted["url"] == ctx.activity.service_url
    assert ctx.sent  # at least one response was attempted


def test_handle_bot_error_sends_once(monkeypatch):
    trusted = {}

    async def failing_send(msg):
        raise RuntimeError("unauthorized")

    class ErrCtx(FakeTurnContext):
        async def send_activity(self, activity):
            return await failing_send(activity)

    monkeypatch.setattr(
        bot_app.MicrosoftAppCredentials,
        "trust_service_url",
        lambda url: trusted.setdefault("url", url),
    )
    ctx = ErrCtx()
    asyncio.run(bot_app.handle_bot_error(ctx, Exception("boom")))
    assert trusted["url"] == ctx.activity.service_url
    # no exception should escape even if send_activity fails


def test_process_activity_invoked(monkeypatch):
    from fastapi.testclient import TestClient

    class MockAdapter:
        def __init__(self):
            self.called = False

        async def process_activity(self, *args, **kwargs):
            self.called = True
            return SimpleNamespace(status=200, body={"ok": True})

    mock_adapter = MockAdapter()

    monkeypatch.setattr(docwain_api, "bot_adapter", mock_adapter)
    monkeypatch.setattr(docwain_api, "BOT_CREDENTIALS_CONFIGURED", True)
    monkeypatch.setattr(docwain_api, "MICROSOFT_APP_ID", "00000000-0000-0000-0000-000000000000")
    docwain_api.Config.Teams.BOT_APP_PASSWORD = "secret"

    activity = {
        "type": "message",
        "serviceUrl": "https://smba.trafficmanager.net/teams/",
        "channelId": "msteams",
        "conversation": {"id": "conv"},
    }
    headers = {"Authorization": "Bearer a.b.c"}
    api_client = TestClient(docwain_api.app)
    response = api_client.post("/api/teams/messages", json=activity, headers=headers)
    assert response.status_code == 200
    assert mock_adapter.called
