import logging
import os
import re
from typing import Any, Dict

try:
    from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
    from botbuilder.core.teams import TeamsActivityHandler
    from botbuilder.schema import Activity, ActivityTypes, ErrorResponseException
    from botframework.connector.auth import MicrosoftAppCredentials

    _BOTBUILDER_AVAILABLE = True
except ImportError:
    _BOTBUILDER_AVAILABLE = False
    BotFrameworkAdapter = None
    BotFrameworkAdapterSettings = None
    TurnContext = None
    TeamsActivityHandler = object  # Allows class definition to succeed
    Activity = None
    ActivityTypes = None
    ErrorResponseException = Exception
    MicrosoftAppCredentials = None

from src.api.config import Config
from src.teams import adapter as legacy_adapter
from src.teams.attachments import ingest_attachments
from src.teams.cards import build_card
from src.teams.logic import TeamsChatService
from src.teams.state import TeamsStateStore
from src.teams.tools import TeamsToolRouter, _card_activity
from src.utils.logging_utils import get_correlation_id, get_logger

logger = logging.getLogger(__name__)

MICROSOFT_APP_ID = Config.Teams.BOT_APP_ID or os.getenv("MICROSOFT_APP_ID")
MICROSOFT_APP_PASSWORD = (
    Config.Teams.BOT_APP_PASSWORD
    or os.getenv("MICROSOFT_APP_PASSWORD")
    or os.getenv("MICROSOFT_APP_PWD")
)
BOT_CREDENTIALS_CONFIGURED = bool(MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD)

if _BOTBUILDER_AVAILABLE and BOT_CREDENTIALS_CONFIGURED:
    adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
    bot_adapter = BotFrameworkAdapter(adapter_settings)
else:
    adapter_settings = None
    bot_adapter = None

state_store = TeamsStateStore()
chat_service = TeamsChatService()
tool_router = TeamsToolRouter(chat_service, state_store)


def _as_activity(payload: Dict[str, Any]):
    """Deserialize a dict into a Bot Framework Activity."""
    if not _BOTBUILDER_AVAILABLE or Activity is None:
        return payload
    try:
        return Activity().deserialize(payload)
    except Exception:  # noqa: BLE001
        return Activity(type=ActivityTypes.message, text=payload.get("text") or "")


def _log_startup_credentials():
    app_id = MICROSOFT_APP_ID or ""
    pwd_set = bool(MICROSOFT_APP_PASSWORD)
    guid_like = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", app_id))
    logger.info(
        "Teams bot credentials loaded | app_id_present=%s guid_like=%s password_set=%s botbuilder=%s",
        bool(app_id),
        guid_like,
        pwd_set,
        _BOTBUILDER_AVAILABLE,
    )


_log_startup_credentials()


class DocWainTeamsBot(TeamsActivityHandler):
    """Teams ActivityHandler that supports text, attachments, and Adaptive Card tool actions."""

    def __init__(self):
        super().__init__()
        self.chat_service = chat_service
        self.state_store = state_store
        self.tool_router = tool_router

    @staticmethod
    async def _send_safe(turn_context, activity, log) -> None:
        if MicrosoftAppCredentials is not None:
            MicrosoftAppCredentials.trust_service_url(turn_context.activity.service_url)
        try:
            await turn_context.send_activity(activity)
        except Exception as exc:  # noqa: BLE001
            is_auth = "unauthorized" in str(exc).lower() or "403" in str(exc)
            if is_auth:
                log.error(
                    "Teams outbound unauthorized | serviceUrl=%s channelId=%s conversation_id=%s app_id=%s error=%s",
                    getattr(turn_context.activity, "service_url", None),
                    getattr(turn_context.activity, "channel_id", None),
                    getattr(getattr(turn_context.activity, "conversation", None), "id", None),
                    MICROSOFT_APP_ID,
                    exc,
                )
                log.warning(
                    "Connector unauthorized; check Azure Bot AppId/secret, Teams manifest botId, "
                    "Teams channel enablement, and endpoint routing."
                )
            else:
                log.error("Failed to send activity: %s", exc, exc_info=True)

    async def _send_typing(self, turn_context) -> None:
        """Send a typing indicator so Teams shows '...' while we process."""
        try:
            if Activity is not None and ActivityTypes is not None:
                typing_activity = Activity(type=ActivityTypes.typing)
                typing_activity.relates_to = getattr(turn_context.activity, "relates_to", None)
                await turn_context.send_activity(typing_activity)
        except Exception:  # noqa: BLE001
            pass  # Typing indicator is best-effort; never block on failure

    async def on_message_activity(self, turn_context):
        activity_dict = turn_context.activity.as_dict()
        correlation_id = get_correlation_id(activity=activity_dict)
        log = get_logger(__name__, correlation_id)
        if MicrosoftAppCredentials is not None:
            MicrosoftAppCredentials.trust_service_url(turn_context.activity.service_url)

        # Send typing indicator immediately so Teams shows "typing..."
        await self._send_typing(turn_context)

        if getattr(Config.Teams, "DIAG_MODE", False):
            log.info(
                "Teams diag: channel=%s serviceUrl=%s conversation=%s activity_id=%s from_id=%s",
                getattr(turn_context.activity, "channel_id", None),
                getattr(turn_context.activity, "service_url", None),
                getattr(turn_context.activity.conversation, "id", None) if turn_context.activity.conversation else None,
                getattr(turn_context.activity, "id", None),
                getattr(turn_context.activity.from_property, "id", None)
                if hasattr(turn_context.activity, "from_property")
                else None,
            )
            service_url = getattr(turn_context.activity, "service_url", "") or ""
            if service_url and "smba.trafficmanager.net" not in service_url:
                log.warning("Service URL is non-standard for Teams: %s", service_url)

        user_id = legacy_adapter.extract_user_id(activity_dict)
        session_id = legacy_adapter.extract_session_id(activity_dict)
        context = self.chat_service.build_context(user_id=user_id, session_id=session_id)

        # Automatic ingestion on file uploads
        if turn_context.activity.attachments:
            try:
                self.chat_service.ensure_collection(context.subscription_id)
                ingestion = await ingest_attachments(
                    activity_dict,
                    turn_context,
                    context,
                    correlation_id,
                    state_store=self.state_store,
                )
                success_card = build_card(
                    "upload_success_card",
                    message="Files ingested. You can start chatting now.",
                    filenames=", ".join(ingestion.filenames),
                    documents_created=str(ingestion.documents_created),
                )
                await self._send_safe(turn_context, _as_activity(_card_activity(success_card, text="Upload complete")), log)
                tools_payload = await self.tool_router.handle_action({"action": "show_tools"}, context)
                await self._send_safe(turn_context, _as_activity(tools_payload), log)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Attachment ingest failed: %s", exc, exc_info=True)
                error_card = build_card("error_card", message="Could not ingest your file. Please try again.")
                await self._send_safe(turn_context, _as_activity(_card_activity(error_card)), log)
                return

        # Handle Adaptive Card Submit actions
        if isinstance(turn_context.activity.value, dict) and turn_context.activity.value:
            payload = await self.tool_router.handle_action(turn_context.activity.value, context)
            await self._send_safe(turn_context, _as_activity(payload), log)
            return

        # Text questions
        question = legacy_adapter.extract_question(activity_dict)
        if not question or question.lower() in {"help", "tools"}:
            if question and question.lower() == "tools":
                tools_payload = await self.tool_router.handle_action({"action": "show_tools"}, context)
                await self._send_safe(turn_context, _as_activity(tools_payload), log)
            else:
                await self._send_safe(turn_context, _as_activity(_card_activity(build_card("help_card"))), log)
            return

        try:
            # Show processing card while we work
            processing_card = build_card("processing_card", status_message="Analyzing your question...")
            processing_activity = _as_activity(_card_activity(processing_card))
            try:
                processing_response = await turn_context.send_activity(processing_activity)
            except Exception:  # noqa: BLE001
                processing_response = None

            answer_result = self.chat_service.answer_question(question, context)
            answer = answer_result.answer
            response_text = answer.get("response") or "I could not generate a response."
            sources_text = legacy_adapter.format_sources(answer.get("sources") or [])

            # Delete the processing card before sending the answer
            if processing_response and getattr(processing_response, "id", None):
                try:
                    await turn_context.delete_activity(processing_response.id)
                except Exception:  # noqa: BLE001
                    pass  # Best-effort cleanup

            card = build_card(
                "answer_card",
                title="Answer",
                text=response_text,
                sources_text=sources_text.replace("\n\nSources:\n", "") if sources_text else "No sources available.",
            )
            await self._send_safe(turn_context, _as_activity(_card_activity(card)), log)

            # Persist conversation history for context-aware follow-ups
            try:
                from src.api.dw_chat import add_message_to_history

                add_message_to_history(
                    user_id=user_id,
                    query=question,
                    answer=answer,
                    session_id=context.session_id,
                    new_session=False,
                )
            except Exception as history_exc:  # noqa: BLE001
                log.debug("Teams BF history persistence failed: %s", history_exc)
        except Exception as exc:  # noqa: BLE001
            log.error("Teams question handling failed: %s", exc, exc_info=True)
            error_card = build_card("error_card", message="I hit a snag answering that. Please try again.")
            await self._send_safe(turn_context, _as_activity(_card_activity(error_card)), log)

    async def on_members_added_activity(self, members_added, turn_context):
        """Send a welcome card when the bot is first added to a conversation."""
        log = get_logger(__name__, get_correlation_id(activity=turn_context.activity.as_dict()))
        if MicrosoftAppCredentials is not None:
            MicrosoftAppCredentials.trust_service_url(turn_context.activity.service_url)

        bot_id = getattr(turn_context.activity.recipient, "id", None)
        bot_was_added = any(getattr(m, "id", None) == bot_id for m in members_added)
        if bot_was_added:
            welcome_card = build_card("welcome_card")
            await self._send_safe(
                turn_context,
                _as_activity(_card_activity(welcome_card, text="Welcome to DocWain!")),
                log,
            )

    async def on_teams_file_consent_accept(self, turn_context, file_consent_card_response):
        log = get_logger(__name__, get_correlation_id(activity=turn_context.activity.as_dict()))
        if Activity is not None and ActivityTypes is not None:
            await self._send_safe(
                turn_context,
                Activity(
                    type=ActivityTypes.message,
                    text="Thanks for approving access. Please upload the file so I can ingest it.",
                ),
                log,
            )

    async def on_teams_file_consent_decline(self, turn_context, file_consent_card_response):
        log = get_logger(__name__, get_correlation_id(activity=turn_context.activity.as_dict()))
        if Activity is not None and ActivityTypes is not None:
            await self._send_safe(
                turn_context,
                Activity(type=ActivityTypes.message, text="File consent was declined. I did not ingest anything."),
                log,
            )


async def handle_bot_error(turn_context, error: Exception):  # type: ignore[func-returns-value]
    """Single-attempt error handler with trusted service URL."""
    if MicrosoftAppCredentials is not None:
        MicrosoftAppCredentials.trust_service_url(getattr(turn_context.activity, "service_url", None))
    logger.error("Bot error: %s", error, exc_info=True)
    try:
        await turn_context.send_activity("Sorry, something went wrong processing your request.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to send error activity: %s", exc)


# Configure adapter error handling
if bot_adapter and _BOTBUILDER_AVAILABLE:
    bot_adapter.on_turn_error = handle_bot_error

docwain_teams_bot = DocWainTeamsBot() if _BOTBUILDER_AVAILABLE else None
