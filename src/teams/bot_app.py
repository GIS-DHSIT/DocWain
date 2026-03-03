import logging
import os
import re
from typing import Any, Dict, Optional

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
from src.teams.attachments import ingest_attachments, DocumentIntelligence, ScreeningSummary
from src.teams.cards import build_card
from src.teams.insights import generate_proactive_insights, get_domain_actions
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
MICROSOFT_APP_TENANT_ID = (
    Config.Teams.BOT_APP_TENANT_ID
    or os.getenv("MICROSOFT_APP_TENANT_ID")
    or os.getenv("MSA_APP_TENANT_ID")
)
MICROSOFT_APP_TYPE = Config.Teams.BOT_APP_TYPE or "SingleTenant"
BOT_CREDENTIALS_CONFIGURED = bool(MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD)

if _BOTBUILDER_AVAILABLE and BOT_CREDENTIALS_CONFIGURED:
    _adapter_kwargs = {
        "app_id": MICROSOFT_APP_ID,
        "app_password": MICROSOFT_APP_PASSWORD,
    }
    # Single-tenant bots MUST set channel_auth_tenant so the adapter validates
    # tokens issued by the specific tenant rather than the multi-tenant endpoint.
    if MICROSOFT_APP_TYPE.lower() == "singletenant" and MICROSOFT_APP_TENANT_ID:
        _adapter_kwargs["channel_auth_tenant"] = MICROSOFT_APP_TENANT_ID
        logger.info(
            "Configuring BotFrameworkAdapter for SingleTenant | tenant=%s",
            MICROSOFT_APP_TENANT_ID,
        )
    elif MICROSOFT_APP_TYPE.lower() == "singletenant" and not MICROSOFT_APP_TENANT_ID:
        logger.warning(
            "Bot is configured as SingleTenant but MICROSOFT_APP_TENANT_ID is not set. "
            "Token validation will likely fail. Set MICROSOFT_APP_TENANT_ID in the environment."
        )
    else:
        logger.info("Configuring BotFrameworkAdapter for %s (no tenant restriction)", MICROSOFT_APP_TYPE)

    adapter_settings = BotFrameworkAdapterSettings(**_adapter_kwargs)
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
    tenant_id = MICROSOFT_APP_TENANT_ID or ""
    app_type = MICROSOFT_APP_TYPE or "unknown"
    guid_like = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", app_id))
    tenant_guid_like = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", tenant_id))
    logger.info(
        "Teams bot credentials loaded | app_id_present=%s guid_like=%s password_set=%s "
        "tenant_id_present=%s tenant_guid_like=%s app_type=%s botbuilder=%s",
        bool(app_id),
        guid_like,
        pwd_set,
        bool(tenant_id),
        tenant_guid_like,
        app_type,
        _BOTBUILDER_AVAILABLE,
    )


_log_startup_credentials()


def _build_screening_card(filenames, screening_results):
    """Build a screening summary card from screening results."""
    if not screening_results:
        return None

    # Aggregate results across all files
    worst_risk = "LOW"
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    max_score = 0.0
    all_findings = []
    total_tools = 0

    for result in screening_results:
        if risk_order.get(result.risk_level, 0) > risk_order.get(worst_risk, 0):
            worst_risk = result.risk_level
        if result.overall_score > max_score:
            max_score = result.overall_score
        all_findings.extend(result.top_findings)
        total_tools = max(total_tools, result.tools_run)

    risk_color = {"LOW": "Good", "MEDIUM": "Warning", "HIGH": "Attention"}.get(worst_risk, "Default")

    if all_findings:
        findings_text = "**Key findings:**\n" + "\n".join(f"- {f}" for f in all_findings[:5])
    else:
        findings_text = "No significant findings detected."

    tools_summary = f"{total_tools} screening tool(s) applied across {len(screening_results)} file(s)"

    return build_card(
        "screening_summary_card",
        risk_level=worst_risk,
        risk_color=risk_color,
        overall_score=f"{max_score:.1f}/100",
        filenames=", ".join(filenames[:3]) + ("..." if len(filenames) > 3 else ""),
        findings_text=findings_text,
        tools_summary=tools_summary,
    )


def _build_di_report_card(ingestion, log) -> Optional[Dict[str, Any]]:
    """Build a Document Intelligence report card from ingestion results."""
    di_results = ingestion.intelligence_results
    if not di_results:
        return None

    # Use the first document's intelligence (primary document)
    primary = di_results[0]
    domain = primary.doc_type or "general"
    domain_badge = f"[{domain.title()}]"

    # Generate proactive insights via LLM (non-blocking — falls back gracefully)
    try:
        insights = generate_proactive_insights(
            doc_type=domain,
            summary=primary.summary,
            key_entities=primary.key_entities,
            key_facts=primary.key_facts,
            filename=ingestion.filenames[0] if ingestion.filenames else "document",
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("Proactive insights generation failed (non-blocking): %s", exc)
        insights = None

    # Summary text
    summary_text = primary.summary or "Document processed successfully."

    # Insights text
    if insights and insights.insights:
        insights_text = "\n".join(f"- {i}" for i in insights.insights)
    elif primary.key_facts:
        insights_text = "\n".join(f"- {f}" for f in primary.key_facts[:3])
    else:
        insights_text = "- Document classified and indexed for Q&A"

    # Security text from screening results
    if ingestion.screening_results:
        sr = ingestion.screening_results[0]
        security_text = f"Security: {sr.risk_level} risk (score: {sr.overall_score:.0f}/100)"
    else:
        security_text = "Security: Screening skipped"

    # Metadata text
    chunks = primary.chunks_created or ingestion.documents_created
    file_count = len(ingestion.filenames)
    filenames_str = ", ".join(ingestion.filenames[:2])
    if file_count > 2:
        filenames_str += f" (+{file_count - 2})"
    metadata_text = f"{filenames_str} | {chunks} chunk(s)"

    # Domain-specific action buttons
    domain_actions = get_domain_actions(domain)
    action_data = {}
    for i, act in enumerate(domain_actions[:3], 1):
        action_data[f"action{i}_title"] = act.get("title", f"Action {i}")
        action_data[f"action{i}_query"] = act.get("query", "")

    return build_card(
        "di_report_card",
        domain_badge=domain_badge,
        summary_text=summary_text,
        insights_text=insights_text,
        security_text=security_text,
        metadata_text=metadata_text,
        **action_data,
    )


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
            resp = await turn_context.send_activity(activity)
            log.info(
                "Teams reply sent | conversation=%s activity_id=%s",
                getattr(getattr(turn_context.activity, "conversation", None), "id", None),
                getattr(resp, "id", None),
            )
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

        # Automatic ingestion on file uploads.
        # Use the raw activity payload from turn_state (preserves attachments that
        # BotBuilder Activity.deserialize() may silently drop) and fall back to
        # the deserialized Activity object.
        _raw_activity = turn_context.turn_state.get("raw_activity") or {}
        # Filter to dicts only — Bot Framework may include non-dict values in the attachments array
        _raw_attachments = [a for a in (_raw_activity.get("attachments") or []) if isinstance(a, dict)]
        _sdk_attachments = turn_context.activity.attachments or []
        log.info(
            "Attachment check | raw_payload=%d sdk_activity=%d",
            len(_raw_attachments),
            len(_sdk_attachments),
        )
        if _raw_attachments:
            def _att_info(a):
                content = a.get("content")
                has_url = bool(a.get("contentUrl") or (content.get("downloadUrl") if isinstance(content, dict) else False))
                return (a.get("contentType"), a.get("name"), has_url)
            log.info("Raw attachments: %s", [_att_info(a) for a in _raw_attachments])

        _SKIP_CONTENT_PREFIXES = ("application/vnd.microsoft.card",)

        def _is_file_attachment_dict(att: dict) -> bool:
            ct = (att.get("contentType") or "").lower()
            if not ct or ct == "unknown":
                return False
            if any(ct.startswith(skip) for skip in _SKIP_CONTENT_PREFIXES):
                return False
            content = att.get("content")
            has_name = bool(att.get("name"))
            has_content_url = bool(att.get("contentUrl"))
            has_download_url = (content.get("downloadUrl") or content.get("download_url")) if isinstance(content, dict) else False
            has_url = has_content_url or bool(has_download_url)
            # Teams file download info (always a real file)
            if "file.download.info" in ct:
                return True
            if "teams" in ct and "file" in ct:
                return True
            # Reject attachments without a name AND without a URL —
            # these are message wrappers (e.g. Teams sends text/html for rich text)
            if not has_name and not has_url:
                return False
            # Standard MIME types (PDF, Word, Excel, images, etc.)
            if any(ct.startswith(p) for p in (
                "application/pdf", "application/msword", "application/vnd.openxmlformats",
                "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
                "text/plain", "text/csv", "text/html",
                "image/png", "image/jpeg", "image/tiff", "image/gif",
                "application/octet-stream",
            )):
                return True
            # Has a downloadable URL (DirectLine/Web Chat uploads)
            if has_url:
                return True
            return False

        # Prefer raw payload (never lost by BotBuilder deserialization)
        _file_attachment_dicts = [a for a in _raw_attachments if _is_file_attachment_dict(a)]

        # Fallback: check SDK activity attachments if raw had nothing
        if not _file_attachment_dicts and _sdk_attachments:
            for a in _sdk_attachments:
                ct = (getattr(a, "content_type", None) or "").lower()
                if not ct or ct == "unknown":
                    continue
                if any(ct.startswith(skip) for skip in _SKIP_CONTENT_PREFIXES):
                    continue
                att_dict = a.as_dict() if hasattr(a, "as_dict") else {"contentType": ct, "name": getattr(a, "name", None), "contentUrl": getattr(a, "content_url", None)}
                if _is_file_attachment_dict(att_dict):
                    _file_attachment_dicts.append(att_dict)

        if _file_attachment_dicts:
            log.info("Detected %d file attachment(s) for ingestion", len(_file_attachment_dicts))

            # Send immediate progress indicator so user knows upload is being processed
            _file_names_preview = ", ".join(
                a.get("name") or "file" for a in _file_attachment_dicts[:3]
            )
            if len(_file_attachment_dicts) > 3:
                _file_names_preview += f" (+{len(_file_attachment_dicts) - 3} more)"
            _progress_card = build_card(
                "processing_card",
                status_message=f"Processing {len(_file_attachment_dicts)} file(s): {_file_names_preview}",
            )
            _progress_response = None
            try:
                _progress_response = await turn_context.send_activity(
                    _as_activity(_card_activity(_progress_card, text="Processing your upload..."))
                )
            except Exception:  # noqa: BLE001
                pass  # Best-effort progress indication

            # Build an activity dict with only file attachments for ingest_attachments()
            _ingest_activity = dict(activity_dict)
            _ingest_activity["attachments"] = _file_attachment_dicts
            try:
                self.chat_service.ensure_collection(context.subscription_id)
                ingestion = await ingest_attachments(
                    _ingest_activity,
                    turn_context,
                    context,
                    correlation_id,
                    state_store=self.state_store,
                )

                # Remove progress card now that processing is done
                if _progress_response and getattr(_progress_response, "id", None):
                    try:
                        await turn_context.delete_activity(_progress_response.id)
                    except Exception:  # noqa: BLE001
                        pass  # Best-effort cleanup

                # Build DI report card (rich domain-aware) or fall back to generic success card
                di_card = _build_di_report_card(ingestion, log)
                if di_card:
                    await self._send_safe(
                        turn_context,
                        _as_activity(_card_activity(di_card, text="Document analyzed")),
                        log,
                    )
                else:
                    success_card = build_card(
                        "upload_success_card",
                        message="Files ingested. You can start chatting now.",
                        filenames=", ".join(ingestion.filenames),
                        documents_created=str(ingestion.documents_created),
                    )
                    await self._send_safe(turn_context, _as_activity(_card_activity(success_card, text="Upload complete")), log)

                # Show screening summary if available (only when no DI card — DI card already includes security)
                if not di_card and ingestion.screening_results:
                    screening_card = _build_screening_card(ingestion.filenames, ingestion.screening_results)
                    if screening_card:
                        await self._send_safe(turn_context, _as_activity(_card_activity(screening_card, text="Security screening complete")), log)
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Attachment ingest failed: %s", exc, exc_info=True)
                # Remove progress card on error too
                if _progress_response and getattr(_progress_response, "id", None):
                    try:
                        await turn_context.delete_activity(_progress_response.id)
                    except Exception:  # noqa: BLE001
                        pass
                error_card = build_card("error_card", message="Could not ingest your file. Please try again.")
                await self._send_safe(turn_context, _as_activity(_card_activity(error_card)), log)
                return

        # Handle Adaptive Card Submit actions
        if isinstance(turn_context.activity.value, dict) and turn_context.activity.value:
            payload = await self.tool_router.handle_action(turn_context.activity.value, context)
            await self._send_safe(turn_context, _as_activity(payload), log)
            return

        # Text questions — handle special commands first
        question = legacy_adapter.extract_question(activity_dict)
        _lower_q = (question or "").lower().strip()
        if not question or _lower_q in {"help", "tools", "delete", "delete docs", "delete documents", "delete all"}:
            if _lower_q == "tools":
                tools_payload = await self.tool_router.handle_action({"action": "show_tools"}, context)
                await self._send_safe(turn_context, _as_activity(tools_payload), log)
            elif _lower_q in {"delete", "delete docs", "delete documents", "delete all"}:
                delete_payload = await self.tool_router.handle_action({"action": "delete_documents"}, context)
                await self._send_safe(turn_context, _as_activity(delete_payload), log)
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
            sources = answer.get("sources") or []
            sources_text = legacy_adapter.format_sources(sources)
            grounded = answer.get("grounded", False)

            # Domain badge from answer metadata
            domain = answer.get("domain") or ""
            domain_badge = f"[{domain.title()}]" if domain and domain != "generic" else ""

            # Confidence indicator
            source_count = len(sources)
            if grounded and source_count >= 3:
                confidence_text = f"High confidence ({source_count} sources)"
            elif grounded and source_count >= 1:
                confidence_text = f"Partial confidence ({source_count} source{'s' if source_count > 1 else ''})"
            elif source_count > 0:
                confidence_text = f"Low confidence ({source_count} source{'s' if source_count > 1 else ''})"
            else:
                confidence_text = ""

            sources_toggle = f"Show sources ({source_count})" if source_count else "Show sources"

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
                domain_badge=domain_badge,
                confidence_text=confidence_text,
                sources_text=sources_text.replace("\n\nSources:\n", "") if sources_text else "No sources available.",
                sources_toggle_title=sources_toggle,
            )
            # Send card with text fallback — text is shown if Adaptive Card can't render
            await self._send_safe(turn_context, _as_activity(_card_activity(card, text=response_text)), log)

            # Persist conversation history for context-aware follow-ups
            try:
                from src.api.dw_chat import add_message_to_history

                add_message_to_history(
                    user_id=user_id,
                    query=question,
                    response=answer,
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
