import hashlib
import hmac
import re
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.api.dw_chat import add_message_to_history
from src.teams.attachments import AttachmentIngestError, IngestionResult, ingest_attachments
from src.teams.logic import TeamsAnswerResult, TeamsChatError, TeamsChatService
from src.teams.state import TeamsStateStore
from src.utils.logging_utils import get_correlation_id, get_logger

logger = get_logger(__name__)
TEAMS_CHAT_SERVICE = TeamsChatService()
STATE_STORE = TeamsStateStore()

class TeamsAuthError(Exception):
    """Raised when the Teams request fails authentication."""

def _get_header(headers: Dict[str, str], key: str) -> str:
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.upper()) or ""

def _is_botframework_jwt(headers: Optional[Dict[str, str]]) -> bool:
    """Detect a Bot Framework bearer token (JWT) to bypass shared-secret auth."""
    if not headers:
        return False
    auth_header = _get_header(headers, "authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header[7:].strip()
    return token.count(".") == 2

def verify_shared_secret(headers: Dict[str, str], raw_body: Optional[bytes] = None) -> None:
    """Check shared secret (simple header-based auth for Teams outbound calls)."""
    if _is_botframework_jwt(headers):
        # Bot Framework JWT present; rely on adapter validation instead.
        return

    expected = Config.Teams.SHARED_SECRET
    if not expected:
        # No secret configured; allow for local/dev use.
        return

    candidate = _get_header(headers, "x-teams-shared-secret")
    if not candidate and not Config.Teams.SIGNATURE_ENABLED:
        candidate = _get_header(headers, "x-teams-signature")
    if not candidate:
        candidate = _get_header(headers, "authorization")
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()

    if not candidate:
        raise TeamsAuthError("Missing Teams shared secret")
    if not hmac.compare_digest(candidate, expected):
        logger.warning("Invalid Teams shared secret provided")
        raise TeamsAuthError("Invalid Teams shared secret")

    if Config.Teams.SIGNATURE_ENABLED:
        signature = _get_header(headers, "x-teams-signature")
        if not signature:
            raise TeamsAuthError("Missing Teams signature")
        signature = signature.replace("sha256=", "")
        computed = hmac.new(expected.encode("utf-8"), raw_body or b"", hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, computed):
            raise TeamsAuthError("Invalid Teams signature")

def _strip_mentions(text: str) -> str:
    """Remove <at> mentions so the question text is clean."""
    if not text:
        return ""
    return re.sub(r"<at[^>]*>.*?</at>", "", text).strip()

def extract_question(activity: Dict[str, Any]) -> str:
    """Pull the user question from the Teams activity payload."""
    return _strip_mentions((activity.get("text") or "").strip())

def extract_user_id(activity: Dict[str, Any]) -> str:
    """Derive a stable user id from Teams activity."""
    user = activity.get("from", {}) or {}
    return (
        user.get("aadObjectId")
        or user.get("id")
        or user.get("userPrincipalName")
        or "teams_user"
    )

def extract_session_id(activity: Dict[str, Any]) -> str:
    """Use conversation id as a lightweight session key."""
    convo = activity.get("conversation") or {}
    channel_data = activity.get("channelData") or {}
    return (
        convo.get("id")
        or channel_data.get("team", {}).get("id")
        or channel_data.get("channel", {}).get("id")
        or channel_data.get("teamsChannelId")
        or "teams-session"
    )

def format_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = []
    for src in sources[:5]:
        name = src.get("source_name") or src.get("source_id") or "Source"
        excerpt = (src.get("excerpt") or "")[:180]
        if excerpt:
            lines.append(f"- {name}: {excerpt}")
        else:
            lines.append(f"- {name}")
    return "\n\nSources:\n" + "\n".join(lines)

def build_teams_message(answer_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a Bot Framework message activity from the DocWain answer."""
    text = answer_payload.get("response") or "I could not generate a response."
    sources_text = format_sources(answer_payload.get("sources") or [])
    if sources_text:
        text = f"{text}\n\n{sources_text}"

    return {
        "type": "message",
        "text": text,
    }

def answer_question(*args, **kwargs):
    """
    Lazy import wrapper to avoid loading heavy RAG deps when the Teams adapter
    is imported in environments without the full stack (e.g., unit tests).
    """
    try:
        from src.api.dw_newron import answer_question as _answer_question  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive guard for missing deps
        raise RuntimeError(
            "DocWain RAG dependencies are missing; install requirements.txt to enable Teams Q&A"
        ) from exc
    return _answer_question(*args, **kwargs)

async def handle_attachment_activity(activity: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Handle Teams activities that include file attachments."""
    correlation_id = correlation_id or get_correlation_id(activity=activity, headers=None)
    session_id = extract_session_id(activity)
    user_id = extract_user_id(activity)
    pref_subscription = session_id if Config.Teams.SESSION_AS_SUBSCRIPTION else Config.Teams.DEFAULT_SUBSCRIPTION
    pref_profile = user_id if Config.Teams.PROFILE_PER_USER else Config.Teams.DEFAULT_PROFILE
    prefs = STATE_STORE.get_preferences(pref_subscription, pref_profile)
    context = TEAMS_CHAT_SERVICE.build_context(
        user_id=user_id,
        session_id=session_id,
        model_name=prefs.get("model_name") or Config.Teams.DEFAULT_MODEL,
        persona=prefs.get("persona") or Config.Teams.DEFAULT_PERSONA,
    )
    TEAMS_CHAT_SERVICE.ensure_collection(context.subscription_id)
    attachments = activity.get("attachments") or []
    image_only = all(
        (att.get("contentType") or "").lower().startswith("image/") and not att.get("contentUrl")
        for att in attachments
    )
    if attachments and image_only:
        return {
            "type": "message",
            "text": "Please upload the image using the file attachment option so DocWain can ingest it.",
        }

    try:
        result: IngestionResult = await ingest_attachments(
            activity,
            turn_context=None,
            context=context,
            correlation_id=correlation_id,
            state_store=STATE_STORE,
        )
    except AttachmentIngestError as exc:
        message = str(exc)
        if "download URL" in message:
            message = "Attachment could not be processed because the download link was missing."
        elif "Unsupported attachment type" in message:
            message = "Unsupported attachment type. Please upload the image or file using the attachment picker."
        return {"type": "message", "text": message}

    filenames = ", ".join(result.filenames)
    return {
        "type": "message",
        "text": (
            f"Successfully processed: {filenames}. "
            f"Ingested {result.documents_created} document(s). "
            "You can start chatting now."
        ),
    }

async def handle_teams_activity(
    activity: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    raw_body: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Entrypoint used by FastAPI route to serve Teams messages."""
    correlation_id = get_correlation_id(activity=activity, headers=headers)
    log = get_logger(__name__, correlation_id)
    if headers:
        verify_shared_secret(headers, raw_body=raw_body)
    elif getattr(Config.Teams, "SHARED_SECRET", ""):
        log.warning("Teams shared secret configured but no headers provided; skipping verification.")

    if activity.get("attachments"):
        try:
            return await handle_attachment_activity(activity)
        except AttachmentIngestError as exc:
            log.error("Teams attachment ingest error: %s", exc)
            return {"type": "message", "text": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected attachment ingest error: %s", exc, exc_info=True)
            return {"type": "message", "text": "Unable to ingest the attachment securely. Please try again."}

    question = extract_question(activity)
    if not question:
        return {"type": "message", "text": "I did not receive a question. Please type your message again."}

    user_id = extract_user_id(activity)
    session_id = extract_session_id(activity)
    pref_subscription = session_id if Config.Teams.SESSION_AS_SUBSCRIPTION else Config.Teams.DEFAULT_SUBSCRIPTION
    pref_profile = user_id if Config.Teams.PROFILE_PER_USER else Config.Teams.DEFAULT_PROFILE
    prefs = STATE_STORE.get_preferences(pref_subscription, pref_profile)
    context = TEAMS_CHAT_SERVICE.build_context(
        user_id=user_id,
        session_id=session_id,
        model_name=prefs.get("model_name") or Config.Teams.DEFAULT_MODEL,
        persona=prefs.get("persona") or Config.Teams.DEFAULT_PERSONA,
    )

    try:
        answer_result: TeamsAnswerResult = TEAMS_CHAT_SERVICE.answer_question(question, context)
        answer = answer_result.answer
        subscription_id = answer_result.subscription_id
        profile_id = answer_result.profile_id

        # Persist conversation history for context-aware follow-ups
        try:
            add_message_to_history(
                user_id=user_id,
                query=question,
                response=answer,
                session_id=context.session_id,
                new_session=False,
            )
        except Exception as history_exc:
            log.debug("Teams history persistence failed: %s", history_exc)
    except TeamsChatError as exc:
        log.error("Failed to answer Teams question: %s", exc, exc_info=True)
        return {
            "type": "message",
            "text": "I hit a snag answering your question. Please try again in a moment.",
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to answer Teams question: %s", exc, exc_info=True)
        return {
            "type": "message",
            "text": "I hit a snag answering your question. Please try again in a moment.",
        }

    # Log session mapping to help debug multi-user chats.
    log.info(
        "Teams message handled | user=%s session=%s subscription=%s profile=%s internet_fallback=%s",
        user_id,
        session_id,
        subscription_id,
        profile_id,
        answer_result.internet_mode,
    )

    return build_teams_message(answer)
