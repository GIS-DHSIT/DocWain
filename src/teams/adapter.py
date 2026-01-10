import hashlib
import hmac
import logging
import re
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.api.dw_chat import add_message_to_history
from src.teams.attachments import handle_attachments
from src.teams.logic import TeamsAnswerResult, TeamsChatError, TeamsChatService
from src.utils.logging_utils import get_correlation_id, get_logger

logger = logging.getLogger(__name__)
TEAMS_CHAT_SERVICE = TeamsChatService()


class TeamsAuthError(Exception):
    """Raised when the Teams request fails authentication."""


def _get_header(headers: Dict[str, str], key: str) -> str:
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.upper()) or ""


def verify_shared_secret(headers: Dict[str, str], raw_body: Optional[bytes] = None) -> None:
    """Check shared secret (simple header-based auth for Teams outbound calls)."""
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


async def handle_attachment_activity(activity: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Handle Teams activities that include file attachments."""
    session_id = extract_session_id(activity)
    user_id = extract_user_id(activity)
    context = TEAMS_CHAT_SERVICE.build_context(user_id=user_id, session_id=session_id)
    TEAMS_CHAT_SERVICE.ensure_collection(context.subscription_id)
    return await handle_attachments(activity, context, correlation_id)


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
        return await handle_attachment_activity(activity, correlation_id)

    question = extract_question(activity)
    if not question:
        return {"type": "message", "text": "I did not receive a question. Please type your message again."}

    user_id = extract_user_id(activity)
    session_id = extract_session_id(activity)
    context = TEAMS_CHAT_SERVICE.build_context(
        user_id=user_id,
        session_id=session_id,
        model_name=Config.Teams.DEFAULT_MODEL,
        persona=Config.Teams.DEFAULT_PERSONA,
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
                answer=answer,
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
