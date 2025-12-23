import logging
import re
from typing import Any, Dict, List

from api.config import Config
from api.dw_newron import answer_question

logger = logging.getLogger(__name__)


class TeamsAuthError(Exception):
    """Raised when the Teams request fails authentication."""


def _get_header(headers: Dict[str, str], key: str) -> str:
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.upper()) or ""


def verify_shared_secret(headers: Dict[str, str]) -> bool:
    """Check shared secret (simple header-based auth for Teams outbound calls)."""
    expected = Config.Teams.SHARED_SECRET
    if not expected:
        # No secret configured; allow for local/dev use.
        return True

    candidate = _get_header(headers, "x-teams-shared-secret") or _get_header(headers, "x-teams-signature")
    if not candidate:
        candidate = _get_header(headers, "authorization")
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()

    if candidate != expected:
        logger.warning("Invalid Teams shared secret provided")
        return False
    return True


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
    return convo.get("id") or "teams-session"


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


def handle_teams_activity(activity: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Entrypoint used by FastAPI route to serve Teams messages."""
    if not verify_shared_secret(headers):
        raise TeamsAuthError("Unauthorized Teams request")

    question = extract_question(activity)
    if not question:
        return {"type": "message", "text": "I did not receive a question. Please type your message again."}

    user_id = extract_user_id(activity)
    session_id = extract_session_id(activity)

    answer = answer_question(
        query=question,
        user_id=user_id,
        profile_id=Config.Teams.DEFAULT_PROFILE,
        subscription_id=Config.Teams.DEFAULT_SUBSCRIPTION,
        model_name=Config.Teams.DEFAULT_MODEL,
        persona=Config.Teams.DEFAULT_PERSONA,
    )

    # Log session mapping to help debug multi-user chats.
    logger.info("Teams message handled | user=%s session=%s", user_id, session_id)

    return build_teams_message(answer)
