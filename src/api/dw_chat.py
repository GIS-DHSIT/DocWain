from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.api.genai_client import generate_text
from src.storage.azure_blob_client import get_chat_container_client, upload_chat_history

logger = get_logger(__name__)

MAX_SESSIONS = int(os.getenv("CHAT_HISTORY_MAX_SESSIONS", "50"))
MAX_MESSAGES_PER_SESSION = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES_PER_SESSION", "200"))
MAX_QUERY_CHARS = int(os.getenv("CHAT_HISTORY_MAX_QUERY_CHARS", "2000"))
MAX_RESPONSE_CHARS = int(os.getenv("CHAT_HISTORY_MAX_RESPONSE_CHARS", "6000"))
MAX_SOURCES_PER_MESSAGE = int(os.getenv("CHAT_HISTORY_MAX_SOURCES_PER_MESSAGE", "8"))
DEFAULT_CONTEXT_MESSAGES = int(os.getenv("CHAT_HISTORY_CONTEXT_MESSAGES", "10"))

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse_iso_to_epoch(value: Optional[str]) -> float:
    raw = (value or "").strip()
    if not raw:
        return 0.0
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return 0.0

def _trim_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text

def _response_text(response: Any) -> str:
    if isinstance(response, dict):
        payload = response.get("response")
        if isinstance(payload, str):
            return payload.strip()
        if payload is not None:
            return str(payload).strip()
        return ""
    if isinstance(response, str):
        return response.strip()
    return str(response or "").strip()

def _sanitize_sources(sources: Any) -> List[Dict[str, Any]]:
    if not isinstance(sources, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for source in sources[:MAX_SOURCES_PER_MESSAGE]:
        if not isinstance(source, dict):
            continue
        item: Dict[str, Any] = {}
        for key in ("source_name", "file_name", "document_id", "page"):
            if key in source and source.get(key) is not None:
                item[key] = source.get(key)
        if not item and source:
            item = dict(source)
        cleaned.append(item)
    return cleaned

def serialize_response(response: Any) -> Any:
    """Serialize responses for storage with bounded size."""
    if isinstance(response, dict):
        text = _trim_text(response.get("response", ""), max_chars=MAX_RESPONSE_CHARS)
        return {
            "response": text,
            "sources": _sanitize_sources(response.get("sources", [])),
        }
    if isinstance(response, str):
        return _trim_text(response, max_chars=MAX_RESPONSE_CHARS)
    return _trim_text(str(response or ""), max_chars=MAX_RESPONSE_CHARS)

def _normalize_message(message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None

    query = _trim_text(message.get("query", ""), max_chars=MAX_QUERY_CHARS)
    if not query:
        return None

    timestamp = str(message.get("timestamp") or _now_iso())
    response = serialize_response(message.get("response"))
    return {
        "query": query,
        "response": response,
        "timestamp": timestamp,
    }

def _normalize_session(session: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(session, dict):
        return None

    raw_messages = session.get("messages") or []
    normalized_messages: List[Dict[str, Any]] = []
    for raw in raw_messages:
        msg = _normalize_message(raw)
        if msg:
            normalized_messages.append(msg)

    if not normalized_messages:
        return None

    if len(normalized_messages) > MAX_MESSAGES_PER_SESSION:
        normalized_messages = normalized_messages[-MAX_MESSAGES_PER_SESSION:]

    created_at = str(session.get("created_at") or normalized_messages[0].get("timestamp") or _now_iso())
    updated_at = str(session.get("updated_at") or normalized_messages[-1].get("timestamp") or created_at)

    session_id = str(session.get("session_id") or f"session_{uuid.uuid4().hex[:12]}")
    title = _trim_text(
        session.get("title") or normalized_messages[0].get("query") or "Chat Session",
        max_chars=80,
    )

    return {
        "session_id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": normalized_messages,
    }

def _normalize_history(history: Any) -> Dict[str, Any]:
    sessions: List[Dict[str, Any]] = []

    # Legacy: list[message]
    if isinstance(history, list) and (not history or isinstance(history[0], dict) and "query" in history[0]):
        if history:
            migrated = _normalize_session(
                {
                    "session_id": f"migrated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "title": history[0].get("query", "Migrated Session"),
                    "created_at": history[0].get("timestamp") or _now_iso(),
                    "updated_at": history[-1].get("timestamp") or _now_iso(),
                    "messages": history,
                }
            )
            if migrated:
                sessions.append(migrated)
    elif isinstance(history, dict):
        raw_sessions = history.get("sessions")
        if isinstance(raw_sessions, list):
            for raw in raw_sessions:
                normalized = _normalize_session(raw)
                if normalized:
                    sessions.append(normalized)

    sessions.sort(key=lambda s: _parse_iso_to_epoch(s.get("updated_at")))
    if len(sessions) > MAX_SESSIONS:
        sessions = sessions[-MAX_SESSIONS:]
    return {"sessions": sessions}

def _latest_session(sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sessions:
        return None
    return max(sessions, key=lambda s: _parse_iso_to_epoch(s.get("updated_at")))

def get_chat_history(user_id: str) -> Dict[str, Any]:
    """Retrieve normalized chat history with sessions from Azure Blob Storage."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        container_client = get_chat_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        payload = json.loads(blob_data.decode("utf-8"))
        history = _normalize_history(payload)
        logger.info("[GET_HISTORY] User=%s sessions=%d", user_id, len(history.get("sessions", [])))
        return history
    except Exception as exc:
        logger.info("[GET_HISTORY] No existing chat history for %s: %s", user_id, exc)
        return {"sessions": []}

def save_chat_history(user_id: str, chat_history: Dict[str, Any]) -> None:
    """Save normalized chat history to Azure Blob Storage."""
    try:
        normalized = _normalize_history(chat_history)
        blob_name = f"chat_history/{user_id}.json"
        payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        chat_data_stream = BytesIO(payload)
        upload_chat_history(blob_name, chat_data_stream.getvalue())
        logger.info("[SAVE_HISTORY] User=%s sessions=%d", user_id, len(normalized.get("sessions", [])))
    except Exception as exc:
        logger.error("Error saving chat history for %s: %s", user_id, exc)

def create_new_session(
    query: Any,
    response: Any,
    session_id: Optional[str] = None,
    initial_message: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new session with the first message.
    Allows supplying a specific session_id and initial message so callers can
    control when a brand-new session is created.
    """
    now = _now_iso()
    message = initial_message or {
        "query": _trim_text(query, max_chars=MAX_QUERY_CHARS),
        "response": serialize_response(response),
        "timestamp": now,
    }
    normalized_message = _normalize_message(message) or {
        "query": _trim_text(query, max_chars=MAX_QUERY_CHARS),
        "response": serialize_response(response),
        "timestamp": now,
    }
    return {
        "session_id": session_id or f"session_{uuid.uuid4().hex[:12]}",
        "title": generate_session_title(normalized_message.get("query", "")),
        "created_at": normalized_message["timestamp"],
        "updated_at": normalized_message["timestamp"],
        "messages": [normalized_message],
    }

def generate_session_title(first_query: Any) -> str:
    """Generate a session title from the first query."""
    title = _trim_text(first_query, max_chars=50)
    return title or "Chat Session"

def _append_message(target_session: Dict[str, Any], new_message: Dict[str, Any]) -> Tuple[int, int]:
    """Append message to a session and return old/new counts."""
    target_session.setdefault("messages", [])
    old_count = len(target_session["messages"])
    target_session["messages"].append(new_message)
    if len(target_session["messages"]) > MAX_MESSAGES_PER_SESSION:
        target_session["messages"] = target_session["messages"][-MAX_MESSAGES_PER_SESSION:]
    target_session["updated_at"] = new_message.get("timestamp") or _now_iso()
    return old_count, len(target_session["messages"])

def add_message_to_history(
    user_id: str,
    query: Any,
    response: Any = None,
    session_id: Optional[str] = None,
    new_session: bool = False,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Session management where frontend controls session IDs.

    Supports legacy callers that pass ``answer=...`` instead of ``response=...``.
    Returns: (history, active_session_id)
    """
    if response is None and "answer" in kwargs:
        response = kwargs.get("answer")

    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    now = _now_iso()

    new_message = _normalize_message(
        {
            "query": query,
            "response": response,
            "timestamp": now,
        }
    )
    if not new_message:
        return history, session_id

    target_session: Optional[Dict[str, Any]] = None
    appended_to_existing = False

    if session_id:
        target_session = next((s for s in sessions if s.get("session_id") == session_id), None)
        if target_session is None:
            session_obj = create_new_session(
                query,
                response,
                session_id=session_id,
                initial_message=new_message,
            )
            sessions.append(session_obj)
            target_session = session_obj
        else:
            _append_message(target_session, new_message)
            appended_to_existing = True
    elif new_session or not sessions:
        session_obj = create_new_session(query, response, initial_message=new_message)
        sessions.append(session_obj)
        target_session = session_obj
    else:
        target_session = _latest_session(sessions)
        if target_session is None:
            session_obj = create_new_session(query, response, initial_message=new_message)
            sessions.append(session_obj)
            target_session = session_obj
        else:
            _append_message(target_session, new_message)
            appended_to_existing = True

    if target_session and not appended_to_existing and target_session.get("updated_at") != new_message.get("timestamp"):
        target_session["updated_at"] = new_message.get("timestamp") or _now_iso()

    normalized_history = _normalize_history({"sessions": sessions})
    save_chat_history(user_id, normalized_history)

    active_session_id = target_session.get("session_id") if target_session else None
    return normalized_history, active_session_id

def get_current_session_context(
    user_id: str,
    session_id: Optional[str] = None,
    max_messages: int = DEFAULT_CONTEXT_MESSAGES,
) -> List[Dict[str, Any]]:
    """Get the latest messages for a session as compact query/response context."""
    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    if not sessions:
        return []

    current_session = None
    if session_id:
        current_session = next((s for s in sessions if s.get("session_id") == session_id), None)
    if current_session is None:
        current_session = _latest_session(sessions)
    if current_session is None:
        return []

    messages = current_session.get("messages", [])
    selected = messages[-max_messages:] if len(messages) > max_messages else messages
    return [
        {
            "query": msg.get("query", ""),
            "response": _response_text(msg.get("response")),
            "timestamp": msg.get("timestamp"),
        }
        for msg in selected
        if isinstance(msg, dict)
    ]

def delete_chat_history(user_id: str) -> Dict[str, str]:
    """Delete chat history from Azure Blob Storage."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        container_client = get_chat_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
        logger.info("Chat history deleted for %s", user_id)
        return {"status": "success", "message": f"Chat history deleted for {user_id}"}
    except Exception as exc:
        logger.error("Error deleting chat history for %s: %s", user_id, exc)
        return {"status": "error", "message": "Error deleting chat history"}

def delete_session(user_id: str, session_id: str) -> Dict[str, str]:
    """Delete a specific session from chat history."""
    try:
        history = get_chat_history(user_id)
        sessions = history.get("sessions", [])
        updated_sessions = [s for s in sessions if s.get("session_id") != session_id]

        if len(updated_sessions) == len(sessions):
            return {"status": "error", "message": "Session not found"}

        updated_history = {"sessions": updated_sessions}
        save_chat_history(user_id, updated_history)

        logger.info("Session %s deleted for %s", session_id, user_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as exc:
        logger.error("Error deleting session %s for %s: %s", session_id, user_id, exc)
        return {"status": "error", "message": "Error deleting session"}

def get_session_list(user_id: str) -> List[Dict[str, Any]]:
    """Get list of all sessions with metadata (for sidebar display)."""
    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    ordered = sorted(sessions, key=lambda s: _parse_iso_to_epoch(s.get("updated_at")), reverse=True)
    return [
        {
            "session_id": session.get("session_id"),
            "title": session.get("title"),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "message_count": len(session.get("messages", [])),
        }
        for session in ordered
    ]

def get_session_by_id(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific session's full chat history."""
    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    for session in sessions:
        if session.get("session_id") == session_id:
            return session
    return None

def search_sessions(
    user_id: str,
    query: str,
    profile_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search sessions by query text and optional profile scope.

    Returns matching sessions with the matching message highlighted.
    """
    if not query or not query.strip():
        return []

    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    query_lower = query.strip().lower()
    results: List[Dict[str, Any]] = []

    for session in sessions:
        # Profile filter if specified
        if profile_id and session.get("profile_id") and session["profile_id"] != profile_id:
            continue

        for msg in session.get("messages", []):
            msg_query = (msg.get("query") or "").lower()
            msg_response = ""
            resp = msg.get("response")
            if isinstance(resp, dict):
                msg_response = (resp.get("response") or "").lower()
            elif isinstance(resp, str):
                msg_response = resp.lower()

            if query_lower in msg_query or query_lower in msg_response:
                results.append({
                    "session_id": session.get("session_id"),
                    "title": session.get("title"),
                    "matched_query": msg.get("query"),
                    "timestamp": msg.get("timestamp"),
                })
                break  # One match per session is enough

    return results


def get_session_summary(user_id: str, session_id: str) -> Optional[str]:
    """Get a progressive summary of a session's conversation.

    Returns a ~200-token running summary of the session's key topics
    and findings, useful for context injection without sending full history.
    """
    session = get_session_by_id(user_id, session_id)
    if not session:
        return None

    messages = session.get("messages", [])
    if not messages:
        return None

    # Build a compact representation for summarization
    exchange_lines: List[str] = []
    for msg in messages[-10:]:  # Last 10 messages max
        q = _trim_text(msg.get("query", ""), max_chars=200)
        r = _response_text(msg.get("response"))[:200]
        exchange_lines.append(f"Q: {q}\nA: {r}")

    exchanges = "\n---\n".join(exchange_lines)

    try:
        prompt = (
            f"Summarize this conversation in 2-3 sentences, focusing on "
            f"the key topics discussed and findings:\n\n{exchanges}"
        )
        text, _ = generate_text(
            api_key=Config.Gemini.GEMINI_API_KEY,
            model="gemini-2.5-flash",
            prompt=prompt,
        )
        return text if text else None
    except Exception as exc:
        logger.error("Error generating session summary: %s", exc)
        return None


def generate_follow_up_questions(retrieved_text: str, num_questions: int = 3):
    """Generates follow-up questions using Gemini 2.5 Flash based on retrieved content."""
    try:
        prompt = f"""
Strictly based on the following retrieved context, suggest {num_questions} relevant follow-up questions.
Keep them short and precise.

Context:
{retrieved_text}
""".strip()

        text, _ = generate_text(
            api_key=Config.Gemini.GEMINI_API_KEY,
            model="gemini-2.5-flash",
            prompt=prompt,
        )
        return text if text else []
    except Exception as exc:
        logger.error("Error generating follow-up questions: %s", exc)
        return []
