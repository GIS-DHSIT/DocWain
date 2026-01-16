import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple

from azure.storage.blob import BlobServiceClient, ContentSettings

from src.api.config import Config
from src.api.genai_client import generate_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

connection_string = Config.AzureBlob.CONNECTION_STRING
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = Config.AzureBlob.CONTAINER_NAME
container_client = blob_service_client.get_container_client(container_name)


def get_chat_history(user_id: str) -> Dict:
    """Retrieve chat history with sessions from Azure Blob Storage."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        chat_history = json.loads(blob_data.decode("utf-8"))

        if isinstance(chat_history, list):
            if chat_history:
                logging.info(f"Migrating old format for {user_id}: {len(chat_history)} messages")
                migrated_session = {
                    "session_id": "migrated_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                    "title": chat_history[0].get("query", "Migrated Session")[:50],
                    "created_at": chat_history[0].get("timestamp", datetime.utcnow().isoformat()),
                    "updated_at": chat_history[-1].get("timestamp", datetime.utcnow().isoformat()),
                    "messages": chat_history,
                }
                chat_history = {"sessions": [migrated_session]}
            else:
                chat_history = {"sessions": []}
        elif "sessions" not in chat_history:
            chat_history = {"sessions": []}

        logging.info(f"[GET_HISTORY] User {user_id}: {len(chat_history.get('sessions', []))} sessions")
        return chat_history
    except Exception as e:
        logging.info(f"[GET_HISTORY] No existing chat history for {user_id}: {e}")
        return {"sessions": []}


def save_chat_history(user_id: str, chat_history: Dict) -> None:
    """Save chat history to Azure Blob Storage."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        content_settings = ContentSettings(content_type="application/json")
        chat_data_stream = BytesIO(json.dumps(chat_history, ensure_ascii=False, indent=2).encode("utf-8"))
        blob_client.upload_blob(chat_data_stream, overwrite=True, content_settings=content_settings)
        logging.info(f"Chat history saved for user {user_id}")
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")


def serialize_response(response):
    """Serialize responses for storage."""
    if isinstance(response, dict):
        return {
            "response": response.get("response", ""),
            "sources": response.get("sources", []),
        }
    if isinstance(response, str):
        return response
    return str(response)


def create_new_session(query, response, session_id=None, initial_message=None):
    """
    Create a new session with the first message.
    Allows supplying a specific session_id and initial message so callers can
    control when a brand-new session is created.
    """
    now = datetime.utcnow()
    message = initial_message or {
        "query": str(query),
        "response": serialize_response(response),
        "timestamp": now.isoformat(),
    }
    return {
        "session_id": session_id or now.strftime("%Y%m%d_%H%M%S"),
        "title": generate_session_title(query),
        "created_at": message["timestamp"],
        "updated_at": message["timestamp"],
        "messages": [message],
    }


def generate_session_title(first_query: str) -> str:
    """Generate a session title from the first query."""
    title = first_query.strip()
    if len(title) > 50:
        title = title[:50] + "..."
    return title


def _append_message(target_session: Dict, new_message: Dict, now: datetime) -> Tuple[int, int]:
    """Append message to a session and return old/new counts."""
    old_count = len(target_session.get("messages", []))
    target_session["messages"].append(new_message)
    target_session["updated_at"] = now.isoformat()
    return old_count, len(target_session["messages"])


def add_message_to_history(user_id, query, response, session_id=None, new_session=False):
    """
    Session management where frontend controls session IDs.

    LOGIC:
    1. Frontend sends session_id (UUID) + new_session flag
    2. If session_id exists in history: continue in that session
    3. If session_id is new: create new session with that ID
    4. If session_id is None: use most recent session (fallback)

    Returns: (history, active_session_id)
    """
    logging.info("[ADD_MSG] ===== START =====")
    logging.info(f"[ADD_MSG] User: {user_id}")
    logging.info(f"[ADD_MSG] Query: {query[:50]}...")
    logging.info(f"[ADD_MSG] Received session_id: {session_id}")
    logging.info(f"[ADD_MSG] New session flag: {new_session}")

    history = get_chat_history(user_id)
    now = datetime.utcnow()
    sessions = history.get("sessions", [])

    logging.info(f"[ADD_MSG] Existing sessions: {len(sessions)}")
    if sessions:
        recent = sessions[-1]
        logging.info(f"[ADD_MSG] Most recent: {recent['session_id']} ({len(recent.get('messages', []))} msgs)")

    new_message = {
        "query": str(query),
        "response": serialize_response(response),
        "timestamp": now.isoformat(),
    }

    target_session = None

    if session_id:
        target_session = next((s for s in sessions if s.get("session_id") == session_id), None)
        if target_session:
            logging.info(f"[ADD_MSG] Found existing session: {session_id}")
            before, after = _append_message(target_session, new_message, now)
            logging.info(f"[ADD_MSG] Messages: {before} -> {after}")
        else:
            logging.info(f"[ADD_MSG] Session ID not found, creating NEW session: {session_id}")
            new_session_obj = create_new_session(
                query,
                response,
                session_id=session_id,
                initial_message=new_message,
            )
            sessions.append(new_session_obj)
            target_session = new_session_obj
            logging.info(f"[ADD_MSG] Created new session with ID: {session_id}")
    else:
        logging.info("[ADD_MSG] No session_id provided")
        if sessions:
            target_session = sessions[-1]
            logging.info(f"[ADD_MSG] Using most recent: {target_session['session_id']}")
            before, after = _append_message(target_session, new_message, now)
            logging.info(f"[ADD_MSG] Messages: {before} -> {after}")
        else:
            logging.info("[ADD_MSG] No sessions exist, creating first")
            new_session_obj = create_new_session(query, response, initial_message=new_message)
            sessions.append(new_session_obj)
            target_session = new_session_obj
            logging.info(f"[ADD_MSG] Created first session: {new_session_obj['session_id']}")

    if len(sessions) > 50:
        logging.info(f"[ADD_MSG] Trimming sessions: {len(sessions)} -> 50")
        sessions = sessions[-50:]

    history["sessions"] = sessions
    save_chat_history(user_id, history)

    final_id = target_session.get("session_id") if target_session else None
    logging.info(f"[ADD_MSG] < Final session_id: {final_id}")
    logging.info(f"[ADD_MSG] Total sessions: {len(sessions)}")
    logging.info("[ADD_MSG] ===== END =====\n")

    return history, final_id


def get_current_session_context(user_id, session_id=None, max_messages=10):
    """Get messages for a specific session (or the latest one)."""
    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    if not sessions:
        return []

    current_session = None
    if session_id:
        current_session = next((s for s in sessions if s.get("session_id") == session_id), None)
    if current_session is None:
        current_session = sessions[-1]

    messages = current_session.get("messages", [])
    return messages[-max_messages:] if len(messages) > max_messages else messages


def delete_chat_history(user_id):
    """Delete chat history from Azure Blob Storage."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
        logging.info(f"Chat history deleted for {user_id}")
        return {"status": "success", "message": f"Chat history deleted for {user_id}"}
    except Exception as e:
        logging.error(f"Error deleting chat history: {e}")
        return {"status": "error", "message": "Error deleting chat history"}


def delete_session(user_id, session_id):
    """Delete a specific session from chat history."""
    try:
        history = get_chat_history(user_id)
        sessions = history.get("sessions", [])
        updated_sessions = [s for s in sessions if s.get("session_id") != session_id]

        if len(updated_sessions) == len(sessions):
            return {"status": "error", "message": "Session not found"}

        history["sessions"] = updated_sessions
        save_chat_history(user_id, history)

        logging.info(f"Session {session_id} deleted for {user_id}")
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        logging.error(f"Error deleting session: {e}")
        return {"status": "error", "message": "Error deleting session"}


def get_session_list(user_id) -> List[Dict]:
    """Get list of all sessions with metadata (for sidebar display)."""
    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    session_list = []
    for session in sessions:
        session_list.append({
            "session_id": session.get("session_id"),
            "title": session.get("title"),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "message_count": len(session.get("messages", [])),
        })
    return session_list


def get_session_by_id(user_id, session_id):
    """Get a specific session's full chat history."""
    history = get_chat_history(user_id)
    sessions = history.get("sessions", [])
    for session in sessions:
        if session.get("session_id") == session_id:
            return session
    return None


def generate_follow_up_questions(retrieved_text, num_questions=3):
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
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}")
        return []
