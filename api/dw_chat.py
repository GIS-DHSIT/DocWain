
import ollama

import json

import logging

from io import BytesIO

from datetime import datetime, timedelta

from api.config import Config

from azure.storage.blob import BlobServiceClient, ContentSettings

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

connection_string = Config.AzureBlob.CONNECTION_STRING

blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = Config.AzureBlob.CONTAINER_NAME

container_client = blob_service_client.get_container_client(container_name)

# Session timeout in minutes (adjust as needed)

SESSION_TIMEOUT_MINUTES = 30


def get_chat_history(user_id):
    """Retrieve chat history with sessions from Azure Blob Storage."""

    try:

        blob_name = f"chat_history/{user_id}.json"

        blob_client = container_client.get_blob_client(blob_name)

        blob_data = blob_client.download_blob().readall()

        chat_history = json.loads(blob_data.decode("utf-8"))

        # Ensure the structure has sessions key

        if isinstance(chat_history, list):

            # Migrate old format to new format

            chat_history = {"sessions": []}

        elif "sessions" not in chat_history:

            chat_history = {"sessions": []}

        return chat_history

    except Exception as e:

        logging.info(f"No existing chat history for {user_id}: {e}")

        return {"sessions": []}


def save_chat_history(user_id, chat_history):
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
    """

    Serialize response properly for storage.

    Handles both dict objects and string responses.

    """

    if isinstance(response, dict):

        # Return a clean JSON-serializable dict

        return {

            "response": response.get("response", ""),

            "sources": response.get("sources", [])

        }

    elif isinstance(response, str):

        # If it's already a string, return as is

        return response

    else:

        # Fallback to string conversion

        return str(response)


def create_new_session(query, response):
    """Create a new session with the first message."""

    now = datetime.utcnow()

    return {

        "session_id": now.strftime("%Y%m%d_%H%M%S"),

        "title": generate_session_title(query),

        "created_at": now.isoformat(),

        "updated_at": now.isoformat(),

        "messages": [

            {

                "query": str(query),

                "response": serialize_response(response),

                "timestamp": now.isoformat()

            }

        ]

    }


def generate_session_title(first_query):
    """Generate a session title from the first query."""

    # Truncate and clean the query for title

    title = first_query.strip()

    if len(title) > 50:
        title = title[:50] + "..."

    return title


def should_create_new_session(last_session):
    """Check if a new session should be created based on timeout."""

    if not last_session or "updated_at" not in last_session:
        return True

    try:

        last_updated = datetime.fromisoformat(last_session["updated_at"])

        time_diff = datetime.utcnow() - last_updated

        return time_diff > timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    except Exception as e:

        logging.error(f"Error checking session timeout: {e}")

        return True


def add_message_to_history(user_id, query, response):
    """Add a message to chat history, creating new session if needed."""

    history = get_chat_history(user_id)

    now = datetime.utcnow()

    # Get the last session

    sessions = history.get("sessions", [])

    new_message = {

        "query": str(query),

        "response": serialize_response(response),

        "timestamp": now.isoformat()

    }

    # Check if we need a new session

    if not sessions or should_create_new_session(sessions[-1]):

        # Create new session

        new_session = create_new_session(query, response)

        sessions.append(new_session)

        logging.info(f"Created new session: {new_session['session_id']}")

    else:

        # Add to existing session

        sessions[-1]["messages"].append(new_message)

        sessions[-1]["updated_at"] = now.isoformat()

        logging.info(f"Added message to session: {sessions[-1]['session_id']}")

    # Keep only last 50 sessions (adjust as needed)

    if len(sessions) > 50:
        sessions = sessions[-50:]

    history["sessions"] = sessions

    save_chat_history(user_id, history)

    return history


def get_current_session_context(user_id, max_messages=10):
    """Get the current session's messages for context."""

    history = get_chat_history(user_id)

    sessions = history.get("sessions", [])

    if not sessions:
        return []

    # Get messages from the current (last) session

    current_session = sessions[-1]

    messages = current_session.get("messages", [])

    # Return last N messages for context

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

        # Filter out the session to delete

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


def get_session_list(user_id):
    """Get list of all sessions with metadata (for sidebar display)."""

    history = get_chat_history(user_id)

    sessions = history.get("sessions", [])

    # Return simplified session info for display

    session_list = []

    for session in sessions:
        session_list.append({

            "session_id": session.get("session_id"),

            "title": session.get("title"),

            "created_at": session.get("created_at"),

            "updated_at": session.get("updated_at"),

            "message_count": len(session.get("messages", []))

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


genai.configure(api_key=Config.Gemini.GEMINI_API_KEY)


def generate_follow_up_questions(retrieved_text, num_questions=3):
    """Generates follow-up questions using Gemini 2.5 Flash based on retrieved content."""

    try:

        prompt = f"""

        Strictly based on the following retrieved context, suggest {num_questions} relevant follow-up questions.

        Keep them short and precise.

        **Context:**

        {retrieved_text}

        **Follow-up Questions:**

        """

        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(prompt)

        return response.text if response and response.text else []

    except Exception as e:

        logging.error(f"Error generating follow-up questions: {e}")

        return []
