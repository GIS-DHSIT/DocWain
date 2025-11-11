import ollama
import logging
import uvicorn
from pydantic import BaseModel
from api.dataHandler import trainData # trainSingleDocument
from api.dataHandler import train_single_document
from api.dw_newron import answer_question
from fastapi import FastAPI, HTTPException
from api.dw_chat import (
    get_chat_history,
    delete_chat_history,
    add_message_to_history,
    get_session_list,
    get_session_by_id,
    delete_session,
    get_current_session_context
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DocWain API")

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    query: str
    user_id: str = 'someone@email.com'
    profile_id: str = "67ac62ddfaa3aee44d38f4a5"
    model_name: str = "gemini-2.5-flash"
    persona: str = "Document Assistant"


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    logging.info(f"[ASK] User: {request.user_id}, Query: {request.query}")
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Get current session context for better responses
    context = get_current_session_context(request.user_id)
    answer = answer_question(
        request.query,
        request.user_id,
        request.profile_id,
        request.model_name,
        request.persona
    )

    # Add message to history (creates new session if needed)
    updated_history = add_message_to_history(request.user_id, request.query, answer)

    logging.info(f"[ASK] User: {request.user_id}, Answer: {answer}")
    return {
        "answer": answer,
        "current_session_id": updated_history["sessions"][-1]["session_id"] if updated_history["sessions"] else None
    }


@app.post("/train/{doc_id}")
def trigger_single_training(doc_id: str):
    """API endpoint to train a single document by its document ID."""
    try:
        logging.info(f"Received single document training request for: {doc_id}")
        result = train_single_document(doc_id)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Single training API error: {e}")
        raise HTTPException(status_code=500, detail="Single document training failed")


@app.get("/train")
def trigger_training():
    """API endpoint to trigger document training."""
    try:
        logging.info("Received training request")
        status_response = trainData()
        logging.info(status_response)
        return {"status": "success", "message": status_response, "response": "Executed"}
    except Exception as e:
        logging.error(f"Training API error: {e}")
        raise HTTPException(status_code=500, detail="Training process failed")


@app.get("/models")
def list_available_models():
    """API endpoint to list available models."""
    try:
        models = ollama.list().model_dump()
        model_list = models['models']
        gemini = {'model': 'gemini-2.5-flash'}
        model_list.append(gemini)
        return {"models": model_list}
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")

'''
@app.get("/chat-history/{user_id}")
def get_chat_history_api(user_id: str):
    """API endpoint to retrieve full chat history for a user."""
    try:
        history = get_chat_history(user_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Chat history not found")

        logging.info(f"[CHAT HISTORY] User: {user_id}")
        return {"user_id": user_id, "chat_history": history}
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")
'''

@app.get("/chat-history/{user_id}")
def get_chat_history_api(user_id: str):
    """
    Returns all chat sessions for a user in the structure expected by NestJS.
    """
    try:
        history = get_chat_history(user_id)
        if not history:
            raise HTTPException(status_code=404, detail="No chat history found")

        # Ensure proper structure
        if isinstance(history, dict) and "sessions" not in history:
            # convert legacy history format to session-style
            sessions = [{
                "session_id": "default",
                "title": history[0]["query"] if isinstance(history, list) and history else "Chat Session",
                "messages": history
            }]
            formatted = {"sessions": sessions}
        elif isinstance(history, list):
            # list of sessions already
            formatted = {"sessions": history}
        else:
            formatted = history  # already correct format

        logging.info(f"[CHAT HISTORY] User: {user_id}, Sessions: {len(formatted['sessions'])}")
        return formatted  # ✅ Now returns {"sessions": [...]}
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.get("/sessions/{user_id}")
def get_sessions_api(user_id: str):
    """API endpoint to get list of all sessions (for sidebar)."""
    try:
        sessions = get_session_list(user_id)
        logging.info(f"[SESSIONS] User: {user_id}, Count: {len(sessions)}")
        return {"user_id": user_id, "sessions": sessions}
    except Exception as e:
        logging.error(f"Failed to retrieve sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@app.get("/session/{user_id}/{session_id}")
def get_session_api(user_id: str, session_id: str):
    """API endpoint to get a specific session's messages."""
    try:
        session = get_session_by_id(user_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        logging.info(f"[SESSION] User: {user_id}, Session: {session_id}")
        return {"user_id": user_id, "session": session}
    except Exception as e:
        logging.error(f"Failed to retrieve session: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@app.delete("/chat-history/{user_id}")
def delete_chat_history_api(user_id: str):
    """API endpoint to delete all chat history for a user."""
    try:
        result = delete_chat_history(user_id)
        return {"user_id": user_id, "message": result}
    except Exception as e:
        logging.error(f"Failed to delete chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


@app.delete("/session/{user_id}/{session_id}")
def delete_session_api(user_id: str, session_id: str):
    """API endpoint to delete a specific session."""
    try:
        result = delete_session(user_id, session_id)
        return {"user_id": user_id, "session_id": session_id, "result": result}
    except Exception as e:
        logging.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
