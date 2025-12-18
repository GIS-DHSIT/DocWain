
import ollama
import logging
import uvicorn
from typing import Optional
from pydantic import BaseModel
from bson.objectid import ObjectId # added by maha/maria
import time
import os, sys
path = os.getcwd()
sys.path.append(path)
from api.dataHandler import trainData # trainSingleDocument
from api.dataHandler import train_single_document, get_pii_stats
from api.dw_newron import answer_question
from fastapi import FastAPI, HTTPException
from api.dw_chat import (
    get_chat_history,
    delete_chat_history,
    add_message_to_history,
    get_session_list,
    get_session_by_id,
    delete_session
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
    subscription_id: str = "default"
    model_name: str = "llama3.2"
    persona: str = "Document Assistant"
    session_id: Optional[str] = None
    new_session: Optional[bool] = False  # Frontend sends flag here


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    logging.info(f"[ASK] ========== START ==========")
    logging.info(f"[ASK] User: {request.user_id}")
    logging.info(f"[ASK] Query: {request.query}")
    logging.info(f"[ASK] Session ID from frontend: {request.session_id}")
    logging.info(f"[ASK] New session flag: {request.new_session}")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Generate answer
    answer = answer_question(
        request.query,
        request.user_id,
        request.profile_id,
        request.subscription_id,
        request.model_name,
        request.persona
    )

    # Add to history - backend uses frontend's session_id
    _history, active_session_id = add_message_to_history(
        request.user_id,
        request.query,
        answer,
        session_id=request.session_id,  # Use frontend's UUID
        new_session=request.new_session  # Use frontend's flag
    )

    logging.info(f"[ASK] Answer generated")
    logging.info(f"[ASK] < Active Session ID: {active_session_id}")
    logging.info(f"[ASK] ========== END ==========\n")

    # Return session_id back to frontend
    return {
        "answer": answer,
        "current_session_id": active_session_id
    }


@app.post("/train/{doc_id}")
def trigger_single_training(doc_id: str, subscription_id: str = "default"):
    """API endpoint to train a single document by its document ID."""
    try:
        logging.info(f"Received single document training request for: {doc_id} (subscription: {subscription_id})")
        result = train_single_document(doc_id)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Single training API error: {e}")
        raise HTTPException(status_code=500, detail="Single document training failed")


@app.get("/train")
def trigger_training(subscription_id: str = "default"):
    """API endpoint to trigger document training."""
    try:
        logging.info(f"Received training request (subscription: {subscription_id})")
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
def get_chat_history_api(user_id: str, subscription_id: str = "default"):
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
def get_sessions_api(user_id: str, subscription_id: str = "default"):
    """API endpoint to get list of all sessions (for sidebar)."""
    try:
        sessions = get_session_list(user_id)
        logging.info(f"[SESSIONS] User: {user_id}, Count: {len(sessions)}")
        return {"user_id": user_id, "sessions": sessions}
    except Exception as e:
        logging.error(f"Failed to retrieve sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@app.get("/session/{user_id}/{session_id}")
def get_session_api(user_id: str, session_id: str, subscription_id: str = "default"):
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
def delete_chat_history_api(user_id: str, subscription_id: str = "default"):
    """API endpoint to delete all chat history for a user."""
    try:
        result = delete_chat_history(user_id)
        return {"user_id": user_id, "message": result}
    except Exception as e:
        logging.error(f"Failed to delete chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


@app.delete("/session/{user_id}/{session_id}")
def delete_session_api(user_id: str, session_id: str, subscription_id: str = "default"):
    """API endpoint to delete a specific session."""
    try:
        result = delete_session(user_id, session_id)
        return {"user_id": user_id, "session_id": session_id, "result": result}
    except Exception as e:
        logging.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

'''
@app.get("/pii/{doc_id}")
def get_pii_info(doc_id: str, subscription_id: str = "default"):
    """API endpoint to retrieve PII masking stats for a document."""
    try:
        stats = get_pii_stats(doc_id)
        if not stats:
            raise HTTPException(status_code=404, detail=f"Document not found for id {doc_id}")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to retrieve PII stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve PII stats")
'''

'-------------added by maha/maria ------------------'
'--------------for PII setting management ------------------'

class PIISettingUpdate(BaseModel):
    pii_enabled: bool


@app.get("/subscription/{subscription_id}/pii-setting")
def get_pii_setting(subscription_id: str):
    """
    Get current PII masking setting for a subscription
    """
    try:
        from api.dataHandler import get_subscription_pii_setting
        pii_enabled = get_subscription_pii_setting(subscription_id)
        return {
            "subscription_id": subscription_id,
            "pii_enabled": pii_enabled,
            "message": f"PII masking is {'ENABLED' if pii_enabled else 'DISABLED'}"
        }
    except Exception as e:
        logging.error(f"Failed to get PII setting: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve PII setting")


@app.put("/subscription/{subscription_id}/pii-setting")
def update_pii_setting(subscription_id: str, setting: PIISettingUpdate):
    """
    Update PII masking setting for a subscription
    Body:
    {
        "pii_enabled": true/false
    }
    """
    try:
        from api.config import Config
        from api.dataHandler import mongoClient, db
        subscriptions_collection = getattr(Config.MongoDB, 'SUBSCRIPTIONS', 'subscriptions')
        collection = db[subscriptions_collection]
        # Find subscription
        subscription = None
        if ObjectId.is_valid(subscription_id):
            subscription = collection.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            subscription = collection.find_one({"subscriptionId": subscription_id})
        if not subscription:
            subscription = collection.find_one({"_id": subscription_id})
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Subscription {subscription_id} not found"
            )
        # Update PII setting
        filter_criteria = {"_id": subscription["_id"]}
        update_operation = {
            "$set": {
                "pii_enabled": setting.pii_enabled,
                "pii_updated_at": time.time()
            }
        }
        result = collection.update_one(filter_criteria, update_operation)
        if result.modified_count > 0:
            logging.info(f"Updated PII setting for subscription {subscription_id}: pii_enabled={setting.pii_enabled}")
            return {
                "status": "success",
                "subscription_id": subscription_id,
                "pii_enabled": setting.pii_enabled,
                "message": f"PII masking is now {'ENABLED' if setting.pii_enabled else 'DISABLED'}"
            }
        else:
            return {
                "status": "no_change",
                "subscription_id": subscription_id,
                "pii_enabled": setting.pii_enabled,
                "message": "PII setting was already at the requested value"
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update PII setting: {e}")
        raise HTTPException(status_code=500, detail="Failed to update PII setting")


@app.post("/subscription/{subscription_id}/reprocess-documents")
def reprocess_documents_with_new_pii_setting(subscription_id: str):
    """
    Reprocess all documents in a subscription with updated PII setting
    This will retrain documents with new PII masking rules
    """
    try:
        from api.dataHandler import trainData, db
        from api.config import Config
        # Get all documents for this subscription
        documents_collection = db[Config.MongoDB.DOCUMENTS]
        # Update all documents in this subscription to UNDER_REVIEW
        # so they get reprocessed with new PII setting
        result = documents_collection.update_many(
            {
                "$or": [
                    {"subscriptionId": subscription_id},
                    {"subscription_id": subscription_id},
                    {"subscription": subscription_id}
                ],
                "status": {"$in": ["TRAINING_COMPLETED", "TRAINING_PARTIALLY_COMPLETED"]}
            },
            {
                "$set": {
                    "status": "UNDER_REVIEW",
                    "reprocess_reason": "PII setting changed",
                    "reprocess_timestamp": time.time()
                }
            }
        )
        logging.info(f"Marked {result.modified_count} documents for reprocessing")
        # Trigger training
        training_result = trainData()
        return {
            "status": "success",
            "subscription_id": subscription_id,
            "documents_marked_for_reprocessing": result.modified_count,
            "training_result": training_result
        }
    except Exception as e:
        logging.error(f"Failed to reprocess documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to reprocess documents")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

