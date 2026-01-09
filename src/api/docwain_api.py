
import json
import logging
import time
from typing import Any, Dict, Optional
import ollama
import uvicorn
from qdrant_client import QdrantClient
from bson.objectid import ObjectId  # added by maha/maria
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.config import Config
from src.teams import adapter as teams_adapter
from src.api.dataHandler import (
    db,
    get_subscription_pii_setting,
    mongoClient,
    trainData,  # trainSingleDocument
    train_single_document,
    delete_embeddings
)
from src.api.dw_chat import (
    add_message_to_history,
    delete_chat_history,
    delete_session,
    get_chat_history,
    get_session_by_id,
    get_session_list,
)
from src.finetune import get_finetune_manager, list_models
from src.finetune.models import FinetuneRequest, AutoFinetuneRequest
from src.finetune.dataset_builder import build_dataset_from_qdrant, discover_collections_and_profiles

app = FastAPI(title="DocWain API")
logger = logging.getLogger(__name__)


def _get_dw_newron():
    """
    Lazy loader for RAG pipeline functions to avoid heavy imports on module load.
    This keeps Teams/chat endpoints responsive even when optional deps are absent
    (tests, lightweight deployments) while preserving behavior for the main API.
    """
    from src.api import dw_newron  # local import to defer heavy deps
    return dw_newron

#  Add CORS middleware
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


def _normalize_answer(answer):
    """Ensure API responses are structured consistently."""
    if isinstance(answer, dict):
        structured = {
            "response": answer.get("response") or answer.get("answer"),
            "sources": answer.get("sources", []),
            "grounded": answer.get("grounded", False),
            "context_found": answer.get("context_found", False),
            "metadata": {k: v for k, v in answer.items() if k not in {"response", "answer", "sources"}},
        }
        return structured
    return {
        "response": str(answer),
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": {},
    }


def _build_text_fallback_activity(raw_body: bytes, headers: Dict[str, str]) -> Dict[str, Any] | None:
    """Construct a minimal Teams-like activity from a plain text payload."""
    try:
        text = raw_body.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not text:
        return None

    convo_id = (
        headers.get("x-teams-conversation-id")
        or headers.get("conversation-id")
        or "teams-text-fallback"
    )
    user_id = headers.get("x-teams-user-id") or "teams_user"

    return {"text": text, "conversation": {"id": convo_id}, "from": {"id": user_id}}


async def _parse_teams_activity(request: Request) -> Dict[str, Any] | None:
    """
    Parse Teams activity payload, tolerating empty/invalid bodies.
    Returns a dict on success; None when the body cannot be parsed.
    """
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    if not raw_body or not raw_body.strip():
        logger.debug(
            "Teams payload missing/empty body; responding with friendly message | content_type=%s",
            content_type,
        )
        return None

    try:
        activity = json.loads(raw_body)
    except json.JSONDecodeError:
        # Fallback: treat text/plain bodies as simple messages
        if content_type.lower().startswith("text/plain"):
            fallback = _build_text_fallback_activity(raw_body, dict(request.headers))
            if fallback:
                logger.debug(
                    "Teams payload decoded as text/plain fallback | content_type=%s | body_len=%d",
                    content_type,
                    len(raw_body or b""),
                )
                return fallback

        logger.debug(
            "Invalid Teams payload: JSON decode failed; responding with friendly message | content_type=%s | body_len=%d",
            content_type,
            len(raw_body or b""),
        )
        return None

    if not isinstance(activity, dict):
        logger.warning("Invalid Teams payload type: %s", type(activity))
        return None

    return activity


@app.post("/teams/messages")
async def handle_teams_messages(request: Request):
    """Endpoint for Microsoft Teams activities (messages, attachments)."""
    activity = await _parse_teams_activity(request)
    if activity is None:
        # Return a Bot-style message so Teams sees a graceful response instead of 4xx
        return {
            "type": "message",
            "text": "I couldn't read that Teams message. Please try sending it again.",
        }

    try:
        return await teams_adapter.handle_teams_activity(activity, headers=dict(request.headers))
    except teams_adapter.TeamsAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    logging.info(f"[ASK] ========== START ==========")
    logging.info(f"[ASK] User: {request.user_id}")
    logging.info(f"[ASK] Query: {request.query}")
    logging.info(f"[ASK] Session ID from frontend: {request.session_id}")
    logging.info(f"[ASK] New session flag: {request.new_session}")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not request.profile_id:
        raise HTTPException(status_code=400, detail="profile_id is required for retrieval")

    # Generate answer
    raw_answer = _get_dw_newron().answer_question(
        request.query,
        request.user_id,
        request.profile_id,
        request.subscription_id,
        request.model_name,
        request.persona
    )
    answer = _normalize_answer(raw_answer)

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


@app.post("/askStream")
def ask_question_stream_api(request: QuestionRequest):
    """
    Streams the LLM response for a query. Uses the same RAG pipeline as /ask but
    streams the generated answer text to the client.
    """
    logging.info(f"[ASK_STREAM] User: {request.user_id} | Query: {request.query}")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not request.profile_id:
        raise HTTPException(status_code=400, detail="profile_id is required for retrieval")

    def _stream():
        try:
            raw_answer = _get_dw_newron().answer_question(
                request.query,
                request.user_id,
                request.profile_id,
                request.subscription_id,
                request.model_name,
                request.persona
            )
            answer = _normalize_answer(raw_answer)
            text = answer.get("response") or ""
            if not text:
                text = "No response generated."
            chunk_size = 256
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]
        except Exception as exc:
            logging.error(f"[ASK_STREAM] Streaming failed: {exc}")
            yield f"[error] {exc}"

    return StreamingResponse(_stream(), media_type="text/plain")


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


@app.post("/finetune")
def trigger_finetune(request: FinetuneRequest):
    """Kick off an Unsloth fine-tune for a profile/domain."""
    manager = get_finetune_manager()
    status = manager.start_job(request)
    return status.dict()


@app.get("/finetune/{job_id}")
def finetune_status(job_id: str):
    manager = get_finetune_manager()
    status = manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status.dict()


@app.get("/models")
def list_available_models():
    return list_models()


@app.post("/finetune/auto")
def auto_finetune(request: Optional[AutoFinetuneRequest] = None):
    """
    Connect to Qdrant, discover collections/profiles, build datasets automatically,
    and kick off Unsloth fine-tunes for each discovered profile.
    """
    manager = get_finetune_manager()
    params = request or AutoFinetuneRequest()
    # Merge legacy single-subscription field with new multi-subscription support
    requested_subs = list(params.subscription_ids or [])
    if params.subscription_id:
        requested_subs.append(params.subscription_id)
    if requested_subs:
        # Remove duplicates while preserving order
        seen = set()
        requested_subs = [s for s in requested_subs if not (s in seen or seen.add(s))]
    else:
        requested_subs = None

    qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    try:
        discovered = discover_collections_and_profiles(
            subscription_ids=requested_subs,
            client=qdrant_client,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to discover Qdrant collections: {exc}")

    if not discovered:
        raise HTTPException(status_code=404, detail="No Qdrant collections with profile data found")

    if params.profile_id:
        filtered = {sub: [pid for pid in pids if pid == params.profile_id] for sub, pids in discovered.items()}
        filtered = {sub: ids for sub, ids in filtered.items() if ids}
        discovered = filtered
        if not discovered:
            raise HTTPException(status_code=404, detail=f"Profile {params.profile_id} not found in Qdrant payloads")

    jobs = []
    dataset_paths = []
    failures = {}

    for subscription_id, profile_ids in discovered.items():
        for profile_id in profile_ids:
            try:
                dataset_path = build_dataset_from_qdrant(
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    max_points=params.max_points,
                    questions_per_chunk=params.questions_per_chunk,
                    generation_model=params.generation_model,
                    client=qdrant_client,
                )
            except Exception as exc:
                failures[f"{subscription_id}:{profile_id}"] = f"dataset_error: {exc}"
                continue

            try:
                payload = params.finetune_payload(profile_id=profile_id, dataset_path=str(dataset_path))
                status = manager.start_job(FinetuneRequest(**payload))
                jobs.append(status.dict())
                dataset_paths.append(str(dataset_path))
            except Exception as exc:  # noqa: BLE001
                failures[f"{subscription_id}:{profile_id}"] = f"finetune_error: {exc}"

    if not jobs:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No finetune jobs started",
                "errors": failures or "Unable to build datasets or schedule jobs",
            },
        )

    response = {
        "jobs": jobs,
        "dataset_paths": dataset_paths,
        "discovered_profiles": discovered,
    }
    if failures:
        response["skipped"] = failures
    return response


@app.delete("/document/{doc_id}/embeddings")
def delete_document_embeddings_api(
        doc_id: str,
        subscription_id: str = "default",
        profile_id: Optional[str] = None
):
    """
    API endpoint to manually delete embeddings for a specific document.
    This is useful when a document is marked as DELETED in MongoDB.
    """
    try:
        from src.api.dataHandler import delete_embeddings, db, Config
        from bson.objectid import ObjectId

        # Fetch document details from MongoDB
        doc = db[Config.MongoDB.DOCUMENTS].find_one({"_id": ObjectId(doc_id)})

        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Get subscription_id from document (override query param if present in doc)
        subscription_id = str(
            doc.get('subscription') or
            doc.get('subscriptionId') or
            subscription_id
        )
        profile_id = str(
            profile_id
            or doc.get('profile')
            or doc.get('profile_id')
            or ""
        )
        if not profile_id:
            raise HTTPException(status_code=400, detail="profile_id is required for deletion")

        logging.info(
            f"[API] Deleting embeddings for doc={doc_id}, "
            f"subscription={subscription_id}, "
            f"profile={profile_id}, "
            f"status={doc.get('status')}"
        )

        # Delete embeddings scoped to subscription and profile
        result = delete_embeddings(
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=doc_id
        )

        if result["status"] == "success":
            logging.info(f" [API] Successfully deleted embeddings for document {doc_id}")
            return {
                "status": "success",
                "document_id": doc_id,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "message": result.get("message", "Embeddings deleted"),
                "details": result
            }
        elif result["status"] == "not_found":
            # Not an error - document just has no embeddings
            return {
                "status": "success",
                "document_id": doc_id,
                "profile_id": profile_id,
                "message": result.get("message", "No embeddings found to delete"),
                "details": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown error during deletion")
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"L [API] Failed to delete embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete embeddings: {str(e)}"
        )


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


@app.get("/metrics")
def get_metrics(days: int = 7):
    """API endpoint to retrieve model/retrieval performance statistics."""
    try:
        summary = _get_dw_newron().metrics_summary(days=days)
        return {
            "status": "success",
            "window_days": days,
            "metrics": summary
        }
    except Exception as e:
        logging.error(f"Failed to retrieve metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

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
        return formatted  #  Now returns {"sessions": [...]}
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
