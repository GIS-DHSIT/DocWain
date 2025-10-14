import ollama
import logging
import uvicorn
from pydantic import BaseModel
from api.dataHandler import trainData
from dw_newron import answer_question
from fastapi import FastAPI, HTTPException
from api.dw_chat import get_chat_history, delete_chat_history

app = FastAPI(title="DocWain API")


class QuestionRequest(BaseModel):
    query: str
    user_id: str = 'someone@email.com'
    profile_id: str = "67ac62ddfaa3aee44d38f4a5"
    model_name: str = "Azure-OpenAI"
    persona: str = "Document Assistant"


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.user_id,
                             request.profile_id, request.model_name, request.persona)
    return answer


@app.get("/train")
def trigger_training():
    """
    API endpoint to trigger document training.

    Request Body:
    - collectionName: Name of the MongoDB collection
    - bucketName: Name of the bucket where documents are stored

    Response:
    - Training status message
    """
    try:
        logging.info(
            f"Received training request")

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
        azure = {'model':'Azure-OpenAI'}
        model_list.append(azure)
        return {"models": models}
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")


@app.get("/chat-history/{user_id}")
def get_chat_history_api(user_id: str):
    """API endpoint to retrieve chat history for a user."""
    try:
        history = get_chat_history(user_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Chat history not found")
        return {"user_id": user_id, "chat_history": history}
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@app.delete("/chat-history/{user_id}")
def delete_chat_history_api(user_id: str):
    """API endpoint to delete chat history for a user."""
    try:
        result = delete_chat_history(user_id)
        return {"user_id": user_id, "message": result}
    except Exception as e:
        logging.error(f"Failed to delete chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

