import ollama
import logging
import uvicorn
from pydantic import BaseModel
from api.dataHandler import trainData
from dw_newron import answer_question
from fastapi import FastAPI, HTTPException

app = FastAPI(title="DocWain API")


class QuestionRequest(BaseModel):
    query: str
    user_id: str = 'someone@email.com'
    profile_id: str = "67ac62ddfaa3aee44d38f4a5"
    model_name: str = "llama3.2"


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.user_id,
                             request.profile_id, request.model_name)
    return {"answer": answer}


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


@app.get("/chat_history")
def get_chat_history(user_id: str):
    """API endpoint to retrieve chat history for a specific user."""
    try:
        history = get_chat_history(user_id)
        return {"history": history}
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)