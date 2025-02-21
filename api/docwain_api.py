import ollama
import logging
import uvicorn
from pydantic import BaseModel
from api.dataHandler import trainData
from fastapi import FastAPI, HTTPException
from dw_newron import answer_question,azure_answer_question

app = FastAPI(title="DocWain API")


class QuestionRequest(BaseModel):
    query: str
    user_id: str = 'someone@email.com'
    profile_id: str = "67ac62ddfaa3aee44d38f4a5"
    model_name: str = "llama3.2"


class AzureQuestionRequest(BaseModel):
    query: str
    user_id: str = 'someone@email.com'
    profile_id: str = "67ac62ddfaa3aee44d38f4a5"
    model_name: str = "OpenAI"
    api_version: str = "2023-07-01-preview"


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.user_id,
                             request.profile_id, request.model_name)
    return {"answer": answer}


@app.post("/askAzure")
def ask_question_api(request: AzureQuestionRequest):
    """Azure openAI API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = azure_answer_question(request.query, request.user_id,
                                   request.profile_id, request.model_name, request.api_version)
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


@app.get("/models")
def list_available_models():
    """API endpoint to list available models."""
    try:
        models = ollama.list().model_dump()
        return {"models": models}
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)