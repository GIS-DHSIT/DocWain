import uvicorn
from pydantic import BaseModel
from dw_newron import answer_question
from fastapi import FastAPI, HTTPException

app = FastAPI(title="DocWain API")


class QuestionRequest(BaseModel):
    query: str
    user_id: str = 'someone@email.com'
    profile_id: str = "67ac62ddfaa3aee44d38f4a5"


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.user_id,
                             request.profile_id)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)