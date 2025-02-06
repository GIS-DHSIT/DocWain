from fastapi import APIRouter, Query
from backend.services.ai_model import generate_answer

router = APIRouter()

@router.get("/chat/")
async def chatbot_query(query: str, response_type: str = Query("detailed", enum=["short", "detailed"])):
    """
    API Endpoint for AI-powered document Q&A.
    """
    result = generate_answer(query, response_type)
    return {
        "response": result["response"],
        "source_documents": result["source_documents"]
    }
