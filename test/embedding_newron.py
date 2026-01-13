import ollama
import faiss
import logging
import numpy as np
from pymongo import MongoClient
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, Filter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.api.config import Config
from src.api.dataHandler import trainData
from src.api.genai_client import generate_text, get_genai_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)

mongoClient = MongoClient(Config.MongoDB.URI)
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
get_genai_client(Config.Model.GEMINI_API_KEY)

app = FastAPI(title="DocWain API")


class QuestionRequest(BaseModel):
    query: str
    schemaName: str = 'actual documents provided '
    profile_id: str = "profile ID"
    model_name: str = "llama3.2"


class GeminiQuestionRequest(BaseModel):  # ✅ renamed from AzureQuestionRequest
    query: str
    schemaName: str = 'actual documents provided '
    profile_id: str = "profile ID"
    model_name: str = "gemini-2.5-flash"


class TrainRequest(BaseModel):
    collectionName: str = 'documents'
    collectionDir: str = 'connectors'
    schemaName: str = 'actual documents provided '


def ensure_qdrant_collection(collection_name, vector_size=384):
    """Ensures the Qdrant collection exists with the correct settings."""
    try:
        collections = qdrant_client.get_collections().collections
        existing_collections = {col.name for col in collections}

        if collection_name not in existing_collections:
            logging.info(f"Creating Qdrant collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error ensuring collection in Qdrant: {e}")


def load_embeddings_from_qdrant(collection_name, tags):
    """Loads embeddings from Qdrant Cloud."""
    logging.info(f"Loading embeddings from Qdrant for tag: {tags}")

    try:
        ensure_qdrant_collection(collection_name)

        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[{"key": "tag", "match": {"value": tags}}]
            ),
            limit=1000
        )

        if not search_result[0]:
            logging.warning("No embeddings found in Qdrant.")
            return None

        embeddings = [point.vector for point in search_result[0]]
        texts = [point.payload["text"] for point in search_result[0]]

        # Create FAISS index
        embeddings = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return {"index": index, "texts": texts}

    except Exception as e:
        logging.error(f"Error loading embeddings from Qdrant: {e}")
        return None


def answer_question(query, collection_name, tag="default", model='llama3.2'):
    """Answers a question based on stored embeddings using Ollama."""
    embeddings = load_embeddings_from_qdrant(collection_name, tag)
    if embeddings is None:
        return "No trained model found!"
    logging.info(f"Processing query: {query} for model: {tag}")
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    D, I = embeddings["index"].search(query_embedding, 3)
    retrieved_text = "\n".join([embeddings["texts"][i] for i in I[0]])
    logging.info(f"Retrieving response using Ollama model {model}.")
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on this context: {retrieved_text}\nQuestion: {query}"}
        ]
    )
    return response["message"]


# ❌ Old Azure function (commented out)
def gemini_answer_question(query, collection_name, tag="default", model="gemini-1.5-flash"):
    """Answers a question based on stored embeddings using Gemini."""
    embeddings = load_embeddings_from_qdrant(collection_name, tag)
    if embeddings is None:
        return "No trained model found!"

    logging.info(f"Processing query: {query} for model: {tag}")
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    D, I = embeddings["index"].search(query_embedding, 3)
    retrieved_text = "\n".join([embeddings["texts"][i] for i in I[0]])

    logging.info(f"Retrieving response using Gemini {model}.")
    prompt = f"Answer based on this context:\n{retrieved_text}\n\nQuestion: {query}"

    text, _ = generate_text(
        api_key=Config.Model.GEMINI_API_KEY,
        model=model,
        prompt=prompt,
    )
    return text


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions (Ollama)."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.schemaName,
                             request.profile_id, request.model_name)
    return {"answer": answer}


@app.post("/ask_gemini")
def ask_gemini_api(request: GeminiQuestionRequest):
    """Gemini API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = gemini_answer_question(request.query, request.schemaName,
                                    request.profile_id, request.model_name)
    return {"answer": answer if isinstance(answer, str) else str(answer)}




@app.post("/train")
def trigger_training(request: TrainRequest):
    """
    API endpoint to trigger document training.
    """
    try:
        logging.info(
            f"Received training request for collection: {request.collectionName}, connector: {request.collectionDir}")

        status_response = trainData(request.collectionDir, request.schemaName, request.collectionName)

        logging.info("Training completed successfully.")
        return {"status": "success", "message": "Training process triggered", "response": "Completed"}

    except Exception as e:
        logging.error(f"Training API error: {e}")
        raise HTTPException(status_code=500, detail="Training process failed")


@app.get("/models")
def list_available_models():
    """API endpoint to list locally available models."""
    try:
        models = ollama.list().model_dump()
        return {"models": models}
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
