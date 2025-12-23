import json
import logging
import numpy as np
import faiss
import boto3
import ollama
from qdrant_client import QdrantClient
from pymongo import MongoClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from src.api.config import Config
import uvicorn
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
mongoClient = MongoClient(Config.MongoDB.URI)

# AWS S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=Config.AWS.ACCESS_KEY,
    aws_secret_access_key=Config.AWS.SECRET_KEY,
    region_name=Config.AWS.REGION
)
S3_BUCKET_NAME = Config.AWS.BUCKET_NAME

app = FastAPI(title="Enhanced RAG Chatbot API")


class ChatRequest(BaseModel):
    user_id: str
    query: str
    schemaName: str = "s3_test"
    profile_id: str = "default"
    model_name: str = "llama3.2"
    use_azure: bool = False


def get_chat_history(user_id: str):
    """Retrieve chat history from S3 for a specific user."""
    try:
        file_key = f"chat_history/{user_id}.json"
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        chat_history = json.loads(response["Body"].read().decode("utf-8"))
        return chat_history
    except s3_client.exceptions.NoSuchKey:
        return []  # Return empty if no history exists
    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}")
        return []


def save_chat_history(user_id: str, chat_history: List[Dict[str, str]]):
    """Save chat history to AWS S3."""
    try:
        file_key = f"chat_history/{user_id}.json"
        chat_data = json.dumps(chat_history)
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_key, Body=chat_data)
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")


def retrieve_context_from_qdrant(collection_name: str, tag: str):
    """Retrieves relevant context from Qdrant."""
    logging.info(f"Retrieving context from Qdrant for tag: {tag}")

    try:
        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[{"key": "tag", "match": {"value": tag}}]),
            limit=1000,
            with_vectors=True
        )

        if not search_result or not search_result[0]:
            logging.warning("No relevant data found in Qdrant.")
            return None

        embeddings = np.array([point.vector for point in search_result[0] if point.vector], dtype=np.float32)
        texts = [point.payload["text"] for point in search_result[0] if "text" in point.payload]

        if embeddings.size == 0:
            logging.warning("Embeddings found but vectors are missing!")
            return None

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return {"index": index, "texts": texts}
    except Exception as e:
        logging.error(f"Error loading embeddings from Qdrant: {e}")
        return None


def generate_rag_response(user_id: str, query: str, collection_name: str, tag: str, model: str, use_azure: bool):
    """Generates response using RAG (retrieved data + chat history)."""
    logging.info(f"Processing query: {query} for user: {user_id}")

    embeddings = retrieve_context_from_qdrant(collection_name, tag)
    if embeddings is None:
        return "No relevant knowledge found in database."

    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    D, I = embeddings["index"].search(query_embedding, 3)
    retrieved_text = "\n".join([embeddings["texts"][i] for i in I[0]])

    # Retrieve and format chat history
    history = get_chat_history(user_id)
    formatted_history = "\n".join([f"User: {h['query']}\nAI: {h['response']}" for h in history[-5:]])

    prompt = f"""
    You are a strict knowledge-based AI assistant. Answer ONLY based on retrieved information from the vector database.
    If the answer is not available, reply with "I don't know."

    **Chat History:** 
    {formatted_history}

    **Retrieved Context:**
    {retrieved_text}

    **User Query:**
    {query}
    """

    # Choose the model
    if use_azure:
        logging.info(f"Using Azure OpenAI for response generation.")
        client = AzureOpenAI(
            api_version="2023-07-01-preview",
            api_key=Config.Model.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.Model.AZURE_OPENAI_ENDPOINT,
        )
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant limited to knowledge from the database."},
                      {"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
    else:
        logging.info(f"Using Ollama model: {model}")
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant limited to knowledge from the database."},
                      {"role": "user", "content": prompt}]
        )
        response_text = response["message"]

    # Save updated chat history to S3
    history.append({"query": query, "response": response_text})
    save_chat_history(user_id, history)

    return response_text


@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    """API endpoint for RAG-based chat."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    answer = generate_rag_response(
        request.user_id, request.query, request.schemaName, request.profile_id, request.model_name, request.use_azure
    )
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




