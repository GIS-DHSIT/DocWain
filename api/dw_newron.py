import ollama
import faiss
import logging
import numpy as np
from typing import List
from api.config import Config
from openai import AzureOpenAI
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
AzureOpenAI.api_key = Config.Model.AZURE_OPENAI_API_KEY
mongoClient = MongoClient(Config.MongoDB.URI)
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)



def rewrite_query(user_id: str, query: str, chat_history: List[dict]):
    """Rewrites user queries to improve retrieval accuracy using chat history & AI-based expansion."""

    # **Check Query Length & Clarity**
    num_words = len(query.split())

    # If query is already detailed (15+ words), no need for rewriting
    if num_words > 15:
        logging.info("Query is already detailed, skipping rewriting.")
        return query

    # **Retrieve Chat History (Last 3 Messages)**
    recent_history = chat_history[-3:] if chat_history else []
    history_context = " ".join([f"User: {h['query']} AI: {h['response']}" for h in recent_history])

    # **Generate Rewritten Query Using AI**
    refined_query = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are an AI specialized in rewriting user queries for improved search results."},
            {"role": "user", "content": f"Rewrite the following query to be more detailed and precise:\n\nQuery: {query}\n\nChat History Context: {history_context}"}
        ]
    )

    rewritten_query = refined_query["message"]
    logging.info(f"Original Query: {query} | Rewritten Query: {rewritten_query}")

    return rewritten_query


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


def load_embeddings_from_qdrant(tags):
    """Loads embeddings from Qdrant Cloud, ensuring vectors are retrieved."""
    logging.info(f"Loading embeddings from Qdrant for tag: {tags}")

    try:

        search_result = qdrant_client.scroll(
            collection_name=tags,
            scroll_filter=Filter(
                must=[{"key": "tag", "match": {"value": tags}}]
            ),
            limit=1000,
            with_vectors=True
        )

        if not search_result or not search_result[0]:
            logging.warning("No embeddings found in Qdrant.")
            return None

        embeddings = []
        texts = []

        for point in search_result[0]:
            if "text" in point.payload and "tag" in point.payload and point.vector:
                embeddings.append(point.vector)
                texts.append(point.payload["text"])
            else:
                logging.warning(f"Skipping invalid point: {point.payload}")

        if not embeddings:
            logging.warning("Embeddings found but vectors are missing!")
            return None

        embeddings = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return {"index": index, "texts": texts}

    except Exception as e:
        logging.error(f"Error loading embeddings from Qdrant: {e}")
        return None


def answer_question(query, user_id, tag="default", model='llama3.2'):
    """Answers a question based on stored embeddings."""

    embeddings = load_embeddings_from_qdrant(tag)
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
            {"role": "user", "content": f"Answer strictly based on this context: {retrieved_text}\nQuestion: {query}"}
        ]
    )
    return response["message"]


def azure_answer_question(query, user_id, tag="default", model="gpt-35-turbo", apiVersion="2023-07-01-preview"):
    """Answers a question based on stored embeddings."""
    embeddings = load_embeddings_from_qdrant(tag)
    if embeddings is None:
        return "No trained model found!"
    logging.info(f"Processing query: {query} for model: {tag}")
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    D, I = embeddings["index"].search(query_embedding, 3)
    retrieved_text = "\n".join([embeddings["texts"][i] for i in I[0]])
    logging.info(f"Retrieving response using Azure OpenAI {model}.")
    client = AzureOpenAI(
        api_version=apiVersion,
        api_key=Config.Model.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.Model.AZURE_OPENAI_ENDPOINT,
    )
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on this context: {retrieved_text}\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content



