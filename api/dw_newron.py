import ollama
import faiss
import logging
import numpy as np
from typing import List
from api.config import Config
from openai import AzureOpenAI
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from api.dw_chat import get_chat_history,save_chat_history


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
AzureOpenAI.api_key = Config.Model.AZURE_OPENAI_API_KEY
mongoClient = MongoClient(Config.MongoDB.URI)
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)


def rewrite_query(user_id: str, query: str, chat_history: List[dict],modelName='llama3.2'):
    """Rewrites user queries to improve retrieval accuracy using chat history & AI-based expansion."""

    num_words = len(query.split())

    if num_words > 15:
        logging.info("Query is already detailed, skipping rewriting.")
        return query
    recent_history = chat_history[-3:] if chat_history else []
    history_context = " ".join([f"User: {h['query']} AI: {h['response']}" for h in recent_history])

    refined_query = ollama.chat(
        model=modelName,
        messages=[
            {"role": "system", "content": "You are an AI specialized in rewriting user queries for improved search results."},
            {"role": "user", "content": f"Rewrite the following query to be more detailed and precise:\n\nQuery: {query}\n\nChat History Context: {history_context}"}
        ]
    )

    rewritten_query = refined_query["message"]
    logging.info(f"Original Query: {query} | Rewritten Query: {rewritten_query}")

    return rewritten_query

def load_embeddings_from_qdrant(tag,query: str, top_k: int = 5):
    try:
        search_result = qdrant_client.scroll(
            collection_name=tag,
            scroll_filter=Filter(must=[{"key": "tag", "match": {"value": tag}}]),
            limit=1000,
            with_vectors=True
        )

        if not search_result or not search_result[0]:
            logging.warning("No relevant data found in Qdrant.")
            return None

        texts = [point.payload["text"] for point in search_result[0] if "text" in point.payload]
        embeddings = np.array([point.vector for point in search_result[0] if point.vector], dtype=np.float32)

        if embeddings.size == 0:
            logging.warning("Embeddings found but vectors are missing!")
            return None

        tokenized_corpus = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.split())

        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        query_embedding = MODEL.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        D, I = index.search(query_embedding, top_k)

        scaler = MinMaxScaler()
        bm25_scores = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
        faiss_scores = scaler.fit_transform(D.flatten().reshape(-1, 1)).flatten()

        hybrid_scores = {}
        for i, idx in enumerate(I[0]):
            hybrid_scores[idx] = faiss_scores[i]
        for i, score in enumerate(bm25_scores):
            if i in hybrid_scores:
                hybrid_scores[i] += score
            else:
                hybrid_scores[i] = score

        sorted_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]
        retrieved_texts = [texts[i] for i in sorted_indices]

        return "\n".join(retrieved_texts)

    except Exception as e:
        logging.error(f"Error in hybrid retrieval: {e}")
        return None


def answer_question(query, user_id, tag, model='llama3.2'):
    """Answers a question based on stored embeddings."""
    history = get_chat_history(user_id) or []
    refined_query = rewrite_query(user_id, query, history)
    retrieved_text  = load_embeddings_from_qdrant(tag,refined_query)
    if not retrieved_text:
        return "No relevant knowledge found in database."

    formatted_history = "\n".join([f"User: {h['query']}\nAI: {h['response']}" for h in history[-5:]])

    prompt = f"""
        You are a strict knowledge-based AI assistant. Answer ONLY based on retrieved information from the vector database.
        If the answer is not available, reply with "I do not have that information provided."

        **Chat History:** 
        {formatted_history}

        **Retrieved Context:**
        {retrieved_text}

        **User Query:**
        {query}
        """
    if model == 'Azure-OpenAI':
        logging.info(f"Retrieving response using Azure OpenAI {model}.")
        client = AzureOpenAI(
            api_version=Config.Model.AZURE_VERSION,
            api_key=Config.Model.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.Model.AZURE_OPENAI_ENDPOINT,
        )
        response = client.chat.completions.create(
            model=Config.Model.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant limited to knowledge from the database."},
                {"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
    else:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant limited to knowledge from the database."},
                      {"role": "user", "content": prompt}]
        )
        response_text = response["message"]
        history.append({"query": query, "response": response_text})
        save_chat_history(user_id, history)
    return response_text



