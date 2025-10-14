import json
import time
import redis
import faiss
import ollama
import logging
import hashlib
import numpy as np
from api.config import Config
from openai import AzureOpenAI
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from api.dw_chat import get_chat_history,save_chat_history,generate_follow_up_questions,rerank_results


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
AzureOpenAI.api_key = Config.Model.AZURE_OPENAI_API_KEY
mongoClient = MongoClient(Config.MongoDB.URI)
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
redis_client = redis.Redis(host=Config.Redis.HOST, port=Config.Redis.PORT, decode_responses=True,
                           username='default', password=Config.Redis.PASSWORD)


def generate_cache_key(query):
    """Generates a Redis cache key based on the query string."""
    return "query:" + hashlib.sha256(query.encode()).hexdigest()


def expand_query(query, tag, top_k=3):
    """
    Expands the query using embedding similarity to retrieve semantically similar terms.
    """
    try:
        query_embedding = MODEL.encode([query], convert_to_numpy=True)
        expanded_queries = [query]

        search_result = qdrant_client.scroll(
            collection_name=tag,
            scroll_filter=Filter(must=[{"key": "tag", "match": {"value": tag}}]),
            limit=1000,
            with_vectors=True
        )

        if not search_result or not search_result[0]:
            return expanded_queries

        texts = [point.payload["text"] for point in search_result[0] if "text" in point.payload]
        embeddings = np.array([point.vector for point in search_result[0] if point.vector], dtype=np.float32)

        if embeddings.size == 0:
            return expanded_queries

        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)

        faiss.normalize_L2(query_embedding)
        D, I = faiss_index.search(query_embedding, top_k)

        for idx in I[0]:
            expanded_queries.append(texts[idx])

        return list(set(expanded_queries))

    except Exception as e:
        logging.error(f"Error in query expansion: {e}")
        return [query]

def load_embeddings_from_qdrant(tag, query: str, top_k: int = 5):
    """Retrieves relevant document chunks using hybrid retrieval (BM25 + FAISS)."""
    try:
        cache_key = generate_cache_key(query)
        cached_result = redis_client.get(cache_key)

        if cached_result:
            logging.info("Cache hit! Returning cached retrieval results.")

            # **Ensure valid cache structure**
            cached_data = json.loads(cached_result)
            if isinstance(cached_data, dict) and "retrieved_texts" in cached_data:
                return cached_data["retrieved_texts"], cached_data.get("bm25_scores", []), cached_data.get("faiss_scores", [])

            logging.warning("Cache format invalid. Fetching new results...")

        expanded_queries = expand_query(query,tag)

        # **Check if Qdrant has stored embeddings**
        search_result = qdrant_client.scroll(
            collection_name=tag,
            scroll_filter=Filter(must=[{"key": "tag", "match": {"value": tag}}]),
            limit=1000,
            with_vectors=True
        )

        if not search_result or not search_result[0]:
            logging.warning(f"No relevant data found in Qdrant for tag: {tag}")
            return None, None, None  # Ensure function returns expected structure

        texts = [point.payload["text"] for point in search_result[0] if "text" in point.payload]
        embeddings = np.array([point.vector for point in search_result[0] if point.vector], dtype=np.float32)

        if embeddings.size == 0:
            logging.warning("Embeddings found but vectors are missing!")
            return None, None, None

        # **Ensure BM25 is properly computed**
        tokenized_corpus = [text.split() for text in texts]
        if len(tokenized_corpus) == 0:
            logging.warning("BM25 tokenized corpus is empty!")
            return None, None, None

        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = np.zeros(len(texts))

        for expanded_query in expanded_queries:
            bm25_scores += bm25.get_scores(expanded_query.split())

        # **Ensure FAISS is initialized properly**
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)

        query_embedding = MODEL.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        D, I = faiss_index.search(query_embedding, top_k)

        if I.shape[1] == 0:
            logging.warning("FAISS retrieval returned empty results!")
            return None, None, None

        query_length = len(query.split())
        alpha = max(0.3, min(0.7, query_length / 20))

        scaler = MinMaxScaler()
        bm25_scores = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        faiss_scores = scaler.fit_transform(D.flatten().reshape(-1, 1)).flatten()

        hybrid_scores = {idx: alpha * faiss_scores[i] + (1 - alpha) * bm25_scores[idx] for i, idx in enumerate(I[0])}
        sorted_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]

        retrieved_texts = [texts[i] for i in sorted_indices]

        # **Diversity-Preserving Filtering**
        seen = set()
        diverse_texts = []
        for text in retrieved_texts:
            words = tuple(text.split()[:10])
            if words not in seen:
                seen.add(words)
                diverse_texts.append(text)

        # **Store Complete Data in Redis Cache**
        cache_data = {
            "retrieved_texts": diverse_texts,
            "bm25_scores": bm25_scores.tolist(),
            "faiss_scores": faiss_scores.tolist()
        }
        redis_client.setex(cache_key, 86400, json.dumps(cache_data))  # Cache expires in 1 day

        return diverse_texts, bm25_scores, faiss_scores  # Ensure function returns 3 values

    except Exception as e:
        logging.error(f"Error in hybrid retrieval: {e}")
        return None, None, None  # Ensure function always returns 3 values


def retrieve_similar_past_queries(query, chat_history, top_k=3):
    """Finds past queries that are similar to the current query using FAISS."""
    try:
        if not chat_history:
            return []

        past_queries = [h["query"] for h in chat_history if "query_embedding" in h]
        past_embeddings = np.array([h["query_embedding"] for h in chat_history if "query_embedding" in h])

        if len(past_embeddings) == 0:
            return []

        faiss_index = faiss.IndexFlatIP(past_embeddings.shape[1])
        faiss.normalize_L2(past_embeddings)
        faiss_index.add(past_embeddings)

        query_embedding = MODEL.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        D, I = faiss_index.search(query_embedding, top_k)

        similar_queries = [past_queries[idx] for idx in I[0]]
        return similar_queries

    except Exception as e:
        logging.error(f"Error retrieving similar past queries: {e}")
        return []


def get_dynamic_cache_expiry(user_id, query):
    """Determines cache expiry time dynamically based on user session activity."""
    last_active_key = f"user:last_active:{user_id}"
    last_active = redis_client.get(last_active_key)
    current_time = int(time.time())

    if last_active:
        last_active = int(last_active)
        time_since_last_active = current_time - last_active
        if time_since_last_active < 600:
            return 600
        elif time_since_last_active < 7200:
            return 7200
    if len(query.split()) > 7:  # If query is long, assume it's an FAQ
        return 604800  # 7 days
    return 3600  # 1 hour


def answer_question(query, user_id, tag, model='llama3.2'):
    """Answers a question based on stored embeddings."""
    # **Ensure Proper Data Types for Redis Storage**
    last_active_key = f"user:last_active:{user_id}"
    current_time = int(time.time())  # Ensure timestamp is integer

    try:
        redis_client.set(last_active_key, str(current_time),
                         ex=604800)  # Store last active time as a valid integer string
    except redis.exceptions.ResponseError as e:
        logging.error(f"Redis Error when setting last active time: {e}")

    # **Determine Cache Expiry Time**
    cache_expiry = get_dynamic_cache_expiry(user_id, query)
    cache_key = generate_cache_key(query)

    # **Check Cached Response**
    cached_response = redis_client.get(cache_key)
    chat_history = get_chat_history(user_id) or []
    similar_queries = retrieve_similar_past_queries(query, chat_history)
    query_embedding = MODEL.encode([query], convert_to_numpy=True).tolist()[0]
    retrieved_texts, bm25_scores, faiss_scores = load_embeddings_from_qdrant(tag, query)
    reranked_texts = rerank_results(retrieved_texts, similar_queries, bm25_scores, faiss_scores)
    follow_up_questions = generate_follow_up_questions(reranked_texts) if reranked_texts else []
    if not retrieved_texts:
        return "No relevant knowledge found in database."
    if cached_response:
        logging.info(f"Cache hit! Returning cached response (Expires in {cache_expiry // 60} min).")
        return {"answer": cached_response, "follow_ups": follow_up_questions, "cached": True}
    formatted_history = "\n".join([f"User: {h['query']}\nAI: {h['response']}" for h in chat_history[-5:]])
    prompt = f"""
       You are an AI assistant. with NO access to outside world. Answer strictly limited only to retrieved text.

       **Chat History:** 
       {formatted_history}

       **Retrieved Context:**
       {reranked_texts}

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
                {"role": "system",
                 "content": "You are a clever assistant and respond only with the knowledge provided."},
                {"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
    else:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system",
                       "content": "You are a clever assistant and respond only with the knowledge provided."},
                      {"role": "user", "content": prompt}]
        )
        response_text = response["message"]
        response["message"] if isinstance(response, dict) else str(response)
        # redis_client.setex(cache_key, 86400, response_text)

        chat_history.append({"query": query, "response": response, "query_embedding": query_embedding})
        save_chat_history(user_id, chat_history)
    return {"answer": response_text, "follow_ups": follow_up_questions}
