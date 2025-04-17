import ollama
import faiss
import logging
import numpy as np
from api.config import Config
from openai import AzureOpenAI
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from api.dw_chat import get_chat_history,save_chat_history,generate_follow_up_questions


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
AzureOpenAI.api_key = Config.Model.AZURE_OPENAI_API_KEY
mongoClient = MongoClient(Config.MongoDB.URI)
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)


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



def answer_question(query, user_id, tag, model='llama3.2',persona="Document Assistant"):
    """Answers a question based on stored embeddings."""

    history = get_chat_history(user_id) or []
    if not isinstance(query, str):
        query = str(query)
    retrieved_text  = load_embeddings_from_qdrant(tag,query)
    # refined_query, follow_up_questions = rewrite_query(query, history, retrieved_text)
    follow_up_questions = generate_follow_up_questions(retrieved_text) if retrieved_text else []
    if not retrieved_text:
        return "No relevant knowledge found in database."

    formatted_history = "\n".join([f"User: {h['query']}\nAI: {h['response']}" for h in history[-5:]])
    prompt = f"""You are an AI assistant. with NO access to outside world. Answer strictly limited only to retrieved text. 

       The answer content should only contain actual responses.

       **Retrieved Context:**
       {retrieved_text}

       **User Query:**
       {query}
       
       **Chat History:**
       {formatted_history}
       """
    if model == 'Azure-OpenAI':
        logging.info(f"Retrieving response using Azure OpenAI {model}.")
        client = AzureOpenAI(
            api_version=Config.AzureGpt4o.AZUREGPT4O_Version,
            api_key=Config.AzureGpt4o.AZUREGPT4O_API_KEY,
            azure_endpoint=Config.AzureGpt4o.AZUREGPT4O_ENDPOINT,
        )
        response = client.chat.completions.create(
            model=Config.AzureGpt4o.AZUREGPT4O_DEPLOYMENT,
            messages=[
                {"role": "system",
                 "content": "You are a {0} and respond only with the knowledge provided. respond only with actual content".format(persona)},
                {"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        history.append({"query": query, "response": response_text})
        save_chat_history(user_id, history)

    else:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system",
                       "content": "You are {0} and respond only with the knowledge provided.respond only with actual content".format(persona)},
                      {"role": "user", "content": prompt}]
        )
        response_text = response["message"]
        history.append({"query": query, "response": response_text['content']})
        save_chat_history(user_id, history)
    return {"answer": response_text, "follow_ups": follow_up_questions}
