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
from api.dw_chat import get_chat_history,save_chat_history


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


def answer_question(query, user_id, tag):
    """Answers a question based on stored embeddings and generates human-like follow-up responses."""
    history = get_chat_history(user_id) or []

    if not isinstance(query, str):
        query = str(query)

    # Retrieve relevant context from embeddings
    retrieved_text = load_embeddings_from_qdrant(tag, query)
    if not retrieved_text:
        return {
            "answer": "I couldn't find anything relevant in my knowledge base. Would you like me to try rephrasing or expanding the search?",
            "follow_ups": ["Can you clarify what you're looking for?", "Would you like me to explain related concepts?",
                           "Do you want me to search in another category?"]
        }

    # Format chat history for continuity
    formatted_history = "\n".join(
        [f"User: {h['query']}\nAI: {h['response']}" for h in history[-5:]]
    ) if history else "No previous chat history."

    # Constructing the final prompt
    prompt = f"""
    You are a friendly AI assistant with NO access to the outside world. Answer the user's question based strictly on the retrieved text.

    **Chat History:**  
    {formatted_history}

    **Retrieved Knowledge:**  
    {retrieved_text}

    **User Query:**  
    {query}

    Respond conversationally, keeping responses engaging, helpful, and natural.
    Also, suggest three follow-up questions to keep the conversation going.
    """

    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=Config.Model.AZURE_VERSION,
            api_key=Config.Model.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.Model.AZURE_OPENAI_ENDPOINT,
        )

        response = client.chat.completions.create(
            model=Config.Model.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a conversational assistant. Keep responses natural and engaging."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract response text
        response_text = response.choices[
            0].message.content if response.choices else "I'm not sure how to respond to that. Can you clarify?"

        # Generate human-like follow-up questions within the same request
        follow_up_prompt = f"""
        Based on the user's question and the retrieved context, suggest 3 natural-sounding follow-up questions.

        **User Query:** {query}
        **Context:** {retrieved_text}

        Format them as a numbered list.
        """

        follow_up_response = client.chat.completions.create(
            model=Config.Model.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system",
                 "content": "Generate three engaging follow-up questions related to the given context."},
                {"role": "user", "content": follow_up_prompt}
            ]
        )

        follow_up_questions = follow_up_response.choices[0].message.content.split(
            "\n") if follow_up_response.choices else []

        # Save chat history
        history.append({"query": query, "response": response_text})
        save_chat_history(user_id, history)

        return {"answer": response_text, "follow_ups": follow_up_questions}

    except Exception as e:
        return {
            "answer": f"Oops, something went wrong: {str(e)}",
            "follow_ups": ["Can you try rephrasing your question?", "Would you like me to summarize instead?",
                           "Let me know how I can assist you better."]
        }

