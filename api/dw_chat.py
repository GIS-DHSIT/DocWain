import ollama
import json
import logging
from typing import Dict, List
from api.config import Config
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

connection_string = Config.AzureBlob.CONNECTION_STRING
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = Config.AzureBlob.CONTAINER_NAME
container_client = blob_service_client.get_container_client(container_name)


def get_chat_history(user_id):
    """Retrieve chat history from Azure Blob Storage or return an empty list if not found."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)

        if not blob_client.exists():
            logging.info(f"No existing chat history found for user {user_id}. Initializing new history.")
            return []

        blob_data = blob_client.download_blob().readall()
        chat_history = json.loads(blob_data.decode("utf-8"))
        return chat_history

    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}")
        return []


def save_chat_history(user_id, chat_history):
    """Save chat history to Azure Blob Storage in a JSON-safe format."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)

        formatted_history = []
        for message in chat_history:
            formatted_message = {
                "query": str(message.get("query", "")),
                "response": str(message.get("response", ""))
            }
            formatted_history.append(formatted_message)

        chat_data = json.dumps(formatted_history, ensure_ascii=False)
        blob_client.upload_blob(chat_data, overwrite=True)
        logging.info(f"Chat history successfully saved for user {user_id}.")

    except Exception as e:
        logging.error(f"Error saving chat history: {e}")

def generate_follow_up_questions(retrieved_text, num_questions=3):
    """Generates follow-up questions based on retrieved content."""
    try:
        prompt = f"""
        Based on the following retrieved context, suggest {num_questions} relevant follow-up questions.

        **Context:** 
        {retrieved_text}

        **Follow-up Questions:**
        """
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "system", "content": "You are an AI specialized in helping users continue conversations effectively."},
                      {"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}")
        return []


def rerank_results(retrieved_texts, similar_queries, bm25_scores, faiss_scores):
    """
    Rerank retrieval results by boosting those that appeared in past interactions.
    """
    try:
        alpha = 0.6
        beta = 0.4
        gamma = 0.3

        final_scores = {}
        for i, text in enumerate(retrieved_texts):
            final_scores[i] = alpha * faiss_scores[i] + beta * bm25_scores[i]

            if any(q in text for q in similar_queries):
                final_scores[i] += gamma

        sorted_indices = sorted(final_scores, key=final_scores.get, reverse=True)
        return [retrieved_texts[i] for i in sorted_indices]

    except Exception as e:
        logging.error(f"Error in reranking: {e}")
        return retrieved_texts
