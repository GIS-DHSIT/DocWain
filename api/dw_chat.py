import ollama
import json
import logging
from io import BytesIO
from api.config import Config
from azure.storage.blob import BlobServiceClient, ContentSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

connection_string = Config.AzureBlob.CONNECTION_STRING
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = Config.AzureBlob.CONTAINER_NAME
container_client = blob_service_client.get_container_client(container_name)


def get_chat_history(user_id):
    """Retrieve chat history from Azure Blob Storage or return an empty list if not found."""
    try:
        blob_name = f"{user_id}.json"
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
        blob_name = f"{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)

        # Ensure the content settings are a proper object
        content_settings = ContentSettings(content_type="application/json")

        # Check if the blob exists, create it if not
        if not blob_client.exists():
            blob_client.upload_blob(b"[]", overwrite=True, content_settings=content_settings)
            logging.info(f"Blob created for user {user_id}.")

        # Format the chat history
        formatted_history = [
            {
                "query": str(message.get("query", "")),
                "response": str(message.get("response", ""))
            }
            for message in chat_history
        ]

        # Serialize JSON data to a BytesIO stream
        chat_data_stream = BytesIO(json.dumps(formatted_history, ensure_ascii=False).encode("utf-8"))

        # Upload the blob with content type
        blob_client.upload_blob(chat_data_stream, overwrite=True, content_settings=content_settings)
        logging.info(f"Chat history successfully saved for user {user_id}.")

    except Exception as e:
        logging.error(f"Error saving chat history: {e}")

def delete_chat_history(user_id):
    """Delete chat history from Azure Blob Storage."""
    try:
        blob_name = f"chat_history/{user_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
        logging.info(f"Chat history successfully deleted for user {user_id}.")
        return {"status": "success", "message": f"Chat history deleted for user {user_id}."}
    except Exception as e:
        logging.error(f"Error deleting chat history: {e}")
        return {"status": "error", "message": "Error deleting chat history."}

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

#
# def rewrite_query(query, chat_history, retrieved_content):
#     """Rewrites queries for improved retrieval and suggests follow-ups."""
#     try:
#         num_words = len(query.split())
#
#         if num_words > 15:
#             logging.info("Query is already detailed, skipping rewrite.")
#             follow_ups = generate_follow_up_questions(retrieved_content) if retrieved_content else []
#             return query, follow_ups
#
#         # **Retrieve Chat History for Context**
#         recent_history = chat_history[-3:] if chat_history else []
#         history_context = " ".join([f"User: {h['query']} AI: {h['response']}" for h in recent_history])
#
#         # **Enhance Query Using an LLM**
#         rewrite_prompt = f"""
#         Improve the following query for better retrieval results. If the query is vague, add relevant details based on the retrieved content
#
#         **User Query:** "{query}"
#         **Recent Chat History:** {history_context}
#         **Relevant data:**{retrieved_content}
#         **Improved Query:**
#         """
#         response = ollama.chat(
#             model="llama3.2",
#             messages=[{"role": "system", "content": "You are an AI specialized in improving search queries. only respond with responses"},
#                       {"role": "user", "content": rewrite_prompt}]
#         )
#
#         refined_query = response['message']['content'].split('**')[-1].replace('"','').strip()
#
#         # **Query Expansion with Sentence Transformer**
#         expanded_query_embeddings = MODEL.encode([query, refined_query])
#         similarity_score = util.cos_sim(expanded_query_embeddings[0], expanded_query_embeddings[1])
#
#         if similarity_score < 0.8:  # If rewritten query differs significantly, return both
#             refined_query = f"{query} | {refined_query}"
#
#         logging.info(f" Original Query: {query} → Rewritten Query: {refined_query}")
#
#         follow_up_questions = generate_follow_up_questions(retrieved_content) if retrieved_content else []
#
#         return refined_query, follow_up_questions
#
#     except Exception as e:
#         logging.error(f" Error in query rewriting: {e}")
#         return query, []
