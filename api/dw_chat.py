import ollama
import boto3 as b3
import json
import logging
from typing import Dict, List
from api.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

s3_client = b3.client(
        "s3",
        aws_access_key_id=Config.AWS.ACCESS_KEY,
        aws_secret_access_key=Config.AWS.SECRET_KEY,
        region_name=Config.AWS.REGION
    )
S3_BUCKET_NAME = Config.AWS.BUCKET_NAME


def get_chat_history(user_id):
    """Retrieve chat history from S3 or return an empty list if not found."""
    try:
        file_key = f"chat_history/{user_id}.json"
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        chat_history = json.loads(response["Body"].read().decode("utf-8"))
        return chat_history
    except s3_client.exceptions.NoSuchKey:
        logging.info(f"No existing chat history found for user {user_id}. Initializing new history.")
        return []
    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}")
        return []


def save_chat_history(user_id, chat_history):
    """Save chat history to AWS S3 in a JSON-safe format."""
    try:
        file_key = f"chat_history/{user_id}.json"
        formatted_history = []
        for message in chat_history:
            formatted_message = {
                "query": str(message.get("query", "")),
                "response": str(message.get("response", ""))
            }
            formatted_history.append(formatted_message)

        chat_data = json.dumps(formatted_history, ensure_ascii=False)

        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_key, Body=chat_data)
        logging.info(f"Chat history successfully saved for user {user_id}.")

    except Exception as e:
        logging.error(f"Error saving chat history: {e}")


def clear_chat_history(user_id, session_id=None):
    """Clear chat history for a specific user or session."""
    file_key = f"chat_history/{user_id}.json"
    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=file_key)

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
