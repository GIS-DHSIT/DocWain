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