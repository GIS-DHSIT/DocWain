import os
from pathlib import Path


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME /"images"
        DUCKDB_DIR = DATABASE_DIR / "duck-db"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Qdrant:
        URL = "https://0a25c9cf-4685-49c7-9382-4c3510754343.europe-west3-0.gcp.cloud.qdrant.io:6333"
        API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ki8Uux4HtKDYj6ebqH9nzu3qg7QVmXTvxT6UkLNwTc4"

    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "gemma2:9b"
        REMOTE_LLM = "llama-3.3-70b-versatile"
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000
        USE_LOCAL = False

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False

    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 160
