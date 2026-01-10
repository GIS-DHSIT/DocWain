import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment overrides from a .env file when present. This makes sure the
# app picks up the deployment-specific connection strings instead of falling
# back to the hard-coded defaults below.
load_dotenv()

DEFAULT_REDIS_CONNECTION_STRING = "docwain-rediscache.redis.cache.windows.net:6380,password=2kwDGVV5OuaOo3YCUD5tGkM5RXgWFU4ROAzCaB5RoFo=,ssl=True,abortConnect=False"


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"
        DUCKDB_DIR = DATABASE_DIR / "duck-db"

    class Qdrant:
        URL = os.getenv("QDRANT_URL", 'https://89f776c3-76fb-493f-8509-c583d9579329.europe-west3-0.gcp.cloud.qdrant.io')
        API = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-cJ2HVTYcH3u5KNuZuxZRNJhhTFfZwqkoVacNCKBYkY")

    class Gemini:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB9jPJeY0W0HJXWbrrNdoQDIAlmrcrzcq8")
        # GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta

    class Model:
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
        SENTENCE_TRANSFORMERS = EMBEDDING_MODEL
        SENTENCE_TRANSFORMERS_FALLBACK = SENTENCE_TRANSFORMERS
        SENTENCE_TRANSFORMERS_CANDIDATES = [
            EMBEDDING_MODEL,
            os.getenv("EMBEDDING_FALLBACK_MODEL", "sentence-transformers/all-mpnet-base-v2"),
        ]
        RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        OCR_ENGINE = os.getenv("OCR_ENGINE", "pytesseract")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "")
        AZURE_VERSION = os.getenv("AZURE_OPENAI_VERSION", "")
        # ✅ Gemini 2.5 Flash configs
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB9jPJeY0W0HJXWbrrNdoQDIAlmrcrzcq8")
        # GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
        GEMINI_MODEL_NAME = "gemini-2.5-flash"


    class AzureGpt4o:
        AZUREGPT4O_ENDPOINT = "https://dw-openai-dev.openai.azure.com/"
        AZUREGPT4O_DEPLOYMENT = "dw-dev1-gpt-4o"
        AZUREGPT4O_API_KEY = '6JSK5oHMv76xL6IAtFwVfgCRykf24MWdvp6oRpxawBk9sGyqXuQYJQQJ99BCACmepeSXJ3w3AAABACOGjB0M'
        AZUREGPT4O_Version = "2024-05-01-preview"

    class MongoDB:
        # Allow overriding the Mongo connection string via env; fall back to a localhost URI
        DEFAULT_URI = 'mongodb+srv://dhsdbadmin:d%21p%40s5w0rd@dw-dev-mongodb.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
        URI = os.getenv("MONGODB_URI", DEFAULT_URI)
        FALLBACK_URI = os.getenv("MONGODB_FALLBACK_URI", "mongodb://localhost:27017")
        DB = os.getenv("MONGODB_DB", 'docwain')
        CONNECTOR = os.getenv("MONGODB_CONNECTORS", 'connectors')
        DOCUMENTS = os.getenv("MONGODB_DOCUMENTS", 'documents')
        PROFILES = os.getenv("MONGODB_PROFILES", 'profiles')
        # Add this new line for subscriptions collection
        # modified by maha/maria
        SUBSCRIPTIONS = os.getenv("MONGODB_SUBSCRIPTIONS_COLLECTION", "subscriptions")

    class Encryption:
        ENCRYPTION_KEY = 'J9cuHrESAz'

    class AWS:
        PROFILE = 'DHS'
        ACCESS_KEY = 'AKIA2UC3EV7PWOLLH4GI'
        SECRET_KEY = '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'
        REGION = 'eu-west-2'
        BUCKET_NAME = 'docwain-chat-history'

    class AzureBlob:
        CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=dwchathistory;AccountKey=1qh9meNrD3PJIbSpfOSC8uMhhg23rMrxBQ0PhL0+QRnE+RUNt1GFx7PCZILc6/XVL5GCrfiZvoQl+ASt3jNtPQ==;EndpointSuffix=core.windows.net"
        CONTAINER_NAME = "chat-history"
        BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=your_account_name;AccountKey=your_account_key;EndpointSuffix=core.windows.net"
        blob_key = "1qh9meNrD3PJIbSpfOSC8uMhhg23rMrxBQ0PhL0+QRnE+RUNt1GFx7PCZILc6/XVL5GCrfiZvoQl+ASt3jNtPQ=="

    class DocAzureBlob:
        AZURE_BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=docwainuploads;AccountKey=+mxYrAnMSQeGjclw7ATpu7/Q6sT/I7twHka6JD4toKGFRlGW0HbX4OcyoproS6TeQ0q6CLSK1Dk6+AStJp0qYA==;EndpointSuffix=core.windows.net"
        AZURE_BLOB_CONTAINER_NAME = "local-uploads"
        AZURE_BLOB_KEY = "+mxYrAnMSQeGjclw7ATpu7/Q6sT/I7twHka6JD4toKGFRlGW0HbX4OcyoproS6TeQ0q6CLSK1Dk6+AStJp0qYA=="
        AZURE_BLOB_ACCOUNT_NAME = "docwainuploads"

    class VettingAzureBlob:
        AZURE_BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=docwainvetting;AccountKey=CM3patdUte0Fm82aFbNBffxtBVHS2i0CzC5JWPy/uQ0C/JtJwGrvHqZ2FO1C/7KH3iduWuCyA/pQ+AStRQkRiQ==;EndpointSuffix=core.windows.net"
        AZURE_BLOB_CONTAINER_NAME = "configuration"
        AZURE_BLOB_KEY = "CM3patdUte0Fm82aFbNBffxtBVHS2i0CzC5JWPy/uQ0C/JtJwGrvHqZ2FO1C/7KH3iduWuCyA/pQ+AStRQkRiQ=="
        AZURE_BLOB_ACCOUNT_NAME = "docwainvetting"
        AZURE_BLOB_FILE_NAME = "default/Vetting conditions_3words.xlsx"

    class Redis:
        CONNECTION_STRING = DEFAULT_REDIS_CONNECTION_STRING
        HOST = "docwain-rediscache.redis.cache.windows.net"
        PORT = 6380
        PASSWORD = "2kwDGVV5OuaOo3YCUD5tGkM5RXgWFU4ROAzCaB5RoFo="
        DB = 0
        SSL = True
        ABORT_CONNECT = False

    class Teams:
        SHARED_SECRET = os.getenv("TEAMS_SHARED_SECRET", "")
        SIGNATURE_ENABLED = os.getenv("TEAMS_SIGNATURE_ENABLED", "false").lower() == "true"
        DEFAULT_PROFILE = os.getenv("TEAMS_DEFAULT_PROFILE", "default")
        DEFAULT_SUBSCRIPTION = os.getenv("TEAMS_DEFAULT_SUBSCRIPTION", "default")
        DEFAULT_MODEL = os.getenv("TEAMS_DEFAULT_MODEL", "llama3.2")
        DEFAULT_PERSONA = os.getenv("TEAMS_DEFAULT_PERSONA", "Document Assistant")
        UPLOAD_DIR = os.getenv("TEAMS_UPLOAD_DIR", "/tmp")
        BLOB_CONNECTION_STRING = os.getenv("TEAMS_BLOB_CONNECTION_STRING", "")
        BLOB_CONTAINER = os.getenv("TEAMS_BLOB_CONTAINER", "local-uploads")
        BLOB_PATH_PREFIX = os.getenv("TEAMS_BLOB_PATH_PREFIX", "teams")
        SESSION_AS_SUBSCRIPTION = os.getenv("TEAMS_SESSION_AS_SUBSCRIPTION", "true").lower() == "true"
        PROFILE_PER_USER = os.getenv("TEAMS_PROFILE_PER_USER", "true").lower() == "true"
        MAX_ATTACHMENT_MB = int(os.getenv("TEAMS_MAX_ATTACHMENT_MB", "50"))
        HTTP_TIMEOUT_SEC = float(os.getenv("TEAMS_HTTP_TIMEOUT_SEC", "20"))
        HTTP_RETRIES = int(os.getenv("TEAMS_HTTP_RETRIES", "2"))
        BOT_ACCESS_TOKEN = os.getenv("TEAMS_BOT_ACCESS_TOKEN", "")

    class Retrieval:
        CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "800"))
        CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "200"))
        MIN_CHUNK_SIZE = int(os.getenv("RETRIEVAL_MIN_CHUNK_SIZE", "150"))
        MAX_CONTEXT_CHUNKS = int(os.getenv("RETRIEVAL_MAX_CONTEXT_CHUNKS", "12"))
        SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.10"))
        USE_SPARSE_VECTORS = os.getenv("RETRIEVAL_USE_SPARSE_VECTORS", "true").lower() == "true"
        USE_ADJACENT_EXPANSION = os.getenv("RETRIEVAL_USE_ADJACENT_EXPANSION", "true").lower() == "true"
        NEIGHBOR_WINDOW = int(os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", "2"))
        NEIGHBOR_MAX_NEW = int(os.getenv("RETRIEVAL_NEIGHBOR_MAX_NEW", "10"))
        HYBRID_WEIGHTS = {
            "dense": float(os.getenv("HYBRID_WEIGHT_DENSE", "0.6")),
            "sparse": float(os.getenv("HYBRID_WEIGHT_SPARSE", "0.4")),
        }
        RERANKER_ENABLED = os.getenv("RETRIEVAL_RERANKER_ENABLED", "true").lower() == "true"

    class LLM:
        TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        TOP_P = float(os.getenv("LLM_TOP_P", "0.85"))
        MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "2"))

    class Finetune:
        AUTO_ENABLED = os.getenv("FINETUNE_AUTO_ENABLED", "false").lower() == "true"
        AUTO_INTERVAL_HOURS = float(os.getenv("FINETUNE_AUTO_INTERVAL_HOURS", "6"))
        TEACHER_MODEL = os.getenv("FINETUNE_TEACHER_MODEL", "")
        MAX_CONCURRENCY = int(os.getenv("FINETUNE_MAX_CONCURRENCY", "4"))
        QA_RETRY_MAX = int(os.getenv("FINETUNE_QA_RETRY_MAX", "2"))
        AUTO_MIN_POINTS = int(os.getenv("FINETUNE_AUTO_MIN_POINTS", "40"))
        AUTO_MIN_RECORDS = int(os.getenv("FINETUNE_AUTO_MIN_RECORDS", "20"))
        AUTO_MAX_PROFILES_PER_RUN = int(os.getenv("FINETUNE_AUTO_MAX_PROFILES_PER_RUN", "10"))
        CLEANUP_ENABLED = os.getenv("FINETUNE_CLEANUP_ENABLED", "false").lower() == "true"
        CLEANUP_KEEP_LAST = int(os.getenv("FINETUNE_CLEANUP_KEEP_LAST", "3"))
