from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    class Path:
        """Path configurations"""
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "files/images"
        LOGS_DIR = APP_HOME / "logs"

    class Database:
        """Database configurations"""
        USE_QDRANT = os.getenv("USE_QDRANT", "false").lower() == "true"
        SAVE_LOCAL_DB = os.getenv("SAVE_LOCAL_DB", "true").lower() == "true"
        DOCUMENTS_COLLECTION = "documents"
        COLLECTION_CONFIG: Dict[str, Any] = {
            "vectors_config": {
                "size": int(os.getenv("QDRANT_COLLECTION_SIZE", "768")),
                "distance": "Cosine"
            }
        }
        QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    class Model:
        """Model configurations"""
        EMBEDDINGS = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")
        RERANKER = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
        LOCAL_LLM = os.getenv("LOCAL_LLM", "gemma2:9b")
        REMOTE_LLM = os.getenv("REMOTE_LLM", "llama-3.3-70b-versatile")
        TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
        MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "8000"))
        USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

    class Retriever:
        """Retriever configurations"""
        USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
        USE_CHAIN_FILTER = os.getenv("USE_CHAIN_FILTER", "false").lower() == "true"
        TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
        SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    class AWS:
        """AWS configurations"""
        ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        S3_BUCKET = os.getenv("AWS_S3_BUCKET")

    # Application settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    CONVERSATION_MESSAGES_LIMIT = int(os.getenv("CONVERSATION_MESSAGES_LIMIT", "100"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        required_vars = [
            "APP_HOME",
            "AWS_ACCESS_KEY_ID" if not cls.Model.USE_LOCAL else None,
            "AWS_SECRET_ACCESS_KEY" if not cls.Model.USE_LOCAL else None,
            "GROQ_API_KEY" if not cls.Model.USE_LOCAL else None,
        ]

        missing_vars = [
            var for var in required_vars
            if var is not None and not os.getenv(var)
        ]

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        return True

    @classmethod
    def setup_directories(cls) -> None:
        """Create required directories"""
        directories = [
            cls.Path.DATABASE_DIR,
            cls.Path.DOCUMENTS_DIR,
            cls.Path.LOGS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create .gitkeep files to preserve empty directories
        for directory in directories:
            gitkeep = directory / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()