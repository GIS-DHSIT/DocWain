import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "files/images"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

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

    _GPU_ENV_DEFAULTS = {
        "CUDA_VISIBLE_DEVICES": "0",
        "FASTEMBED_DEVICE": "cuda",
        "FASTEMBED_USE_GPU": "1",
        "FASTEMBED_BATCH_SIZE": "64",
    }

    @classmethod
    def _ensure_directories(cls) -> None:
        directories: Iterable[Path] = (
            cls.Path.DATABASE_DIR,
            cls.Path.DOCUMENTS_DIR,
            cls.Path.IMAGES_DIR,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _configure_gpu_environment(cls) -> None:
        load_dotenv()

        for env_var, default_value in cls._GPU_ENV_DEFAULTS.items():
            os.environ.setdefault(env_var, default_value)

        # Avoid noisy tokenizer warnings when running on multi-core GPU machines.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @classmethod
    def bootstrap(cls) -> None:
        """Ensure required directories exist and GPU env variables are set."""
        cls._configure_gpu_environment()
        cls._ensure_directories()


# Automatically prepare the runtime environment when the configuration is imported.
Config.bootstrap()
