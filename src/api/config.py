import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REDIS_CONNECTION_STRING = "docwain-rediscache.redis.cache.windows.net:6380,password=2kwDGVV5OuaOo3YCUD5tGkM5RXgWFU4ROAzCaB5RoFo=,ssl=True,abortConnect=False"


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"
        DUCKDB_DIR = DATABASE_DIR / "duck-db"

    class API:
        # Allow explicit whitelist; fall back to localhost defaults when not provided.
        ALLOW_ORIGINS = [
            origin.strip()
            for origin in os.getenv(
                "API_ALLOWED_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if origin.strip()
        ]
        # Permit "*" only when explicitly enabled for local development.
        ALLOW_ALL_ORIGINS = os.getenv("API_ALLOW_ALL_ORIGINS", "").lower() in {"1", "true", "yes"}

    class Execution:
        DEFAULT_AGENT_MODE = os.getenv("DEFAULT_AGENT_MODE", "false").lower() in {"1", "true", "yes", "on"}
        ALLOW_AGENT_MODE = os.getenv("ALLOW_AGENT_MODE", "true").lower() in {"1", "true", "yes", "on"}
        MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
        MAX_AGENT_EVIDENCE = int(os.getenv("MAX_AGENT_EVIDENCE", "20"))
        AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "nemotron-3-nano")
        RETRIEVER_MAX_WORKERS = int(os.getenv("RETRIEVER_MAX_WORKERS", "4"))

    class DialogueIntel:
        PERSONA_ENABLED = os.getenv("DOCWAIN_PERSONA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        SENTIMENT_ENABLED = os.getenv("DOCWAIN_SENTIMENT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        SMALLTALK_MODEL = os.getenv("DOCWAIN_SMALLTALK_MODEL", "")
        INTENT_THRESHOLD = float(os.getenv("DOCWAIN_INTENT_THRESHOLD", "0.65"))

    class Qdrant:
        URL = os.getenv("QDRANT_URL", 'https://89f776c3-76fb-493f-8509-c583d9579329.europe-west3-0.gcp.cloud.qdrant.io')
        API = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-cJ2HVTYcH3u5KNuZuxZRNJhhTFfZwqkoVacNCKBYkY")

    class Neo4j:
        URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        USER = os.getenv("NEO4J_USER", "neo4j")
        PASSWORD = os.getenv("NEO4J_PASSWORD", "dhs@welcome01")
        DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    class KnowledgeGraph:
        QDRANT_COLLECTION = os.getenv("KG_QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "default"))

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
        HF_HUB_READ_TIMEOUT = int(os.getenv("HF_HUB_READ_TIMEOUT", "30"))
        HF_HUB_CONNECT_TIMEOUT = int(os.getenv("HF_HUB_CONNECT_TIMEOUT", "10"))
        HF_HUB_MAX_RETRIES = int(os.getenv("HF_HUB_MAX_RETRIES", "3"))
        HF_DISABLE_TELEMETRY = os.getenv("HF_HUB_DISABLE_TELEMETRY", "false").lower() in {"1", "true", "yes", "on"}
        TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "false").lower() in {"1", "true", "yes", "on"}
        DISABLE_HF = os.getenv("DISABLE_HF", "false").lower() in {"1", "true", "yes", "on"}

    class Azure:
        AZURE_SUBSCRIPTION_ID = "249bb11f-9b6e-4c0e-a844-500d627b80b3"
        AZURE_RESOURCE_GROUP = "rg-docwain-dev"
        AZURE_PROJECT_NAME = "dhs-ai-competency"
        AZURE_AI_ENDPOINT = "https://dhs-ai-competency-resource.services.ai.azure.com/api/projects/dhs-ai-competency"
        AZURE_AI_KEY = "CpQaDiSSIBQBXcIysi5DpdoalYW8ghtDLwU8nGOsagVlkulbKvCsJQQJ99CAAC77bzfXJ3w3AAAAACOGG3Yu"

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
        CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=docwainuploads;AccountKey=+mxYrAnMSQeGjclw7ATpu7/Q6sT/I7twHka6JD4toKGFRlGW0HbX4OcyoproS6TeQ0q6CLSK1Dk6+AStJp0qYA==;EndpointSuffix=core.windows.net"
        AZURE_BLOB_KEY = "+mxYrAnMSQeGjclw7ATpu7/Q6sT/I7twHka6JD4toKGFRlGW0HbX4OcyoproS6TeQ0q6CLSK1Dk6+AStJp0qYA=="
        AZURE_BLOB_ACCOUNT_NAME = "docwainuploads"
        CONTAINER_NAME = "chat-history"
        DOCUMENT_CONTAINER_NAME = "document-content"

        @classmethod
        def validate(cls) -> None:
            if not cls.CONNECTION_STRING:
                raise ValueError(
                    "AzureBlob.CONNECTION_STRING is missing or empty. Set AZURE_STORAGE_CONNECTION_STRING."
                )


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
        DEFAULT_SUBSCRIPTION = os.getenv("TEAMS_DEFAULT_SUBSCRIPTION", "15e0c724-4de0-492e-9861-9e637b3f9076")
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
        # Support common Azure Bot env var spellings; defaults intentionally blank to force explicit configuration
        BOT_APP_ID = os.getenv("MICROSOFT_APP_ID")
        BOT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD") or os.getenv("MICROSOFT_APP_PWD")
        WEB_APP_URL = os.getenv("DOCWAIN_WEB_URL", os.getenv("TEAMS_WEB_APP_URL", "https://www.docwain.ai"))
        DIAG_MODE = os.getenv("TEAMS_DIAG_MODE", "").lower() in {"1", "true", "yes", "on"}

    class Retrieval:
        CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "800"))
        CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "200"))
        MIN_CHUNK_SIZE = int(os.getenv("RETRIEVAL_MIN_CHUNK_SIZE", "150"))
        MIN_CHUNK_CHARS = int(os.getenv("RETRIEVAL_MIN_CHUNK_CHARS", "40"))
        MIN_CHUNK_QUALITY = float(os.getenv("RETRIEVAL_MIN_CHUNK_QUALITY", "0.2"))
        MAX_SYMBOL_RATIO = float(os.getenv("RETRIEVAL_MAX_SYMBOL_RATIO", "0.6"))
        CHUNK_COVERAGE_THRESHOLD = float(os.getenv("CHUNK_COVERAGE_THRESHOLD", "0.98"))
        TOPK_DENSE = int(os.getenv("TOPK_DENSE", "50"))
        TOPK_RERANK = int(os.getenv("TOPK_RERANK", "20"))
        FINAL_CONTEXT_CHUNKS = int(os.getenv("FINAL_CONTEXT_CHUNKS", "8"))
        HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.75"))
        MIN_CONFIDENCE_TO_ANSWER = float(os.getenv("MIN_CONFIDENCE_TO_ANSWER", "0.62"))
        DEDUP_THRESHOLD = float(os.getenv("RETRIEVAL_DEDUP_THRESHOLD", "0.92"))
        MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))
        MAX_CONTEXT_CHUNKS = int(os.getenv("RETRIEVAL_MAX_CONTEXT_CHUNKS", "12"))
        SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.10"))
        USE_SPARSE_VECTORS = os.getenv("RETRIEVAL_USE_SPARSE_VECTORS", "true").lower() == "true"
        USE_ADJACENT_EXPANSION = os.getenv("RETRIEVAL_USE_ADJACENT_EXPANSION", "true").lower() == "true"
        NEIGHBOR_WINDOW = int(os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", "2"))
        NEIGHBOR_MAX_NEW = int(os.getenv("RETRIEVAL_NEIGHBOR_MAX_NEW", "10"))
        BROAD_RECALL_MULTIPLIER = float(os.getenv("RETRIEVAL_BROAD_RECALL_MULTIPLIER", "1.5"))
        BROAD_RECALL_THRESHOLD = float(os.getenv("RETRIEVAL_BROAD_RECALL_THRESHOLD", "0.02"))
        DIVERSITY_THRESHOLD = float(os.getenv("RETRIEVAL_DIVERSITY_THRESHOLD", "0.6"))
        MIN_OCR_CONFIDENCE = float(os.getenv("RETRIEVAL_MIN_OCR_CONFIDENCE", "60"))
        CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.55"))
        HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_HIGH_CONFIDENCE_THRESHOLD", "0.75"))
        QUERY_INTELLIGENCE_USE_LLM = os.getenv("RETRIEVAL_QUERY_INTELLIGENCE_LLM", "true").lower() == "true"
        REASONING_LAYER_ENABLED = os.getenv("RETRIEVAL_REASONING_LAYER_ENABLED", "true").lower() == "true"
        MIN_SUPPORT_SCORE = float(os.getenv("RETRIEVAL_MIN_SUPPORT_SCORE", "0.15"))
        MIN_CITATION_COVERAGE = float(os.getenv("RETRIEVAL_MIN_CITATION_COVERAGE", "0.75"))
        HYBRID_WEIGHTS = {
            "dense": float(os.getenv("HYBRID_WEIGHT_DENSE", "0.6")),
            "sparse": float(os.getenv("HYBRID_WEIGHT_SPARSE", "0.4")),
        }
        RERANKER_ENABLED = os.getenv("RETRIEVAL_RERANKER_ENABLED", "true").lower() == "true"
        METADATA_FALLBACK_LIMIT = int(os.getenv("RETRIEVAL_METADATA_FALLBACK_LIMIT", "200"))
        METADATA_FALLBACK_MIN_SCORE = float(os.getenv("RETRIEVAL_METADATA_FALLBACK_MIN_SCORE", "0.02"))
        MIN_QUERY_OVERLAP = float(os.getenv("RETRIEVAL_MIN_QUERY_OVERLAP", "0.06"))
        RELEVANCE_KEEP_TOP_K = int(os.getenv("RETRIEVAL_RELEVANCE_KEEP_TOP_K", "12"))
        MIN_COMPARISON_DOCS = int(os.getenv("RETRIEVAL_MIN_COMPARISON_DOCS", "2"))
        RETRIEVAL_GUARD_BUDGET_MS = int(os.getenv("RETRIEVAL_GUARD_BUDGET_MS", "40"))
        RETRIEVAL_QUALITY_THRESH_HIGH = float(os.getenv("RETRIEVAL_QUALITY_THRESH_HIGH", "0.75"))
        RETRIEVAL_QUALITY_THRESH_LOW = float(os.getenv("RETRIEVAL_QUALITY_THRESH_LOW", "0.45"))
        RETRIEVAL_EVIDENCE_MIN_COVERAGE = float(os.getenv("RETRIEVAL_EVIDENCE_MIN_COVERAGE", "0.6"))
        RETRIEVAL_EVIDENCE_STRICT_COVERAGE = float(os.getenv("RETRIEVAL_EVIDENCE_STRICT_COVERAGE", "0.75"))
        RETRIEVAL_RERANK_ON_LOW_QUALITY = os.getenv("RETRIEVAL_RERANK_ON_LOW_QUALITY", "true").lower() == "true"
        RETRIEVAL_RERANK_ON_HIGH_STAKES = os.getenv("RETRIEVAL_RERANK_ON_HIGH_STAKES", "true").lower() == "true"
        KG_PROBE_TIMEOUT_MS = int(os.getenv("KG_PROBE_TIMEOUT_MS", "80"))
        KG_PROBE_TTL_SECONDS = int(os.getenv("KG_PROBE_TTL_SECONDS", "1200"))
        KG_PROBE_LIMIT = int(os.getenv("KG_PROBE_LIMIT", "20"))
        KG_DOC_FILTER_LIMIT = int(os.getenv("KG_DOC_FILTER_LIMIT", "8"))
        KG_RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("KG_RETRIEVAL_CACHE_TTL_SECONDS", "240"))
        RETRIEVAL_FALLBACK_REWRITE = os.getenv("RETRIEVAL_FALLBACK_REWRITE", "true").lower() == "true"
        RETRIEVAL_FALLBACK_MAX_ATTEMPTS = int(os.getenv("RETRIEVAL_FALLBACK_MAX_ATTEMPTS", "1"))

    class Quality:
        ENABLED = os.getenv("QUALITY_EVAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        AUTO_REPAIR_ENABLED = os.getenv("QUALITY_AUTO_REPAIR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        EVAL_BUDGET_MS = int(os.getenv("QUALITY_EVAL_BUDGET_MS", "40"))
        REPAIR_BUDGET_MS = int(os.getenv("QUALITY_REPAIR_BUDGET_MS", "500"))
        MAX_REPAIR_ATTEMPTS = int(os.getenv("QUALITY_MAX_REPAIR_ATTEMPTS", "2"))
        LEX_SUPPORT_TH = float(os.getenv("QUALITY_LEX_SUPPORT_TH", "0.18"))
        SUPPORTED_RATIO_TH = float(os.getenv("QUALITY_SUPPORTED_RATIO_TH", "0.60"))
        CRITICAL_SUPPORTED_RATIO_TH = float(os.getenv("QUALITY_CRITICAL_SUPPORTED_RATIO_TH", "0.75"))
        OVERALL_SCORE_TH = float(os.getenv("QUALITY_OVERALL_SCORE_TH", "0.72"))
        HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("QUALITY_HIGH_CONFIDENCE_THRESHOLD", "0.75"))

    class Chat:
        MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "6"))
        SUMMARY_TURNS = int(os.getenv("CHAT_SUMMARY_TURNS", "6"))
        CONTEXT_TURNS = int(os.getenv("CHAT_CONTEXT_TURNS", "4"))

    class Companion:
        CLASSIFIER_TTL_SECONDS = int(os.getenv("COMPANION_CLASSIFIER_TTL_SECONDS", "600"))
        CLASSIFIER_USE_LLM = os.getenv("COMPANION_CLASSIFIER_USE_LLM", "false").lower() in {"1", "true", "yes", "on"}
        CLASSIFIER_MODEL = os.getenv("COMPANION_CLASSIFIER_MODEL", "")
        CLASSIFIER_TIMEOUT_SEC = float(os.getenv("COMPANION_CLASSIFIER_TIMEOUT_SEC", "0.4"))

    class LLM:
        TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        TOP_P = float(os.getenv("LLM_TOP_P", "0.85"))
        MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "2"))

    class ModelArbitration:
        ENABLED = os.getenv("MODEL_ARBITRATION_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        MAX_PARALLEL = int(os.getenv("MODEL_ARBITRATION_MAX_PARALLEL", "3"))
        MODELS_RAW = os.getenv("MODEL_ARBITRATION_MODELS", "").strip()
        MODELS = []
        if MODELS_RAW:
            try:
                MODELS = json.loads(MODELS_RAW)
            except Exception:
                MODELS = []
        EMBEDDING_MAP_RAW = os.getenv("MODEL_EMBEDDING_MAP", "").strip()
        EMBEDDING_MAP = {}
        if EMBEDDING_MAP_RAW:
            try:
                EMBEDDING_MAP = json.loads(EMBEDDING_MAP_RAW)
            except Exception:
                EMBEDDING_MAP = {}
        INDEX_SUFFIX_MAP_RAW = os.getenv("EMBEDDING_INDEX_SUFFIX_MAP", "").strip()
        INDEX_SUFFIX_MAP = {}
        if INDEX_SUFFIX_MAP_RAW:
            try:
                INDEX_SUFFIX_MAP = json.loads(INDEX_SUFFIX_MAP_RAW)
            except Exception:
                INDEX_SUFFIX_MAP = {}

    class Finetune:
        AUTO_ENABLED = os.getenv("FINETUNE_AUTO_ENABLED", "false").lower() == "true"
        AUTO_INTERVAL_HOURS = float(os.getenv("FINETUNE_AUTO_INTERVAL_HOURS", "6"))
        TEACHER_MODEL = os.getenv("FINETUNE_TEACHER_MODEL", "")
        MAX_CONCURRENCY = int(os.getenv("FINETUNE_MAX_CONCURRENCY", "4"))
        QA_RETRY_MAX = int(os.getenv("FINETUNE_QA_RETRY_MAX", "2"))
        MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "200"))
        MIN_MERGED_TOKENS = int(os.getenv("MIN_MERGED_TOKENS", "120"))
        MIN_PAIRS_PER_PROFILE = int(os.getenv("MIN_PAIRS_PER_PROFILE", "5"))
        MAX_PAIRS_PER_PROFILE = int(os.getenv("MAX_PAIRS_PER_PROFILE", "40"))
        MERGE_WINDOW = int(os.getenv("MERGE_WINDOW", "4"))
        DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.92"))
        GENERATION_MODEL = os.getenv("GENERATION_MODEL", "")
        GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
        BOILERPLATE_THRESHOLD = float(os.getenv("BOILERPLATE_THRESHOLD", "0.6"))
        ALLOW_NON_ENGLISH = os.getenv("ALLOW_NON_ENGLISH", "true").lower() == "true"
        AUTO_MIN_POINTS = int(os.getenv("FINETUNE_AUTO_MIN_POINTS", "40"))
        AUTO_MIN_RECORDS = int(os.getenv("FINETUNE_AUTO_MIN_RECORDS", "20"))
        AUTO_MAX_PROFILES_PER_RUN = int(os.getenv("FINETUNE_AUTO_MAX_PROFILES_PER_RUN", "10"))
        CLEANUP_ENABLED = os.getenv("FINETUNE_CLEANUP_ENABLED", "false").lower() == "true"
        CLEANUP_KEEP_LAST = int(os.getenv("FINETUNE_CLEANUP_KEEP_LAST", "3"))
        AGENTIC_ENABLED = os.getenv("FINETUNE_AGENTIC_ENABLED", "false").lower() == "true"
        ORCHESTRATOR_MODEL = os.getenv("FINETUNE_ORCHESTRATOR_MODEL", "nemotron-3-nano")
        AGENT_MAX_STEPS = int(os.getenv("FINETUNE_AGENT_MAX_STEPS", "12"))
        AGENT_TIMEOUT_S = int(os.getenv("FINETUNE_AGENT_TIMEOUT_S", "900"))
        AGENT_FALLBACK_TO_LEGACY = os.getenv("FINETUNE_AGENT_FALLBACK_TO_LEGACY", "true").lower() == "true"
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        OLLAMA_API = os.getenv("OLLAMA_API", "455d65a864a84e3bba92c0faea74f027.t3cJBT6-bjULKiySSIZQx4Dg")
