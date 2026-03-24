import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REDIS_CONNECTION_STRING = os.getenv(
    "REDIS_CONNECTION_STRING",
    "",
)


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
        AGENT_AUTO_TOOLS = os.getenv("AGENT_AUTO_TOOLS", "true").lower() in {"1", "true", "yes", "on"}
        AGENT_MAX_AUTO_TOOLS = int(os.getenv("AGENT_MAX_AUTO_TOOLS", "3"))

    class DialogueIntel:
        PERSONA_ENABLED = os.getenv("DOCWAIN_PERSONA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        SENTIMENT_ENABLED = os.getenv("DOCWAIN_SENTIMENT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        SMALLTALK_MODEL = os.getenv("DOCWAIN_SMALLTALK_MODEL", "")
        INTENT_THRESHOLD = float(os.getenv("DOCWAIN_INTENT_THRESHOLD", "0.65"))

    class Features:
        DOMAIN_SPECIFIC_ENABLED = os.getenv("DOCWAIN_DOMAIN_SPECIFIC_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    class Qdrant:
        URL = os.getenv("QDRANT_URL", "")
        API = os.getenv("QDRANT_API_KEY", "")

    class Neo4j:
        URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        USER = os.getenv("NEO4J_USER", "neo4j")
        PASSWORD = os.getenv("NEO4J_PASSWORD", "")
        DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    class KnowledgeGraph:
        QDRANT_COLLECTION = os.getenv("KG_QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "default"))
        ENABLED = os.getenv("KG_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MAX_EXPANSION_ENTITIES = int(os.getenv("KG_MAX_EXPANSION_ENTITIES", "8"))
        MAX_GRAPH_SNIPPETS = int(os.getenv("KG_MAX_GRAPH_SNIPPETS", "10"))
        GRAPH_SCORE_ALPHA = float(os.getenv("KG_GRAPH_SCORE_ALPHA", "0.7"))
        MAX_GRAPH_RESULTS = int(os.getenv("KG_MAX_GRAPH_RESULTS", "200"))

    class Intelligence:
        ENABLED = os.getenv("DWX_INTEL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        SESSION_TTL_SECONDS = int(os.getenv("DWX_SESSION_TTL_SECONDS", "604800"))
        CATALOG_TTL_SECONDS = int(os.getenv("DWX_CATALOG_TTL_SECONDS", "2592000"))
        SUMMARY_TTL_SECONDS = int(os.getenv("DWX_SUMMARY_TTL_SECONDS", "2592000"))
        ENTITIES_TTL_SECONDS = int(os.getenv("DWX_ENTITIES_TTL_SECONDS", "2592000"))
        ROUTE_HISTORY_MAX = int(os.getenv("DWX_ROUTE_HISTORY_MAX", "20"))
        ENTITY_HISTORY_MAX = int(os.getenv("DWX_ENTITY_HISTORY_MAX", "50"))
        SECTION_SUMMARY_VECTORS_ENABLED = os.getenv("DWX_SECTION_SUMMARY_VECTORS", "true").lower() in {"1", "true", "yes", "on"}
        SECTION_SUMMARY_MAX_CHARS = int(os.getenv("DWX_SECTION_SUMMARY_MAX_CHARS", "700"))
        SECTION_SUMMARY_TOPK = int(os.getenv("DWX_SECTION_SUMMARY_TOPK", "6"))
        SECTION_RETRIEVAL_ENABLED = os.getenv("DWX_SECTION_RETRIEVAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        REASONING_ENGINE_ENABLED = os.getenv("DWX_REASONING_ENGINE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        REASONING_FAST_PATH_ENABLED = os.getenv("DWX_REASONING_FAST_PATH", "true").lower() in {"1", "true", "yes", "on"}
        VERIFY_CONFIDENCE_THRESHOLD = float(os.getenv("DWX_VERIFY_CONFIDENCE_THRESHOLD", "0.8"))

    class Gemini:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        # GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta

    class Model:
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
        OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
        SENTENCE_TRANSFORMERS = EMBEDDING_MODEL
        SENTENCE_TRANSFORMERS_FALLBACK = SENTENCE_TRANSFORMERS
        SENTENCE_TRANSFORMERS_CANDIDATES = [
            EMBEDDING_MODEL,
            os.getenv("EMBEDDING_FALLBACK_MODEL", "BAAI/bge-base-en-v1.5"),
        ]
        RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        OCR_ENGINE = os.getenv("OCR_ENGINE", "pytesseract")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "")
        AZURE_VERSION = os.getenv("AZURE_OPENAI_VERSION", "")
        # ✅ Gemini 2.5 Flash configs
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        # GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
        GEMINI_MODEL_NAME = "gemini-2.5-flash"
        HF_HUB_READ_TIMEOUT = int(os.getenv("HF_HUB_READ_TIMEOUT", "30"))
        HF_HUB_CONNECT_TIMEOUT = int(os.getenv("HF_HUB_CONNECT_TIMEOUT", "10"))
        HF_HUB_MAX_RETRIES = int(os.getenv("HF_HUB_MAX_RETRIES", "3"))
        HF_DISABLE_TELEMETRY = os.getenv("HF_HUB_DISABLE_TELEMETRY", "false").lower() in {"1", "true", "yes", "on"}
        TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "false").lower() in {"1", "true", "yes", "on"}
        DISABLE_HF = os.getenv("DISABLE_HF", "false").lower() in {"1", "true", "yes", "on"}
        OFFLINE_ONLY = os.getenv("DOCWAIN_OFFLINE_ONLY", "true").lower() in {"1", "true", "yes", "on"}

    class VisionOCR:
        ENABLED = os.getenv("VISION_OCR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MODEL = os.getenv("VISION_OCR_MODEL", "glm-ocr:latest")
        MIN_IMAGE_WIDTH = int(os.getenv("VISION_OCR_MIN_WIDTH", "100"))
        MIN_IMAGE_HEIGHT = int(os.getenv("VISION_OCR_MIN_HEIGHT", "100"))
        FALLBACK_TO_TRADITIONAL = os.getenv("VISION_OCR_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}
        OCR_CONTENT_IMAGES = os.getenv("VISION_OCR_CONTENT_IMAGES", "true").lower() in {"1", "true", "yes", "on"}

    class Azure:
        AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
        AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-docwain-dev")
        AZURE_PROJECT_NAME = os.getenv("AZURE_PROJECT_NAME", "dhs-ai-competency")
        AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "")
        AZURE_AI_KEY = os.getenv("AZURE_AI_KEY", "")

    class AzureGpt4o:
        AZUREGPT4O_ENDPOINT = os.getenv("AZUREGPT4O_ENDPOINT", "")
        AZUREGPT4O_DEPLOYMENT = os.getenv("AZUREGPT4O_DEPLOYMENT", "dw-dev1-gpt-4o")
        AZUREGPT4O_API_KEY = os.getenv("AZUREGPT4O_API_KEY", "")
        AZUREGPT4O_Version = os.getenv("AZUREGPT4O_VERSION", "2024-05-01-preview")

    class MongoDB:
        DEFAULT_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        URI = DEFAULT_URI
        FALLBACK_URI = os.getenv("MONGODB_FALLBACK_URI", "mongodb://localhost:27017")
        DB = os.getenv("MONGODB_DB", 'test')
        CONNECTOR = os.getenv("MONGODB_CONNECTORS", 'connectors')
        DOCUMENTS = os.getenv("MONGODB_DOCUMENTS", 'documents')
        PROFILES = os.getenv("MONGODB_PROFILES", 'profiles')
        # Add this new line for subscriptions collection
        # modified by maha/maria
        SUBSCRIPTIONS = os.getenv("MONGODB_SUBSCRIPTIONS_COLLECTION", "subscriptions")

    class Encryption:
        ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

    class AWS:
        PROFILE = os.getenv("AWS_PROFILE", "DHS")
        ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
        SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
        REGION = os.getenv("AWS_REGION", "eu-west-2")
        BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "docwain-chat-history")

    class AzureBlob:
        CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        AZURE_BLOB_KEY = os.getenv("AZURE_BLOB_KEY", "")
        AZURE_BLOB_ACCOUNT_NAME = os.getenv("AZURE_BLOB_ACCOUNT_NAME", "docwainuploads")
        CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "chat-history")
        DOCUMENT_CONTAINER_NAME = os.getenv("AZURE_BLOB_DOCUMENT_CONTAINER", "document-content")

        @classmethod
        def validate(cls) -> None:
            if not cls.CONNECTION_STRING:
                raise ValueError(
                    "AzureBlob.CONNECTION_STRING is missing or empty. Set AZURE_STORAGE_CONNECTION_STRING."
                )


    class Redis:
        CONNECTION_STRING = DEFAULT_REDIS_CONNECTION_STRING
        HOST = os.getenv("REDIS_HOST", "localhost")
        PORT = int(os.getenv("REDIS_PORT", "6380"))
        PASSWORD = os.getenv("REDIS_PASSWORD", "")
        DB = int(os.getenv("REDIS_DB", "0"))
        SSL = os.getenv("REDIS_SSL", "true").lower() in {"1", "true", "yes", "on"}
        ABORT_CONNECT = False
        CLEAR_UNSAFE_ON_STARTUP = os.getenv("REDIS_CLEAR_UNSAFE_ON_STARTUP", "true").lower() in {"1", "true", "yes", "on"}
        UNSAFE_KEY_PATTERNS = os.getenv("REDIS_UNSAFE_KEY_PATTERNS", "dw:plan:*,rag:*").strip()
        CLEAR_SCAN_COUNT = int(os.getenv("REDIS_CLEAR_SCAN_COUNT", "200"))
        CLEAR_MAX_KEYS = int(os.getenv("REDIS_CLEAR_MAX_KEYS", "5000"))

    class Teams:
        SHARED_SECRET = os.getenv("TEAMS_SHARED_SECRET", "")
        SIGNATURE_ENABLED = os.getenv("TEAMS_SIGNATURE_ENABLED", "false").lower() == "true"
        DEFAULT_PROFILE = os.getenv("TEAMS_DEFAULT_PROFILE", "default")
        DEFAULT_SUBSCRIPTION = os.getenv("TEAMS_DEFAULT_SUBSCRIPTION") or "15e0c724-4de0-492e-9861-9e637b3f9076"
        DEFAULT_MODEL = os.getenv("TEAMS_DEFAULT_MODEL", "DHS/DocWain")
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
        BOT_APP_TENANT_ID = os.getenv("MICROSOFT_APP_TENANT_ID") or os.getenv("MSA_APP_TENANT_ID")
        BOT_APP_TYPE = os.getenv("MICROSOFT_APP_TYPE", "SingleTenant")  # SingleTenant | MultiTenant | UserAssignedMSI
        WEB_APP_URL = os.getenv("DOCWAIN_WEB_URL", os.getenv("TEAMS_WEB_APP_URL", "https://www.docwain.ai"))
        DIAG_MODE = os.getenv("TEAMS_DIAG_MODE", "").lower() in {"1", "true", "yes", "on"}

    class Tools:
        LLM_ENABLED = os.getenv("TOOLS_LLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        LLM_TIMEOUT = float(os.getenv("TOOLS_LLM_TIMEOUT", "30.0"))
        LLM_MAX_INPUT_CHARS = int(os.getenv("TOOLS_LLM_MAX_INPUT_CHARS", "3500"))

    class Retrieval:
        CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "800"))
        CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "200"))
        MIN_CHUNK_SIZE = int(os.getenv("RETRIEVAL_MIN_CHUNK_SIZE", "150"))
        MIN_CHUNK_CHARS = int(os.getenv("RETRIEVAL_MIN_CHUNK_CHARS", "40"))

    class Reranker:
        DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
        TIMEOUT_S = float(os.getenv("RERANKER_TIMEOUT_S", "6.0"))

    class RagV3:
        DEBUG_LOGS = os.getenv("DOCWAIN_RAG_V3_DEBUG_LOGS", "false").lower() in {"1", "true", "yes", "on"}
        DEBUG_SCHEMA = os.getenv("DOCWAIN_RAG_V3_DEBUG_SCHEMA", "false").lower() in {"1", "true", "yes", "on"}
        MIN_CHARS = int(os.getenv("RETRIEVAL_MIN_CHARS", "80"))
        MIN_TOKENS = int(os.getenv("RETRIEVAL_MIN_TOKENS", "15"))
        MIN_REQUIRED_CHUNKS = int(
            os.getenv("RETRIEVAL_MIN_REQUIRED_CHUNKS", os.getenv("RETRIEVAL_MIN_VALID_CHUNKS_PER_DOC", "3"))
        )
        MIN_VALID_CHUNKS_PER_DOC = MIN_REQUIRED_CHUNKS
        FALLBACK_CHUNK_SIZE = int(os.getenv("RETRIEVAL_FALLBACK_CHUNK_SIZE", "600"))
        FALLBACK_OVERLAP = int(os.getenv("RETRIEVAL_FALLBACK_CHUNK_OVERLAP", "80"))
        MIN_CHUNK_QUALITY = float(os.getenv("RETRIEVAL_MIN_CHUNK_QUALITY", "0.2"))
        MAX_SYMBOL_RATIO = float(os.getenv("RETRIEVAL_MAX_SYMBOL_RATIO", "0.6"))
        CHUNK_COVERAGE_THRESHOLD = float(os.getenv("CHUNK_COVERAGE_THRESHOLD", "0.98"))
        TOPK_DENSE = int(os.getenv("TOPK_DENSE", "50"))
        TOPK_RERANK = int(os.getenv("TOPK_RERANK", "20"))
        FINAL_CONTEXT_CHUNKS = int(os.getenv("FINAL_CONTEXT_CHUNKS", "12"))
        HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.75"))
        MIN_CONFIDENCE_TO_ANSWER = float(os.getenv("MIN_CONFIDENCE_TO_ANSWER", "0.62"))
        DEDUP_THRESHOLD = float(os.getenv("RETRIEVAL_DEDUP_THRESHOLD", "0.92"))
        MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
        MAX_CONTEXT_CHUNKS = int(os.getenv("RETRIEVAL_MAX_CONTEXT_CHUNKS", "16"))
        MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
        SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.10"))
        USE_SPARSE_VECTORS = os.getenv("RETRIEVAL_USE_SPARSE_VECTORS", "true").lower() == "true"
        USE_ADJACENT_EXPANSION = os.getenv("RETRIEVAL_USE_ADJACENT_EXPANSION", "true").lower() == "true"
        NEIGHBOR_WINDOW = int(os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", "2"))
        RETRIEVAL_PLANNER_ENABLED = os.getenv("RETRIEVAL_PLANNER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        RETRIEVAL_PLANNER_MAX_RETRIES = int(os.getenv("RETRIEVAL_PLANNER_MAX_RETRIES", "2"))
        RETRIEVAL_PLANNER_BACKOFF = float(os.getenv("RETRIEVAL_PLANNER_BACKOFF", "0.4"))
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
        EVIDENCE_SYNTHESIZER_ENABLED = os.getenv("EVIDENCE_SYNTHESIZER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        EVIDENCE_SYNTHESIZER_MAX_RETRIES = int(os.getenv("EVIDENCE_SYNTHESIZER_MAX_RETRIES", "1"))
        EVIDENCE_SYNTHESIZER_BACKOFF = float(os.getenv("EVIDENCE_SYNTHESIZER_BACKOFF", "0.3"))
        EVIDENCE_SYNTHESIZER_MAX_EXCERPTS_PER_FILE = int(os.getenv("EVIDENCE_SYNTHESIZER_MAX_EXCERPTS_PER_FILE", "6"))
        EVIDENCE_SYNTHESIZER_EXCERPT_CHARS = int(os.getenv("EVIDENCE_SYNTHESIZER_EXCERPT_CHARS", "800"))
        METADATA_FALLBACK_MIN_SCORE = float(os.getenv("RETRIEVAL_METADATA_FALLBACK_MIN_SCORE", "0.02"))
        MIN_QUERY_OVERLAP = float(os.getenv("RETRIEVAL_MIN_QUERY_OVERLAP", "0.06"))
        RELEVANCE_KEEP_TOP_K = int(os.getenv("RETRIEVAL_RELEVANCE_KEEP_TOP_K", "12"))
        USE_UNIFIED_RETRIEVER = os.getenv("USE_UNIFIED_RETRIEVER", "false").lower() in {"1", "true", "yes", "on"}
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
        GROUNDING_GATE_ENABLED = os.getenv("GROUNDING_GATE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        GROUNDING_GATE_CRITICAL_TH = float(os.getenv("GROUNDING_GATE_CRITICAL_TH", "0.30"))

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
        MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
        DISABLE_EXTERNAL = os.getenv("LLM_DISABLE_EXTERNAL", "true").lower() in {"1", "true", "yes", "on"}

    class VLLM:
        """vLLM serving config — Qwen3-14B-AWQ via OpenAI-compatible API."""
        ENABLED = os.getenv("VLLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8001/v1/chat/completions")
        MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
        API_KEY = os.getenv("VLLM_API_KEY", "")
        TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "30"))

    class DocumentProfiler:
        """Ingestion-time document profiling via LLM."""
        ENABLED = os.getenv("DOC_PROFILER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    class TaskRouting:
        ENABLED = os.getenv("TASK_ROUTING_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        QUERY_REWRITE_MODEL = os.getenv("TASK_ROUTE_QUERY_REWRITE", "")
        INTENT_PARSE_MODEL = os.getenv("TASK_ROUTE_INTENT_PARSE", "")
        RESPONSE_GENERATION_MODEL = os.getenv("TASK_ROUTE_RESPONSE_GENERATION", "")
        STRUCTURED_EXTRACTION_MODEL = os.getenv("TASK_ROUTE_STRUCTURED_EXTRACTION", "")
        TOOL_EXECUTION_MODEL = os.getenv("TASK_ROUTE_TOOL_EXECUTION", "")
        ANSWER_JUDGING_MODEL = os.getenv("TASK_ROUTE_ANSWER_JUDGING", "")
        GROUNDING_VERIFY_MODEL = os.getenv("TASK_ROUTE_GROUNDING_VERIFY", "")
        CONTENT_GENERATION_MODEL = os.getenv("TASK_ROUTE_CONTENT_GENERATION", "")
        QUERY_CLASSIFICATION_MODEL = os.getenv("TASK_ROUTE_QUERY_CLASSIFICATION", "")
        CONVERSATION_SUMMARY_MODEL = os.getenv("TASK_ROUTE_CONVERSATION_SUMMARY", "")
        DOCUMENT_UNDERSTANDING_MODEL = os.getenv("TASK_ROUTE_DOCUMENT_UNDERSTANDING", "")
        GENERAL_MODEL = os.getenv("TASK_ROUTE_GENERAL", "")
        FALLBACK_MODEL = os.getenv("TASK_ROUTE_FALLBACK", "DocWain-Agent:latest")

    class MultiAgent:
        ENABLED = os.getenv("MULTI_AGENT_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        CLASSIFIER_MODEL = os.getenv("MULTI_AGENT_CLASSIFIER", "DocWain-Agent:latest")
        EXTRACTOR_MODEL = os.getenv("MULTI_AGENT_EXTRACTOR", "DocWain-Agent:latest")
        GENERATOR_MODEL = os.getenv("MULTI_AGENT_GENERATOR", "DocWain-Agent:latest")
        VERIFIER_MODEL = os.getenv("MULTI_AGENT_VERIFIER", "DocWain-Agent:latest")
        DEFAULT_MODEL = os.getenv("MULTI_AGENT_DEFAULT", "DocWain-Agent:latest")
        VERIFIER_ENABLED = os.getenv("MULTI_AGENT_VERIFIER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        CLASSIFIER_TIMEOUT = float(os.getenv("MULTI_AGENT_CLASSIFIER_TIMEOUT", "15.0"))
        VERIFIER_TIMEOUT = float(os.getenv("MULTI_AGENT_VERIFIER_TIMEOUT", "30.0"))
        CLASSIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("MULTI_AGENT_CLASSIFIER_CONFIDENCE", "0.7"))

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

    class RAGV2:
        ENABLED = os.getenv("DOCWAIN_RAG_V2_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    class RAGV3:
        ENABLED = os.getenv("RAG_V3_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

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
        OLLAMA_API = os.getenv("OLLAMA_API", "")

    class Evolve:
        ENABLED = os.getenv("EVOLVE_ENABLED", "false").lower() == "true"
        SIGNALS_DIR = os.getenv("EVOLVE_SIGNALS_DIR", "signals")
        ARTIFACT_DIR = os.getenv("EVOLVE_ARTIFACT_DIR", "finetune_artifacts")
        REGISTRY_PATH = os.getenv("EVOLVE_REGISTRY_PATH", "registry.yaml")
        CONFIG_PATH = os.getenv("EVOLVE_CONFIG_PATH", "src/finetune/evolve_config.yaml")

    class V2:
        BASE_MODEL = os.getenv("V2_BASE_MODEL", "unsloth/Qwen3-14B-bnb-4bit")
        VISION_ENCODER = os.getenv("V2_VISION_ENCODER", "google/siglip-so400m-patch14-384")
        ARTIFACT_DIR = os.getenv("V2_ARTIFACT_DIR", "finetune_artifacts/v2")

    class FollowUp:
        ENABLED = os.getenv("FOLLOWUP_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MAX_SUGGESTIONS = int(os.getenv("FOLLOWUP_MAX_SUGGESTIONS", "3"))
        LLM_TIMEOUT = float(os.getenv("FOLLOWUP_LLM_TIMEOUT", "3.0"))

    class QueryPlanner:
        ENABLED = os.getenv("QUERY_PLANNER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MAX_STEPS = int(os.getenv("QUERY_PLANNER_MAX_STEPS", "3"))
        LLM_TIMEOUT = float(os.getenv("QUERY_PLANNER_LLM_TIMEOUT", "5.0"))

    class HallucinationCorrector:
        ENABLED = os.getenv("HALLUCINATION_CORRECTOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        SCORE_THRESHOLD = float(os.getenv("HALLUCINATION_SCORE_THRESHOLD", "0.5"))
        MAX_CORRECTIONS = int(os.getenv("HALLUCINATION_MAX_CORRECTIONS", "3"))

    class Confidence:
        ENABLED = os.getenv("CONFIDENCE_SCORING_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    class WebSearch:
        ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        ENGINE = os.getenv("WEB_SEARCH_ENGINE", "duckduckgo")
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
        MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
        TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "30.0"))
        MAX_URL_FETCH_CHARS = int(os.getenv("WEB_SEARCH_MAX_URL_FETCH_CHARS", "6000"))
        FALLBACK_ON_NO_RESULTS = os.getenv("WEB_SEARCH_FALLBACK_ON_NO_RESULTS", "true").lower() in {"1", "true", "yes", "on"}

    class Synthesis:
        ENABLED = os.getenv("SYNTHESIS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        TIMEOUT = float(os.getenv("SYNTHESIS_TIMEOUT", "10.0"))
        MIN_DOCUMENTS = int(os.getenv("SYNTHESIS_MIN_DOCUMENTS", "2"))

    class Verification:
        ENABLED = os.getenv("VERIFICATION_ENABLED", "false").lower() in {"1", "true", "yes", "on"}

    class LLMCache:
        ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))

    class DomainKnowledge:
        ENABLED = os.getenv("DOMAIN_KNOWLEDGE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        WEB_ENRICHMENT = os.getenv("DOMAIN_KNOWLEDGE_WEB_ENRICHMENT", "false").lower() in {"1", "true", "yes", "on"}
        CACHE_TTL = int(os.getenv("DOMAIN_KNOWLEDGE_CACHE_TTL", "3600"))
        INJECT_INTO_PROMPTS = os.getenv("DOMAIN_KNOWLEDGE_INJECT_PROMPTS", "true").lower() in {"1", "true", "yes", "on"}

    class CloudLLM:
        ENABLED = os.getenv("DOCWAIN_CLOUD_LLM_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
        CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        COMPLEXITY_THRESHOLD_T2 = float(os.getenv("CLOUD_THRESHOLD_T2", "0.4"))
        COMPLEXITY_THRESHOLD_T3 = float(os.getenv("CLOUD_THRESHOLD_T3", "0.7"))
        CIRCUIT_BREAKER_FAILURES = int(os.getenv("CLOUD_CIRCUIT_BREAKER_FAILURES", "3"))
        CIRCUIT_BREAKER_COOLDOWN = int(os.getenv("CLOUD_CIRCUIT_BREAKER_COOLDOWN", "60"))

    class DocumentProcessing:
        MAX_CONCURRENT_DEEP_ANALYSIS = int(os.getenv("DOC_PROCESSING_MAX_CONCURRENT", "2"))
        KG_INGEST_ASYNC = os.getenv("KG_INGEST_ASYNC", "true").lower() in {"1", "true", "yes", "on"}

    class DeepAnalysis:
        ENABLED = os.getenv("DEEP_ANALYSIS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        BACKGROUND_ENABLED = os.getenv("DEEP_ANALYSIS_BACKGROUND_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MAX_ENTITIES = int(os.getenv("DEEP_ANALYSIS_MAX_ENTITIES", "100"))
        QUALITY_GRADING = os.getenv("DEEP_ANALYSIS_QUALITY_GRADING", "true").lower() in {"1", "true", "yes", "on"}

    class ProfileDomain:
        ENABLED = os.getenv("PROFILE_DOMAIN_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MAJORITY_THRESHOLD = float(os.getenv("PROFILE_DOMAIN_MAJORITY_THRESHOLD", "0.80"))
        MIN_SIGNAL_SCORE = float(os.getenv("PROFILE_DOMAIN_MIN_SIGNAL_SCORE", "0.25"))

    class ExtractionPipeline:
        ENABLED = os.getenv("EXTRACTION_PIPELINE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        DEFAULT_STORE = os.getenv("EXTRACTION_PIPELINE_DEFAULT_STORE", "all")
        MAX_FILE_SIZE_MB = int(os.getenv("EXTRACTION_PIPELINE_MAX_FILE_SIZE_MB", "100"))

    class DiagramExtraction:
        ENABLED = os.getenv("DIAGRAM_EXTRACTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        DETECTION_THRESHOLD = float(os.getenv("DIAGRAM_DETECTION_THRESHOLD", "0.5"))
        USE_THINKING = os.getenv("DIAGRAM_USE_THINKING", "true").lower() in {"1", "true", "yes", "on"}

    class ThinkingModel:
        """lfm2.5-thinking — fast reasoning sub-agent for MoE routing."""
        ENABLED = os.getenv("THINKING_MODEL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MODEL = os.getenv("THINKING_MODEL", "lfm2.5-thinking:latest")
        KEEP_ALIVE = os.getenv("THINKING_MODEL_KEEP_ALIVE", "24h")
        DEFAULT_TEMPERATURE = float(os.getenv("THINKING_MODEL_TEMPERATURE", "0.05"))
        MAX_PREDICT = int(os.getenv("THINKING_MODEL_MAX_PREDICT", "512"))
        USE_FOR_JUDGING = os.getenv("THINKING_MODEL_USE_FOR_JUDGING", "true").lower() in {"1", "true", "yes", "on"}
        USE_FOR_AGENT_STEPS = os.getenv("THINKING_MODEL_USE_FOR_AGENT_STEPS", "true").lower() in {"1", "true", "yes", "on"}
        USE_FOR_VERIFICATION = os.getenv("THINKING_MODEL_USE_FOR_VERIFICATION", "true").lower() in {"1", "true", "yes", "on"}

    class VisionAnalysis:
        """glm-ocr extended for rich image analysis (charts, tables, diagrams)."""
        ENABLED = os.getenv("VISION_ANALYSIS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MODEL = os.getenv("VISION_ANALYSIS_MODEL", "glm-ocr:latest")
        CHART_ANALYSIS = os.getenv("VISION_ANALYSIS_CHART", "true").lower() in {"1", "true", "yes", "on"}
        TABLE_ANALYSIS = os.getenv("VISION_ANALYSIS_TABLE", "true").lower() in {"1", "true", "yes", "on"}
        DIAGRAM_ANALYSIS = os.getenv("VISION_ANALYSIS_DIAGRAM", "true").lower() in {"1", "true", "yes", "on"}
        PHOTO_ANALYSIS = os.getenv("VISION_ANALYSIS_PHOTO", "true").lower() in {"1", "true", "yes", "on"}
        MAX_IMAGE_TOKENS = int(os.getenv("VISION_ANALYSIS_MAX_IMAGE_TOKENS", "4096"))

    class VisualIntelligence:
        """ML-based visual document understanding (second-pass enrichment)."""
        ENABLED = os.getenv("VISUAL_INTELLIGENCE_ENABLED", "true")
        GPU_DEVICE = os.getenv("VISUAL_INTELLIGENCE_GPU_DEVICE", "cuda:0")
        CPU_FALLBACK = os.getenv("VISUAL_INTELLIGENCE_CPU_FALLBACK", "true")
        MAX_CONCURRENT_PAGES = int(os.getenv("VISUAL_INTELLIGENCE_MAX_CONCURRENT_PAGES", "4"))
        TIER1_MODELS = os.getenv("VISUAL_INTELLIGENCE_TIER1_MODELS", "dit").split(",")
        TIER2_MODELS = os.getenv("VISUAL_INTELLIGENCE_TIER2_MODELS", "dit,table_transformer,trocr,layoutlmv3").split(",")
        DIT_MODEL = os.getenv("VISUAL_INTELLIGENCE_DIT_MODEL", "microsoft/dit-large-finetuned-publaynet")
        TABLE_DET_MODEL = os.getenv("VISUAL_INTELLIGENCE_TABLE_DET_MODEL", "microsoft/table-transformer-detection")
        TABLE_STR_MODEL = os.getenv("VISUAL_INTELLIGENCE_TABLE_STR_MODEL", "microsoft/table-transformer-structure-recognition")
        TROCR_PRINTED_MODEL = os.getenv("VISUAL_INTELLIGENCE_TROCR_PRINTED_MODEL", "microsoft/trocr-base-printed")
        TROCR_HANDWRITTEN_MODEL = os.getenv("VISUAL_INTELLIGENCE_TROCR_HANDWRITTEN_MODEL", "microsoft/trocr-base-handwritten")
        LAYOUTLMV3_MODEL = os.getenv("VISUAL_INTELLIGENCE_LAYOUTLMV3_MODEL", "microsoft/layoutlmv3-base")
        RENDER_DPI = int(os.getenv("VISUAL_INTELLIGENCE_RENDER_DPI", "300"))
        COMPLEXITY_OCR_HIGH = float(os.getenv("VISUAL_INTELLIGENCE_OCR_HIGH", "0.85"))
        COMPLEXITY_OCR_LOW = float(os.getenv("VISUAL_INTELLIGENCE_OCR_LOW", "0.70"))

    class Agents:
        """Per-agent feature flags for enhanced agent capabilities."""
        RESUMES_INTERNET_ENABLED = os.getenv("DOCWAIN_RESUMES_INTERNET_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MEDICAL_NICE_ENABLED = os.getenv("DOCWAIN_MEDICAL_NICE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        MEDICAL_NICE_MAX_LOOKUPS = int(os.getenv("DOCWAIN_MEDICAL_NICE_MAX_LOOKUPS", "3"))

    class Celery:
        BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
        EXTRACTION_CONCURRENCY = int(os.getenv("CELERY_EXTRACTION_CONCURRENCY", "2"))
        SCREENING_CONCURRENCY = int(os.getenv("CELERY_SCREENING_CONCURRENCY", "4"))
        KG_CONCURRENCY = int(os.getenv("CELERY_KG_CONCURRENCY", "4"))
        EMBEDDING_CONCURRENCY = int(os.getenv("CELERY_EMBEDDING_CONCURRENCY", "2"))
        BACKFILL_CONCURRENCY = int(os.getenv("CELERY_BACKFILL_CONCURRENCY", "4"))

    class CloudPlatform:
        """SharePoint + cloud platform integration settings."""
        SHAREPOINT_ENABLED = os.getenv("DOCWAIN_SHAREPOINT_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        SHAREPOINT_TENANT_ID = os.getenv("DOCWAIN_SHAREPOINT_TENANT_ID", "")
        SHAREPOINT_CLIENT_ID = os.getenv("DOCWAIN_SHAREPOINT_CLIENT_ID", "")
        SHAREPOINT_CLIENT_SECRET = os.getenv("DOCWAIN_SHAREPOINT_CLIENT_SECRET", "")
