import os
from pathlib import Path


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME /"images"
        DUCKDB_DIR = DATABASE_DIR / "duck-db"


    class Qdrant:
        URL = "https://0a25c9cf-4685-49c7-9382-4c3510754343.europe-west3-0.gcp.cloud.qdrant.io:6333"
        API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ki8Uux4HtKDYj6ebqH9nzu3qg7QVmXTvxT6UkLNwTc4"

    class Model:
        SENTENCE_TRANSFORMERS = 'all-roberta-large-v1'
        AZURE_OPENAI_ENDPOINT = "https://nhspoc.openai.azure.com/"
        AZURE_OPENAI_API_KEY = "1r0Tggwh4wemD9VU7CDP2YQgeri8Z8tYnvVADu07EtQ8W0GAnvVtJQQJ99AKAC77bzfXJ3w3AAABACOGUkAC"
        AZURE_DEPLOYMENT_NAME = "gpt-35-turbo"
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "gemma2:9b"
        REMOTE_LLM = "llama-3.3-70b-versatile"
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000
        USE_LOCAL = False


    class MongoDB:
        URI = 'mongodb+srv://admin:nj4pJfO3e1FcGXDo@cluster0.47nz8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        DB = 'docwain'
        CONNECTOR = 'connectors'
        DOCUMENTS = 'documents'
        PROFILES = 'profiles'

    class Encryption:
        ENCRYPTION_KEY = 'J9cuHrESAz'
    class AWS:
        profile = 'DHS'
        accessKey = 'AKIA2UC3EV7PWOLLH4GI'
        secretKey = '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'
        region = 'eu-west-2'