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
        AZURE_VERSION = "2023-07-01-preview"

    class AzureGpt4o:
        AZUREGPT4O_ENDPOINT = "https://dw-openai-dev.openai.azure.com/"
        AZUREGPT4O_DEPLOYMENT = "dw-dev1-gpt-4o"
        AZUREGPT4O_API_KEY = '6JSK5oHMv76xL6IAtFwVfgCRykf24MWdvp6oRpxawBk9sGyqXuQYJQQJ99BCACmepeSXJ3w3AAABACOGjB0M'
        AZUREGPT4O_Version = "2024-05-01-preview"

    class MongoDB:
        URI = 'mongodb+srv://admin:nj4pJfO3e1FcGXDo@cluster0.47nz8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        DB = 'docwain'
        CONNECTOR = 'connectors'
        DOCUMENTS = 'documents'
        PROFILES = 'profiles'

    class Encryption:
        ENCRYPTION_KEY = 'J9cuHrESAz'

    class AWS:
        PROFILE = 'DHS'
        ACCESS_KEY = 'AKIA2UC3EV7PWOLLH4GI'
        SECRET_KEY = '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'
        REGION = 'eu-west-2'
        BUCKET_NAME = 'docwain-chat-history'