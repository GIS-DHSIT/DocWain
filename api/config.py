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
        URL = "https://6fe03a10-2d12-4853-bd91-7f355d4fe4e5.uksouth-0.azure.cloud.qdrant.io:6333"
        API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.mcy7r_Ej4crS9bQ1wAKXdOwOVE4CbJYsDtmUNnRXark"

    class Model:
        SENTENCE_TRANSFORMERS = 'all-roberta-large-v1'
        AZURE_OPENAI_ENDPOINT = "https://nhspoc.openai.azure.com/"
        AZURE_OPENAI_API_KEY = "1r0Tggwh4wemD9VU7CDP2YQgeri8Z8tYnvVADu07EtQ8W0GAnvVtJQQJ99AKAC77bzfXJ3w3AAABACOGUkAC"
        AZURE_DEPLOYMENT_NAME = "gpt-35-turbo"
        AZURE_VERSION = "2023-07-01-preview"

    class MongoDB:
        URI = 'mongodb+srv://admin:nj4pJfO3e1FcGXDo@cluster0.47nz8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        DB = 'docwain'
        CONNECTOR = 'connectors'
        DOCUMENTS = 'documents'
        PROFILES = 'profiles'

    class Redis:
        HOST = 'redis-12404.c251.east-us-mz.azure.redns.redis-cloud.com'
        PORT = 12404
        ACCOUNT_KEY = 'A3dbedbev3g3e9rhbgctc4ylc9zqpblrjty1nvnyskfwmpscpzx'
        API_KEY = 'Skb4e4ll3dkjbnr6k7fd32ozuevafn59ed8icy4ykfojhu0m22'
        PASSWORD = 'bBxadyqsRx8qwtMIqsIzZc2sYeIA6Xvm'

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