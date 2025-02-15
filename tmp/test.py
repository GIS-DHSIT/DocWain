from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://0a25c9cf-4685-49c7-9382-4c3510754343.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ki8Uux4HtKDYj6ebqH9nzu3qg7QVmXTvxT6UkLNwTc4",
)

print(qdrant_client.get_collection('documents'))
# print(qdrant_client.retrieve('documents'))