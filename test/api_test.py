import requests
from qdrant_client import QdrantClient

def qudrantTest():
    qdrant_client = QdrantClient(
        url="https://0a25c9cf-4685-49c7-9382-4c3510754343.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ki8Uux4HtKDYj6ebqH9nzu3qg7QVmXTvxT6UkLNwTc4",
    )

    print(qdrant_client.get_collection('documents'))

def endpointTest():
    headers = {
        'Content-Type': 'application/json',
    }

    json_data = {
        'query': 'What is the document about?',
        'mongo_client': 'test',
        'mongo_collection': 'documents',
        'profile_id': '67ac62ddfaa3aee44d38f4a5',
        'model_name': 'OpenAI'
    }

    response = requests.post('http://127.0.0.1:8000/ask', headers=headers, json=json_data)
    print(response.content)
