from azure.storage.blob import BlobServiceClient
from api.config import Config

connection_string = Config.AzureBlob.CONNECTION_STRING
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "chat-history"

# Get the container client
container_client = blob_service_client.get_container_client(container_name)

# Check if the container exists
if container_client.exists():
    print(f"Connected to container: {container_name}")
else:
    print(f"Container '{container_name}' does not exist.")