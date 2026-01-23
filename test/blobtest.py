from src.storage.azure_blob_client import get_chat_container_client

# Get the container client
container_client = get_chat_container_client()
container_name = container_client.container_name

# Check if the container exists
if container_client.exists():
    print(f"Connected to container: {container_name}")
else:
    print(f"Container '{container_name}' does not exist.")
