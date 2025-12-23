from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# _QDRANT_CLIENT = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
from src.api.config import Config

client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)


def delete_all_collections():
    try:
        # Get all collections
        response = client.get_collections()
        collections = [c.name for c in response.collections]

        if not collections:
            print("No collections found in Qdrant.")
            return

        print("Found collections:")
        for name in collections:
            print(f" - {name}")

        # Delete each collection
        for name in collections:
            try:
                client.delete_collection(name)
                print(f"🗑️ Deleted collection: {name}")
            except UnexpectedResponse as e:
                print(f"❌ Qdrant error while deleting '{name}': {e}")
            except Exception as e:
                print(f"❌ Unexpected error while deleting '{name}': {e}")

        print("\n✅ All collections processed.")

    except Exception as e:
        print(f"Failed to list collections: {e}")


if __name__ == "__main__":
    delete_all_collections()
