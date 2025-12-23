"""
Diagnostic script to test and debug retrieval system
Run this to verify your setup is working correctly
"""

import logging
import sys
from src.api.config import Config
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_qdrant_connection():
    """Test Qdrant connection"""
    try:
        logger.info("=" * 80)
        logger.info("Testing Qdrant Connection")
        logger.info("=" * 80)

        client = QdrantClient(
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
            timeout=120
        )

        # List collections
        collections = client.get_collections()
        logger.info(f"✅ Connected to Qdrant successfully")
        logger.info(f"Found {len(collections.collections)} collections:")
        for col in collections.collections:
            logger.info(f"  - {col.name}")

        return client
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        return None


def test_collection_details(client, collection_name):
    """Test specific collection"""
    try:
        logger.info("=" * 80)
        logger.info(f"Testing Collection: {collection_name}")
        logger.info("=" * 80)

        # Get collection info
        info = client.get_collection(collection_name)
        logger.info(f"✅ Collection exists")
        logger.info(f"Points count: {info.points_count}")

        # Get config
        config = info.config
        params = config.params

        # Check vectors
        if hasattr(params, 'vectors'):
            vectors = params.vectors
            if hasattr(vectors, '__dict__'):
                logger.info(f"Vector config: {vectors.__dict__}")
            else:
                logger.info(f"Vector config: {vectors}")

        # Sample a few points
        logger.info("\nSampling 3 points:")
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )

        if hasattr(scroll_result, 'points'):
            points = scroll_result.points
        elif isinstance(scroll_result, tuple):
            points = scroll_result[0]
        else:
            points = []

        for i, pt in enumerate(points, 1):
            payload = pt.payload or {}
            logger.info(f"\nPoint {i}:")
            logger.info(f"  ID: {pt.id}")
            logger.info(f"  Profile ID: {payload.get('profile_id')}")
            logger.info(f"  Document ID: {payload.get('document_id')}")
            logger.info(f"  Text preview: {payload.get('text', '')[:100]}...")

        return True
    except Exception as e:
        logger.error(f"❌ Collection test failed: {e}")
        return False


def test_model_loading():
    """Test model loading"""
    try:
        logger.info("=" * 80)
        logger.info("Testing Model Loading")
        logger.info("=" * 80)

        model_name = getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2")
        logger.info(f"Loading model: {model_name}")

        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Embedding dimension: {dim}")

        # Test encoding
        test_text = "This is a test sentence for encoding"
        embedding = model.encode(test_text, convert_to_numpy=True, normalize_embeddings=True)
        logger.info(f"Test encoding shape: {embedding.shape}")
        logger.info(f"Test encoding sample: {embedding[:5]}")

        return model, dim
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None, None


def test_retrieval(client, model, collection_name, profile_id):
    """Test actual retrieval"""
    try:
        logger.info("=" * 80)
        logger.info("Testing Retrieval")
        logger.info("=" * 80)

        test_query = "What are the key features?"
        logger.info(f"Query: {test_query}")
        logger.info(f"Profile ID: {profile_id}")

        # Encode query
        query_vector = model.encode(
            test_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32').tolist()

        logger.info(f"Query vector dimension: {len(query_vector)}")

        # Test 1: Basic search without threshold
        logger.info("\n--- Test 1: Basic search (no threshold) ---")
        filter_dict = {
            "must": [
                {"key": "profile_id", "match": {"value": str(profile_id)}}
            ]
        }

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="content_vector",
            limit=10,
            query_filter=filter_dict,
            with_payload=True,
            with_vectors=False
        )

        if results and hasattr(results, 'points'):
            logger.info(f"✅ Retrieved {len(results.points)} results")
            for i, pt in enumerate(results.points[:3], 1):
                logger.info(f"  {i}. Score: {pt.score:.4f} | Text: {(pt.payload.get('text', ''))[:80]}...")
        else:
            logger.warning("❌ No results returned")

        # Test 2: Check filter is working
        logger.info("\n--- Test 2: Testing filter ---")
        results_all = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="content_vector",
            limit=10,
            with_payload=True,
            with_vectors=False
        )

        if results_all and hasattr(results_all, 'points'):
            logger.info(f"Without filter: {len(results_all.points)} results")
            logger.info(f"With filter: {len(results.points) if results and hasattr(results, 'points') else 0} results")

            if results_all.points:
                profiles_found = set(pt.payload.get('profile_id') for pt in results_all.points if pt.payload)
                logger.info(f"Profile IDs found in collection: {profiles_found}")

        # Test 3: Try enhanced retrieval
        logger.info("\n--- Test 3: Enhanced Adaptive Retrieval ---")
        try:
            from src.api.enhanced_retrieval import AdaptiveRetriever

            retriever = AdaptiveRetriever(client, model, use_sparse=True)
            chunks = retriever.retrieve_adaptive(
                collection_name=collection_name,
                query=test_query,
                profile_id=profile_id,
                top_k=10
            )

            logger.info(f"✅ Enhanced retrieval returned {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3], 1):
                logger.info(f"  {i}. Score: {chunk['score']:.4f} | Method: {chunk.get('method', 'unknown')}")
                logger.info(f"     Text: {chunk['text'][:80]}...")
        except ImportError:
            logger.error("❌ enhanced_retrieval module not found - make sure to create it!")
        except Exception as e:
            logger.error(f"❌ Enhanced retrieval failed: {e}")

        return True
    except Exception as e:
        logger.error(f"❌ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    logger.info("\n" + "=" * 80)
    logger.info("RETRIEVAL DIAGNOSTIC TOOL")
    logger.info("=" * 80 + "\n")

    # Test 1: Qdrant connection
    client = test_qdrant_connection()
    if not client:
        logger.error("Cannot proceed without Qdrant connection")
        sys.exit(1)

    # Test 2: Model loading
    model, dim = test_model_loading()
    if not model:
        logger.error("Cannot proceed without model")
        sys.exit(1)

    # Get collection name from command line or use default
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    else:
        # Try to find a collection
        collections = client.get_collections()
        if collections.collections:
            collection_name = collections.collections[0].name
            logger.info(f"Using collection: {collection_name}")
        else:
            logger.error("No collections found")
            sys.exit(1)

    # Get profile_id from command line or use default
    if len(sys.argv) > 2:
        profile_id = sys.argv[2]
    else:
        # Try to find a profile_id from data
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if hasattr(scroll_result, 'points'):
            points = scroll_result.points
        elif isinstance(scroll_result, tuple):
            points = scroll_result[0]
        else:
            points = []

        if points:
            profile_id = points[0].payload.get('profile_id', 'unknown')
            logger.info(f"Using profile_id: {profile_id}")
        else:
            profile_id = "67ac62ddfaa3aee44d38f4a5"  # Default from your logs
            logger.info(f"Using default profile_id: {profile_id}")

    # Test 3: Collection details
    test_collection_details(client, collection_name)

    # Test 4: Retrieval
    test_retrieval(client, model, collection_name, profile_id)

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
































