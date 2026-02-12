"""Live retrieval test — connects to real Qdrant and runs through the new pipeline.

Gated behind ``LIVE_TEST=1`` environment variable so it never runs in CI.

Usage::

    LIVE_TEST=1 pytest tests/test_live_retrieval.py -v -s
"""
from __future__ import annotations

import os
import pprint
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("LIVE_TEST", "") not in {"1", "true", "yes"},
    reason="Live test skipped (set LIVE_TEST=1 to enable)",
)


@pytest.fixture(scope="module")
def qdrant_client():
    from qdrant_client import QdrantClient
    from src.api.config import Config

    return QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=15)


@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer("BAAI/bge-large-en-v1.5")
    except (RuntimeError, Exception) as exc:
        if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
            return SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        raise


def test_retrieve_and_extract(qdrant_client, embedder):
    """Retrieve chunks from Qdrant and run through the new LLM-first pipeline."""
    from src.rag_v3.retrieve import retrieve_chunks
    from src.rag_v3.types import Chunk

    # Use a test subscription/profile — adjust IDs for your deployment
    subscription_id = os.getenv("TEST_SUBSCRIPTION_ID", "67fde0754e36c00b14cea7f5")
    profile_id = os.getenv("TEST_PROFILE_ID", "698c46e6bcae2c45eca1d8d9")
    query = os.getenv("TEST_QUERY", "What are the candidate's skills and experience?")

    chunks = retrieve_chunks(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        qdrant_client=qdrant_client,
        embedder=embedder,
        top_k=10,
    )

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Retrieved {len(chunks)} chunks")
    print(f"{'='*60}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} (score={chunk.score:.4f}) ---")
        print(f"  ID: {chunk.id}")
        print(f"  Source: {chunk.source.document_name} p.{chunk.source.page}")
        print(f"  Text: {chunk.text[:200]}...")
        print(f"  Meta keys: {list(chunk.meta.keys())}")

    assert len(chunks) > 0, "Expected at least one chunk from Qdrant"

    # Verify slim payload on retrieved chunks (new embeddings)
    for chunk in chunks:
        meta = chunk.meta
        # Core fields should be present
        assert meta.get("subscription_id") or meta.get("subscriptionId"), \
            f"Missing subscription_id in chunk {chunk.id}"
        assert meta.get("profile_id") or meta.get("profileId"), \
            f"Missing profile_id in chunk {chunk.id}"

    print("\n[OK] Retrieval passed")


def test_full_pipeline_live(qdrant_client, embedder):
    """Run the full RAG v3 pipeline end-to-end against live data."""
    from src.rag_v3.pipeline import run

    subscription_id = os.getenv("TEST_SUBSCRIPTION_ID", "67fde0754e36c00b14cea7f5")
    profile_id = os.getenv("TEST_PROFILE_ID", "698c46e6bcae2c45eca1d8d9")
    query = os.getenv("TEST_QUERY", "What are the candidate's skills and experience?")

    result = run(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        llm_client=None,  # deterministic fallback only
        qdrant_client=qdrant_client,
        redis_client=None,
        embedder=embedder,
        cross_encoder=None,
    )

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print(f"\nResponse:\n{result.get('response', '')[:500]}")
    print(f"\nSources: {len(result.get('sources', []))}")
    print(f"Metadata: {pprint.pformat(result.get('metadata', {}))}")
    print(f"Context found: {result.get('context_found')}")
    print(f"Grounded: {result.get('grounded')}")

    assert result.get("response"), "Expected non-empty response"
    assert isinstance(result.get("sources"), list), "Expected sources list"

    print("\n[OK] Full pipeline passed")
