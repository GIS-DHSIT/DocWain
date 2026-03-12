"""Gateway tool bridge — register content generation as a gateway tool.

Exposes content generation through the unified gateway via:
  - tool:content_generate — main generation entry point
  - tool:content_types — list available content types
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional

from src.tools.base import register_tool

from .engine import ContentGenerationEngine
from .registry import (
    CONTENT_TYPE_REGISTRY,
    DOMAINS,
    get_content_type,
    list_content_types,
    list_domains,
)

logger = get_logger(__name__)

def _get_chunks_from_payload(payload: Dict[str, Any]) -> list:
    """Extract chunks from tool payload, supporting multiple sources."""
    input_data = payload.get("input", payload)

    # Direct chunks
    chunks = input_data.get("chunks")
    if chunks:
        return chunks

    # If profile_id + query provided, retrieve from pipeline
    profile_id = (
        input_data.get("profile_id")
        or payload.get("profile_id")
        or (payload.get("context") or {}).get("profile_id")
    )
    collection_id = input_data.get("collection_id") or (payload.get("context") or {}).get("collection_id")
    query = input_data.get("query", "")

    if profile_id and query:
        try:
            from src.rag_v3.retrieve import vector_retrieve
            from src.api.config import Config

            collection = collection_id or Config.Qdrant.DEFAULT_COLLECTION
            results = vector_retrieve(
                query=query,
                collection_name=collection,
                profile_id=profile_id,
                top_k=10,
            )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to retrieve chunks for content generation: %s", exc)

    return []

def _get_llm_client() -> Optional[Any]:
    """Lazy-load LLM client."""
    try:
        from src.api.config import Config
        from src.api.dw_newron import DwNewron

        # Try to get a cached instance's LLM client
        if hasattr(DwNewron, "_llm_client"):
            return DwNewron._llm_client
    except Exception:  # noqa: BLE001
        pass
    return None

@register_tool("content_generate")
async def content_generate_handler(
    payload: Dict[str, Any],
    *,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate document-grounded content.

    Input fields:
        query (str): What to generate (e.g., "write a cover letter for Abinaya")
        content_type (str, optional): Explicit content type ID
        chunks (list, optional): Pre-retrieved document chunks
        profile_id (str, optional): Profile for auto-retrieval
        collection_id (str, optional): Qdrant collection
        extra_instructions (str, optional): Additional generation instructions
    """
    input_data = payload.get("input", payload)
    query = input_data.get("query", "")
    if not query:
        return {
            "result": {"error": "query is required"},
            "sources": [],
            "grounded": False,
            "context_found": False,
        }

    content_type_id = input_data.get("content_type")
    extra_instructions = input_data.get("extra_instructions", "")
    chunk_domain = input_data.get("domain")

    # Get chunks
    chunks = _get_chunks_from_payload(payload)

    # Get LLM client
    llm_client = _get_llm_client()

    engine = ContentGenerationEngine(llm_client=llm_client)
    result = engine.generate(
        query=query,
        chunks=chunks,
        content_type_id=content_type_id,
        chunk_domain=chunk_domain,
        extra_instructions=extra_instructions,
        correlation_id=correlation_id,
    )

    return {
        "result": {
            "response": result.get("response", ""),
            "content_type": (result.get("metadata") or {}).get("content_type"),
            "domain": (result.get("metadata") or {}).get("domain"),
            "verification": (result.get("metadata") or {}).get("verification"),
        },
        "sources": result.get("sources", []),
        "grounded": result.get("grounded", False),
        "context_found": result.get("context_found", False),
        "warnings": result.get("warnings", []),
    }

@register_tool("content_types")
async def content_types_handler(
    payload: Dict[str, Any],
    *,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List available content types, optionally filtered by domain.

    Input fields:
        domain (str, optional): Filter by domain (e.g., "hr", "invoice")
    """
    input_data = payload.get("input", payload)
    domain = input_data.get("domain")

    types = list_content_types(domain=domain)
    domains = list_domains()

    return {
        "result": {
            "content_types": [
                {
                    "id": ct.id,
                    "domain": ct.domain,
                    "name": ct.name,
                    "description": ct.description,
                    "supports_multi_doc": ct.supports_multi_doc,
                }
                for ct in types
            ],
            "domains": domains,
            "total": len(types),
        },
        "sources": [],
        "grounded": True,
        "context_found": True,
    }
