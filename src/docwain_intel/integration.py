"""Integration hooks for wiring intelligence pipeline into DocWain upload/query flows.

This module provides clean integration points without modifying core API files directly.
The hooks are designed to be called from embedding_service.py and documents_api.py.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .models import ExtractedDocumentJSON
from .intel_pipeline import process_document, ProcessingResult
from .query_router import route_query, QueryAnalysis, QueryRoute
from .response_assembler import assemble_response, AssembledResponse

logger = logging.getLogger(__name__)

# Feature flag — set DOCWAIN_INTEL_PIPELINE=1 to enable
INTEL_PIPELINE_ENABLED = os.getenv("DOCWAIN_INTEL_PIPELINE", "0") == "1"


def run_intel_pipeline_hook(
    *,
    extracted_doc: ExtractedDocumentJSON,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    vector_store: Any = None,
    graph_store: Any = None,
) -> Optional[ProcessingResult]:
    """Hook for document upload flow. Call after raw extraction.

    Returns ProcessingResult if pipeline enabled and succeeds, None otherwise.
    Never raises — errors are logged and None is returned so upload flow continues.
    """
    if not INTEL_PIPELINE_ENABLED:
        return None

    try:
        result = process_document(
            extracted_doc=extracted_doc,
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            vector_store=vector_store,
            graph_store=graph_store,
        )
        logger.info(
            "Intel pipeline completed: doc=%s stage=%s entities=%d facts=%d",
            document_id, result.stage_reached,
            len(result.extraction.entities) if result.extraction else 0,
            len(result.extraction.facts) if result.extraction else 0,
        )
        return result
    except Exception as exc:
        logger.error("Intel pipeline hook failed for doc=%s: %s", document_id, exc, exc_info=True)
        return None


def route_and_assemble(
    *,
    query: str,
    facts: List[Dict[str, Any]] | None = None,
    chunks: List[Dict[str, Any]] | None = None,
) -> AssembledResponse:
    """Query-time hook: route query and assemble response from pre-computed data.

    Returns AssembledResponse with text, sources, and confidence.
    """
    analysis = route_query(query)
    return assemble_response(
        query=query,
        route=analysis.route,
        facts=facts,
        chunks=chunks,
        is_comparison=analysis.is_comparison,
        is_aggregation=analysis.is_aggregation,
    )
