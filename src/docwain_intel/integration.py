"""Integration hooks for wiring intelligence pipeline into DocWain upload/query flows.

This module provides clean integration points without modifying core API files directly.
The hooks are designed to be called from embedding_service.py and documents_api.py.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
import threading
from typing import Any, Dict, List, Optional

from .models import ExtractedDocumentJSON
from .intel_pipeline import process_document, ProcessingResult
from .query_router import route_query, QueryAnalysis, QueryRoute
from .response_assembler import assemble_response, AssembledResponse

logger = get_logger(__name__)

# Feature flag — set DOCWAIN_INTEL_PIPELINE=1 to enable
INTEL_PIPELINE_ENABLED = os.getenv("DOCWAIN_INTEL_PIPELINE", "0") == "1"

# ---------------------------------------------------------------------------
# Intelligence Engine V2 singleton
# ---------------------------------------------------------------------------

_engine_lock = threading.Lock()
_engine_instance = None

def get_intelligence_engine():
    """Return the singleton IntelligenceEngine, or None if disabled.

    Gated by the ``DOCWAIN_INTEL_V2`` environment variable.
    Set ``DOCWAIN_INTEL_V2=1`` to enable.
    """
    global _engine_instance
    if os.getenv("DOCWAIN_INTEL_V2", "0") != "1":
        return None

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:
        if _engine_instance is not None:
            return _engine_instance
        from .intelligence import IntelligenceEngine
        _engine_instance = IntelligenceEngine()
        return _engine_instance

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

    When DOCWAIN_INTEL_V2 is enabled, routes through the full intelligence engine
    first (with pronoun resolution, geometry analysis, etc.). Falls back to the
    legacy path on failure or when the engine is disabled.

    Returns AssembledResponse with text, sources, and confidence.
    """
    engine = get_intelligence_engine()
    if engine is not None:
        try:
            result = engine.process_query(
                query=query,
                subscription_id="default",
                profile_id="default",
                session_id="default",
                chunks=chunks,
                facts=facts,
            )
            if result.text:
                return AssembledResponse(
                    text=result.text,
                    sources=result.sources,
                    confidence=result.confidence,
                    route_used=result.route_used,
                    fact_count=len(facts or []),
                    chunk_count=len(chunks or []),
                )
        except Exception:
            logger.debug("Intelligence engine failed, falling back", exc_info=True)

    # Legacy path
    analysis = route_query(query)
    return assemble_response(
        query=query,
        route=analysis.route,
        facts=facts,
        chunks=chunks,
        is_comparison=analysis.is_comparison,
        is_aggregation=analysis.is_aggregation,
    )
