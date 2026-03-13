"""
Hybrid RAG Extraction Strategy

Combines:
1. Document-level extraction (complete resume data) - PRIMARY
2. Chunk-based extraction (Qdrant fragments) - FALLBACK

This ensures accurate retrieval even when Qdrant chunks are incomplete.
"""

from typing import Any, Dict, List, Optional
from src.utils.logging_utils import get_logger

from .document_extraction import extract_hr_from_complete_document
from .extract import _extract_hr, Candidate, HRSchema, _candidates_field

logger = get_logger(__name__)

def extract_hr_hybrid(
    chunks: List[Any],
    document_data: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
) -> HRSchema:
    """
    Extract HR data using hybrid strategy:
    1. If complete document data is available, use document-level extraction (ACCURATE)
    2. Fallback to chunk-based extraction if only Qdrant chunks available

    Args:
        chunks: Retrieved chunks from Qdrant (may be incomplete)
        document_data: Complete extracted document data (preferred)
        document_id: Document ID (for logging/debugging)

    Returns:
        HRSchema with extracted candidate information
    """

    candidates = []

    # STRATEGY 1: Use complete document data if available (PREFERRED)
    if document_data:
        logger.info(f"Using document-level extraction for {document_id}")
        try:
            candidate_data = extract_hr_from_complete_document(document_data)
            candidates.append(_convert_to_candidate(candidate_data))
        except Exception as e:
            logger.debug(f"Document-level extraction failed for {document_id}: {e}. Falling back to chunks.")
            candidates = []

    # STRATEGY 2: Fallback to chunk-based extraction
    if not candidates:
        logger.info(f"Using chunk-based extraction for {document_id}")
        schema = _extract_hr(chunks)
        candidates = (schema.candidates.items if schema.candidates else None) or []

    # Return as HRSchema
    return HRSchema(candidates=_candidates_field(candidates))

def _convert_to_candidate(data: Dict[str, Any]) -> Candidate:
    """Convert document extraction result to Candidate model."""
    return Candidate(
        name=data.get("name"),
        role=None,
        details=None,
        total_years_experience=data.get("total_years_experience"),
        experience_summary=data.get("experience_summary"),
        technical_skills=data.get("technical_skills") or [],
        functional_skills=data.get("functional_skills") or [],
        certifications=data.get("certifications") or [],
        education=data.get("education") or [],
        achievements=data.get("achievements") or [],
        emails=data.get("email") or [],
        phones=data.get("phone") or [],
        linkedins=data.get("linkedin") or [],
        source_type=data.get("source_type"),
        evidence_spans=[],  # Not available from document-level extraction
        missing_reason={},
    )

