"""6-stage document intelligence pipeline orchestrator."""
from __future__ import annotations

import enum
from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .models import (
    ExtractedDocumentJSON, StructuredDocument, ExtractionResult,
    DocumentFingerprint, VerificationResult,
)
from .structure_parser import parse_structure
from .entity_engine import extract_entities_and_facts
from .entity_resolver import resolve_entities
from .document_fingerprint import compute_fingerprint
from .verification import verify_extraction

logger = get_logger(__name__)

class PipelineStage(str, enum.Enum):
    STRUCTURED = "STRUCTURED"
    ENTITIES_EXTRACTED = "ENTITIES_EXTRACTED"
    ENTITIES_RESOLVED = "ENTITIES_RESOLVED"
    FINGERPRINTED = "FINGERPRINTED"
    VERIFIED = "VERIFIED"
    STORED = "STORED"
    AUDITED = "AUDITED"
    FAILED = "FAILED"

class ProcessingResult(BaseModel):
    document_id: str
    subscription_id: str
    profile_id: str
    stage_reached: str = PipelineStage.FAILED.value
    structured_doc: Optional[StructuredDocument] = None
    extraction: Optional[ExtractionResult] = None
    fingerprint: Optional[DocumentFingerprint] = None
    verification: Optional[VerificationResult] = None
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None

def process_document(
    *,
    extracted_doc: ExtractedDocumentJSON,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    vector_store: Any = None,
    graph_store: Any = None,
) -> ProcessingResult:
    """Run the 6-stage document intelligence pipeline.

    Stages:
    1. Structure parsing (blocks -> semantic units)
    2. Entity + fact extraction (spaCy + patterns + textacy SVO)
    3. Entity resolution (cross-document dedup via Jaro-Winkler)
    4. Document fingerprinting (auto-tagging)
    5. Verification (conflict detection + provenance validation)
    6. Storage + audit (Qdrant vectors + Neo4j KG) -- skipped if no stores provided
    """
    result = ProcessingResult(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
    )

    try:
        # Stage 1: Structure parsing
        t0 = time.monotonic()
        structured = parse_structure(extracted_doc, document_id=document_id)
        result.stage_timings["structure"] = time.monotonic() - t0
        result.structured_doc = structured
        result.stage_reached = PipelineStage.STRUCTURED.value
        logger.info(
            "Pipeline stage 1 (structure): %d units in %.2fs",
            len(structured.units), result.stage_timings["structure"],
        )

        # Stage 2: Entity + fact extraction
        t0 = time.monotonic()
        extraction = extract_entities_and_facts(structured)
        result.stage_timings["extraction"] = time.monotonic() - t0
        result.extraction = extraction
        result.stage_reached = PipelineStage.ENTITIES_EXTRACTED.value
        logger.info(
            "Pipeline stage 2 (extraction): %d entities, %d facts in %.2fs",
            len(extraction.entities), len(extraction.facts),
            result.stage_timings["extraction"],
        )

        # Stage 3: Entity resolution
        t0 = time.monotonic()
        resolved = resolve_entities(extraction.entities)
        extraction.entities = resolved
        result.stage_timings["resolution"] = time.monotonic() - t0
        result.stage_reached = PipelineStage.ENTITIES_RESOLVED.value
        logger.info(
            "Pipeline stage 3 (resolution): %d resolved entities in %.2fs",
            len(resolved), result.stage_timings["resolution"],
        )

        # Stage 4: Document fingerprinting
        t0 = time.monotonic()
        fingerprint = compute_fingerprint(structured, extraction)
        result.stage_timings["fingerprint"] = time.monotonic() - t0
        result.fingerprint = fingerprint
        result.stage_reached = PipelineStage.FINGERPRINTED.value
        logger.info(
            "Pipeline stage 4 (fingerprint): %d tags in %.2fs",
            len(fingerprint.auto_tags), result.stage_timings["fingerprint"],
        )

        # Stage 5: Verification
        t0 = time.monotonic()
        verification = verify_extraction(extraction, structured)
        result.stage_timings["verification"] = time.monotonic() - t0
        result.verification = verification
        result.stage_reached = PipelineStage.VERIFIED.value
        logger.info(
            "Pipeline stage 5 (verification): valid=%s, score=%.2f in %.2fs",
            verification.is_valid, verification.quality_score,
            result.stage_timings["verification"],
        )

        # Stage 6: Storage + audit (skip if no stores)
        # -- graph population (best-effort) ----------------------------------
        try:
            from .graph_adapter import get_graph_adapter
            from .graph_populator import populate_graph

            graph_adapter = graph_store or get_graph_adapter()
            if graph_adapter is not None:
                t0_graph = time.monotonic()
                fp_tags = fingerprint.auto_tags if fingerprint else []
                populate_graph(
                    adapter=graph_adapter,
                    extraction=extraction,
                    structured_doc=structured,
                    document_id=document_id,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    fingerprint_tags=fp_tags,
                )
                result.stage_timings["graph_populate"] = time.monotonic() - t0_graph
                logger.info(
                    "Pipeline stage 6 (graph): %d entities, %d facts written in %.2fs",
                    len(extraction.entities), len(extraction.facts),
                    result.stage_timings["graph_populate"],
                )
        except Exception as graph_exc:
            logger.warning("Graph population failed (non-fatal): %s", graph_exc)

        # -- integrity audit -------------------------------------------------
        if vector_store or graph_store:
            from .integrity_audit import run_integrity_audit
            t0 = time.monotonic()
            audit = run_integrity_audit(
                structured, extraction,
                vector_store=vector_store, graph_store=graph_store,
            )
            result.stage_timings["audit"] = time.monotonic() - t0
            result.stage_reached = PipelineStage.AUDITED.value
            logger.info(
                "Pipeline stage 6 (audit): passed=%s in %.2fs",
                audit.passed, result.stage_timings["audit"],
            )
        else:
            result.stage_reached = PipelineStage.VERIFIED.value

    except Exception as exc:
        result.error = str(exc)
        result.stage_reached = PipelineStage.FAILED.value
        logger.error("Pipeline failed at %s: %s", result.stage_reached, exc, exc_info=True)

    return result
