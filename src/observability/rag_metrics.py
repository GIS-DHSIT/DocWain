"""
RAG Pipeline Metrics and Timing Utilities.

This module provides:
- RAGPipelineMetrics: Dataclass for aggregating pipeline metrics
- StageTimer: Context manager for timing pipeline stages
- log_pipeline_summary: Function to log execution summary

Usage:
    from src.observability.rag_metrics import RAGPipelineMetrics, StageTimer

    metrics = RAGPipelineMetrics(correlation_id="abc-123")

    with StageTimer(metrics, "retrieval"):
        results = retrieve_documents(query)

    with StageTimer(metrics, "reranking"):
        reranked = rerank_results(results)

    log_pipeline_summary(metrics)
"""

from __future__ import annotations

import logging

from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

@dataclass
class RAGPipelineMetrics:
    """
    Aggregate metrics for a RAG pipeline execution.

    Collects timing information, counts, and quality scores
    for each stage of the pipeline.
    """

    # Identification
    correlation_id: str = ""
    query: str = ""

    # Stage timings (in milliseconds)
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # Retrieval metrics
    chunks_retrieved: int = 0
    chunks_after_filter: int = 0
    chunks_after_rerank: int = 0
    documents_accessed: int = 0

    # Quality scores
    confidence_score: float = 0.0
    relevance_scores: List[float] = field(default_factory=list)
    avg_chunk_score: float = 0.0

    # Flags
    used_cache: bool = False
    used_fallback: bool = False
    reranking_applied: bool = False

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def record_stage_timing(self, stage: str, duration_ms: float) -> None:
        """Record timing for a pipeline stage."""
        self.stage_timings[stage] = duration_ms

    def add_error(self, error: str) -> None:
        """Record an error that occurred during processing."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Record a warning during processing."""
        self.warnings.append(warning)

    def finalize(self) -> None:
        """Mark the pipeline execution as complete."""
        self.end_time = time.time()

    @property
    def total_duration_ms(self) -> float:
        """Total pipeline execution time in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def is_successful(self) -> bool:
        """Whether the pipeline completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/serialization."""
        return {
            "correlation_id": self.correlation_id,
            "query_length": len(self.query),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "stage_timings": {k: round(v, 2) for k, v in self.stage_timings.items()},
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_after_filter": self.chunks_after_filter,
            "chunks_after_rerank": self.chunks_after_rerank,
            "documents_accessed": self.documents_accessed,
            "confidence_score": round(self.confidence_score, 3),
            "avg_chunk_score": round(self.avg_chunk_score, 3),
            "used_cache": self.used_cache,
            "used_fallback": self.used_fallback,
            "reranking_applied": self.reranking_applied,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "is_successful": self.is_successful,
        }

class StageTimer:
    """
    Context manager for timing pipeline stages.

    Automatically records the duration of a stage in the metrics object.

    Usage:
        metrics = RAGPipelineMetrics()
        with StageTimer(metrics, "retrieval"):
            # Do retrieval work
            pass
        # metrics.stage_timings["retrieval"] now contains the duration
    """

    def __init__(
        self,
        metrics: RAGPipelineMetrics,
        stage_name: str,
        log_on_exit: bool = False,
    ):
        """
        Initialize the stage timer.

        Args:
            metrics: The metrics object to record timing in.
            stage_name: Name of the stage being timed.
            log_on_exit: Whether to log timing on context exit.
        """
        self.metrics = metrics
        self.stage_name = stage_name
        self.log_on_exit = log_on_exit
        self.start_time: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> "StageTimer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and record the duration."""
        end_time = time.perf_counter()
        self.duration_ms = (end_time - self.start_time) * 1000
        self.metrics.record_stage_timing(self.stage_name, self.duration_ms)

        if exc_type is not None:
            self.metrics.add_error(f"{self.stage_name}: {exc_type.__name__}: {exc_val}")

        if self.log_on_exit:
            logger.debug(
                "Stage '%s' completed in %.2f ms",
                self.stage_name,
                self.duration_ms,
                extra={"correlation_id": self.metrics.correlation_id},
            )

def log_pipeline_summary(
    metrics: RAGPipelineMetrics,
    level: int = logging.INFO,
) -> None:
    """
    Log a summary of the pipeline execution.

    Args:
        metrics: The metrics object containing execution data.
        level: Logging level to use.
    """
    metrics.finalize()
    summary = metrics.to_dict()

    # Build summary message
    status = "SUCCESS" if metrics.is_successful else "FAILED"
    msg = (
        f"RAG Pipeline {status} | "
        f"duration={summary['total_duration_ms']:.0f}ms | "
        f"chunks={summary['chunks_retrieved']}->{summary['chunks_after_rerank']} | "
        f"confidence={summary['confidence_score']:.2f}"
    )

    if metrics.used_cache:
        msg += " | cached"
    if metrics.used_fallback:
        msg += " | fallback"
    if metrics.errors:
        msg += f" | errors={len(metrics.errors)}"

    logger.log(
        level,
        msg,
        extra={
            "correlation_id": metrics.correlation_id,
            "metrics": summary,
        },
    )

def create_metrics(
    correlation_id: str = "",
    query: str = "",
) -> RAGPipelineMetrics:
    """
    Factory function to create a new metrics object.

    Args:
        correlation_id: Request correlation ID.
        query: The user query being processed.

    Returns:
        A new RAGPipelineMetrics instance.
    """
    return RAGPipelineMetrics(
        correlation_id=correlation_id,
        query=query,
    )

__all__ = [
    "RAGPipelineMetrics",
    "StageTimer",
    "log_pipeline_summary",
    "create_metrics",
]
