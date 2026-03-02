"""
Observability package for DocWain.

Provides metrics collection, tracing, and monitoring utilities for the RAG pipeline.
"""

from src.observability.rag_metrics import (
    RAGPipelineMetrics,
    StageTimer,
    log_pipeline_summary,
)

__all__ = [
    "RAGPipelineMetrics",
    "StageTimer",
    "log_pipeline_summary",
]
