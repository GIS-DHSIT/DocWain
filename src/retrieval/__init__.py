"""Intent-aware retrieval helpers + unified retrieval system."""

# --- Legacy exports (used by existing modules) ---
from .query_analyzer import QueryAnalyzer, QueryAnalysis
from .evidence_constraints import EvidenceConstraints, EvidenceRequirements, EvidenceCoverage
from .hybrid_ranker import HybridRanker, HybridRankerConfig
from .retrieval_quality import RetrievalQualityScorer, RetrievalQualityResult
from .context_assembler import ContextAssembler, ContextBuildResult
from .fallback_repair import FallbackRepair, FallbackResult
from .evidence_synthesizer import EvidenceSynthesizer
from .intent_filter import (
    extract_required_attributes,
    filter_chunks_by_intent,
    extract_answer_requirements,
    validate_answer_requirements,
    build_intent_miss_response,
)

# --- New unified retrieval system ---
from .retriever import EvidenceChunk, RetrievalResult, UnifiedRetriever
from .reranker import rerank_chunks
from .context_builder import build_context

__all__ = [
    # Legacy
    "QueryAnalyzer",
    "QueryAnalysis",
    "EvidenceConstraints",
    "EvidenceRequirements",
    "EvidenceCoverage",
    "HybridRanker",
    "HybridRankerConfig",
    "RetrievalQualityScorer",
    "RetrievalQualityResult",
    "ContextAssembler",
    "ContextBuildResult",
    "FallbackRepair",
    "FallbackResult",
    "EvidenceSynthesizer",
    "extract_required_attributes",
    "filter_chunks_by_intent",
    "extract_answer_requirements",
    "validate_answer_requirements",
    "build_intent_miss_response",
    # New
    "EvidenceChunk",
    "RetrievalResult",
    "UnifiedRetriever",
    "rerank_chunks",
    "build_context",
]
