"""
DocWain Intelligence Layer.

This package provides comprehensive document intelligence capabilities:
- Structured document extraction with metadata capture
- Entity and relationship extraction (NER)
- Domain classification
- Knowledge graph construction
- Auto Q&A generation for quality retrieval
- Response formatting with acknowledgement
"""

from src.intelligence.document_intelligence import (
    DocumentIntelligence,
    DocumentMetadata,
    StructuredDocument,
    ExtractedEntities,
)
from src.intelligence.qa_generator import (
    QAGenerator,
    GeneratedQA,
)
from src.intelligence.response_formatter import (
    ResponseFormatter,
    FormattedResponse,
    QueryIntent,
    format_acknowledged_response,
)
from src.intelligence.integration import (
    DocumentIntelligenceProcessor,
    IntelligenceResult,
    KnowledgeGraphBuilder,
    process_document_intelligence,
)
from src.intelligence.task_spec import TaskSpec
from src.intelligence.query_understanding import understand_query

__all__ = [
    # Document Intelligence
    "DocumentIntelligence",
    "DocumentMetadata",
    "StructuredDocument",
    "ExtractedEntities",
    # Q&A Generation
    "QAGenerator",
    "GeneratedQA",
    # Response Formatting
    "ResponseFormatter",
    "FormattedResponse",
    "QueryIntent",
    "format_acknowledged_response",
    # Integration
    "DocumentIntelligenceProcessor",
    "IntelligenceResult",
    "KnowledgeGraphBuilder",
    "process_document_intelligence",
    # Query Understanding
    "TaskSpec",
    "understand_query",
]
