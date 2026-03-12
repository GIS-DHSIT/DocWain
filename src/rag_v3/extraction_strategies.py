"""
Extraction Strategies for DocWain RAG Pipeline.

This module provides multiple strategies for extracting structured data
from documents, with automatic fallback between strategies.

Strategies (in order of preference):
1. DocumentDataStrategy - Extract from pre-parsed document data
2. ChunkBasedStrategy - Regex/heuristic extraction from chunks
3. LLMExtractionStrategy - LLM-based extraction for complex cases

The ExtractionOrchestrator coordinates these strategies with fallback.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = get_logger(__name__)

@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    strategy_used: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Merge another result into this one, combining data."""
        merged_data = {**self.data, **other.data}
        merged_errors = self.errors + other.errors
        return ExtractionResult(
            success=self.success or other.success,
            data=merged_data,
            confidence=max(self.confidence, other.confidence),
            strategy_used=f"{self.strategy_used},{other.strategy_used}",
            errors=merged_errors,
            metadata={**self.metadata, **other.metadata},
        )

class ExtractionStrategy(ABC):
    """Base class for extraction strategies."""

    name: str = "base"
    priority: int = 0  # Higher = tried first

    @abstractmethod
    def can_extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
    ) -> bool:
        """Check if this strategy can extract the requested field."""
        pass

    @abstractmethod
    def extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
        **kwargs,
    ) -> ExtractionResult:
        """Extract the requested field using this strategy."""
        pass

class DocumentDataStrategy(ExtractionStrategy):
    """
    Extract from pre-parsed document data.

    This is the highest-accuracy strategy as it uses structured data
    that was extracted during document ingestion.
    """

    name = "document_data"
    priority = 100

    # Mapping of field aliases to canonical names
    FIELD_ALIASES: Dict[str, List[str]] = {
        "name": ["full_name", "candidate_name", "applicant_name", "person_name"],
        "email": ["email_address", "contact_email", "e-mail"],
        "phone": ["phone_number", "telephone", "mobile", "contact_number"],
        "title": ["job_title", "position", "role", "current_title"],
        "company": ["employer", "organization", "current_company"],
        "experience": ["work_experience", "professional_experience", "employment_history"],
        "education": ["educational_background", "qualifications", "degrees"],
        "skills": ["skill_set", "technical_skills", "competencies"],
    }

    def can_extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
    ) -> bool:
        """Check if field exists in document data."""
        if not document_data:
            return False

        field_lower = field_name.lower()
        if field_lower in document_data:
            return True

        # Check aliases
        for canonical, aliases in self.FIELD_ALIASES.items():
            if field_lower == canonical or field_lower in aliases:
                for alias in [canonical] + aliases:
                    if alias in document_data:
                        return True

        return False

    def extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
        **kwargs,
    ) -> ExtractionResult:
        """Extract field from document data."""
        if not document_data:
            return ExtractionResult(
                success=False,
                errors=["No document data available"],
                strategy_used=self.name,
            )

        field_lower = field_name.lower()
        value = None

        # Direct lookup
        if field_lower in document_data:
            value = document_data[field_lower]

        # Check aliases
        if value is None:
            for canonical, aliases in self.FIELD_ALIASES.items():
                if field_lower == canonical or field_lower in aliases:
                    for alias in [canonical] + aliases:
                        if alias in document_data and document_data[alias]:
                            value = document_data[alias]
                            break
                    break

        if value is not None:
            return ExtractionResult(
                success=True,
                data={field_name: value},
                confidence=0.95,
                strategy_used=self.name,
            )

        return ExtractionResult(
            success=False,
            errors=[f"Field '{field_name}' not found in document data"],
            strategy_used=self.name,
        )

class ChunkBasedStrategy(ExtractionStrategy):
    """
    Extract using regex and heuristics on document chunks.

    Uses pattern matching to find specific data types in text.
    """

    name = "chunk_based"
    priority = 50

    # Regex patterns for common fields
    PATTERNS: Dict[str, List[re.Pattern]] = {
        "email": [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ],
        "phone": [
            re.compile(r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            re.compile(r'\b\+?[0-9]{1,3}[-.\s]?[0-9]{2,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'),
        ],
        "url": [
            re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
        ],
        "date": [
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.I),
        ],
        "money": [
            re.compile(r'\$\s?[\d,]+(?:\.\d{2})?'),
            re.compile(r'[\d,]+(?:\.\d{2})?\s?(?:USD|EUR|GBP|CAD)'),
        ],
    }

    def can_extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
    ) -> bool:
        """Check if we have patterns for this field type."""
        return chunks and field_name.lower() in self.PATTERNS

    def extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
        **kwargs,
    ) -> ExtractionResult:
        """Extract using regex patterns."""
        if not chunks:
            return ExtractionResult(
                success=False,
                errors=["No chunks available"],
                strategy_used=self.name,
            )

        field_lower = field_name.lower()
        patterns = self.PATTERNS.get(field_lower, [])
        if not patterns:
            return ExtractionResult(
                success=False,
                errors=[f"No patterns defined for '{field_name}'"],
                strategy_used=self.name,
            )

        matches = []
        combined_text = "\n".join(chunks)

        for pattern in patterns:
            found = pattern.findall(combined_text)
            matches.extend(found)

        if matches:
            # Deduplicate while preserving order
            unique_matches = list(dict.fromkeys(matches))
            return ExtractionResult(
                success=True,
                data={field_name: unique_matches if len(unique_matches) > 1 else unique_matches[0]},
                confidence=0.7,
                strategy_used=self.name,
                metadata={"match_count": len(matches)},
            )

        return ExtractionResult(
            success=False,
            errors=[f"No matches found for '{field_name}'"],
            strategy_used=self.name,
        )

class LLMExtractionStrategy(ExtractionStrategy):
    """
    Extract using LLM for complex or ambiguous cases.

    This is the fallback strategy when structured and regex
    extraction fails.
    """

    name = "llm"
    priority = 10

    def __init__(self, llm_client=None):
        """Initialize with optional LLM client."""
        self.llm_client = llm_client

    def can_extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
    ) -> bool:
        """LLM can attempt to extract any field from chunks."""
        return bool(chunks)

    def extract(
        self,
        document_data: Optional[Dict[str, Any]],
        chunks: Optional[List[str]],
        field_name: str,
        **kwargs,
    ) -> ExtractionResult:
        """Extract using LLM inference."""
        if not chunks:
            return ExtractionResult(
                success=False,
                errors=["No chunks available for LLM extraction"],
                strategy_used=self.name,
            )

        # Build context from chunks
        context = "\n\n".join(chunks[:5])  # Limit context size

        # Build extraction prompt
        prompt = f"""Extract the following information from the document:
Field: {field_name}

Document content:
{context}

Return only the extracted value, or "NOT_FOUND" if the information is not present."""

        try:
            if self.llm_client:
                response = self.llm_client.generate(prompt)
                value = response.strip()

                if value and value.upper() != "NOT_FOUND":
                    return ExtractionResult(
                        success=True,
                        data={field_name: value},
                        confidence=0.6,
                        strategy_used=self.name,
                    )
            else:
                logger.debug("LLM client not configured, skipping LLM extraction")

        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return ExtractionResult(
                success=False,
                errors=[f"LLM extraction error: {str(e)}"],
                strategy_used=self.name,
            )

        return ExtractionResult(
            success=False,
            errors=[f"LLM could not extract '{field_name}'"],
            strategy_used=self.name,
        )

class ExtractionOrchestrator:
    """
    Coordinates multiple extraction strategies with fallback.

    Tries strategies in priority order until successful extraction
    or all strategies exhausted.
    """

    def __init__(
        self,
        strategies: Optional[List[ExtractionStrategy]] = None,
        llm_client=None,
    ):
        """
        Initialize the orchestrator.

        Args:
            strategies: Optional list of strategies. If None, uses defaults.
            llm_client: Optional LLM client for LLM-based extraction.
        """
        if strategies is None:
            self.strategies = [
                DocumentDataStrategy(),
                ChunkBasedStrategy(),
                LLMExtractionStrategy(llm_client),
            ]
        else:
            self.strategies = strategies

        # Sort by priority (highest first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

    def extract(
        self,
        field_name: str,
        document_data: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[str]] = None,
        require_success: bool = False,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract a field using available strategies.

        Args:
            field_name: The field to extract.
            document_data: Optional pre-parsed document data.
            chunks: Optional document chunks.
            require_success: If True, tries all strategies even after success.
            **kwargs: Additional arguments passed to strategies.

        Returns:
            ExtractionResult with extracted data or errors.
        """
        all_errors: List[str] = []
        strategies_tried: List[str] = []

        for strategy in self.strategies:
            if not strategy.can_extract(document_data, chunks, field_name):
                continue

            strategies_tried.append(strategy.name)
            logger.debug("Trying %s strategy for field '%s'", strategy.name, field_name)

            result = strategy.extract(
                document_data, chunks, field_name, **kwargs
            )

            if result.success:
                logger.debug(
                    "Extraction successful with %s (confidence: %.2f)",
                    strategy.name, result.confidence
                )
                return result

            all_errors.extend(result.errors)

        # All strategies failed
        return ExtractionResult(
            success=False,
            errors=all_errors,
            strategy_used=",".join(strategies_tried),
            metadata={"strategies_tried": strategies_tried},
        )

    def extract_multiple(
        self,
        field_names: List[str],
        document_data: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, ExtractionResult]:
        """
        Extract multiple fields.

        Args:
            field_names: List of fields to extract.
            document_data: Optional pre-parsed document data.
            chunks: Optional document chunks.
            **kwargs: Additional arguments passed to strategies.

        Returns:
            Dictionary mapping field names to their extraction results.
        """
        results = {}
        for field_name in field_names:
            results[field_name] = self.extract(
                field_name, document_data, chunks, **kwargs
            )
        return results

__all__ = [
    "ExtractionResult",
    "ExtractionStrategy",
    "DocumentDataStrategy",
    "ChunkBasedStrategy",
    "LLMExtractionStrategy",
    "ExtractionOrchestrator",
]
