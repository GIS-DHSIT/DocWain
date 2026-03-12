"""
Error Handling for DocWain RAG Pipeline.

This module provides:
- ErrorCategory: Enum for classifying errors
- RAGError: Structured error class with context
- @with_error_handling: Decorator for consistent error handling
- @with_retry: Decorator for retrying transient failures
- User-friendly fallback responses
"""

from __future__ import annotations

import functools
import logging

from src.utils.logging_utils import get_logger
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = get_logger(__name__)

T = TypeVar("T")

class ErrorCategory(Enum):
    """Categories of errors that can occur in the RAG pipeline."""

    # Retrieval errors
    RETRIEVAL_FAILED = "retrieval_failed"
    NO_RESULTS = "no_results"
    LOW_CONFIDENCE = "low_confidence"

    # LLM errors
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMITED = "llm_rate_limited"
    LLM_INVALID_RESPONSE = "llm_invalid_response"

    # Database errors
    QDRANT_ERROR = "qdrant_error"
    MONGODB_ERROR = "mongodb_error"
    REDIS_ERROR = "redis_error"

    # Input errors
    INVALID_QUERY = "invalid_query"
    MISSING_CONTEXT = "missing_context"
    INVALID_PROFILE = "invalid_profile"

    # Processing errors
    EXTRACTION_FAILED = "extraction_failed"
    PARSING_ERROR = "parsing_error"
    TIMEOUT = "timeout"

    # Unknown
    UNKNOWN = "unknown"

@dataclass
class RAGError:
    """
    Structured error with context for RAG pipeline failures.

    Provides consistent error representation with:
    - Category for programmatic handling
    - User-friendly message
    - Technical details for debugging
    - Recovery suggestions
    """

    category: ErrorCategory
    message: str
    user_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    retry_after: Optional[float] = None
    correlation_id: Optional[str] = None
    traceback: Optional[str] = None

    def __post_init__(self):
        """Set default user message if not provided."""
        if self.user_message is None:
            self.user_message = self._default_user_message()

    def _default_user_message(self) -> str:
        """Generate a user-friendly message based on category."""
        messages = {
            ErrorCategory.RETRIEVAL_FAILED: (
                "I'm having trouble searching the documents right now. "
                "Please try again in a moment."
            ),
            ErrorCategory.NO_RESULTS: (
                "I couldn't find relevant information in the documents. "
                "Could you try rephrasing your question?"
            ),
            ErrorCategory.LOW_CONFIDENCE: (
                "I found some information but I'm not confident about the answer. "
                "You may want to verify this manually."
            ),
            ErrorCategory.LLM_UNAVAILABLE: (
                "The AI service is temporarily unavailable. "
                "Please try again in a few moments."
            ),
            ErrorCategory.LLM_TIMEOUT: (
                "The request took too long to process. "
                "Please try a simpler question or try again later."
            ),
            ErrorCategory.LLM_RATE_LIMITED: (
                "We're experiencing high demand. "
                "Please wait a moment before trying again."
            ),
            ErrorCategory.INVALID_QUERY: (
                "I couldn't understand your question. "
                "Could you rephrase it?"
            ),
            ErrorCategory.MISSING_CONTEXT: (
                "I need more context to answer this question. "
                "Could you provide more details?"
            ),
            ErrorCategory.TIMEOUT: (
                "The request timed out. Please try again."
            ),
            ErrorCategory.UNKNOWN: (
                "Something went wrong. Please try again later."
            ),
        }
        return messages.get(self.category, messages[ErrorCategory.UNKNOWN])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        correlation_id: Optional[str] = None,
        include_traceback: bool = False,
    ) -> "RAGError":
        """
        Create a RAGError from an exception.

        Args:
            exc: The exception to convert.
            correlation_id: Optional correlation ID for tracing.
            include_traceback: Whether to include the traceback.

        Returns:
            A RAGError instance.
        """
        # Determine category based on exception type
        category = cls._categorize_exception(exc)

        return cls(
            category=category,
            message=str(exc),
            correlation_id=correlation_id,
            traceback=traceback.format_exc() if include_traceback else None,
            details={"exception_type": type(exc).__name__},
        )

    @staticmethod
    def _categorize_exception(exc: Exception) -> ErrorCategory:
        """Categorize an exception."""
        exc_type = type(exc).__name__.lower()
        exc_msg = str(exc).lower()

        # Timeout detection
        if "timeout" in exc_type or "timeout" in exc_msg:
            return ErrorCategory.TIMEOUT

        # Rate limiting
        if "rate" in exc_msg and ("limit" in exc_msg or "exceeded" in exc_msg):
            return ErrorCategory.LLM_RATE_LIMITED

        # Connection errors
        if "connection" in exc_type or "connection" in exc_msg:
            if "qdrant" in exc_msg:
                return ErrorCategory.QDRANT_ERROR
            if "mongo" in exc_msg:
                return ErrorCategory.MONGODB_ERROR
            if "redis" in exc_msg:
                return ErrorCategory.REDIS_ERROR
            return ErrorCategory.LLM_UNAVAILABLE

        return ErrorCategory.UNKNOWN

class RAGException(Exception):
    """Exception wrapper for RAGError."""

    def __init__(self, error: RAGError):
        self.error = error
        super().__init__(error.message)

def with_error_handling(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    user_message: Optional[str] = None,
    log_level: int = logging.ERROR,
    reraise: bool = False,
    default_return: Any = None,
):
    """
    Decorator for consistent error handling in RAG pipeline functions.

    Args:
        category: Default error category if exception occurs.
        user_message: User-friendly message to include in error.
        log_level: Logging level for errors.
        reraise: Whether to re-raise the exception after handling.
        default_return: Value to return on error if not reraising.

    Example:
        @with_error_handling(category=ErrorCategory.RETRIEVAL_FAILED)
        def search_documents(query: str) -> List[Document]:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, RAGError]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, RAGError]:
            try:
                return func(*args, **kwargs)
            except RAGException:
                raise  # Re-raise RAG exceptions as-is
            except Exception as exc:
                error = RAGError(
                    category=category,
                    message=f"{func.__name__} failed: {exc}",
                    user_message=user_message,
                    details={
                        "function": func.__name__,
                        "exception_type": type(exc).__name__,
                    },
                    traceback=traceback.format_exc(),
                )
                logger.log(
                    log_level,
                    "Error in %s: %s",
                    func.__name__,
                    error.message,
                    extra={"error": error.to_dict()},
                )
                if reraise:
                    raise RAGException(error) from exc
                return default_return if default_return is not None else error
        return wrapper
    return decorator

def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 0.5,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying functions on transient failures.

    Args:
        max_attempts: Maximum number of attempts.
        backoff_factor: Multiplier for exponential backoff.
        retryable_exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback called on each retry.

    Example:
        @with_retry(max_attempts=3, backoff_factor=0.5)
        def call_external_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt < max_attempts:
                        wait_time = backoff_factor * (2 ** (attempt - 1))
                        logger.warning(
                            "Attempt %d/%d failed for %s, retrying in %.2fs: %s",
                            attempt, max_attempts, func.__name__, wait_time, exc
                        )
                        if on_retry:
                            on_retry(exc, attempt)
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts, func.__name__, exc
                        )
            raise last_exception
        return wrapper
    return decorator

# Pre-defined fallback responses for common scenarios
FALLBACK_RESPONSES: Dict[ErrorCategory, str] = {
    ErrorCategory.NO_RESULTS: (
        "I searched through the available documents but couldn't find "
        "information directly related to your question. Could you try "
        "asking in a different way, or let me know if there's a specific "
        "document you'd like me to focus on?"
    ),
    ErrorCategory.LOW_CONFIDENCE: (
        "Based on the available information, I can provide a partial answer, "
        "but I'm not fully confident in its accuracy. Please verify this "
        "information independently before making any decisions."
    ),
    ErrorCategory.RETRIEVAL_FAILED: (
        "I'm experiencing some technical difficulties accessing the document "
        "database. Please try your question again in a moment. If the problem "
        "persists, you may want to contact support."
    ),
    ErrorCategory.LLM_UNAVAILABLE: (
        "The AI service is currently unavailable. Your documents are safe, "
        "and I'll be able to help you as soon as the service is restored. "
        "Please try again in a few minutes."
    ),
    ErrorCategory.TIMEOUT: (
        "Your request took longer than expected to process. This might happen "
        "with complex questions. Please try breaking your question into smaller "
        "parts or try again."
    ),
}

def get_fallback_response(
    category: ErrorCategory,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get a user-friendly fallback response for an error category.

    Args:
        category: The error category.
        context: Optional context for customizing the response.

    Returns:
        A user-friendly response string.
    """
    base_response = FALLBACK_RESPONSES.get(
        category,
        "I encountered an unexpected issue. Please try again later."
    )

    # Customize based on context if provided
    if context:
        query = context.get("query", "")
        if query and category == ErrorCategory.NO_RESULTS:
            base_response = (
                f"I couldn't find information about \"{query[:50]}...\" "
                "in the available documents. Would you like to try a different "
                "question or search in a specific document?"
            )

    return base_response

__all__ = [
    "ErrorCategory",
    "RAGError",
    "RAGException",
    "with_error_handling",
    "with_retry",
    "get_fallback_response",
    "FALLBACK_RESPONSES",
]
