"""
Correlation ID middleware for request tracing.

This middleware:
- Extracts correlation ID from incoming request headers (X-Correlation-Id, X-Request-Id)
- Generates a new UUID if no correlation ID is provided
- Makes the correlation ID available via context variable for logging
- Adds the correlation ID to response headers

Usage:
    from src.middleware.correlation import get_correlation_id

    # In any async context during a request
    correlation_id = get_correlation_id()
    logger.info("Processing request", extra={"correlation_id": correlation_id})
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

# Context variable to store correlation ID for the current request
correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="",
)

logger = logging.getLogger(__name__)

# Header names to check for correlation ID (in order of preference)
CORRELATION_ID_HEADERS = [
    "x-correlation-id",
    "x-request-id",
    "request-id",
    "trace-id",
]


def get_correlation_id() -> str:
    """
    Get the current correlation ID from context.

    Returns:
        The correlation ID for the current request, or "unknown" if not set.
    """
    return correlation_id_ctx.get() or "unknown"


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def extract_correlation_id(request: Request) -> Optional[str]:
    """
    Extract correlation ID from request headers.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The correlation ID if found in headers, None otherwise.
    """
    headers = request.headers
    for header_name in CORRELATION_ID_HEADERS:
        value = headers.get(header_name)
        if value:
            return value
    return None


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle correlation ID for request tracing.

    This middleware ensures every request has a correlation ID for distributed tracing.
    The ID is extracted from headers or generated, then made available via context var.
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Correlation-Id",
        generator: Optional[Callable[[], str]] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application.
            header_name: Header name to use in responses.
            generator: Optional custom correlation ID generator.
        """
        super().__init__(app)
        self.header_name = header_name
        self.generator = generator or generate_correlation_id

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """
        Process the request with correlation ID tracking.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The response with correlation ID header added.
        """
        # Extract or generate correlation ID
        correlation_id = extract_correlation_id(request) or self.generator()

        # Set in context variable
        token = correlation_id_ctx.set(correlation_id)

        # Store in request state for handler access
        request.state.correlation_id = correlation_id

        start = time.perf_counter()

        try:
            # Log request start (at debug level to avoid noise)
            logger.debug(
                "Request started: %s %s",
                request.method,
                request.url.path,
                extra={"correlation_id": correlation_id},
            )

            # Process request
            response = await call_next(request)

            # Calculate request duration
            duration_ms = (time.perf_counter() - start) * 1000

            # Add correlation ID and response time to response headers
            response.headers[self.header_name] = correlation_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

            # Log request completion at INFO level with latency
            logger.info(
                "Request completed: %s %s -> %s (%.2fms)",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                extra={"correlation_id": correlation_id, "duration_ms": duration_ms, "status_code": response.status_code},
            )

            return response

        except Exception as exc:
            # Calculate request duration
            duration_ms = (time.perf_counter() - start) * 1000

            # Log exception with correlation ID and latency
            logger.error(
                "Request failed: %s %s -> %s (%.2fms)",
                request.method,
                request.url.path,
                type(exc).__name__,
                duration_ms,
                extra={"correlation_id": correlation_id, "duration_ms": duration_ms},
                exc_info=True,
            )
            raise

        finally:
            # Reset context variable
            correlation_id_ctx.reset(token)


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to log records.

    Usage:
        handler = logging.StreamHandler()
        handler.addFilter(CorrelationIdFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the log record."""
        record.correlation_id = get_correlation_id()
        return True


__all__ = [
    "CorrelationIdMiddleware",
    "CorrelationIdFilter",
    "correlation_id_ctx",
    "get_correlation_id",
    "generate_correlation_id",
    "extract_correlation_id",
]
