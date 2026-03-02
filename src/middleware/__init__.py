"""
Middleware components for the DocWain API.

This package contains FastAPI middleware for:
- Correlation ID tracking
- Request/response logging
- Performance metrics
"""

from src.middleware.correlation import (
    CorrelationIdMiddleware,
    correlation_id_ctx,
    get_correlation_id,
)

__all__ = [
    "CorrelationIdMiddleware",
    "correlation_id_ctx",
    "get_correlation_id",
]
