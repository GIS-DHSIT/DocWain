"""
Enhanced logging utilities with correlation ID support and JSON formatting.

This module provides:
- CorrelationLoggerAdapter: Logger adapter that auto-injects correlation ID
- JSONFormatter: Structured JSON log formatter for production
- configure_logging(): Application-wide logging configuration

Usage:
    from src.utils.logging_utils import get_logger, configure_logging

    # At application startup
    configure_logging(json_format=True, log_level="INFO")

    # In any module
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"user_id": "123"})
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


def get_correlation_id_from_context() -> str:
    """
    Get correlation ID from the middleware context.

    Returns:
        Correlation ID or "n/a" if not in request context.
    """
    try:
        from src.middleware.correlation import get_correlation_id
        return get_correlation_id()
    except ImportError:
        return "n/a"


def get_correlation_id(
    activity: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve a correlation id from headers/activity or generate a new one.

    This is a legacy function maintained for backward compatibility.
    For new code, use the middleware context via get_correlation_id_from_context().

    Args:
        activity: Optional Teams activity dict.
        headers: Optional request headers dict.

    Returns:
        A correlation ID string.
    """
    header_id = None
    if headers:
        header_id = (
            headers.get("x-correlation-id")
            or headers.get("x-request-id")
            or headers.get("X-Correlation-Id")
            or headers.get("X-Request-Id")
        )
    activity_id = None
    if activity:
        activity_id = activity.get("id") or activity.get("activityId")
    return str(header_id or activity_id or uuid.uuid4())


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects correlation ID into log records.

    This adapter retrieves the correlation ID from the middleware context
    and adds it to every log message.

    Usage:
        logger = CorrelationLoggerAdapter(logging.getLogger(__name__))
        logger.info("Processing request")  # correlation_id auto-added
    """

    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            logger: The underlying logger.
            extra: Optional extra fields to include in all log records.
        """
        super().__init__(logger, extra or {})

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any],
    ) -> Tuple[str, MutableMapping[str, Any]]:
        """
        Process log message to add correlation ID.

        Args:
            msg: The log message.
            kwargs: Keyword arguments passed to the logging call.

        Returns:
            Tuple of (message, kwargs) with correlation_id added.
        """
        extra = dict(self.extra) if self.extra else {}
        extra["correlation_id"] = get_correlation_id_from_context()

        # Merge with any extra provided in the log call
        if "extra" in kwargs:
            extra.update(kwargs["extra"])
        kwargs["extra"] = extra

        return msg, kwargs


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs as single-line JSON objects for easy parsing by log aggregators.

    Example output:
        {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Request processed", ...}
    """

    def __init__(
        self,
        include_extra: bool = True,
        timestamp_format: str = "iso",
    ):
        """
        Initialize the JSON formatter.

        Args:
            include_extra: Whether to include extra fields in output.
            timestamp_format: "iso" for ISO 8601, "epoch" for Unix timestamp.
        """
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data: Dict[str, Any] = {
            "timestamp": self._format_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add location info
        log_data["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            extra_fields = self._extract_extra_fields(record)
            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)

    def _format_timestamp(self, record: logging.LogRecord) -> str | float:
        """Format the timestamp based on configuration."""
        if self.timestamp_format == "epoch":
            return record.created
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat()

    def _extract_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields from the log record."""
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "correlation_id", "taskName",
        }
        return {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }


class ConsoleFormatter(logging.Formatter):
    """
    Enhanced console formatter with optional correlation ID.

    Format: [TIMESTAMP] LEVEL [correlation_id] logger - message
    """

    def __init__(self, include_correlation_id: bool = True):
        """
        Initialize the console formatter.

        Args:
            include_correlation_id: Whether to include correlation ID in output.
        """
        self.include_correlation_id = include_correlation_id
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for console output."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        correlation_id = getattr(record, "correlation_id", "n/a")

        if self.include_correlation_id:
            prefix = f"[{timestamp}] {record.levelname:<8} [{correlation_id[:8]}] {record.name}"
        else:
            prefix = f"[{timestamp}] {record.levelname:<8} {record.name}"

        message = record.getMessage()

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            return f"{prefix} - {message}\n{exc_text}"

        return f"{prefix} - {message}"


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    include_correlation_id: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Use JSON formatter for structured logging.
        include_correlation_id: Include correlation ID in log output.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Set formatter
    if json_format:
        formatter = JSONFormatter(include_extra=True)
    else:
        formatter = ConsoleFormatter(include_correlation_id=include_correlation_id)

    console_handler.setFormatter(formatter)

    # Add correlation ID filter
    try:
        from src.middleware.correlation import CorrelationIdFilter
        console_handler.addFilter(CorrelationIdFilter())
    except ImportError:
        pass

    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("pymongo.client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(
    name: str,
    correlation_id: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    Return a logger adapter that injects correlation_id into log records.

    If correlation_id is not provided, it will be automatically retrieved
    from the request context.

    Args:
        name: Logger name (usually __name__).
        correlation_id: Optional explicit correlation ID.

    Returns:
        A LoggerAdapter with correlation ID support.
    """
    if correlation_id:
        # Use simple adapter with explicit correlation ID
        extra = {"correlation_id": correlation_id}
        return logging.LoggerAdapter(logging.getLogger(name), extra)

    # Use auto-injecting adapter
    return CorrelationLoggerAdapter(logging.getLogger(name))


__all__ = [
    "get_correlation_id",
    "get_logger",
    "configure_logging",
    "CorrelationLoggerAdapter",
    "JSONFormatter",
    "ConsoleFormatter",
]
