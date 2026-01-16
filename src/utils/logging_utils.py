import logging
import uuid
from typing import Any, Dict, Optional


def get_correlation_id(
    activity: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve a correlation id from headers/activity or generate a new one."""
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


def get_logger(name: str, correlation_id: Optional[str] = None) -> logging.LoggerAdapter:
    """Return a logger adapter that injects correlation_id into log records."""
    extra = {"correlation_id": correlation_id or "n/a"}
    return logging.LoggerAdapter(logging.getLogger(name), extra)
