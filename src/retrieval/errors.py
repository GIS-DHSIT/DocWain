from __future__ import annotations


class RetrievalFilterError(RuntimeError):
    """Raised when a filtered retrieval fails (e.g., missing payload indexes)."""

    def __init__(self, message: str, *, code: str = "RETRIEVAL_FILTER_FAILED", details: str | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or ""


__all__ = ["RetrievalFilterError"]
