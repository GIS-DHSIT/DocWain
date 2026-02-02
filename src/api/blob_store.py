"""Compatibility helpers for blob configuration and metadata."""

from __future__ import annotations

from src.storage.azure_blob_client import BlobInfo, extract_document_id, has_blob_credentials, is_trusted_blob


class BlobConfigurationError(ValueError):
    """Raised when blob auth config is missing or invalid."""


def blob_storage_configured() -> bool:
    return has_blob_credentials()


__all__ = [
    "BlobConfigurationError",
    "BlobInfo",
    "blob_storage_configured",
    "extract_document_id",
    "is_trusted_blob",
]
