from __future__ import annotations

from typing import Optional

from qdrant_client.models import Filter

from src.api.vector_store import build_qdrant_filter as _build_qdrant_filter
from src.retrieval.errors import RetrievalFilterError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_qdrant_filter(
    *,
    subscription_id: str,
    profile_id: str,
    document_id: Optional[object] = None,
    doc_domain: Optional[object] = None,
    section_kind: Optional[object] = None,
    chunk_kind: Optional[object] = None,
) -> Filter:
    logger.debug(
        "build_qdrant_filter: subscription_id=%s, profile_id=%s, document_id=%s, doc_domain=%s, section_kind=%s, chunk_kind=%s",
        subscription_id, profile_id, document_id, doc_domain, section_kind, chunk_kind,
    )
    if not profile_id or not str(profile_id).strip():
        raise RetrievalFilterError(
            "profile_id is required for retrieval",
            code="MISSING_PROFILE_SCOPE",
            details="profile_id missing",
        )
    return _build_qdrant_filter(
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
        doc_domain=doc_domain,
        section_kind=section_kind,
        chunk_kind=chunk_kind,
    )


__all__ = ["build_qdrant_filter"]
