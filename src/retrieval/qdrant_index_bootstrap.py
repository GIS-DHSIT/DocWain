from __future__ import annotations

from typing import Any, Dict, Iterable

from qdrant_client import QdrantClient

from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS, ensure_payload_indexes
from src.retrieval.filter_builder import build_qdrant_filter


def ensure_qdrant_indexes(
    *,
    client: QdrantClient,
    collection_name: str,
    fields: Iterable[str] = REQUIRED_PAYLOAD_INDEX_FIELDS,
    create_missing: bool = True,
) -> Dict[str, Any]:
    return ensure_payload_indexes(
        client=client,
        collection_name=collection_name,
        required_fields=list(fields),
        create_missing=create_missing,
    )


def build_filter(
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str | None = None,
    doc_domain: str | None = None,
    section_kind: str | None = None,
    chunk_kind: str | None = None,
):
    return build_qdrant_filter(
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
        doc_domain=doc_domain,
        section_kind=section_kind,
        chunk_kind=chunk_kind,
    )


__all__ = ["ensure_qdrant_indexes", "build_filter"]
