from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.vector_store import build_collection_name, build_qdrant_filter
from src.retrieval.profile_document_index import ProfileDocumentIndex
from src.utils.logging_utils import get_logger
from src.utils.payload_utils import get_canonical_text, get_source_name

logger = get_logger(__name__)


@dataclass
class RetrievalPlan:
    scope: str
    target_document_ids: List[str]
    matched_document_names: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _name_variants(name: str) -> List[str]:
    raw = name.strip()
    base = raw.lower()
    no_ext = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
    variants = {raw.lower(), base}
    if len(no_ext) >= 4:
        variants.add(no_ext)
    return list(variants)


def _extract_identifier_tokens(query: str) -> List[str]:
    patterns = [
        r"\b(?:invoice|inv|po|purchase order|case|tax|vat|gst|ein|ssn)\s*[#:]\s*([A-Za-z0-9-]+)",
        r"\b(?:invoice|inv|po|purchase order|case|tax|vat|gst|ein|ssn)\s+([A-Za-z0-9-]+)",
    ]
    results: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            results.append(match)
    return list({token for token in results if token})


def route_query(user_query: str, pdi: ProfileDocumentIndex) -> RetrievalPlan:
    logger.debug("route_query: profile_id=%s, doc_count=%d", pdi.profile_id, len(pdi.document_ids))
    normalized_query = _normalize(user_query)
    matched_doc_ids: List[str] = []
    matched_names: List[str] = []
    reasons: List[str] = []

    for doc_id in pdi.document_ids:
        if doc_id and doc_id.lower() in normalized_query:
            matched_doc_ids.append(doc_id)
            reasons.append("document_id_match")

    if not matched_doc_ids:
        for doc_id, entry in pdi.documents.items():
            if not entry.source_name:
                continue
            normalized_name = _normalize(entry.source_name)
            if len(normalized_name) < 4 and "." not in normalized_name:
                continue
            for variant in _name_variants(entry.source_name):
                if variant and variant in normalized_query:
                    matched_doc_ids.append(doc_id)
                    matched_names.append(entry.source_name)
                    reasons.append("source_name_match")
                    break

    if not matched_doc_ids:
        if " in " in normalized_query or " from " in normalized_query:
            for doc_id, entry in pdi.documents.items():
                if not entry.source_name:
                    continue
                name_norm = _normalize(entry.source_name)
                if name_norm and name_norm in normalized_query:
                    matched_doc_ids.append(doc_id)
                    matched_names.append(entry.source_name)
                    reasons.append("inline_source_match")

    if not matched_doc_ids:
        identifier_tokens = _extract_identifier_tokens(user_query)
        if identifier_tokens:
            matched_doc_ids = _find_docs_with_identifiers(pdi, identifier_tokens)
            if matched_doc_ids:
                reasons.append("identifier_match")

    if matched_doc_ids:
        unique_doc_ids = sorted(set(matched_doc_ids))
        logger.debug("route_query: scope=DOCUMENT, matched_docs=%d, reasons=%s", len(unique_doc_ids), reasons)
        return RetrievalPlan(scope="DOCUMENT", target_document_ids=unique_doc_ids, matched_document_names=matched_names, reasons=reasons)

    logger.debug("route_query: scope=PROFILE, all_docs=%d", len(pdi.document_ids))
    return RetrievalPlan(
        scope="PROFILE",
        target_document_ids=list(pdi.document_ids),
        matched_document_names=[],
        reasons=["default_profile_scope"],
    )


def _find_docs_with_identifiers(pdi: ProfileDocumentIndex, identifiers: Iterable[str]) -> List[str]:
    identifiers_lower = {identifier.lower() for identifier in identifiers if identifier}
    if not identifiers_lower:
        return []
    logger.debug("_find_docs_with_identifiers: identifiers=%d", len(identifiers_lower))
    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
    collection = build_collection_name(pdi.subscription_id)
    scroll_filter = build_qdrant_filter(
        subscription_id=str(pdi.subscription_id),
        profile_id=str(pdi.profile_id),
    )
    offset: Optional[Any] = None
    matched_docs: set[str] = set()
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = get_canonical_text(payload)
            lowered = text.lower()
            if any(token in lowered for token in identifiers_lower):
                doc_id = payload.get("document_id")
                if doc_id:
                    matched_docs.add(str(doc_id))
        if next_offset is None:
            break
        offset = next_offset
    logger.debug("_find_docs_with_identifiers: matched_docs=%d", len(matched_docs))
    return sorted(matched_docs)


def fetch_document_corpus(subscription_id: str, profile_id: str, document_id: str) -> List[Dict[str, Any]]:
    logger.debug("fetch_document_corpus: subscription_id=%s, profile_id=%s, document_id=%s", subscription_id, profile_id, document_id)
    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
    collection = build_collection_name(subscription_id)
    scroll_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_id=str(document_id),
    )
    offset: Optional[Any] = None
    corpus: List[Dict[str, Any]] = []
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = get_canonical_text(payload)
            corpus.append(
                {
                    "document_id": str(document_id),
                    "chunk_id": (payload.get("chunk") or {}).get("id") or payload.get("chunk_id") or str(point.id),
                    "text": text,
                    "source_name": get_source_name(payload) or payload.get("source_name"),
                    "chunk_role": (payload.get("chunk") or {}).get("role") or payload.get("chunk_kind"),
                    "chunk_type": (payload.get("chunk") or {}).get("type") or payload.get("chunk_type"),
                    "section_title": (payload.get("section") or {}).get("title") or payload.get("section_title"),
                    "page_start": (payload.get("provenance") or {}).get("page_start") or payload.get("page_start"),
                    "page_end": (payload.get("provenance") or {}).get("page_end") or payload.get("page_end"),
                }
            )
        if next_offset is None:
            break
        offset = next_offset
    logger.debug("fetch_document_corpus: returning %d chunks", len(corpus))
    return corpus


__all__ = ["RetrievalPlan", "route_query", "fetch_document_corpus"]
