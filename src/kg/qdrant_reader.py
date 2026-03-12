import hashlib
from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.utils.payload_utils import get_document_type, get_source_name

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from src.api.config import Config

logger = get_logger(__name__)

TEXT_FIELD_CANDIDATES = [
    "text",
    "content",
    "page_content",
    "chunk",
    "chunk_text",
    "document_text",
    "raw_text",
]

@dataclass(frozen=True)
class QdrantChunk:
    chunk_id: str
    chunk_hash: Optional[str]
    text: str
    document_id: str
    filename: Optional[str]
    source_file: Optional[str]
    section_title: Optional[str]
    section_path: Optional[str]
    chunk_index: Optional[int]
    page_start: Optional[int]
    page_end: Optional[int]
    doc_type: Optional[str]
    document_type: Optional[str]
    chunk_char_len: Optional[int]
    prev_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    subscription_id: Optional[str]
    profile_id: Optional[str]

@dataclass(frozen=True)
class QdrantBatch:
    points: List[QdrantChunk]
    next_offset: Optional[Any]

def _get_first(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None

def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _normalize_payload(point_id: Any, payload: Dict[str, Any]) -> Optional[QdrantChunk]:
    if payload is None:
        payload = {}

    document_id = _get_first(payload, ["document_id", "doc_id", "documentId", "documentID"])
    if document_id is None:
        logger.debug("Skipping payload without document_id (point_id=%s)", point_id)
        return None

    text_val = _get_first(payload, TEXT_FIELD_CANDIDATES) or ""
    chunk_id = _get_first(payload, ["chunk_id", "chunkId", "chunkID"]) or point_id
    if chunk_id is None:
        logger.debug("Skipping payload without chunk_id (document_id=%s)", document_id)
        return None

    chunk_hash = _get_first(payload, ["chunk_hash", "chunkHash"])
    if not chunk_hash:
        chunk_hash = (payload.get("chunk") or {}).get("hash")
    if not chunk_hash and text_val:
        chunk_hash = hashlib.sha256(str(text_val).encode("utf-8")).hexdigest()

    chunk_index = _coerce_int(_get_first(payload, ["chunk_index", "chunkIndex"]))
    page_start = _coerce_int(_get_first(payload, ["page_start", "pageStart", "page"]))
    page_end = _coerce_int(_get_first(payload, ["page_end", "pageEnd", "page"]))

    chunk_char_len = _coerce_int(_get_first(payload, ["chunk_char_len", "chunkCharLen"]))
    if chunk_char_len is None and text_val:
        chunk_char_len = len(text_val)

    return QdrantChunk(
        chunk_id=str(chunk_id),
        chunk_hash=str(chunk_hash) if chunk_hash else None,
        text=str(text_val or ""),
        document_id=str(document_id),
        filename=get_source_name(payload),
        source_file=get_source_name(payload),
        section_title=_get_first(payload, ["section_title", "sectionTitle", "section"]),
        section_path=_get_first(payload, ["section_path", "sectionPath"]),
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        doc_type=get_document_type(payload),
        document_type=get_document_type(payload),
        chunk_char_len=chunk_char_len,
        prev_chunk_id=_get_first(payload, ["prev_chunk_id", "prevChunkId"]),
        next_chunk_id=_get_first(payload, ["next_chunk_id", "nextChunkId"]),
        subscription_id=_get_first(payload, ["subscription_id", "subscriptionId"]),
        profile_id=_get_first(payload, ["profile_id", "profileId"]),
    )

class QdrantKGReader:
    def __init__(self, collection_name: str, client: Optional[QdrantClient] = None):
        if not collection_name:
            raise ValueError("collection_name is required")
        self.collection_name = collection_name
        self.client = client or QdrantClient(
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
            timeout=120,
        )

    def scroll_batches(
        self,
        *,
        batch_size: int,
        offset: Optional[Any] = None,
        max_points: Optional[int] = None,
        scroll_filter: Optional[Filter] = None,
    ) -> Iterable[QdrantBatch]:
        processed = 0
        current_offset = offset
        while True:
            if max_points is not None and processed >= max_points:
                break
            limit = batch_size
            if max_points is not None:
                limit = max(0, min(batch_size, max_points - processed))
            if limit <= 0:
                break
            try:
                batch, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=current_offset,
                    with_vectors=False,
                    with_payload=True,
                    scroll_filter=scroll_filter,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Qdrant scroll failed: %s", exc)
                raise

            if not batch:
                break

            normalized: List[QdrantChunk] = []
            for point in batch:
                point_payload = getattr(point, "payload", None) or {}
                point_id = getattr(point, "id", None)
                normalized_point = _normalize_payload(point_id, point_payload)
                if normalized_point is None:
                    continue
                normalized.append(normalized_point)

            yield QdrantBatch(points=normalized, next_offset=next_offset)
            processed += len(batch)
            current_offset = next_offset
            if next_offset is None:
                break
