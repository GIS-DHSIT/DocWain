from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from src.utils.payload_utils import get_canonical_text

logger = get_logger(__name__)

def _get_nested(value: Dict[str, Any], path: List[str]) -> Any:
    cur = value
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

@dataclass
class RetrievedChunk:
    profile_id: str
    doc_id: Optional[str]
    source_file: Optional[str]
    chunk_index: Optional[int]
    text: str
    payload: Dict[str, Any]
    vector: Optional[List[float]]
    point_id: Any

class QdrantRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        text_path: List[str],
        profile_path: List[str],
        profile_type: str = "str",
        with_vectors: bool = True,
        page_size: int = 128,
        max_points: Optional[int] = None,
    ):
        self.client = client
        self.collection = collection
        self.text_path = text_path
        self.profile_path = profile_path
        self.profile_type = profile_type
        self.with_vectors = with_vectors
        self.page_size = page_size
        self.max_points = max_points

    def _convert_profile_value(self, value: Any) -> Any:
        if value is None:
            return None
        if self.profile_type == "int":
            try:
                return int(value)
            except Exception:
                return value
        if self.profile_type == "float":
            try:
                return float(value)
            except Exception:
                return value
        return str(value)

    def discover_profiles(self, scan_limit: int = 2048) -> List[str]:
        profiles = set()
        count = 0
        offset = None
        while True:
            if self.max_points and count >= self.max_points:
                break
            limit = min(self.page_size, (self.max_points - count) if self.max_points else self.page_size)
            batch, offset = self.client.scroll(
                ScrollRequest(
                    collection_name=self.collection,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            if not batch:
                break
            count += len(batch)
            for pt in batch:
                payload = pt.payload or {}
                val = _get_nested(payload, self.profile_path)
                if val is not None:
                    profiles.add(str(self._convert_profile_value(val)))
            if offset is None or count >= scan_limit:
                break
        return sorted(profiles)

    def _scroll(self, filt: Optional[Filter] = None) -> Generator[RetrievedChunk, None, None]:
        offset = None
        seen = 0
        while True:
            if self.max_points and seen >= self.max_points:
                break
            limit = min(self.page_size, (self.max_points - seen) if self.max_points else self.page_size)
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=self.with_vectors,
                scroll_filter=filt,
            )
            if not batch:
                break
            seen += len(batch)
            for pt in batch:
                payload = pt.payload or {}
                text = _get_nested(payload, self.text_path)
                if not text:
                    text = get_canonical_text(payload)
                profile_val = _get_nested(payload, self.profile_path)
                profile_id = str(self._convert_profile_value(profile_val)) if profile_val is not None else "unknown"
                yield RetrievedChunk(
                    profile_id=profile_id,
                    doc_id=str(payload.get("document_id") or payload.get("doc_id") or payload.get("doc") or ""),
                    source_file=str(payload.get("source_file") or payload.get("source") or ""),
                    chunk_index=self._safe_int(payload.get("chunk_index")),
                    text=str(text) if text is not None else "",
                    payload=payload,
                    vector=pt.vector if hasattr(pt, "vector") else None,
                    point_id=pt.id,
                )
            if offset is None:
                break

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    def retrieve_profile(self, profile_id: str) -> Generator[RetrievedChunk, None, None]:
        converted = self._convert_profile_value(profile_id)
        cond = FieldCondition(
            key=".".join(self.profile_path),
            match=MatchValue(value=converted),
        )
        filt = Filter(must=[cond])
        return self._scroll(filt=filt)

    def retrieve_all(self) -> Generator[RetrievedChunk, None, None]:
        return self._scroll(filt=None)
