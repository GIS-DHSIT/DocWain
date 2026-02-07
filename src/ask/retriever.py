from __future__ import annotations

import hashlib
from typing import Any, List, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

from qdrant_client.models import FieldCondition, MatchAny

from src.api.vector_store import build_qdrant_filter
from .models import EvidenceChunk


class DocWainRetriever:
    def __init__(self, qdrant_client: Any, embedder: Any) -> None:
        self.qdrant_client = qdrant_client
        self.embedder = embedder

    def retrieve(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        top_k: int = 50,
        document_ids: Optional[List[str]] = None,
        section_ids: Optional[List[str]] = None,
        chunk_kinds: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[EvidenceChunk]:
        collection = collection_name or str(subscription_id)
        vector = self._embed(query)
        q_filter = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id), document_id=document_ids)
        if section_ids:
            q_filter.must.append(FieldCondition(key="section_id", match=MatchAny(any=[str(s) for s in section_ids if s])))
        if chunk_kinds:
            q_filter.must.append(FieldCondition(key="chunk_kind", match=MatchAny(any=[str(k) for k in chunk_kinds if k])))

        results = self.qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            using="content_vector",
            query_filter=q_filter,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(results, "points", None) or []
        return [self._to_evidence(point) for point in points if point is not None]

    def _embed(self, query: str) -> List[float]:
        if hasattr(self.embedder, "encode"):
            vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=False)
            return _vector_to_list(vec)
        raise RuntimeError("Embedder missing encode")

    @staticmethod
    def _to_evidence(point: Any) -> EvidenceChunk:
        payload = getattr(point, "payload", None) or {}
        text = (
            payload.get("canonical_text")
            or payload.get("content")
            or payload.get("text")
            or payload.get("embedding_text")
            or ""
        )
        file_name = payload.get("source_name") or (payload.get("source") or {}).get("name") or "document"
        document_id = payload.get("document_id") or ""
        section_id = payload.get("section_id") or (payload.get("section") or {}).get("id") or ""
        page = payload.get("page")
        chunk_kind = payload.get("chunk_kind") or "section_text"
        snippet = _snippet(text)
        snippet_sha = hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:12]
        return EvidenceChunk(
            text=text,
            score=float(getattr(point, "score", 0.0) or 0.0),
            metadata=payload,
            file_name=file_name,
            document_id=str(document_id),
            section_id=str(section_id),
            page=int(page) if isinstance(page, int) else None,
            chunk_kind=str(chunk_kind),
            snippet=snippet,
            snippet_sha=snippet_sha,
        )


def _snippet(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()


def _vector_to_list(vec: Any) -> List[float]:
    if torch is not None and torch.is_tensor(vec):
        return [float(v) for v in vec.detach().cpu().flatten().tolist()]
    if np is not None and isinstance(vec, np.ndarray):
        return [float(v) for v in vec.reshape(-1).tolist()]
    if isinstance(vec, list) and vec:
        first = vec[0]
        if torch is not None and torch.is_tensor(first):
            return [float(v) for v in first.detach().cpu().flatten().tolist()]
        if np is not None and isinstance(first, np.ndarray):
            return [float(v) for v in first.reshape(-1).tolist()]
        if isinstance(first, list):
            return [float(v) for v in first]
        if isinstance(first, tuple):
            return [float(v) for v in list(first)]
    if isinstance(vec, list):
        return [float(v) for v in vec]
    if isinstance(vec, tuple):
        return [float(v) for v in list(vec)]
    raise ValueError("Embedder returned unsupported vector type")


__all__ = ["DocWainRetriever"]
