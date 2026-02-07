from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.api.vector_store import build_collection_name, build_qdrant_filter

from .types import Chunk, ChunkSource

logger = logging.getLogger(__name__)

MIN_RESULTS = 8
FALLBACK_LIMIT = 200
MAX_UNION_RESULTS = 24
LOW_SCORE_THRESHOLD = 0.2


def retrieve_chunks(
    *,
    query: str,
    raw_query: Optional[str] = None,
    subscription_id: str,
    profile_id: Optional[str],
    qdrant_client: Any,
    embedder: Any,
    document_id: Optional[str] = None,
    top_k: int = 50,
    correlation_id: Optional[str] = None,
    intent_type: Optional[str] = None,
) -> List[Chunk]:
    if not subscription_id or not str(subscription_id).strip():
        raise ValueError("subscription_id is required for retrieval")
    if not profile_id or not str(profile_id).strip():
        raise ValueError("profile_id is required for retrieval")
    raw_query = raw_query or query
    collection = build_collection_name(subscription_id)
    vector = _embed(query, embedder)
    domain = _infer_domain(raw_query)
    wants_skill_rank = _wants_skill_ranking(raw_query, intent_type)
    skill_focus = wants_skill_rank or _needs_skill_focus(raw_query)
    q_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_id=document_id,
        doc_domain=domain if domain != "generic" else None,
    )
    logger.info(
        "RAG v3 retrieval filter enforced (subscription_id=%s profile_id=%s document_id=%s)",
        subscription_id,
        profile_id,
        document_id,
        extra={"stage": "retrieve_filter", "correlation_id": correlation_id},
    )

    results = _query(qdrant_client, collection, vector, q_filter, top_k, correlation_id)
    if domain != "generic" and len(results) < MIN_RESULTS:
        fallback_filter = build_qdrant_filter(
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            document_id=document_id,
        )
        broad = _query(qdrant_client, collection, vector, fallback_filter, top_k, correlation_id, label="retrieve_broad")
        results = _merge_dedupe(results, broad)
    if _needs_expansion(results):
        expanded_query = _expand_query(raw_query)
        if expanded_query and expanded_query != query:
            expanded_vector = _embed(expanded_query, embedder)
            expanded_results = _query(qdrant_client, collection, expanded_vector, q_filter, top_k, correlation_id, label="retrieve_expand")
            results = _merge_results(results, expanded_results)

    if domain == "hr" and (wants_skill_rank or skill_focus):
        results = _boost_skill_chunks(results)
        results = _boost_section_title(results)
    results = _apply_domain_gate(results, raw_query, correlation_id)
    _log_top5(results, correlation_id, label="vector")

    if _needs_hybrid_fallback(results):
        fallback_tokens = _query_tokens(raw_query)
        if domain == "hr" and (wants_skill_rank or skill_focus):
            fallback_tokens = _merge_tokens(fallback_tokens, _skill_tokens())
        fallback = _keyword_fallback(
            qdrant_client=qdrant_client,
            collection=collection,
            q_filter=q_filter,
            tokens=fallback_tokens,
            limit=FALLBACK_LIMIT,
            correlation_id=correlation_id,
        )
        merged = _merge_dedupe(results, fallback)
        merged = _apply_domain_gate(merged, query, correlation_id)
        merged = sorted(merged, key=lambda c: c.score, reverse=True)[:MAX_UNION_RESULTS]
        _log_top5(merged, correlation_id, label="hybrid")
        return merged

    return results


def retrieve(
    *,
    query: str,
    raw_query: Optional[str] = None,
    subscription_id: str,
    profile_id: Optional[str],
    qdrant_client: Any,
    embedder: Any,
    document_id: Optional[str] = None,
    top_k: int = 50,
    correlation_id: Optional[str] = None,
    intent_type: Optional[str] = None,
) -> List[Chunk]:
    return retrieve_chunks(
        query=query,
        raw_query=raw_query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        qdrant_client=qdrant_client,
        embedder=embedder,
        document_id=document_id,
        top_k=top_k,
        correlation_id=correlation_id,
        intent_type=intent_type,
    )


def _query(
    qdrant_client: Any,
    collection: str,
    vector: List[float],
    q_filter: Any,
    top_k: int,
    correlation_id: Optional[str],
    label: str = "retrieve",
) -> List[Chunk]:
    try:
        response = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            using="content_vector",
            query_filter=q_filter,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", None) or []
        chunks = [_to_chunk(point) for point in points if point is not None]
        chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        logger.info(
            "RAG v3 retrieval %s returned %s chunks",
            label,
            len(chunks),
            extra={"stage": label, "correlation_id": correlation_id},
        )
        return chunks
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 retrieval failed: %s",
            exc,
            extra={"stage": label, "correlation_id": correlation_id},
        )
        return []


def _embed(query: str, embedder: Any) -> List[float]:
    if hasattr(embedder, "encode"):
        vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=False)
        return _to_vector(vec)
    raise RuntimeError("Embedder missing encode")


def _to_vector(vec: Any) -> List[float]:
    if vec is None:
        raise RuntimeError("Embedder returned no vector")
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, (list, tuple)):
        if not vec:
            raise RuntimeError("Embedder returned empty vector")
        first = vec[0]
        if hasattr(first, "tolist") and not isinstance(first, (int, float)):
            first = first.tolist()
        if isinstance(first, (list, tuple)):
            vec = first
        return [float(v) for v in vec]
    if isinstance(vec, (int, float)):
        return [float(vec)]
    raise RuntimeError("Unsupported embedding output type")


def _to_chunk(point: Any) -> Chunk:
    payload = getattr(point, "payload", None) or {}
    text = (
        payload.get("canonical_text")
        or payload.get("content")
        or payload.get("text")
        or payload.get("embedding_text")
        or ""
    )
    document_name = payload.get("source_name") or (payload.get("source") or {}).get("name") or "document"
    page = payload.get("page")
    chunk_id = payload.get("chunk_id") or payload.get("id") or getattr(point, "id", None) or ""
    return Chunk(
        id=str(chunk_id),
        text=text or "",
        score=float(getattr(point, "score", 0.0) or 0.0),
        source=ChunkSource(document_name=str(document_name), page=int(page) if isinstance(page, int) else None),
        meta=payload,
    )


def _needs_expansion(chunks: List[Chunk]) -> bool:
    if not chunks:
        return True
    if len(chunks) < 4:
        return True
    top_score = max((c.score for c in chunks), default=0.0)
    return top_score < 0.2


def _needs_hybrid_fallback(chunks: List[Chunk]) -> bool:
    if len(chunks) < MIN_RESULTS:
        return True
    top_score = max((c.score for c in chunks), default=0.0)
    return top_score < LOW_SCORE_THRESHOLD


def _expand_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "from", "about"}
    keywords = [t for t in tokens if t not in stop and len(t) > 2]
    if not keywords:
        return query
    deduped: List[str] = []
    seen = set()
    for tok in keywords:
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return " ".join(deduped) if deduped else query


def _merge_results(primary: List[Chunk], secondary: List[Chunk]) -> List[Chunk]:
    merged: Dict[str, Chunk] = {}
    for chunk in primary + secondary:
        if chunk.id in merged:
            if chunk.score > merged[chunk.id].score:
                merged[chunk.id] = chunk
        else:
            merged[chunk.id] = chunk
    return list(merged.values())


def _keyword_fallback(
    *,
    qdrant_client: Any,
    collection: str,
    q_filter: Any,
    tokens: List[str],
    limit: int,
    correlation_id: Optional[str],
) -> List[Chunk]:
    if not tokens:
        return []
    points = _scroll_points(qdrant_client, collection, q_filter, limit, correlation_id)
    scored: List[Chunk] = []
    for point in points:
        payload = getattr(point, "payload", None) or {}
        text = _payload_text(payload)
        score = _keyword_score(text, tokens)
        if score <= 0:
            continue
        chunk = _to_chunk(point)
        chunk.score = float(score)
        scored.append(chunk)
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:MAX_UNION_RESULTS]


def _scroll_points(
    qdrant_client: Any,
    collection: str,
    q_filter: Any,
    limit: int,
    correlation_id: Optional[str],
) -> List[Any]:
    try:
        response = qdrant_client.scroll(
            collection_name=collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            query_filter=q_filter,
            scroll_filter=q_filter,
        )
        if isinstance(response, tuple):
            points = response[0] or []
        else:
            points = getattr(response, "points", None) or response or []
        logger.info(
            "RAG v3 fallback scroll returned %s points",
            len(points),
            extra={"stage": "retrieve_fallback", "correlation_id": correlation_id},
        )
        return points
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 fallback scroll failed: %s",
            exc,
            extra={"stage": "retrieve_fallback", "correlation_id": correlation_id},
        )
        return []


def _payload_text(payload: Dict[str, Any]) -> str:
    return (
        payload.get("canonical_text")
        or payload.get("content")
        or payload.get("text")
        or payload.get("embedding_text")
        or ""
    )


def _query_tokens(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "from", "about"}
    return [tok for tok in tokens if tok not in stop and len(tok) > 2]


def _skill_tokens() -> List[str]:
    return [
        "skills",
        "technical",
        "functional",
        "tools",
        "technologies",
        "certifications",
        "certified",
        "languages",
        "frameworks",
        "platforms",
    ]


def _needs_skill_focus(query: str) -> bool:
    lowered = (query or "").lower()
    return any(
        token in lowered
        for token in (
            "skills",
            "technical skills",
            "functional skills",
            "certifications",
            "education",
            "experience summary",
            "years of experience",
        )
    )


def _merge_tokens(base: List[str], extra: List[str]) -> List[str]:
    merged = list(base)
    seen = set(base)
    for tok in extra:
        if tok in seen:
            continue
        seen.add(tok)
        merged.append(tok)
    return merged


def _wants_skill_ranking(query: str, intent_type: Optional[str]) -> bool:
    lowered = (query or "").lower()
    if intent_type in {"rank", "compare"}:
        return True
    return "rank" in lowered and "skill" in lowered


def _boost_skill_chunks(chunks: List[Chunk]) -> List[Chunk]:
    if not chunks:
        return chunks
    terms = _skill_tokens()
    for chunk in chunks:
        text = (chunk.text or "").lower()
        hits = sum(1 for term in terms if term in text)
        if hits:
            chunk.score += 0.05 * hits
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def _boost_section_title(chunks: List[Chunk]) -> List[Chunk]:
    if not chunks:
        return chunks
    for chunk in chunks:
        meta = chunk.meta or {}
        title = str(meta.get("section_title") or meta.get("section.name") or "").lower()
        if any(token in title for token in ("skill", "experience", "education", "certification")):
            chunk.score += 0.1
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def _keyword_score(text: str, tokens: List[str]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    matches = 0
    for tok in tokens:
        if tok in lowered:
            matches += 1
    if not tokens:
        return 0.0
    return matches / float(len(tokens) or 1)


def _merge_dedupe(primary: List[Chunk], secondary: List[Chunk]) -> List[Chunk]:
    merged: Dict[Tuple[str, str], Chunk] = {}
    for chunk in primary + secondary:
        key = _chunk_key(chunk)
        existing = merged.get(key)
        if existing is None or chunk.score > existing.score:
            merged[key] = chunk
    return list(merged.values())


def _chunk_key(chunk: Chunk) -> Tuple[str, str]:
    meta = chunk.meta or {}
    doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or "")
    chunk_id = str(meta.get("chunk_id") or chunk.id or "")
    return doc_id, chunk_id


def _apply_domain_gate(chunks: List[Chunk], query: str, correlation_id: Optional[str]) -> List[Chunk]:
    domain = _infer_domain(query)
    if domain == "generic" or not chunks:
        return chunks
    top5 = chunks[:5]
    mismatched = sum(1 for chunk in top5 if _chunk_domain(chunk) != domain)
    if mismatched >= 3:
        filtered = [chunk for chunk in chunks if _chunk_domain(chunk) == domain]
        if len(filtered) >= MIN_RESULTS:
            chunks = sorted(filtered, key=lambda c: c.score, reverse=True)
        else:
            for chunk in chunks:
                if _chunk_domain(chunk) != domain:
                    chunk.score *= 0.6
            chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        logger.info(
            "RAG v3 domain gate applied for %s (mismatched=%s)",
            domain,
            mismatched,
            extra={"stage": "retrieve_domain_gate", "correlation_id": correlation_id},
        )
    return chunks


def _chunk_domain(chunk: Chunk) -> str:
    meta = chunk.meta or {}
    return str(meta.get("doc_domain") or meta.get("doc_type") or meta.get("document.type") or "").lower()


def _infer_domain(query: str) -> str:
    lowered = (query or "").lower()
    if any(token in lowered for token in ("invoice", "amount due", "subtotal", "billing", "payment terms")):
        return "invoice"
    if any(token in lowered for token in ("resume", "cv", "candidate", "experience", "education")):
        return "hr"
    if any(token in lowered for token in ("agreement", "contract", "clause", "warranty", "liability")):
        return "legal"
    return "generic"


def _log_top5(chunks: List[Chunk], correlation_id: Optional[str], label: str) -> None:
    previews = []
    for chunk in chunks[:5]:
        meta = chunk.meta or {}
        doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
        chunk_kind = meta.get("chunk_kind") or meta.get("chunk_type") or ""
        text = " ".join((chunk.text or "").split())[:120]
        previews.append(f"({chunk.score:.4f}, {doc_id}, {chunk_kind}, {text})")
    logger.info(
        "RAG v3 retrieval top5 %s: %s",
        label,
        "; ".join(previews),
        extra={"stage": "retrieve_top5", "correlation_id": correlation_id},
    )
