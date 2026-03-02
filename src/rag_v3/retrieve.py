from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from src.api.vector_store import build_collection_name, build_qdrant_filter

from .types import Chunk, ChunkSource

logger = logging.getLogger(__name__)

# Retrieval configuration - tuned for accuracy over recall
MIN_RESULTS = 6  # Reduced: prefer quality over quantity
FALLBACK_LIMIT = 100  # Reduced from 200
MAX_UNION_RESULTS = 16  # Reduced from 24
LOW_SCORE_THRESHOLD = 0.45  # Increased from 0.2 - reject low-quality matches
HIGH_SCORE_THRESHOLD = 0.7  # New: high-confidence threshold
MAX_EXPANDED_DOCS = 3
MAX_DOC_CHUNKS = 15  # Reduced from 20
MAX_FULL_SCAN_DOCS = 2  # Reduced from 3
MAX_FULL_SCAN_CHUNKS = 40  # Reduced from 160 - critical fix for noise reduction
MAX_PROFILE_SCAN_CHUNKS = 500  # Reduced from 5000
MAX_UNSCOPED_SCAN_CHUNKS = 200  # Reduced from 2000
SECTION_KEYWORDS_BY_DOMAIN = {
    "hr": ["experience", "summary", "skills", "certification", "education", "project"],
    "invoice": ["total", "amount", "payment", "invoice", "bill to", "due"],
    "legal": ["clause", "section", "term", "liability", "warranty"],
    "generic": [],
}


_QUERY_SECTION_MAP = [
    (("skills", "technical skills", "tech stack", "tools", "technologies", "frameworks", "programming"), "skills_technical"),
    (("soft skills", "functional skills", "communication", "leadership"), "skills_functional"),
    (("education", "degree", "university", "academic", "qualification"), "education"),
    (("certification", "certified", "credential", "license"), "certifications"),
    (("experience", "work history", "employment", "career", "job history"), "experience"),
    (("summary", "objective", "profile", "overview", "about"), "summary_objective"),
    (("contact", "email", "phone", "address"), "identity_contact"),
    (("achievement", "award", "accomplishment"), "achievements"),
]


def _infer_query_section_kind(query: str) -> Optional[str]:
    lowered = (query or "").lower()
    for keywords, kind in _QUERY_SECTION_MAP:
        if any(kw in lowered for kw in keywords):
            return kind
    return None


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

    # Enrich query for better matching with context-enriched embeddings
    embedding_query = _enrich_query_for_embedding(query, correlation_id)
    vector = _embed(embedding_query, embedder)
    wants_skill_rank = _wants_skill_ranking(raw_query, intent_type)
    skill_focus = wants_skill_rank or _needs_skill_focus(raw_query)
    q_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_id=document_id,
    )
    logger.info(
        "RAG v3 retrieval filter enforced (subscription_id=%s profile_id=%s document_id=%s)",
        subscription_id,
        profile_id,
        document_id,
        extra={"stage": "retrieve_filter", "correlation_id": correlation_id},
    )

    results = _query(qdrant_client, collection, vector, q_filter, top_k, correlation_id)
    if _needs_expansion(results):
        expanded_query = _expand_query(raw_query)
        if expanded_query and expanded_query != query:
            expanded_vector = _embed(expanded_query, embedder)
            expanded_results = _query(qdrant_client, collection, expanded_vector, q_filter, top_k, correlation_id, label="retrieve_expand")
            results = _merge_results(results, expanded_results)

    if wants_skill_rank or skill_focus:
        results = _boost_skill_chunks(results)
        results = _boost_section_title(results)
    results = _boost_by_section_kind(results, raw_query)
    _log_top5(results, correlation_id, label="vector")

    if _needs_hybrid_fallback(results):
        fallback_tokens = _query_tokens(raw_query)
        if wants_skill_rank or skill_focus:
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
        merged = sorted(merged, key=lambda c: c.score, reverse=True)[:MAX_UNION_RESULTS]
        _log_top5(merged, correlation_id, label="hybrid")
        results = merged

    results = _expand_by_document(
        qdrant_client=qdrant_client,
        collection=collection,
        base_chunks=results,
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        correlation_id=correlation_id,
        domain="generic",
    )

    # Single defense-in-depth scope check (Qdrant filters already enforce scoping)
    results = [c for c in results
               if str((c.meta or {}).get("profile_id") or "") == str(profile_id)
               and str((c.meta or {}).get("subscription_id") or "") == str(subscription_id)]

    # Apply quality filtering pipeline for accuracy
    results = apply_quality_pipeline(results, raw_query, correlation_id)

    return results


def retrieve_entity_scoped(
    query: str,
    entity_name: str,
    collection: str,
    subscription_id: str,
    profile_id: str,
    *,
    top_k: int = 20,
    embedder: Any = None,
    qdrant_client: Any = None,
    correlation_id: str = "",
) -> List[Chunk]:
    """Retrieve chunks scoped to documents mentioning a specific entity.

    First finds documents containing the entity name, then runs vector search
    restricted to those documents. Falls back to unscoped search if no
    entity-matching documents are found.
    """
    if qdrant_client is None or embedder is None:
        raise ValueError("qdrant_client and embedder are required")

    base_filter = build_qdrant_filter(subscription_id, profile_id)

    # Step 1: Find documents containing the entity name via scroll + text match
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=collection,
            scroll_filter=base_filter,
            limit=200,
            with_payload=["document_id", "source_name", "canonical_text", "embedding_text"],
        )
        points = scroll_result[0] if scroll_result else []
    except Exception:  # noqa: BLE001
        points = []

    entity_lower = entity_name.lower()
    matching_doc_ids: set = set()
    for pt in points:
        payload = getattr(pt, "payload", None) or {}
        text = (payload.get("canonical_text") or payload.get("embedding_text") or "").lower()
        source = (payload.get("source_name") or "").lower()
        if entity_lower in text or entity_lower in source:
            doc_id = payload.get("document_id")
            if doc_id:
                matching_doc_ids.add(str(doc_id))

    # Step 2: Vector search scoped to matching documents
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    if matching_doc_ids:
        entity_filter = Filter(must=[
            FieldCondition(key="subscription_id", match=MatchValue(value=subscription_id)),
            FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
            FieldCondition(key="document_id", match=MatchAny(any=list(matching_doc_ids))),
        ])
    else:
        entity_filter = base_filter

    try:
        query_vector = _embed(query, embedder)
    except Exception:  # noqa: BLE001
        return []

    try:
        search_result = qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            using="content_vector",
            query_filter=entity_filter,
            limit=top_k,
            with_payload=True,
        )
        result_points = getattr(search_result, "points", None) or []
    except Exception:  # noqa: BLE001
        result_points = []

    return [_to_chunk(pt) for pt in result_points if _to_chunk(pt).text.strip()]


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


def expand_full_scan_by_document(
    *,
    qdrant_client: Any,
    collection: str,
    base_chunks: List[Chunk],
    subscription_id: str,
    profile_id: str,
    correlation_id: Optional[str],
    max_docs: int = MAX_FULL_SCAN_DOCS,
    max_chunks_per_doc: int = MAX_FULL_SCAN_CHUNKS,
) -> List[Chunk]:
    doc_ids = []
    for chunk in base_chunks:
        doc_id = _chunk_doc_id(chunk)
        if not doc_id:
            continue
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    doc_ids = doc_ids[:max_docs]
    if not doc_ids:
        return base_chunks

    def _fetch(doc_id: str) -> List[Chunk]:
        q_filter = build_qdrant_filter(
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=doc_id,
        )
        collected: List[Any] = []
        offset = None
        remaining = max_chunks_per_doc
        while remaining > 0:
            limit = min(remaining, 64)
            try:
                response, offset = qdrant_client.scroll(
                    collection_name=collection,
                    scroll_filter=q_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "RAG v3 full scan failed for doc_id=%s: %s",
                    doc_id,
                    exc,
                    extra={"stage": "retrieve_full_scan", "correlation_id": correlation_id},
                )
                break
            if not response:
                break
            collected.extend(response)
            remaining -= len(response)
            if offset is None:
                break
        chunks = [_to_chunk(point) for point in collected if point is not None]
        return chunks

    expanded: List[Chunk] = []
    with ThreadPoolExecutor(max_workers=min(4, len(doc_ids))) as executor:
        futures = {executor.submit(_fetch, doc_id): doc_id for doc_id in doc_ids}
        for future in as_completed(futures):
            try:
                expanded.extend(future.result() or [])
            except Exception:
                continue

    merged = _merge_dedupe(base_chunks, expanded)
    merged = sorted(merged, key=lambda c: c.score, reverse=True)[:MAX_UNION_RESULTS]
    _log_top5(merged, correlation_id, label="full_scan")
    return merged


def expand_full_scan_by_profile(
    *,
    qdrant_client: Any,
    collection: str,
    subscription_id: str,
    profile_id: str,
    correlation_id: Optional[str],
    max_chunks: int = MAX_PROFILE_SCAN_CHUNKS,
) -> List[Chunk]:
    q_filter = build_qdrant_filter(
        subscription_id=subscription_id,
        profile_id=profile_id,
    )
    collected: List[Any] = []
    offset = None
    remaining = max_chunks
    while remaining > 0:
        limit = min(remaining, 64)
        try:
            response, offset = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=q_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "RAG v3 profile scan failed: %s",
                exc,
                extra={"stage": "retrieve_profile_scan", "correlation_id": correlation_id},
            )
            break
        if not response:
            break
        collected.extend(response)
        remaining -= len(response)
        if offset is None:
            break
    chunks = [_to_chunk(point) for point in collected if point is not None]
    chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    # Fallback: if filtered scroll returned 0 chunks, try unscoped scroll
    # + Python-side profile filtering.  This handles Qdrant payload field-name
    # mismatches (e.g. the filter checks 'profile_id' but the payload stores
    # 'profileId') that cause the server-side filter to miss all points.
    if not chunks:
        logger.warning(
            "Profile scan filter returned 0 chunks from %s; trying unscoped fallback",
            collection,
            extra={"stage": "retrieve_profile_scan_fallback", "correlation_id": correlation_id},
        )
        all_points = _scroll_unscoped_raw(
            qdrant_client, collection, max_chunks=min(max_chunks, 200), correlation_id=correlation_id,
        )
        unscoped_chunks = [_to_chunk(p) for p in all_points if p is not None]
        chunks = filter_chunks_by_profile_scope(
            unscoped_chunks, profile_id=profile_id, subscription_id=subscription_id,
        )
        if chunks:
            logger.info(
                "Unscoped fallback recovered %d chunks for profile %s",
                len(chunks), profile_id,
                extra={"stage": "retrieve_profile_scan_fallback", "correlation_id": correlation_id},
            )
        chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    _log_top5(chunks, correlation_id, label="profile_scan")
    return chunks


def _scroll_unscoped_raw(
    qdrant_client: Any,
    collection: str,
    max_chunks: int = 200,
    correlation_id: Optional[str] = None,
) -> List[Any]:
    """Scroll collection without filter — returns raw Qdrant points."""
    collected: List[Any] = []
    offset = None
    remaining = max_chunks
    while remaining > 0:
        limit = min(remaining, 64)
        try:
            response, offset = qdrant_client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "RAG v3 unscoped raw scroll failed: %s", exc,
                extra={"stage": "retrieve_unscoped_raw", "correlation_id": correlation_id},
            )
            break
        if not response:
            break
        collected.extend(response)
        remaining -= len(response)
        if offset is None:
            break
    return collected


def expand_full_scan_unscoped(
    *,
    qdrant_client: Any,
    collection: str,
    correlation_id: Optional[str],
    max_chunks: int = MAX_UNSCOPED_SCAN_CHUNKS,
) -> List[Chunk]:
    collected: List[Any] = []
    offset = None
    remaining = max_chunks
    while remaining > 0:
        limit = min(remaining, 64)
        try:
            response, offset = qdrant_client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "RAG v3 unscoped scan failed: %s",
                exc,
                extra={"stage": "retrieve_unscoped_scan", "correlation_id": correlation_id},
            )
            break
        if not response:
            break
        collected.extend(response)
        remaining -= len(response)
        if offset is None:
            break
    chunks = [_to_chunk(point) for point in collected if point is not None]
    chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    _log_top5(chunks, correlation_id, label="unscoped_scan")
    return chunks


def filter_chunks_by_profile_scope(
    chunks: List[Chunk],
    *,
    profile_id: str,
    subscription_id: str,
) -> List[Chunk]:
    if not chunks:
        return chunks
    profile_id = str(profile_id)
    subscription_id = str(subscription_id)

    def _extract_values(payload: Dict[str, Any], keys: List[str]) -> List[str]:
        values = []
        for key in keys:
            val = payload.get(key)
            if val is not None and str(val).strip():
                values.append(str(val))
        return values

    def _nested_value(payload: Dict[str, Any], key: str, subkey: str) -> Optional[str]:
        nested = payload.get(key)
        if isinstance(nested, dict):
            val = nested.get(subkey)
            if val is not None and str(val).strip():
                return str(val)
        return None

    profile_keys = ["profile_id", "profileId", "profileID", "profile.id"]
    subscription_keys = ["subscription_id", "subscriptionId", "subscriptionID", "subscription.id"]

    any_profile_key = False
    any_subscription_key = False
    scoped: List[Chunk] = []
    for chunk in chunks:
        payload = chunk.meta or {}
        profile_values = _extract_values(payload, profile_keys)
        subscription_values = _extract_values(payload, subscription_keys)
        nested_profile = _nested_value(payload, "profile", "id")
        nested_subscription = _nested_value(payload, "subscription", "id")
        if nested_profile:
            profile_values.append(nested_profile)
        if nested_subscription:
            subscription_values.append(nested_subscription)

        if profile_values:
            any_profile_key = True
        if subscription_values:
            any_subscription_key = True

        profile_ok = not profile_values or profile_id in profile_values
        subscription_ok = not subscription_values or subscription_id in subscription_values
        if profile_ok and subscription_ok:
            scoped.append(chunk)

    if any_profile_key or any_subscription_key:
        return scoped
    # No metadata keys found — chunks lack profile scoping.
    # Return empty to prevent cross-profile leakage.
    logger.warning(
        "filter_chunks_by_profile_scope: no profile/subscription keys in chunk metadata; "
        "returning empty to prevent cross-profile leakage (chunks=%d)",
        len(chunks),
    )
    return []


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


def _enrich_query_for_embedding(query: str, correlation_id: Optional[str] = None) -> str:
    """Return query as-is.

    Section prefixes were removed to match the rebuilt embedding format
    where prefixes are only added for high-confidence title-derived kinds.
    Symmetric query/embedding treatment maximizes cosine similarity.
    """
    return query


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


_METADATA_GARBAGE_MARKERS = (
    "'chunk_type':", "'section_id':", "'section_title':", "'page': None",
    "section_id :", "chunk_type :", "section_title :",
    "Chunk Candidate text", "chunk_candidates Chunk",
    # Space-delimited format (no colon between key and value)
    ", section_id ", ", chunk_type ", ", section_title ",
    ", start_page ", ", end_page ", "text : ",
)
# Strong markers: a single occurrence means garbage (never in real document text)
_STRONG_GARBAGE_MARKERS = (
    "Extracted Document full_text",
    "Extracted Document (full_text=",
    "ExtractedDocument(",
    "Section section_id",
    "start_page 1, end_page",
    "Chunk Candidate text",
)
_EXTRACTED_DOC_REPR_RE = re.compile(r"^Extracted\s*Document\s*\(\s*full_text='", re.IGNORECASE)
_EXTRACTED_DOC_FULLTEXT_RE = re.compile(r"full_text='(.*?)(?:',\s*\w+=|'\s*\)$)", re.DOTALL)


def _is_metadata_garbage(text: str) -> bool:
    """Detect text that is actually stringified chunk metadata dicts or ExtractedDocument repr."""
    if not text or len(text) < 30:
        return False
    if _EXTRACTED_DOC_REPR_RE.match(text):
        return True
    if any(m in text for m in _STRONG_GARBAGE_MARKERS):
        return True
    # Multiple "text :" or ", text " segments = metadata key contamination
    if text.count("text : ") >= 2 or text.count(", text ") >= 2:
        return True
    return sum(1 for marker in _METADATA_GARBAGE_MARKERS if marker in text) >= 2


def _extract_text_from_repr(text: str) -> str:
    """Extract actual document text from ExtractedDocument repr string."""
    if not text:
        return ""
    m = _EXTRACTED_DOC_FULLTEXT_RE.search(text)
    if m:
        extracted = m.group(1).replace("\\n", "\n").strip()
        if extracted and len(extracted) > 20:
            return extracted
    # Fallback: strip the "Extracted Document (full_text='" prefix and trailing "')"
    if _EXTRACTED_DOC_REPR_RE.match(text):
        stripped = re.sub(r"^Extracted\s*Document\s*\(\s*full_text='", "", text)
        stripped = re.sub(r"'\s*\)\s*$", "", stripped)
        return stripped.replace("\\n", "\n").strip()
    return ""


def _salvage_text_from_garbage(payload: dict) -> str:
    """Try to extract actual text content from a garbage payload."""
    # Try extracting from ExtractedDocument repr in any field
    for key in ("canonical_text", "embedding_text"):
        val = payload.get(key) or ""
        if val and _EXTRACTED_DOC_REPR_RE.match(val):
            extracted = _extract_text_from_repr(val)
            if extracted:
                return extracted

    # Try embedding_text after stripping section prefix and metadata fragments
    emb = payload.get("embedding_text") or ""
    if emb:
        emb_clean = _strip_embedding_prefix(emb)
        # After prefix strip, check if it's now an ExtractedDocument repr
        if _EXTRACTED_DOC_REPR_RE.match(emb_clean):
            extracted = _extract_text_from_repr(emb_clean)
            if extracted and len(extracted) > 20:
                return extracted
        emb_clean = _clean_metadata_from_text(emb_clean)
        # Also strip ExtractedDocument(...) wrapper
        emb_clean = re.sub(r"ExtractedDocument\(", "", emb_clean)
        emb_clean = re.sub(r"^\[.*?\]\s*(?:Section\s+\d+\s*:\s*(?:[\w\s]+:\s*)?)?", "", emb_clean).strip()
        if emb_clean and len(emb_clean) > 20 and not _is_metadata_garbage(emb_clean):
            return emb_clean

    for key in ("text_raw", "text_clean", "text"):
        val = payload.get(key)
        if val and isinstance(val, str) and not _is_metadata_garbage(val):
            return val
    return ""


_EMBEDDING_PREFIX_RE = re.compile(r"^\[[\w\s]+\]\s*(?:Section\s*\d+\s*:\s*)?(?:[\w\s&/,-]+:\s*)?")

_METADATA_KEY_RE = re.compile(
    r"^(?:section_id|section_title|chunk_type|page|start_page|end_page"
    r"|tables|figures|level|Section section_id|title)\s",
    re.IGNORECASE,
)
_CHUNK_CANDIDATE_RE = re.compile(
    r"^(?:chunk_candidates\s+)?Chunk\s+Candidate\s+text\s*(.*)",
    re.IGNORECASE,
)
# "text <actual content>" — strip key prefix, keep value
_TEXT_KEY_RE = re.compile(r"^text\s+:?\s*", re.IGNORECASE)


def _strip_embedding_prefix(text: str) -> str:
    """Strip the [Section Kind] Section N: prefix added by the embedding pipeline."""
    return _EMBEDDING_PREFIX_RE.sub("", text).strip() if text else ""


_EXTRACTED_DOC_PREFIX_RE = re.compile(
    r"^(?:Extracted\s+Document\s+(?:full_text|[\w.]+)\s*)", re.IGNORECASE
)


def _clean_metadata_from_text(text: str) -> str:
    """Strip metadata key-value fragments that are comma-separated with actual content."""
    if not text:
        return ""
    # Strip "Extracted Document full_text" prefix
    text = _EXTRACTED_DOC_PREFIX_RE.sub("", text).strip()
    text = text.lstrip("-").strip()
    # Split by comma, classify each segment as metadata or content
    parts = re.split(r"\s*,\s*", text)
    content_parts: list = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Skip pure metadata key-value segments (only short ones — long segments are real content)
        if _METADATA_KEY_RE.match(part) and len(part) < 40:
            continue
        # Strip "Chunk Candidate text" prefix but keep the actual text after it
        m = _CHUNK_CANDIDATE_RE.match(part)
        if m:
            remainder = m.group(1).strip()
            if remainder:
                content_parts.append(remainder)
            continue
        # "text <content>" — strip key prefix, keep the value
        tm = _TEXT_KEY_RE.match(part)
        if tm:
            remainder = part[tm.end():].strip()
            if remainder and len(remainder) > 3:
                content_parts.append(remainder)
            continue
        content_parts.append(part)
    result = ", ".join(content_parts) if content_parts else ""
    return re.sub(r"\s{2,}", " ", result).strip()


def _to_chunk(point: Any) -> Chunk:
    payload = getattr(point, "payload", None) or {}

    # Primary: canonical_text (clean after rebuild)
    text = payload.get("canonical_text") or ""

    # Fallback for legacy data: embedding_text with prefix stripped
    if not text or _is_metadata_garbage(text):
        emb = payload.get("embedding_text") or ""
        if emb:
            text = _strip_embedding_prefix(emb)
            text = _clean_metadata_from_text(text)

    # Last resort: salvage (kept for any remaining pre-rebuild data)
    if not text or _is_metadata_garbage(text):
        text = _salvage_text_from_garbage(payload) or ""

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


def _needs_doc_expansion(chunks: List[Chunk]) -> bool:
    if not chunks:
        return False
    if len(chunks) < MIN_RESULTS:
        return True
    unique_docs = {(_chunk_doc_id(c) or "") for c in chunks}
    if len(unique_docs) < 2:
        return True
    top_score = max((c.score for c in chunks), default=0.0)
    return top_score < 0.25


def _expand_by_document(
    *,
    qdrant_client: Any,
    collection: str,
    base_chunks: List[Chunk],
    subscription_id: str,
    profile_id: str,
    correlation_id: Optional[str],
    domain: str,
) -> List[Chunk]:
    if not _needs_doc_expansion(base_chunks):
        return base_chunks

    doc_ids = []
    for chunk in base_chunks:
        doc_id = _chunk_doc_id(chunk)
        if not doc_id:
            continue
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    doc_ids = doc_ids[:MAX_EXPANDED_DOCS]
    if not doc_ids:
        return base_chunks

    def _fetch(doc_id: str) -> List[Chunk]:
        q_filter = build_qdrant_filter(
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=doc_id,
        )
        try:
            response, _next = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=q_filter,
                limit=MAX_DOC_CHUNKS,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "RAG v3 doc expansion failed for doc_id=%s: %s",
                doc_id,
                exc,
                extra={"stage": "retrieve_expand_doc", "correlation_id": correlation_id},
            )
            return []
        chunks = [_to_chunk(point) for point in response or [] if point is not None]
        return _filter_section_chunks(chunks, domain)

    expanded: List[Chunk] = []
    with ThreadPoolExecutor(max_workers=min(4, len(doc_ids))) as executor:
        futures = {executor.submit(_fetch, doc_id): doc_id for doc_id in doc_ids}
        for future in as_completed(futures):
            try:
                expanded.extend(future.result() or [])
            except Exception:
                continue

    merged = _merge_dedupe(base_chunks, expanded)
    merged = sorted(merged, key=lambda c: c.score, reverse=True)[:MAX_UNION_RESULTS]
    _log_top5(merged, correlation_id, label="expand_doc")
    return merged


def _filter_section_chunks(chunks: List[Chunk], domain: str) -> List[Chunk]:
    keywords = SECTION_KEYWORDS_BY_DOMAIN.get(domain) or []
    if not keywords:
        return chunks
    filtered: List[Chunk] = []
    for chunk in chunks:
        meta = chunk.meta or {}
        section_title = str(meta.get("section_title") or meta.get("section.title") or "")
        section_kind = str(meta.get("section_kind") or meta.get("section.kind") or "")
        haystack = f"{section_title} {section_kind}".lower()
        if any(k in haystack for k in keywords):
            filtered.append(chunk)
    return filtered or chunks


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


def _boost_by_section_kind(chunks: List[Chunk], query: str, boost: float = 0.12) -> List[Chunk]:
    query_kind = _infer_query_section_kind(query)
    if not query_kind or not chunks:
        return chunks
    for chunk in chunks:
        stored_kind = (chunk.meta or {}).get("section_kind") or \
                      ((chunk.meta or {}).get("section") or {}).get("kind", "")
        if stored_kind == query_kind:
            chunk.score += boost
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def _keyword_score(text: str, tokens: List[str]) -> float:
    if not text or not tokens:
        return 0.0
    lowered = text.lower()
    matches = sum(1 for tok in tokens if tok in lowered)
    if matches == 0:
        return 0.0
    return max(matches / float(len(tokens)), 0.2)


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


def _normalize_chunk_domain(chunk: Chunk) -> str:
    """Map stored Qdrant doc_domain to internal retrieval domain for comparison."""
    raw = _chunk_domain(chunk)
    return _QDRANT_TO_RETRIEVAL_DOMAIN.get(raw, raw)


def _apply_domain_gate(chunks: List[Chunk], query: str, correlation_id: Optional[str]) -> List[Chunk]:
    domain = _infer_domain(query)
    if domain == "generic" or not chunks:
        return chunks
    top5 = chunks[:5]
    mismatched = sum(1 for chunk in top5 if _normalize_chunk_domain(chunk) != domain)
    if mismatched >= 3:
        filtered = [chunk for chunk in chunks if _normalize_chunk_domain(chunk) == domain]
        if len(filtered) >= MIN_RESULTS:
            chunks = sorted(filtered, key=lambda c: c.score, reverse=True)
        else:
            for chunk in chunks:
                if _normalize_chunk_domain(chunk) != domain:
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


def _chunk_doc_id(chunk: Chunk) -> Optional[str]:
    meta = chunk.meta or {}
    for key in ("document_id", "doc_id", "docId"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


# Maps Qdrant-stored doc_domain values to internal retrieval domain names
_QDRANT_TO_RETRIEVAL_DOMAIN = {
    "resume": "hr",
    "invoice": "invoice",
    "purchase_order": "invoice",
    "legal": "legal",
    "generic": "generic",
}

# Maps internal retrieval domain names to Qdrant-stored doc_domain values
_RETRIEVAL_TO_QDRANT_DOMAIN = {
    "hr": "resume",
    "invoice": "invoice",
    "legal": "legal",
    "generic": None,
}


def _infer_domain(query: str) -> str:
    lowered = (query or "").lower()
    if any(token in lowered for token in ("invoice", "amount due", "subtotal", "billing", "payment terms")):
        return "invoice"
    # Only strong HR signals for domain inference
    if any(token in lowered for token in ("resume", "cv", "candidate")):
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


# ============================================================================
# Quality Filtering Functions - Critical for accuracy
# ============================================================================

def filter_by_score(
    chunks: List[Chunk],
    min_score: float = LOW_SCORE_THRESHOLD,
    min_results: int = MIN_RESULTS,
) -> List[Chunk]:
    """
    Filter chunks by minimum score threshold.

    If filtering would leave fewer than min_results, keeps top chunks regardless of score.
    This ensures we always have something to work with while preferring quality.
    """
    if not chunks:
        return chunks

    # Try strict filtering first
    filtered = [c for c in chunks if c.score >= min_score]

    if len(filtered) >= min_results:
        return filtered

    # Fall back to keeping top N if strict filtering removes too many
    return sorted(chunks, key=lambda c: c.score, reverse=True)[:max(min_results, len(filtered))]


def filter_high_quality(
    chunks: List[Chunk],
    high_threshold: float = HIGH_SCORE_THRESHOLD,
    low_threshold: float = LOW_SCORE_THRESHOLD,
    prefer_quality: bool = True,
    min_keep: int = 3,
) -> List[Chunk]:
    """
    Smart quality filter that adapts based on available results.

    If we have multiple high-quality matches, prefer those.
    Otherwise, fall back to low_threshold filtering.
    Always keeps at least ``min_keep`` chunks (sorted by score descending)
    so that downstream stages never receive an empty list.
    """
    if not chunks:
        return chunks

    high_quality = [c for c in chunks if c.score >= high_threshold]
    medium_quality = [c for c in chunks if low_threshold <= c.score < high_threshold]

    # Only drop medium-quality chunks when we have plenty of high-quality ones
    if len(high_quality) >= 8 and prefer_quality:
        logger.debug(
            "Using %d high-quality chunks only (score >= %.2f)",
            len(high_quality),
            high_threshold,
        )
        return high_quality

    # Keep both high + medium for adequate coverage
    if high_quality or medium_quality:
        combined = high_quality + medium_quality
        logger.debug(
            "Using %d high + %d medium chunks (threshold %.2f/%.2f)",
            len(high_quality),
            len(medium_quality),
            high_threshold,
            low_threshold,
        )
        return combined

    # Fallback — keep top min_keep chunks even if all are below threshold
    # so that downstream extraction always has something to work with.
    logger.debug(
        "All %d chunks below low threshold %.2f; keeping top %d by score",
        len(chunks),
        low_threshold,
        min_keep,
    )
    return sorted(chunks, key=lambda c: c.score, reverse=True)[:min_keep]


def score_query_relevance(
    chunks: List[Chunk],
    query: str,
    boost_factor: float = 0.15,
) -> List[Chunk]:
    """
    Boost chunk scores based on query term presence.

    This helps prioritize chunks that directly mention query terms
    over those that are only semantically similar.
    """
    if not chunks or not query:
        return chunks

    # Extract meaningful query terms
    query_terms = set(_query_tokens(query))
    if not query_terms:
        return chunks

    for chunk in chunks:
        text_lower = (chunk.text or "").lower()

        # Count exact query term matches
        term_hits = sum(1 for term in query_terms if term in text_lower)

        if term_hits > 0:
            # Proportional boost based on term coverage
            coverage = term_hits / len(query_terms)
            chunk.score += boost_factor * coverage

    return sorted(chunks, key=lambda c: c.score, reverse=True)


def deduplicate_by_content(
    chunks: List[Chunk],
    similarity_threshold: float = 0.78,
) -> List[Chunk]:
    """
    Remove near-duplicate chunks based on content similarity.

    Keeps the highest-scoring version of similar chunks.
    """
    if len(chunks) <= 1:
        return chunks

    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    def _jaccard_similarity(a: str, b: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    # Sort by score descending - keep best version of duplicates
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    kept: List[Chunk] = []
    kept_texts: List[str] = []

    for chunk in sorted_chunks:
        normalized = _normalize_text(chunk.text or "")

        # Check if similar to any kept chunk
        is_duplicate = False
        for kept_text in kept_texts:
            if _jaccard_similarity(normalized, kept_text) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(chunk)
            kept_texts.append(normalized)

    return kept


def apply_quality_pipeline(
    chunks: List[Chunk],
    query: str,
    correlation_id: Optional[str] = None,
) -> List[Chunk]:
    """
    Apply the full quality filtering pipeline to chunks.

    Pipeline:
    1. Score by query relevance (boost term matches)
    2. Filter by quality threshold
    3. Deduplicate similar content
    4. Limit to reasonable size
    """
    if not chunks:
        return chunks

    original_count = len(chunks)

    # Step 1: Boost chunks that match query terms
    chunks = score_query_relevance(chunks, query)

    # Step 2: Apply quality filtering
    chunks = filter_high_quality(chunks)

    # Step 3: Remove near-duplicates
    chunks = deduplicate_by_content(chunks)

    # Step 4: Limit to max results
    chunks = chunks[:MAX_UNION_RESULTS]

    logger.info(
        "Quality pipeline: %d -> %d chunks",
        original_count,
        len(chunks),
        extra={"stage": "quality_pipeline", "correlation_id": correlation_id},
    )

    return chunks


# ============================================================================
# Section-Filtered Retrieval
# ============================================================================

def retrieve_section_filtered(
    query: str,
    collection: str,
    subscription_id: str,
    profile_id: str,
    *,
    section_kind: Optional[str] = None,
    doc_domain: Optional[str] = None,
    top_k: int = 20,
    embedder=None,
    qdrant_client=None,
    correlation_id: str = "",
) -> List[Chunk]:
    """Retrieve chunks with pre-applied section_kind and doc_domain filters.

    Applies structural metadata filters BEFORE vector search for precision.
    Falls back to unfiltered search if filtered results are empty.
    """
    if not subscription_id or not str(subscription_id).strip():
        raise ValueError("subscription_id is required for retrieval")
    if not profile_id or not str(profile_id).strip():
        raise ValueError("profile_id is required for retrieval")
    if embedder is None:
        raise ValueError("embedder is required for section-filtered retrieval")
    if qdrant_client is None:
        raise ValueError("qdrant_client is required for section-filtered retrieval")

    client = qdrant_client
    enc = embedder

    # Build the filtered query filter with section_kind and/or doc_domain
    filtered_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        section_kind=section_kind,
        doc_domain=doc_domain,
    )

    try:
        query_vector = _embed(query, enc)
    except Exception:
        logger.warning(
            "Section-filtered retrieval: embedding failed for query",
            extra={"stage": "retrieve_section_filtered", "correlation_id": correlation_id},
        )
        return []

    try:
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            using="content_vector",
            query_filter=filtered_filter,
            limit=top_k,
            with_payload=True,
        )
        points = getattr(result, "points", None) or []
    except Exception as exc:
        logger.warning(
            "Section-filtered retrieval failed: %s",
            exc,
            extra={"stage": "retrieve_section_filtered", "correlation_id": correlation_id},
        )
        points = []

    # Fallback: if section/domain filter returned nothing, retry without those filters
    if not points and (section_kind or doc_domain):
        logger.info(
            "Section filter returned 0 results for section_kind=%s doc_domain=%s; falling back to base filter",
            section_kind,
            doc_domain,
            extra={"stage": "retrieve_section_filtered_fallback", "correlation_id": correlation_id},
        )
        base_filter = build_qdrant_filter(
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
        )
        try:
            result = client.query_points(
                collection_name=collection,
                query=query_vector,
                using="content_vector",
                query_filter=base_filter,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(result, "points", None) or []
        except Exception as exc:
            logger.warning(
                "Section-filtered fallback retrieval failed: %s",
                exc,
                extra={"stage": "retrieve_section_filtered_fallback", "correlation_id": correlation_id},
            )
            points = []

    chunks = [_to_chunk(pt) for pt in points if pt is not None]
    # Filter out empty-text chunks
    return [c for c in chunks if c.text.strip()]
