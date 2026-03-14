from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from src.api.vector_store import build_collection_name, build_qdrant_filter

from .types import Chunk, ChunkSource

logger = get_logger(__name__)

# Retrieval configuration - tuned for accuracy over recall
MIN_RESULTS = 6  # Reduced: prefer quality over quantity
FALLBACK_LIMIT = 100  # Reduced from 200
MAX_UNION_RESULTS = 16  # Reduced from 24
LOW_SCORE_THRESHOLD = 0.30  # Balanced: reject garbage, keep borderline-relevant chunks
HIGH_SCORE_THRESHOLD = 0.7  # High-confidence threshold
MAX_EXPANDED_DOCS = 4
MAX_DOC_CHUNKS = 20  # Allow more chunks per doc for complete context
MAX_FULL_SCAN_DOCS = 3
MAX_FULL_SCAN_CHUNKS = 60  # Moderate scan to capture relevant chunks without noise
MAX_PROFILE_SCAN_CHUNKS = 800
MAX_UNSCOPED_SCAN_CHUNKS = 400
SECTION_KEYWORDS_BY_DOMAIN = {
    "hr": ["experience", "summary", "skills", "certification", "education", "project",
           "achievement", "reference", "objective", "contact", "language"],
    "invoice": ["total", "amount", "payment", "invoice", "bill to", "due",
                "line item", "subtotal", "tax", "discount", "purchase order"],
    "legal": ["clause", "section", "term", "liability", "warranty",
              "indemnification", "termination", "confidentiality", "governing law",
              "obligation", "penalty", "notice"],
    "medical": ["diagnosis", "medication", "prescription", "treatment", "lab result",
                "allergy", "vitals", "symptom", "history", "assessment"],
    "policy": ["coverage", "premium", "deductible", "exclusion", "benefit",
               "claim", "rider", "policyholder", "insured"],
    "generic": [],
}

_QUERY_SECTION_MAP = [
    # HR/Resume sections
    (("skills", "technical skills", "tech stack", "tools", "technologies", "frameworks", "programming"), "skills_technical"),
    (("soft skills", "functional skills", "communication", "leadership"), "skills_functional"),
    (("education", "degree", "university", "academic", "qualification"), "education"),
    (("certification", "certified", "credential", "license"), "certifications"),
    (("experience", "work history", "employment", "career", "job history"), "experience"),
    (("summary", "objective", "profile", "overview", "about"), "summary_objective"),
    (("contact", "email", "phone", "address"), "identity_contact"),
    (("achievement", "award", "accomplishment"), "achievements"),
    # Medical sections
    (("diagnosis", "condition", "assessment", "impression"), "diagnosis"),
    (("medication", "prescription", "drug", "dosage", "treatment", "therapy"), "treatment"),
    (("lab result", "test result", "blood work", "vitals", "laboratory"), "lab_results"),
    (("allergy", "allergies", "medical history", "surgical history"), "medical_history"),
    # Legal sections
    (("clause", "provision", "article"), "clause"),
    (("liability", "indemnification", "indemnity"), "liability"),
    (("termination", "cancellation"), "termination"),
    (("confidentiality", "non-disclosure", "nda"), "confidentiality"),
    # Invoice sections
    (("line item", "product", "service"), "items"),
    (("total", "subtotal", "amount due", "balance"), "totals"),
    (("payment term", "due date"), "payment_terms"),
    # Policy sections
    (("coverage", "benefit", "protection"), "coverage"),
    (("exclusion", "exception", "limitation"), "exclusions"),
    (("claim", "filing"), "claims"),
]

def _infer_query_section_kind(query: str) -> Optional[str]:
    lowered = (query or "").lower()
    for keywords, kind in _QUERY_SECTION_MAP:
        if any(kw in lowered for kw in keywords):
            return kind
    return None

_COMPARISON_PATTERN = re.compile(
    r'\b(compare|comparison|vs\.?|versus|difference between|similarities between)\b',
    re.IGNORECASE,
)
_COMPARISON_WORDS = {"compare", "comparison", "difference", "similarities", "between"}
_AND_ENTITY_PATTERN = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:and|vs\.?|versus|&)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)
_POSSESSIVE_MULTI = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+(?:and\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s",
)

def _decompose_comparison_query(query: str) -> List[str]:
    """Decompose comparison queries into per-entity sub-queries.

    Returns list of sub-queries (1 per entity) or empty list if not a comparison.
    """
    if not _COMPARISON_PATTERN.search(query):
        # Also check possessive multi-entity pattern
        match = _POSSESSIVE_MULTI.search(query)
        if not match:
            return []
        entities = [match.group(1), match.group(2)]
    else:
        match = _AND_ENTITY_PATTERN.search(query)
        if not match:
            return []
        # Strip comparison verbs that regex may capture as part of entity name
        entities = []
        for g in [match.group(1), match.group(2)]:
            parts = [w for w in g.split() if w.lower() not in _COMPARISON_WORDS]
            if parts:
                entities.append(" ".join(parts))

    # Build topic by removing comparison words and entity names
    _skip_words = {'compare', 'comparison', 'vs', 'vs.', 'versus', 'and', 'between',
                   'difference', 'similarities', "'s"}
    _entity_words = set()
    for e in entities:
        for w in e.split():
            _entity_words.add(w.lower())
    topic_words = []
    for word in query.split():
        clean = word.rstrip("'s").rstrip("'s")
        if word.lower() not in _skip_words and clean.lower() not in _entity_words:
            topic_words.append(word)
    topic = " ".join(topic_words).strip()

    sub_queries = []
    for entity in entities:
        sub_q = f"{entity} {topic}".strip()
        if sub_q:
            sub_queries.append(sub_q)

    return sub_queries[:3]  # Max 3 entities

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

    # Multi-entity comparison: run sub-queries for each entity to improve recall
    sub_queries = _decompose_comparison_query(raw_query)
    if sub_queries and len(sub_queries) >= 2:
        for sq in sub_queries:
            sq_vector = _embed(sq, embedder)
            sq_results = _query(qdrant_client, collection, sq_vector, q_filter, top_k // 2, correlation_id, label="retrieve_entity_sub")
            results = _merge_results(results, sq_results)
        logger.info(
            "Comparison decomposition: %d sub-queries, %d total chunks",
            len(sub_queries), len(results),
            extra={"stage": "retrieve_decompose", "correlation_id": correlation_id},
        )

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
    results = _boost_exact_query_terms(results, raw_query)
    _log_top5(results, correlation_id, label="vector")

    if _needs_hybrid_fallback(results, raw_query=raw_query):
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

    # Step 1: Find documents containing the entity name via paginated scroll
    # Use MAX_PROFILE_SCAN_CHUNKS (500) to avoid missing entities on large profiles
    # (was limited to 200, causing silent fallback to unscoped search)
    points: list = []
    try:
        _scroll_limit = min(MAX_PROFILE_SCAN_CHUNKS, 500)
        _offset = None
        _scrolled = 0
        while _scrolled < _scroll_limit:
            _batch_size = min(100, _scroll_limit - _scrolled)
            scroll_result = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=base_filter,
                limit=_batch_size,
                offset=_offset,
                with_payload=["document_id", "source_name", "canonical_text", "embedding_text"],
            )
            _batch = scroll_result[0] if scroll_result else []
            if not _batch:
                break
            points.extend(_batch)
            _scrolled += len(_batch)
            _offset = scroll_result[1] if len(scroll_result) > 1 else None
            if _offset is None:
                break
    except Exception:  # noqa: BLE001
        points = []

    entity_lower = entity_name.lower()
    # Also try underscore-normalized and possessive-stripped variants
    _entity_variants = {entity_lower, entity_lower.replace(" ", "_"), entity_lower.rstrip("s")}
    if entity_lower.endswith("'s"):
        _entity_variants.add(entity_lower[:-2])
    matching_doc_ids: set = set()
    for pt in points:
        payload = getattr(pt, "payload", None) or {}
        text = (payload.get("canonical_text") or payload.get("embedding_text") or "").lower()
        source = (payload.get("source_name") or "").lower().replace("_", " ")
        if any(v in text or v in source for v in _entity_variants):
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
        logger.warning(
            "Entity '%s' not found in %d scrolled points — falling back to unscoped search | cid=%s",
            entity_name, len(points), correlation_id,
        )
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
    domain_hint: Optional[str] = None,
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

    # Domain-scoped filtering: when a domain is known, prioritize chunks from
    # matching documents to prevent cross-domain noise (e.g. invoice chunks
    # diluting a resume query).  Keep non-matching chunks as fallback only if
    # domain-matching yields too few results.
    if domain_hint and chunks:
        _dh_lower = domain_hint.lower()
        _domain_match = []
        _domain_other = []
        for _ch in chunks:
            _ch_domain = ""
            _m = getattr(_ch, "meta", None) or {}
            _ch_domain = str(
                _m.get("doc_domain") or _m.get("domain") or _m.get("category") or ""
            ).lower()
            if _ch_domain and _ch_domain == _dh_lower:
                _domain_match.append(_ch)
            else:
                _domain_other.append(_ch)
        # Use domain-matched chunks if we have enough; otherwise mix in others
        if len(_domain_match) >= 3:
            chunks = _domain_match + _domain_other[:max(0, max_chunks // 4)]
            logger.info(
                "Domain filter: %d matched '%s', %d other kept | cid=%s",
                len(_domain_match), domain_hint, min(len(_domain_other), max_chunks // 4),
                correlation_id,
            )
        else:
            logger.debug(
                "Domain filter: only %d matched '%s' — keeping all %d chunks | cid=%s",
                len(_domain_match), domain_hint, len(chunks), correlation_id,
            )

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
    """Enrich query for better embedding match.

    Adds lightweight domain-context hint when query clearly targets a domain,
    improving cosine similarity with domain-specific chunks.  The hint is a
    single descriptor word prepended — keeps the query close to its original
    semantic meaning while biasing the embedding toward relevant document types.
    """
    if not query or len(query.split()) < 3:
        return query

    ql = query.lower()

    # Domain context hints — single word that biases the embedding
    _DOMAIN_HINTS = {
        "resume": ("candidate", {"resume", "candidate", "applicant", "experience", "skills", "education", "certification"}),
        "medical": ("clinical", {"patient", "diagnosis", "medication", "lab", "clinical", "treatment", "symptoms", "vitals"}),
        "invoice": ("financial", {"invoice", "payment", "vendor", "amount", "total", "due", "bill"}),
        "legal": ("legal", {"clause", "agreement", "liability", "contract", "obligation", "jurisdiction", "indemnification"}),
        "policy": ("insurance", {"coverage", "premium", "exclusion", "deductible", "policyholder", "benefit", "claim"}),
    }

    # Strong keywords: a single occurrence is sufficient for domain identification
    _STRONG_KEYWORDS = frozenset({
        "patient", "diagnosis", "prescription", "invoice", "indemnification",
        "policyholder", "deductible", "candidate", "applicant", "resume",
        "medication", "liability", "coverage", "premium",
    })

    for domain, (hint, keywords) in _DOMAIN_HINTS.items():
        matched_words = [kw for kw in keywords if kw in ql]
        matched = len(matched_words)
        # Single strong keyword or 2+ regular keywords trigger enrichment
        if matched >= 2 or (matched == 1 and matched_words[0] in _STRONG_KEYWORDS):
            # Only prepend if hint word not already in query
            if hint not in ql:
                return f"{hint} {query}"
            break

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

def _needs_hybrid_fallback(chunks: List[Chunk], raw_query: str = "") -> bool:
    if len(chunks) < MIN_RESULTS:
        return True
    scores = [c.score for c in chunks]
    top_score = max(scores, default=0.0)
    if top_score < LOW_SCORE_THRESHOLD:
        return True
    # Wide score spread with very low median indicates uncertain retrieval
    if len(scores) >= 4:
        sorted_scores = sorted(scores, reverse=True)
        median_score = sorted_scores[len(sorted_scores) // 2]
        if top_score - median_score > 0.35 and median_score < 0.20:
            return True
    # Entity-aware: if query contains proper nouns that aren't found in top chunks,
    # hybrid keyword fallback may recover entity-specific results
    if raw_query and chunks:
        _entities = [w for w in raw_query.split()
                     if w[0:1].isupper() and len(w) > 2
                     and w.lower() not in {"the", "what", "how", "who", "where", "when",
                                           "which", "can", "could", "does", "list", "show",
                                           "compare", "find", "get"}]
        if _entities:
            _top_text = " ".join(c.text.lower() for c in chunks[:6])
            _found = sum(1 for e in _entities if e.lower() in _top_text)
            if _found < len(_entities):
                return True  # Not all entities found in top chunks
    return False

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

_DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    # Medical
    "medication": ["drug", "prescription", "medicine"],
    "drug": ["medication", "prescription"],
    "diagnosis": ["condition", "disease", "disorder"],
    "symptom": ["complaint", "sign", "presentation"],
    "treatment": ["therapy", "intervention", "procedure"],
    "patient": ["individual", "subject"],
    "allergy": ["sensitivity", "reaction", "intolerance"],
    "lab": ["laboratory", "test", "diagnostic"],
    # Legal
    "clause": ["provision", "section", "term"],
    "liability": ["obligation", "responsibility", "exposure"],
    "indemnification": ["indemnity", "compensation"],
    "termination": ["cancellation", "expiry", "end"],
    "contract": ["agreement", "deed"],
    "party": ["signatory", "counterparty"],
    "breach": ["violation", "default", "infringement"],
    # HR/Resume
    "experience": ["background", "history", "tenure"],
    "qualification": ["credential", "certification"],
    "skill": ["competency", "proficiency", "expertise"],
    "education": ["degree", "academic", "university"],
    "salary": ["compensation", "pay", "remuneration"],
    "candidate": ["applicant", "prospect"],
    # Invoice/Financial
    "invoice": ["bill", "statement"],
    "payment": ["remittance", "disbursement"],
    "vendor": ["supplier", "provider"],
    "amount": ["sum", "total", "value"],
    "discount": ["rebate", "reduction"],
    "overdue": ["delinquent", "past due", "late"],
    # Policy/Insurance
    "coverage": ["protection", "benefit"],
    "premium": ["cost", "rate"],
    "exclusion": ["exception", "limitation"],
    "deductible": ["excess", "copay"],
    "claim": ["request", "submission"],
    "policyholder": ["insured", "subscriber"],
}

_EXPAND_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on",
    "with", "from", "about", "what", "how", "who", "when", "where",
    "which", "why", "does", "did", "are", "was", "were", "been",
    "has", "have", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "not", "all", "any", "each",
    "this", "that", "these", "those", "his", "her", "its", "their",
    "our", "your", "my", "but", "than", "then", "also", "only",
    "just", "very", "much", "more", "most", "some", "other",
    "tell", "show", "give", "get", "find", "please",
})

def _expand_query(query: str) -> str:
    """Expand query with domain-aware synonyms for better recall."""
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    keywords = [t for t in tokens if t not in _EXPAND_STOP and len(t) > 2]
    if not keywords:
        return query
    deduped: List[str] = []
    seen: set[str] = set()
    for tok in keywords:
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)

    # Add domain synonyms (max 1 synonym per keyword, max 2 total additions)
    expansions: List[str] = []
    for tok in deduped:
        synonyms = _DOMAIN_SYNONYMS.get(tok, [])
        added = 0
        for syn in synonyms:
            if syn not in seen and added < 1 and len(expansions) < 2:
                expansions.append(syn)
                seen.add(syn)
                added += 1

    result = deduped + expansions
    return " ".join(result) if result else query

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

def _boost_exact_query_terms(chunks: List[Chunk], query: str, boost: float = 0.08) -> List[Chunk]:
    """Boost chunks that contain exact query content words.

    Complements embedding similarity (which captures semantics) with
    lexical matching (which captures exact terminology). Useful when
    query uses specific technical terms, names, or numbers that
    embeddings may not perfectly capture.
    """
    if not chunks or not query:
        return chunks
    _stop = frozenset({
        "what", "is", "are", "the", "a", "an", "of", "for", "in", "how",
        "do", "does", "can", "about", "from", "with", "tell", "me", "show",
        "please", "give", "could", "would", "all", "each", "every", "this",
        "that", "those", "these", "my", "their", "your",
    })
    q_words = [w.lower().rstrip("?,!.") for w in query.split()
               if w.lower().rstrip("?,!.") not in _stop and len(w) > 2]
    if not q_words:
        return chunks
    # Section-aware boost: if query targets a specific section, boost matching chunks
    _target_section = None
    _ql = query.lower()
    for keywords, section_name in _QUERY_SECTION_MAP:
        if any(kw in _ql for kw in keywords):
            _target_section = section_name
            break

    for chunk in chunks:
        text_lower = ((getattr(chunk, "text", "") or "") + " " +
                      str((chunk.meta or {}).get("section_title") or "")).lower()
        # Word-boundary match to avoid partial matches ("skill" in "skilled")
        matches = sum(1 for w in q_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
        if matches > 0:
            # Proportional boost: more matching terms = higher boost
            ratio = matches / len(q_words)
            chunk.score += boost * ratio

        # Section boost: if chunk's section matches query target, add small boost
        if _target_section:
            _chunk_section = ((chunk.meta or {}).get("section_kind") or
                              (chunk.meta or {}).get("chunk_type") or "").lower()
            if _target_section.lower() in _chunk_section or _chunk_section in _target_section.lower():
                chunk.score += boost * 0.5

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
    # Fallback: use text hash when both IDs are missing to prevent
    # all unidentified chunks from collapsing into a single entry
    if not doc_id and not chunk_id:
        text = getattr(chunk, "text", "") or ""
        chunk_id = f"_hash_{hash(text[:200])}"
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
    "medical": "medical",
    "clinical": "medical",
    "policy": "policy",
    "insurance": "policy",
    "generic": "generic",
}

# Maps internal retrieval domain names to Qdrant-stored doc_domain values
_RETRIEVAL_TO_QDRANT_DOMAIN = {
    "hr": "resume",
    "invoice": "invoice",
    "legal": "legal",
    "medical": "medical",
    "policy": "policy",
    "generic": None,
}

def _infer_domain(query: str) -> str:
    lowered = (query or "").lower()
    if any(token in lowered for token in ("invoice", "amount due", "subtotal", "billing", "payment terms")):
        return "invoice"
    if any(token in lowered for token in ("resume", "cv", "candidate")):
        return "hr"
    if any(token in lowered for token in ("agreement", "contract", "clause", "warranty", "liability")):
        return "legal"
    if any(token in lowered for token in ("diagnosis", "patient", "medication", "prescription", "symptom", "lab result")):
        return "medical"
    if any(token in lowered for token in ("policy", "premium", "deductible", "coverage", "insured", "policyholder")):
        return "policy"
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
    similarity_threshold: float = 0.85,
) -> List[Chunk]:
    """
    Remove near-duplicate chunks based on content similarity.

    Keeps the highest-scoring version of similar chunks.
    Threshold 0.85 preserves chunks with shared vocabulary but different facts
    (e.g., two job experiences at same company).
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

def _boost_entity_name_match(chunks: List[Chunk], query: str, boost: float = 0.12) -> List[Chunk]:
    """Boost chunks that contain multi-word entity names from the query.

    Detects capitalized multi-word names (e.g., "John Smith", "Acme Corp")
    and gives a stronger boost to chunks containing the full name vs. partial.
    """
    if not chunks or not query:
        return chunks
    # Extract multi-word capitalized sequences (potential entity names)
    _name_re = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    names = _name_re.findall(query)
    if not names:
        return chunks
    for chunk in chunks:
        text_lower = (chunk.text or "").lower()
        for name in names:
            name_lower = name.lower()
            if name_lower in text_lower:
                chunk.score += boost  # Full name match — strong boost
            else:
                # Partial: check individual words
                parts = name_lower.split()
                matched = sum(1 for p in parts if p in text_lower)
                if matched > 0 and matched < len(parts):
                    chunk.score += boost * 0.3  # Partial match — weaker
    return sorted(chunks, key=lambda c: c.score, reverse=True)

def apply_quality_pipeline(
    chunks: List[Chunk],
    query: str,
    correlation_id: Optional[str] = None,
) -> List[Chunk]:
    """
    Apply the full quality filtering pipeline to chunks.

    Pipeline:
    1. Score by query relevance (boost term matches)
    1b. Boost entity name matches
    2. Filter by quality threshold
    3. Deduplicate similar content
    4. Limit to reasonable size
    """
    if not chunks:
        return chunks

    original_count = len(chunks)

    # Step 1: Boost chunks that match query terms
    chunks = score_query_relevance(chunks, query)

    # Step 1b: Boost chunks containing entity names from query
    chunks = _boost_entity_name_match(chunks, query)

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
