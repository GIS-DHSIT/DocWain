"""
Enhanced Retrieval System - FIXED FOR MULTI-DOCUMENT ACCURACY
Key Fixes:
1. Proper OR logic for document filtering
2. Document-aware adjacent chunk expansion
3. Strict source boundary preservation
"""

from src.utils.logging_utils import get_logger
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from src.api.vector_store import build_qdrant_filter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from src.embedding.pipeline.embed_pipeline import compute_stable_chunk_id
from src.kg.entity_extractor import EntityExtractor
from src.kg.neo4j_store import Neo4jStore
from src.utils.redis_cache import RedisJsonCache, hash_query, stamp_cache_payload
from src.utils.payload_utils import get_canonical_text, get_source_name

logger = get_logger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for each chunk to preserve context"""
    chunk_id: str
    doc_name: str
    chunk_index: int
    total_chunks: int
    section_title: str
    document_id: str  # ADDED: Track document ID
    prev_chunk_id: str = None
    next_chunk_id: str = None
    chunk_type: str = "text"
    page_number: int = None

class EnhancedSemanticChunker:
    """Advanced chunking that preserves document structure"""

    def __init__(self, chunk_size: int = 800, overlap: int = 200, min_chunk_size: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.heading_pattern = re.compile(r'^(?:#+\s+|[A-Z][A-Z\s]{2,}:|\d+\.\s+[A-Z])', re.MULTILINE)
        self.list_pattern = re.compile(r'^[\s]*[-"*]\s+|\d+\.\s+', re.MULTILINE)
        self.table_pattern = re.compile(r'\|.*\|.*\||(?:\s{2,}[^\s]+){3,}', re.MULTILINE)

    def _detect_structure(self, text: str) -> List[Dict[str, Any]]:
        structures = []
        for match in self.heading_pattern.finditer(text):
            structures.append({
                'type': 'heading',
                'start': match.start(),
                'end': match.end(),
                'text': match.group().strip()
            })
        for match in self.table_pattern.finditer(text):
            structures.append({
                'type': 'table',
                'start': match.start(),
                'end': match.end(),
                'text': match.group().strip()
            })
        return sorted(structures, key=lambda x: x['start'])

    def _split_into_semantic_units(self, text: str) -> List[Tuple[str, str]]:
        structures = self._detect_structure(text)
        units = []
        current_section = "Introduction"
        last_pos = 0

        for struct in structures:
            if struct['start'] > last_pos:
                chunk_text = text[last_pos:struct['start']].strip()
                if chunk_text:
                    units.append((chunk_text, current_section))
            if struct['type'] == 'heading':
                current_section = struct['text']
            units.append((struct['text'], current_section))
            last_pos = struct['end']

        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                units.append((remaining, current_section))

        return units

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _create_chunk_with_context(self, current_text: str, prev_text: str, next_text: str, section_title: str,
                                   doc_name: str) -> str:
        """FIXED: Add document name to context"""
        parts = []

        # CRITICAL: Always include document identifier
        parts.append(f"[DOCUMENT: {doc_name}]")

        if section_title and section_title != "Introduction":
            parts.append(f"[SECTION: {section_title}]")
        if prev_text:
            prev_snippet = prev_text[-100:].strip()
            if prev_snippet:
                parts.append(f"[CONTEXT: ...{prev_snippet}]")
        parts.append(current_text)
        if next_text:
            next_snippet = next_text[:100].strip()
            if next_snippet:
                parts.append(f"[CONTINUES: {next_snippet}...]")
        return " ".join(parts)

    def chunk_document(self, text: str, doc_name: str, document_id: str, page_number: int = None) -> List[
        Tuple[str, ChunkMetadata]]:
        """FIXED: Added document_id parameter"""
        if not text or not text.strip():
            logger.debug(f"Empty text for document: {doc_name}")
            return []

        units = self._split_into_semantic_units(text)
        if not units:
            logger.warning(f"No semantic units extracted from: {doc_name}")
            return []

        chunks = []
        current_chunk_units = []
        current_tokens = 0
        current_section = units[0][1] if units else "Introduction"

        for i, (unit_text, section_title) in enumerate(units):
            unit_tokens = self._estimate_tokens(unit_text)

            if current_tokens + unit_tokens > self.chunk_size and current_chunk_units:
                chunk_text = " ".join([u[0] for u in current_chunk_units])
                prev_text = chunks[-1][0] if chunks else ""
                next_text = unit_text if i < len(units) else ""
                contextualized_text = self._create_chunk_with_context(
                    chunk_text, prev_text, next_text, current_section, doc_name
                )

                metadata = ChunkMetadata(
                    chunk_id=f"{doc_name}_chunk_{len(chunks)}",
                    doc_name=doc_name,
                    document_id=document_id,  # ADDED
                    chunk_index=len(chunks),
                    total_chunks=-1,
                    section_title=current_section,
                    prev_chunk_id=chunks[-1][1].chunk_id if chunks else None,
                    next_chunk_id=None,
                    page_number=page_number
                )

                chunks.append((contextualized_text, metadata))
                if len(chunks) > 1:
                    chunks[-2][1].next_chunk_id = metadata.chunk_id

                overlap_units = []
                overlap_tokens = 0
                for u in reversed(current_chunk_units):
                    u_tokens = self._estimate_tokens(u[0])
                    if overlap_tokens + u_tokens <= self.overlap:
                        overlap_units.insert(0, u)
                        overlap_tokens += u_tokens
                    else:
                        break

                current_chunk_units = overlap_units
                current_tokens = overlap_tokens

            current_chunk_units.append((unit_text, section_title))
            current_tokens += unit_tokens
            current_section = section_title

        if current_chunk_units:
            chunk_text = " ".join([u[0] for u in current_chunk_units])
            prev_text = chunks[-1][0] if chunks else ""
            contextualized_text = self._create_chunk_with_context(
                chunk_text, prev_text, "", current_section, doc_name
            )

            metadata = ChunkMetadata(
                chunk_id=f"{doc_name}_chunk_{len(chunks)}",
                doc_name=doc_name,
                document_id=document_id,  # ADDED
                chunk_index=len(chunks),
                total_chunks=-1,
                section_title=current_section,
                prev_chunk_id=chunks[-1][1].chunk_id if chunks else None,
                page_number=page_number
            )

            chunks.append((contextualized_text, metadata))
            if len(chunks) > 1:
                chunks[-2][1].next_chunk_id = metadata.chunk_id

        total = len(chunks)
        for _, meta in chunks:
            meta.total_chunks = total

        logger.info(f"Created {len(chunks)} chunks from {len(units)} units for {doc_name}")
        return chunks

def chunk_text_for_embedding(text: str, doc_name: str, document_id: str) -> List[Tuple[str, dict]]:
    """
     FIXED: document_id is now REQUIRED (no default value)

    Args:
        text: Document text to chunk
        doc_name: Source filename
        document_id: MongoDB _id (REQUIRED - must be unique per document)

    Returns:
        List of (chunk_text, metadata) tuples
    """
    #  REMOVED fallback - document_id is mandatory
    if not document_id:
        raise ValueError(f"document_id is required for {doc_name}")

    #  Validate document_id is not a filename
    if document_id.endswith(('.pdf', '.docx', '.txt', '.csv', '.xlsx')):
        raise ValueError(
            f"Invalid document_id '{document_id}' - looks like a filename. "
            f"Must be MongoDB _id (e.g., '695a5ce55a3e77b5c85c144a')"
        )

    chunker = EnhancedSemanticChunker(chunk_size=800, overlap=200, min_chunk_size=150)
    chunks = chunker.chunk_document(text, doc_name, document_id)

    result = []
    for chunk_text, metadata in chunks:
        meta_dict = {
            'chunk_id': metadata.chunk_id,
            'doc_name': metadata.doc_name,
            'document_id': metadata.document_id,  #  Now guaranteed to be MongoDB _id
            'chunk_index': metadata.chunk_index,
            'total_chunks': metadata.total_chunks,
            'section_title': metadata.section_title,
            'prev_chunk_id': metadata.prev_chunk_id,
            'next_chunk_id': metadata.next_chunk_id,
            'page_number': metadata.page_number
        }
        result.append((chunk_text, meta_dict))

    logger.info(f" Created {len(result)} chunks with document_id={document_id} for {doc_name}")

    #  VERIFICATION: Check all chunks have same document_id
    doc_ids = set(m['document_id'] for _, m in result)
    if len(doc_ids) > 1:
        raise ValueError(f"L CRITICAL: Multiple document_ids in chunks: {doc_ids}")

    return result

def normalize_chunk_links(
        chunk_metadata: List[dict],
        subscription_id: str,
        profile_id: str,
        document_id: str,
        doc_name: str,
        chunks: List[str],
) -> List[dict]:
    """
    Recompute chunk_id/prev/next to ensure they are consistent and deterministic.
    """
    if len(chunk_metadata) != len(chunks):
        raise ValueError("chunk_metadata and chunks length mismatch")

    computed_ids = []
    for idx in range(len(chunks)):
        meta = chunk_metadata[idx] or {}
        chunk_hash = meta.get("chunk_hash") or hashlib.sha256(chunks[idx].encode("utf-8")).hexdigest()
        section_id = meta.get("section_id") or meta.get("section_path") or meta.get("section_title") or "section"
        computed_ids.append(
            compute_stable_chunk_id(subscription_id, profile_id, document_id, str(section_id), idx, chunk_hash)
        )

    normalized = []
    for idx, meta in enumerate(chunk_metadata):
        m = dict(meta) if meta else {}
        m["chunk_index"] = idx
        m["chunk_id"] = computed_ids[idx]
        m["prev_chunk_id"] = computed_ids[idx - 1] if idx > 0 else None
        m["next_chunk_id"] = computed_ids[idx + 1] if idx < len(computed_ids) - 1 else None
        m["document_id"] = document_id
        normalized.append(m)

    return normalized

class AdaptiveRetriever:
    """
    Multi-strategy retrieval - FIXED FOR MULTI-DOCUMENT
    """

    def __init__(self, qdrant_client: QdrantClient, model: SentenceTransformer):
        self.client = qdrant_client
        self.model = model
        self.tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
        self._fit_tfidf()

    def _fit_tfidf(self):
        """Pre-fit TF-IDF on sample corpus"""
        sample_corpus = [
            "machine learning artificial intelligence",
            "data science analysis python",
            "software engineer developer",
            "professional summary experience skills",
            "location address city country"
        ]
        try:
            self.tfidf.fit(sample_corpus)
        except Exception as e:
            logger.warning(f"TF-IDF pre-fitting failed: {e}")

    def _build_filter(
            self,
            subscription_id: str,
            profile_id: str,
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None
    ) -> Optional[Filter]:
        """
        FIXED: Use MatchAny for OR logic with multiple documents
        """
        base = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))
        conditions = list(getattr(base, "must", []) or [])
        should = list(getattr(base, "should", []) or [])

        # FIXED: Use MatchAny for document filtering (OR logic)
        if document_ids and len(document_ids) > 0:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=document_ids)  # FIXED: OR logic
                )
            )

        # FIXED: Use MatchAny for source file filtering (OR logic)
        if source_files and len(source_files) > 0:
            should.append(
                FieldCondition(
                    key="source.name",
                    match=MatchAny(any=source_files)
                )
            )
            should.append(
                FieldCondition(
                    key="source_file",
                    match=MatchAny(any=source_files)
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions, should=should or None)

    def _dense_search(
            self,
            collection_name: str,
            query: str,
            subscription_id: str,
            profile_id: str,
            top_k: int,
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None,
            score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Dense vector search with proper document filtering"""
        try:
            # Encode query
            query_vector = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()

            # Build filter with OR logic
            query_filter = self._build_filter(subscription_id, profile_id, document_ids, source_files)

            # Log filter for debugging
            if query_filter:
                logger.info(f"Filter: profile={profile_id}, docs={document_ids}, sources={source_files}")

            # Search using query_points
            kwargs = {
                "collection_name": collection_name,
                "query": query_vector,
                "using": "content_vector",
                "limit": top_k,
                "with_payload": True,
                "with_vectors": False
            }

            if query_filter:
                kwargs["query_filter"] = query_filter

            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            results = self.client.query_points(**kwargs)

            chunks = []
            for pt in (results.points or []):
                payload = pt.payload or {}
                chunks.append({
                    'id': str(pt.id),
                    'text': get_canonical_text(payload),
                    'score': float(pt.score),
                    'metadata': payload,
                    'method': 'dense'
                })

            logger.info(f"Dense search returned {len(chunks)} results")
            if chunks:
                # ADDED: Log which documents were returned
                returned_docs = list(set([get_source_name(c.get("metadata") or {}) or "unknown" for c in chunks[:10]]))
                logger.info(f"Returned documents: {returned_docs}")
                logger.info(f"Top 3 scores: {[round(c['score'], 4) for c in chunks[:3]]}")

            return chunks

        except Exception as e:
            logger.error(f"Dense search failed: {e}", exc_info=True)
            return []

    def _keyword_boost_rerank(self, chunks: List[Dict], query: str, boost_factor: float = 0.15) -> List[Dict]:
        """Boost scores for exact keyword matches"""
        query_terms = set(query.lower().split())

        for chunk in chunks:
            text_lower = chunk['text'].lower()
            matched_terms = sum(1 for term in query_terms if term in text_lower)

            if matched_terms > 0:
                match_ratio = matched_terms / len(query_terms)
                chunk['score'] = chunk['score'] + (boost_factor * match_ratio)
                chunk['keyword_boost'] = match_ratio

        chunks.sort(key=lambda x: x['score'], reverse=True)
        return chunks

@dataclass
class KGProbeResult:
    document_ids: List[str]
    section_paths: List[str]
    hits: Dict[str, int]
    source: str = "neo4j"

class GraphGuidedRetriever:
    """Lightweight KG probe + score boost layer for retrieval."""

    def __init__(
        self,
        *,
        neo4j_store: Optional[Neo4jStore],
        cache: Optional[RedisJsonCache],
        entity_extractor: Optional[EntityExtractor] = None,
        probe_limit: int = 20,
        probe_timeout_ms: int = 80,
        cache_ttl_seconds: int = 1200,
    ) -> None:
        self.neo4j_store = neo4j_store
        self.cache = cache
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.probe_limit = int(probe_limit)
        self.probe_timeout_ms = int(probe_timeout_ms)
        self.cache_ttl_seconds = int(cache_ttl_seconds)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").strip().lower().split())

    def _cache_key(self, tenant: str, collection: str, normalized_query: str) -> str:
        return f"kgprobe:{tenant}:{collection}:{hash_query(normalized_query)}"

    def _entity_cache_key(self, tenant: str, normalized_query: str) -> str:
        return f"kgentities:{tenant}:{hash_query(normalized_query)}"

    def get_cached_probe(self, *, tenant: str, collection: str, query: str) -> Optional[KGProbeResult]:
        if not self.cache:
            return None
        normalized = self._normalize_query(query)
        cached = self.cache.get_json(self._cache_key(tenant, collection, normalized), feature="kgprobe")
        if not cached:
            return None
        return KGProbeResult(
            document_ids=cached.get("document_ids") or [],
            section_paths=cached.get("section_paths") or [],
            hits=cached.get("hits") or {},
            source="cache",
        )

    def _extract_entities(self, *, tenant: str, query: str) -> List[str]:
        normalized = self._normalize_query(query)
        if self.cache:
            cached = self.cache.get_json(self._entity_cache_key(tenant, normalized), feature="kgprobe")
            if cached:
                return cached.get("entity_ids") or []
        entities = self.entity_extractor.extract(query)
        entity_ids = [ent.entity_id for ent in entities]
        if self.cache:
            self.cache.set_json(
                self._entity_cache_key(tenant, normalized),
                stamp_cache_payload({"entity_ids": entity_ids}),
                feature="kgprobe",
                ttl=max(300, self.cache_ttl_seconds // 2),
            )
        return entity_ids

    def probe(self, *, tenant: str, collection: str, query: str) -> KGProbeResult:
        normalized = self._normalize_query(query)
        if self.cache:
            cached = self.cache.get_json(self._cache_key(tenant, collection, normalized), feature="kgprobe")
            if cached:
                return KGProbeResult(
                    document_ids=cached.get("document_ids") or [],
                    section_paths=cached.get("section_paths") or [],
                    hits=cached.get("hits") or {},
                    source="cache",
                )

        entity_ids = self._extract_entities(tenant=tenant, query=normalized)
        if not entity_ids or not self.neo4j_store:
            result = KGProbeResult(document_ids=[], section_paths=[], hits={}, source="empty")
            if self.cache:
                self.cache.set_json(
                    self._cache_key(tenant, collection, normalized),
                    stamp_cache_payload(result.__dict__),
                    feature="kgprobe",
                    ttl=self.cache_ttl_seconds,
                )
            return result

        start = time.time()
        try:
            hits = self.neo4j_store.probe_entities(
                entity_ids=entity_ids,
                limit=self.probe_limit,
                timeout_ms=self.probe_timeout_ms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("KG probe failed: %s", exc)
            hits = []

        doc_hits: Dict[str, int] = {}
        section_paths: List[str] = []
        for row in hits:
            doc_id = str(row.get("document_id") or "")
            section_path = row.get("section_path")
            hit_count = int(row.get("hits") or 0)
            if doc_id:
                doc_hits[doc_id] = doc_hits.get(doc_id, 0) + hit_count
            if section_path:
                section_paths.append(str(section_path))

        ranked_docs = [doc for doc, _ in sorted(doc_hits.items(), key=lambda item: item[1], reverse=True)]
        unique_sections = list(dict.fromkeys(section_paths))
        result = KGProbeResult(
            document_ids=ranked_docs,
            section_paths=unique_sections,
            hits=doc_hits,
            source="neo4j",
        )
        if self.cache:
            self.cache.set_json(
                self._cache_key(tenant, collection, normalized),
                stamp_cache_payload(result.__dict__),
                feature="kgprobe",
                ttl=self.cache_ttl_seconds,
            )
        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms > self.probe_timeout_ms:
            logger.debug("KG probe exceeded budget: %.1fms", elapsed_ms)
        return result

    @staticmethod
    def apply_boosts(
        chunks: List[Dict[str, Any]],
        probe: KGProbeResult,
        *,
        doc_boost: float = 0.12,
        section_boost: float = 0.08,
    ) -> List[Dict[str, Any]]:
        if not chunks or not probe.document_ids:
            return chunks
        doc_set = set(probe.document_ids)
        section_set = set([s.lower() for s in (probe.section_paths or [])])
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            doc_id = str(meta.get("document_id") or "")
            if doc_id and doc_id in doc_set:
                chunk["score"] = float(chunk.get("score", 0.0)) + doc_boost
                meta["kg_boost"] = True
            section_path = (meta.get("section_path") or meta.get("section_title") or "").strip().lower()
            if section_path and section_path in section_set:
                chunk["score"] = float(chunk.get("score", 0.0)) + section_boost
                meta["kg_section_boost"] = True
            chunk["metadata"] = meta
        chunks.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return chunks

    def _expand_with_adjacent_chunks(
            self,
            collection_name: str,
            chunks: List[Dict],
            subscription_id: str,
            profile_id: str,
            max_adjacent: int = 1
    ) -> List[Dict]:
        """
        FIXED: Only expand within SAME document
        """
        try:
            expanded = []
            chunk_ids_seen = set()

            for chunk in chunks:
                chunk_id = chunk['metadata'].get('chunk_id')
                current_doc_id = chunk['metadata'].get('document_id')  # ADDED

                if not chunk_id or chunk_id in chunk_ids_seen:
                    expanded.append(chunk)
                    continue

                chunk_ids_seen.add(chunk_id)
                expanded.append(chunk)

                # Get adjacent IDs
                prev_id = chunk['metadata'].get('prev_chunk_id')
                next_id = chunk['metadata'].get('next_chunk_id')

                adjacent_ids = []
                if prev_id and prev_id not in chunk_ids_seen:
                    adjacent_ids.append(prev_id)
                if next_id and next_id not in chunk_ids_seen:
                    adjacent_ids.append(next_id)

                # Fetch adjacent chunks
                for adj_id in adjacent_ids[:max_adjacent]:
                    try:
                        # FIXED: Filter by both chunk_id AND document_id
                        base = build_qdrant_filter(
                            subscription_id=str(subscription_id),
                            profile_id=str(profile_id),
                            document_id=str(current_doc_id),
                        )
                        must = list(getattr(base, "must", []) or [])
                        must.append(FieldCondition(key="chunk_id", match=MatchValue(value=adj_id)))
                        scroll_filter = Filter(must=must)

                        results = self.client.scroll(
                            collection_name=collection_name,
                            scroll_filter=scroll_filter,
                            limit=1,
                            with_payload=True,
                            with_vectors=False
                        )

                        if hasattr(results, 'points'):
                            points = results.points or []
                        elif isinstance(results, tuple):
                            points = results[0] if len(results) > 0 else []
                        else:
                            points = []

                        for pt in points:
                            if adj_id not in chunk_ids_seen:
                                payload = pt.payload or {}
                                # VERIFY: Adjacent chunk is from same document
                                adj_doc_id = payload.get('document_id')
                                if adj_doc_id == current_doc_id:  # ADDED CHECK
                                    expanded.append({
                                        'id': str(pt.id),
                                        'text': get_canonical_text(payload),
                                        'score': chunk['score'] * 0.95,
                                        'metadata': payload,
                                        'method': f"{chunk['method']}_adjacent"
                                    })
                                    chunk_ids_seen.add(adj_id)
                                else:
                                    logger.warning(
                                        f"Skipped adjacent chunk from different document: {adj_doc_id} vs {current_doc_id}")

                    except Exception as adj_err:
                        logger.debug(f"Failed to fetch adjacent chunk {adj_id}: {adj_err}")

            logger.info(f"Expanded from {len(chunks)} to {len(expanded)} chunks (same-document only)")
            return expanded

        except Exception as e:
            logger.error(f"Adjacent expansion failed: {e}", exc_info=True)
            return chunks

    def retrieve_adaptive(
            self,
            collection_name: str,
            query: str,
            subscription_id: str,
            profile_id: str,
            top_k: int = 30,
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None,
            use_expansion: bool = True,
            use_keyword_boost: bool = True
    ) -> List[Dict]:
        """
        Main adaptive retrieval - FIXED FOR MULTI-DOCUMENT
        """
        logger.info(f"Adaptive retrieval for query: {query[:100]}")
        if document_ids:
            logger.info(f"Filtering by document_ids: {document_ids}")
        if source_files:
            logger.info(f"Filtering by source_files: {source_files}")

        # Strategy 1: Dense search with moderate threshold
        chunks = self._dense_search(
            collection_name, query, subscription_id, profile_id, top_k,
            document_ids=document_ids,
            source_files=source_files,
            score_threshold=0.15
        )

        # Strategy 2: Fallback without threshold
        if not chunks:
            logger.debug("No results with threshold, retrying without")
            chunks = self._dense_search(
                collection_name, query, subscription_id, profile_id, top_k,
                document_ids=document_ids,
                source_files=source_files,
                score_threshold=None
            )

        # Strategy 3: Remove document filter if still empty
        if not chunks and (document_ids or source_files):
            logger.debug("No results with document filter, retrying without filter")
            chunks = self._dense_search(
                collection_name, query, subscription_id, profile_id, top_k * 2,
                score_threshold=None
            )

        if not chunks:
            logger.error("All retrieval strategies failed")
            return []

        # Keyword boosting
        if use_keyword_boost:
            chunks = self._keyword_boost_rerank(chunks, query)

        # Adjacent chunk expansion (FIXED: same-document only)
        if use_expansion and len(chunks) > 0:
            chunks = self._expand_with_adjacent_chunks(
                collection_name,
                chunks,
                subscription_id=subscription_id,
                profile_id=profile_id,
                max_adjacent=1,
            )

        # Add methods metadata
        for chunk in chunks:
            if 'methods' not in chunk:
                chunk['methods'] = [chunk.get('method', 'unknown')]

        logger.info(f"Final retrieval: {len(chunks)} chunks")
        if chunks:
            unique_docs = list(set([get_source_name(c.get("metadata") or {}) or "unknown" for c in chunks]))
            logger.info(f"Unique documents in results: {unique_docs}")
            logger.info(f"Top 5 scores: {[round(c['score'], 4) for c in chunks[:5]]}")

        return chunks
