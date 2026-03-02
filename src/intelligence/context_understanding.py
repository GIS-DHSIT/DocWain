"""ML-based context understanding for document intelligence.

Uses embeddings and semantic analysis to deeply understand document content
before LLM extraction. This module is document-type agnostic and works with
any domain (HR, medical, legal, invoice, policy, generic).

Key capabilities:
  - Semantic chunk clustering: groups chunks by topic/section using cosine similarity
  - Query-evidence alignment: scores how well each chunk addresses the query
  - Entity salience: detects which entities are most relevant to the query
  - Cross-document relationships: finds connections between documents
  - Context distillation: produces a structured context summary for LLM prompts

The output is a ContextUnderstanding object that enriches the LLM prompt with
structured knowledge, enabling more accurate and intelligent responses.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────
_CLUSTER_SIMILARITY_THRESHOLD = 0.72  # cosine sim to merge into same cluster
_MAX_CLUSTERS = 8
_MIN_ALIGNMENT_SCORE = 0.25  # min query-chunk alignment to consider relevant
_MAX_CONTEXT_ENTITIES = 12
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_KV_RE = re.compile(r"^(.{3,40})\s*[:]\s+(.{2,})", re.MULTILINE)
_NUMBER_RE = re.compile(r"\b\d[\d,.]+\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\w+ \d{1,2},?\s*\d{4}|"
    r"\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b"
)

# Common stopwords to exclude from topic extraction
_STOP = frozenset({
    "the", "and", "for", "are", "was", "were", "has", "have", "had",
    "this", "that", "with", "from", "not", "but", "all", "can", "will",
    "been", "being", "more", "than", "each", "which", "their", "also",
    "into", "some", "only", "other", "very", "just", "about", "would",
    "could", "should", "these", "those", "between", "through", "during",
})


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class TopicCluster:
    """A semantic cluster of related chunks."""
    topic_label: str
    chunk_indices: List[int]
    representative_terms: List[str]
    avg_score: float = 0.0


@dataclass
class EntitySalience:
    """An entity with its relevance score to the query."""
    name: str
    entity_type: str  # "person", "org", "number", "date", "term"
    salience: float  # 0.0 - 1.0
    mentions: int
    source_documents: List[str]


@dataclass
class QueryAlignment:
    """How well a chunk aligns with the query."""
    chunk_index: int
    alignment_score: float  # 0.0 - 1.0
    matching_aspects: List[str]


@dataclass
class DocumentRelationship:
    """Relationship between two documents."""
    doc_a: str
    doc_b: str
    relationship_type: str  # "similar_topic", "shared_entities", "complementary"
    strength: float


@dataclass
class StructuredFact:
    """A key-value fact extracted from document content."""
    key: str
    value: str
    source_doc: str
    confidence: float


@dataclass
class ContextUnderstanding:
    """Complete context understanding for a query + chunks."""
    topic_clusters: List[TopicCluster]
    entity_salience: List[EntitySalience]
    query_alignments: List[QueryAlignment]
    document_relationships: List[DocumentRelationship]
    structured_facts: List[StructuredFact]
    content_summary: str
    document_count: int
    total_chunks: int
    dominant_domain: str
    key_topics: List[str]

    def to_prompt_section(self) -> str:
        """Render as a structured section for LLM prompt injection."""
        parts: List[str] = []

        if self.content_summary:
            parts.append(f"DOCUMENT INTELLIGENCE:\n{self.content_summary}")

        if self.key_topics:
            parts.append(f"KEY TOPICS: {', '.join(self.key_topics[:6])}")

        if self.entity_salience:
            salient = [
                f"{e.name} ({e.entity_type}, relevance: {e.salience:.0%})"
                for e in self.entity_salience[:8]
                if e.salience >= 0.3
            ]
            if salient:
                parts.append(f"KEY ENTITIES: {'; '.join(salient)}")

        if self.structured_facts:
            top_facts = sorted(
                self.structured_facts, key=lambda f: f.confidence, reverse=True
            )[:10]
            fact_lines = [f"  - {f.key}: {f.value}" for f in top_facts]
            if fact_lines:
                parts.append("EXTRACTED FACTS:\n" + "\n".join(fact_lines))

        if self.document_relationships:
            rels = [
                f"  - {r.doc_a} ↔ {r.doc_b}: {r.relationship_type}"
                for r in self.document_relationships[:5]
            ]
            if rels:
                parts.append("DOCUMENT RELATIONSHIPS:\n" + "\n".join(rels))

        if self.topic_clusters and len(self.topic_clusters) > 1:
            cluster_desc = [
                f"  - {c.topic_label}: {', '.join(c.representative_terms[:4])}"
                for c in self.topic_clusters[:5]
            ]
            parts.append("CONTENT STRUCTURE:\n" + "\n".join(cluster_desc))

        return "\n\n".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_count": self.document_count,
            "total_chunks": self.total_chunks,
            "dominant_domain": self.dominant_domain,
            "key_topics": self.key_topics,
            "entity_count": len(self.entity_salience),
            "cluster_count": len(self.topic_clusters),
            "fact_count": len(self.structured_facts),
            "relationship_count": len(self.document_relationships),
        }


# ── Core analysis functions ───────────────────────────────────────────

def _get_chunk_text(chunk: Any) -> str:
    """Extract text from a chunk object."""
    return (getattr(chunk, "text", "") or "").strip()


def _get_chunk_doc_name(chunk: Any) -> str:
    """Extract document name from a chunk object."""
    meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", None) or {}
    source = getattr(chunk, "source", None)
    return (
        meta.get("source_name")
        or meta.get("document_name")
        or (getattr(source, "document_name", "") if source else "")
        or "Document"
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract_key_terms(text: str, top_k: int = 8) -> List[str]:
    """Extract top-k key terms from text using TF scoring."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    words = [w for w in words if w not in _STOP and len(w) > 2]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_k)]


def _extract_entities_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text using pattern matching.

    Returns (name, type) tuples. This is a fast heuristic — for production
    accuracy, the spaCy NER in the pipeline supplements this.
    """
    entities: List[Tuple[str, str]] = []

    # Proper noun entities
    for match in _ENTITY_RE.finditer(text):
        name = match.group()
        if len(name) > 2 and name.lower() not in _STOP:
            entities.append((name, "entity"))

    # Numeric values
    for match in _NUMBER_RE.finditer(text):
        val = match.group()
        if len(val) > 1:
            entities.append((val, "number"))

    # Dates
    for match in _DATE_RE.finditer(text):
        entities.append((match.group(), "date"))

    return entities


def _extract_structured_facts(text: str, doc_name: str) -> List[StructuredFact]:
    """Extract key-value facts from text using colon heuristic."""
    facts: List[StructuredFact] = []
    for match in _KV_RE.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if len(key) < 3 or len(value) < 2:
            continue
        # Skip if key looks like a URL or code
        if "://" in key or key.startswith("http"):
            continue
        # Estimate confidence based on structure clarity
        confidence = 0.8 if len(value) > 5 else 0.5
        facts.append(StructuredFact(
            key=key, value=value[:200], source_doc=doc_name, confidence=confidence,
        ))
    return facts


# ── Semantic clustering ───────────────────────────────────────────────

def _cluster_chunks(
    embeddings: np.ndarray,
    chunks: List[Any],
    threshold: float = _CLUSTER_SIMILARITY_THRESHOLD,
) -> List[TopicCluster]:
    """Cluster chunks by semantic similarity using greedy agglomerative approach.

    No external clustering library needed — uses simple cosine similarity
    with a greedy merge strategy.
    """
    n = len(chunks)
    if n == 0:
        return []
    if n == 1:
        text = _get_chunk_text(chunks[0])
        terms = _extract_key_terms(text, top_k=4)
        return [TopicCluster(
            topic_label=terms[0] if terms else "content",
            chunk_indices=[0],
            representative_terms=terms,
            avg_score=getattr(chunks[0], "score", 0.0) or 0.0,
        )]

    # Greedy single-linkage clustering
    assigned = [False] * n
    clusters: List[TopicCluster] = []

    for i in range(n):
        if assigned[i]:
            continue

        cluster_indices = [i]
        assigned[i] = True
        centroid = embeddings[i].copy()

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = _cosine_similarity(centroid, embeddings[j])
            if sim >= threshold:
                cluster_indices.append(j)
                assigned[j] = True
                # Update centroid as running average
                centroid = (centroid * (len(cluster_indices) - 1) + embeddings[j]) / len(cluster_indices)

        # Extract representative terms from cluster chunks
        cluster_text = " ".join(_get_chunk_text(chunks[idx]) for idx in cluster_indices)
        terms = _extract_key_terms(cluster_text, top_k=6)
        avg_score = np.mean([getattr(chunks[idx], "score", 0.0) or 0.0 for idx in cluster_indices])

        clusters.append(TopicCluster(
            topic_label=terms[0] if terms else f"topic_{len(clusters)}",
            chunk_indices=cluster_indices,
            representative_terms=terms,
            avg_score=float(avg_score),
        ))

        if len(clusters) >= _MAX_CLUSTERS:
            # Assign remaining to nearest cluster
            for k in range(n):
                if not assigned[k]:
                    best_sim = -1.0
                    best_ci = 0
                    for ci, c in enumerate(clusters):
                        c_centroid = np.mean(embeddings[c.chunk_indices], axis=0)
                        s = _cosine_similarity(c_centroid, embeddings[k])
                        if s > best_sim:
                            best_sim = s
                            best_ci = ci
                    clusters[best_ci].chunk_indices.append(k)
                    assigned[k] = True
            break

    # Sort clusters by average score (most relevant first)
    clusters.sort(key=lambda c: c.avg_score, reverse=True)
    return clusters


# ── Query-evidence alignment ─────────────────────────────────────────

def _compute_query_alignment(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: List[Any],
    query: str,
) -> List[QueryAlignment]:
    """Score how well each chunk addresses the query using embedding similarity + keyword overlap."""
    alignments: List[QueryAlignment] = []
    query_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower())) - _STOP

    for i, chunk in enumerate(chunks):
        # Semantic alignment via cosine similarity
        sem_score = _cosine_similarity(query_embedding, chunk_embeddings[i])

        # Keyword overlap boost
        chunk_text = _get_chunk_text(chunk).lower()
        chunk_terms = set(re.findall(r"[a-zA-Z]{3,}", chunk_text)) - _STOP
        if query_terms:
            overlap = len(query_terms & chunk_terms) / len(query_terms)
        else:
            overlap = 0.0

        # Combined score (70% semantic, 30% keyword)
        alignment_score = 0.7 * sem_score + 0.3 * overlap

        # Identify matching aspects
        matching = sorted(query_terms & chunk_terms)[:5]

        alignments.append(QueryAlignment(
            chunk_index=i,
            alignment_score=float(alignment_score),
            matching_aspects=matching,
        ))

    alignments.sort(key=lambda a: a.alignment_score, reverse=True)
    return alignments


# ── Entity salience ───────────────────────────────────────────────────

def _compute_entity_salience(
    chunks: List[Any],
    query: str,
    query_embedding: np.ndarray,
    embedder: Any,
) -> List[EntitySalience]:
    """Detect entities and score their relevance to the query."""
    entity_mentions: Dict[str, Dict[str, Any]] = {}  # name -> {type, count, docs}
    query_lower = query.lower()

    for chunk in chunks:
        text = _get_chunk_text(chunk)
        doc_name = _get_chunk_doc_name(chunk)
        entities = _extract_entities_from_text(text)

        for name, etype in entities:
            key = name.lower()
            if key not in entity_mentions:
                entity_mentions[key] = {
                    "name": name, "type": etype, "count": 0, "docs": set(),
                }
            entity_mentions[key]["count"] += 1
            entity_mentions[key]["docs"].add(doc_name)

    if not entity_mentions:
        return []

    # Compute salience: frequency + query relevance (keyword-based, no extra GPU encoding)
    results: List[EntitySalience] = []
    max_count = max(e["count"] for e in entity_mentions.values())

    for key, info in entity_mentions.items():
        freq_score = info["count"] / max(max_count, 1)

        # Query mention boost (keyword overlap — no GPU encoding needed)
        query_boost = 0.3 if key in query_lower else 0.0

        # Source diversity boost
        doc_boost = min(0.2, len(info["docs"]) * 0.05)

        salience = min(1.0, 0.4 * freq_score + query_boost + doc_boost + 0.1)

        results.append(EntitySalience(
            name=info["name"],
            entity_type=info["type"],
            salience=salience,
            mentions=info["count"],
            source_documents=sorted(info["docs"]),
        ))

    results.sort(key=lambda e: e.salience, reverse=True)
    return results[:_MAX_CONTEXT_ENTITIES]


# ── Cross-document relationships ──────────────────────────────────────

def _compute_document_relationships(
    chunks: List[Any],
    chunk_embeddings: np.ndarray,
) -> List[DocumentRelationship]:
    """Find relationships between documents using embedding similarity."""
    # Group chunk indices by document
    doc_chunks: Dict[str, List[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        doc_name = _get_chunk_doc_name(chunk)
        doc_chunks[doc_name].append(i)

    doc_names = list(doc_chunks.keys())
    if len(doc_names) < 2:
        return []

    relationships: List[DocumentRelationship] = []

    # Compute pairwise document similarity via centroid
    for i in range(min(len(doc_names), 10)):
        for j in range(i + 1, min(len(doc_names), 10)):
            indices_a = doc_chunks[doc_names[i]]
            indices_b = doc_chunks[doc_names[j]]

            centroid_a = np.mean(chunk_embeddings[indices_a], axis=0)
            centroid_b = np.mean(chunk_embeddings[indices_b], axis=0)
            sim = _cosine_similarity(centroid_a, centroid_b)

            # Check shared entities
            texts_a = " ".join(_get_chunk_text(chunks[k]) for k in indices_a)
            texts_b = " ".join(_get_chunk_text(chunks[k]) for k in indices_b)
            ents_a = {name.lower() for name, _ in _extract_entities_from_text(texts_a)}
            ents_b = {name.lower() for name, _ in _extract_entities_from_text(texts_b)}
            shared = ents_a & ents_b

            if sim > 0.7:
                rel_type = "similar_topic"
            elif shared:
                rel_type = "shared_entities"
            elif sim > 0.4:
                rel_type = "complementary"
            else:
                continue

            relationships.append(DocumentRelationship(
                doc_a=doc_names[i],
                doc_b=doc_names[j],
                relationship_type=rel_type,
                strength=float(sim),
            ))

    relationships.sort(key=lambda r: r.strength, reverse=True)
    return relationships[:10]


# ── Content summary ───────────────────────────────────────────────────

def _build_content_summary(
    chunks: List[Any],
    clusters: List[TopicCluster],
    query: str,
    alignments: List[QueryAlignment],
) -> str:
    """Build a concise content summary for LLM context."""
    doc_names = sorted(set(_get_chunk_doc_name(c) for c in chunks))
    n_docs = len(doc_names)
    n_chunks = len(chunks)

    # Count highly aligned chunks
    high_align = sum(1 for a in alignments if a.alignment_score >= _MIN_ALIGNMENT_SCORE)

    parts: List[str] = []
    parts.append(
        f"Analyzing {n_chunks} text segments from {n_docs} document(s). "
        f"{high_align} segments are directly relevant to the query."
    )

    if clusters:
        topic_desc = ", ".join(c.topic_label for c in clusters[:5])
        parts.append(f"Content covers: {topic_desc}.")

    if n_docs > 1:
        short_names = [n.replace(".pdf", "").replace(".docx", "")[:30] for n in doc_names[:6]]
        parts.append(f"Documents: {', '.join(short_names)}")

    return " ".join(parts)


# ── Main entry point ──────────────────────────────────────────────────

_MAX_CHUNKS_FOR_UNDERSTANDING = 16  # Limit to avoid OOM on large profiles


def _safe_encode(embedder: Any, texts: List[str]) -> Optional[np.ndarray]:
    """Encode texts with GPU OOM fallback to CPU."""
    try:
        return embedder.encode(texts, normalize_embeddings=True)
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "out of memory" in str(exc):
            # Clear GPU cache and retry on CPU
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                return embedder.encode(
                    texts, normalize_embeddings=True, device="cpu",
                )
            except Exception:
                pass
        return None


def understand_context(
    *,
    query: str,
    chunks: List[Any],
    embedder: Any,
    domain_hint: Optional[str] = None,
) -> Optional[ContextUnderstanding]:
    """Perform deep ML-based understanding of document context.

    This is the main entry point. It computes semantic clusters, query alignment,
    entity salience, document relationships, and structured facts from the chunks.

    Args:
        query: User query
        chunks: Retrieved and reranked chunks
        embedder: SentenceTransformer model for encoding
        domain_hint: Optional domain hint (hr, medical, legal, etc.)

    Returns:
        ContextUnderstanding object or None on failure
    """
    if not chunks or not embedder:
        return None

    try:
        # Limit chunks to avoid OOM on large profiles
        if len(chunks) > _MAX_CHUNKS_FOR_UNDERSTANDING:
            # Take top chunks by score
            scored = sorted(chunks, key=lambda c: getattr(c, "score", 0.0) or 0.0, reverse=True)
            chunks = scored[:_MAX_CHUNKS_FOR_UNDERSTANDING]

        # Encode query and chunks
        texts = [_get_chunk_text(c) for c in chunks]
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return None

        # Batch encode all texts + query together for efficiency
        all_texts = [query] + [t for _, t in non_empty]
        all_embeddings = _safe_encode(embedder, all_texts)
        if all_embeddings is None:
            return None

        query_embedding = all_embeddings[0]
        chunk_embeddings_map = {}
        for idx, (orig_i, _) in enumerate(non_empty):
            chunk_embeddings_map[orig_i] = all_embeddings[idx + 1]

        # Build full embedding array (zero for empty chunks)
        dim = all_embeddings.shape[1]
        chunk_embeddings = np.zeros((len(chunks), dim))
        for orig_i, emb in chunk_embeddings_map.items():
            chunk_embeddings[orig_i] = emb

        # 1. Semantic clustering
        valid_embeddings = all_embeddings[1:]  # skip query
        valid_chunks = [chunks[i] for i, _ in non_empty]
        clusters = _cluster_chunks(valid_embeddings, valid_chunks)

        # Remap cluster indices back to original chunk indices
        for cluster in clusters:
            cluster.chunk_indices = [non_empty[i][0] for i in cluster.chunk_indices if i < len(non_empty)]

        # 2. Query-evidence alignment
        alignments = _compute_query_alignment(query_embedding, chunk_embeddings, chunks, query)

        # 3. Entity salience
        entities = _compute_entity_salience(chunks, query, query_embedding, embedder)

        # 4. Cross-document relationships
        relationships = _compute_document_relationships(chunks, chunk_embeddings)

        # 5. Structured facts extraction
        all_facts: List[StructuredFact] = []
        for chunk in chunks:
            text = _get_chunk_text(chunk)
            doc_name = _get_chunk_doc_name(chunk)
            facts = _extract_structured_facts(text, doc_name)
            all_facts.extend(facts)
        # Deduplicate by key (keep highest confidence)
        seen_keys: Dict[str, StructuredFact] = {}
        for f in all_facts:
            key_lower = f.key.lower().strip()
            if key_lower not in seen_keys or f.confidence > seen_keys[key_lower].confidence:
                seen_keys[key_lower] = f
        all_facts = sorted(seen_keys.values(), key=lambda f: f.confidence, reverse=True)

        # 6. Content summary
        summary = _build_content_summary(chunks, clusters, query, alignments)

        # 7. Key topics (from top clusters)
        key_topics: List[str] = []
        for c in clusters[:5]:
            key_topics.extend(c.representative_terms[:2])
        key_topics = list(dict.fromkeys(key_topics))[:8]  # dedupe, top 8

        # 8. Detect dominant domain from chunk metadata
        domain_counts: Counter = Counter()
        for chunk in chunks:
            meta = getattr(chunk, "meta", None) or {}
            d = meta.get("doc_domain") or meta.get("domain") or ""
            if d:
                domain_counts[d.lower()] += 1
        dominant_domain = domain_hint or (domain_counts.most_common(1)[0][0] if domain_counts else "generic")

        doc_names = set(_get_chunk_doc_name(c) for c in chunks)

        return ContextUnderstanding(
            topic_clusters=clusters,
            entity_salience=entities,
            query_alignments=alignments,
            document_relationships=relationships,
            structured_facts=all_facts[:20],
            content_summary=summary,
            document_count=len(doc_names),
            total_chunks=len(chunks),
            dominant_domain=dominant_domain,
            key_topics=key_topics,
        )
    except Exception as exc:
        logger.warning("Context understanding failed: %s", exc)
        return None


def understand_context_for_prompt(
    *,
    query: str,
    chunks: List[Any],
    embedder: Any,
    domain_hint: Optional[str] = None,
    max_chars: int = 1200,
) -> str:
    """Convenience: compute context understanding and return prompt section.

    Returns empty string on failure (safe for prompt injection).
    """
    understanding = understand_context(
        query=query, chunks=chunks, embedder=embedder, domain_hint=domain_hint,
    )
    if understanding is None:
        return ""
    section = understanding.to_prompt_section()
    if len(section) > max_chars:
        section = section[:max_chars] + "\n..."
    return section
