"""
Embedding Pipeline Enhancement - Integrates enhanced embedding into the training flow.

This module provides the integration layer between the enhanced embedding components
and the existing training pipeline. Uses UniversalEmbeddingEnhancer for document-agnostic
improvements that work across resumes, invoices, legal documents, and all other document types.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = get_logger(__name__)

# Import universal enhancer for document-agnostic improvements
try:
    from src.embedding.universal_enhancer import (
        UniversalEmbeddingEnhancer,
        EnhancedEmbeddingResult,
        ContentTypeDetector,
        SemanticFieldExtractor,
        QualityScorer,
        enrich_query as universal_enrich_query,
    )
    UNIVERSAL_ENHANCER_AVAILABLE = True
except ImportError:
    UNIVERSAL_ENHANCER_AVAILABLE = False
    logger.debug("Universal enhancer not available, using fallback enhancement")

@dataclass
class EmbeddingEnhancementResult:
    """Result of embedding enhancement process."""
    enhanced_texts: List[str]  # Texts to embed (context-enriched)
    enhanced_metadata: List[Dict[str, Any]]  # Updated chunk metadata
    original_count: int
    deduplicated_count: int
    average_quality_score: float
    enhancement_stats: Dict[str, Any]

def enhance_chunks_for_embedding(
    texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
    document_metadata: Dict[str, Any],
    domain: str = "generic",
    embedder=None,
) -> EmbeddingEnhancementResult:
    """
    Enhance chunks before embedding for better retrieval quality.

    This function:
    1. Deduplicates similar chunks
    2. Detects content type automatically (contact, skills, education, etc.)
    3. Enriches embedding text with section context
    4. Extracts semantic keywords, entities, and key phrases
    5. Calculates multi-dimensional quality scores
    6. Stores enhanced semantic fields for filtering/boosting

    Args:
        texts: List of chunk texts
        chunk_metadata: List of chunk metadata dictionaries
        document_metadata: Document-level metadata
        domain: Document domain (resume, invoice, legal, etc.)

    Returns:
        EmbeddingEnhancementResult with enhanced data
    """
    if not texts:
        return EmbeddingEnhancementResult(
            enhanced_texts=[],
            enhanced_metadata=[],
            original_count=0,
            deduplicated_count=0,
            average_quality_score=0.0,
            enhancement_stats={},
        )

    # Use universal enhancer for document-agnostic improvements
    if UNIVERSAL_ENHANCER_AVAILABLE:
        return _enhance_with_universal_enhancer(
            texts=texts,
            chunk_metadata=chunk_metadata,
            document_metadata=document_metadata,
            domain=domain,
            embedder=embedder,
        )

    # Fallback to legacy enhancement
    return _enhance_with_legacy_pipeline(
        texts=texts,
        chunk_metadata=chunk_metadata,
        document_metadata=document_metadata,
        domain=domain,
    )

def _enhance_with_universal_enhancer(
    texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
    document_metadata: Dict[str, Any],
    domain: str = "generic",
    embedder=None,
) -> EmbeddingEnhancementResult:
    """
    Enhance chunks using the universal enhancer for document-agnostic improvements.
    """
    try:
        # Initialize universal enhancer (with optional ML embedder)
        enhancer = UniversalEmbeddingEnhancer(embedder=embedder)

        # Prepare chunks for enhancement
        chunks_for_enhancement = []
        for idx, (text, meta) in enumerate(zip(texts, chunk_metadata)):
            chunks_for_enhancement.append({
                "text": text,
                "section_title": meta.get("section_title", ""),
                "section_path": meta.get("section_path", ""),
                "section_kind": meta.get("section_kind", ""),
                "page": meta.get("page", 1),
                "chunk_index": idx,
            })

        # Deduplicate first
        deduped_texts, deduped_metadata, deduped_chunks = deduplicate_chunks_with_info(
            texts=texts,
            metadata=chunk_metadata,
            chunks=chunks_for_enhancement,
            similarity_threshold=0.85,
        )

        # Batch-detect content types (single encode call instead of per-chunk)
        section_titles_list = [c.get("section_title", "") for c in deduped_chunks]
        content_infos = enhancer.content_detector.detect_ml_batch(
            deduped_texts, section_titles_list,
        )

        # Enhance each chunk with universal enhancer (content type pre-computed)
        enhanced_texts = []
        enhanced_metadata = []
        quality_scores = []
        content_type_counts = {}

        doc_type = document_metadata.get("document_type", domain)

        for idx, (text, meta, chunk) in enumerate(zip(deduped_texts, deduped_metadata, deduped_chunks)):
            # Enhance chunk with pre-computed content info
            result = enhancer.enhance_chunk(
                text=text,
                section_title=chunk.get("section_title", ""),
                section_path=chunk.get("section_path", ""),
                document_type=doc_type,
                document_domain=domain,
                _content_info=content_infos[idx] if idx < len(content_infos) else None,
            )

            # Use context-enriched embedding text for vector generation
            enhanced_texts.append(result.embedding_text)

            # Build enhanced payload with semantic fields
            enhanced_payload = enhancer.build_enhanced_payload(result, base_payload=meta)

            # Add original chunk index for tracking
            original_idx = chunk.get("chunk_index", idx)
            enhanced_payload["chunk_index"] = original_idx
            enhanced_payload["page"] = chunk.get("page", 1)
            enhanced_payload["section_title"] = chunk.get("section_title", "")
            enhanced_payload["section_path"] = chunk.get("section_path", "")

            enhanced_metadata.append(enhanced_payload)
            quality_scores.append(result.quality_score.overall)

            # Track content type distribution
            ct = result.content_type
            content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

        # Calculate statistics
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        deduped_count = len(texts) - len(enhanced_texts)

        stats = {
            "domain": domain,
            "document_type": doc_type,
            "original_chunks": len(texts),
            "enhanced_chunks": len(enhanced_texts),
            "deduplicated": deduped_count,
            "average_quality": avg_quality,
            "quality_distribution": {
                "high": sum(1 for s in quality_scores if s >= 0.7),
                "medium": sum(1 for s in quality_scores if 0.4 <= s < 0.7),
                "low": sum(1 for s in quality_scores if s < 0.4),
            },
            "content_types": content_type_counts,
            "enhancer": "universal",
        }

        logger.info(
            f"Universal enhanced {len(texts)} chunks -> {len(enhanced_texts)} "
            f"(deduped: {deduped_count}, avg quality: {avg_quality:.2f})"
        )

        return EmbeddingEnhancementResult(
            enhanced_texts=enhanced_texts,
            enhanced_metadata=enhanced_metadata,
            original_count=len(texts),
            deduplicated_count=len(enhanced_texts),
            average_quality_score=avg_quality,
            enhancement_stats=stats,
        )

    except Exception as e:
        logger.error(f"Universal enhancement failed: {e}. Falling back to legacy.", exc_info=True)
        return _enhance_with_legacy_pipeline(
            texts=texts,
            chunk_metadata=chunk_metadata,
            document_metadata=document_metadata,
            domain=domain,
        )

def _enhance_with_legacy_pipeline(
    texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
    document_metadata: Dict[str, Any],
    domain: str = "generic",
) -> EmbeddingEnhancementResult:
    """
    Fallback legacy enhancement pipeline.
    """
    try:
        from src.embedding.enhanced_embedding import EnhancedEmbeddingBuilder, build_enhanced_payload
        from src.embedding.query_aware_embedding import QueryAwareEmbedder

        # Initialize builders
        embedding_builder = EnhancedEmbeddingBuilder(domain=domain)
        query_embedder = QueryAwareEmbedder(domain=domain)

        # Prepare chunks for enhancement
        chunks_for_enhancement = []
        for idx, (text, meta) in enumerate(zip(texts, chunk_metadata)):
            chunks_for_enhancement.append({
                "text": text,
                "content": text,
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "section_title": meta.get("section_title", ""),
                "section_path": meta.get("section_path", ""),
                "section_kind": meta.get("section_kind", ""),
                "page": meta.get("page", 1),
                **meta,
            })

        # Build enhanced chunks
        enhanced_chunks = embedding_builder.build_enhanced_chunks(
            chunks=chunks_for_enhancement,
            document_metadata=document_metadata,
        )

        # Prepare results
        enhanced_texts = []
        enhanced_metadata = []
        quality_scores = []

        for chunk in enhanced_chunks:
            # Use embedding text (context-enriched) for vector generation
            enhanced_texts.append(chunk.embedding_text)

            # Update metadata with enhancements
            chunk_meta = {
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "section_path": chunk.section_path,
                "section_type": chunk.section_type,
                "content_type": chunk.content_type,
                "quality_score": chunk.quality_score,
                "content_hash": chunk.content_hash,
                "semantic_keywords": chunk.semantic_keywords,
                "entities": chunk.entities,
                "key_phrases": chunk.key_phrases,
                "embedding_text": chunk.embedding_text,
                "canonical_text": chunk.canonical_text,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }

            # Merge with original metadata
            original_meta = chunk_metadata[chunk.chunk_index] if chunk.chunk_index < len(chunk_metadata) else {}
            merged_meta = {**original_meta, **chunk_meta}
            enhanced_metadata.append(merged_meta)

            quality_scores.append(chunk.quality_score)

        # Calculate statistics
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        dedupe_count = len(texts) - len(enhanced_texts)

        stats = {
            "domain": domain,
            "original_chunks": len(texts),
            "enhanced_chunks": len(enhanced_texts),
            "deduplicated": dedupe_count,
            "average_quality": avg_quality,
            "quality_distribution": {
                "high": sum(1 for s in quality_scores if s >= 0.7),
                "medium": sum(1 for s in quality_scores if 0.4 <= s < 0.7),
                "low": sum(1 for s in quality_scores if s < 0.4),
            },
            "enhancer": "legacy",
        }

        logger.info(
            f"Legacy enhanced {len(texts)} chunks -> {len(enhanced_texts)} "
            f"(deduped: {dedupe_count}, avg quality: {avg_quality:.2f})"
        )

        return EmbeddingEnhancementResult(
            enhanced_texts=enhanced_texts,
            enhanced_metadata=enhanced_metadata,
            original_count=len(texts),
            deduplicated_count=len(enhanced_texts),
            average_quality_score=avg_quality,
            enhancement_stats=stats,
        )

    except ImportError as e:
        logger.debug(f"Enhanced embedding not available: {e}. Using original texts.")
        return EmbeddingEnhancementResult(
            enhanced_texts=list(texts),
            enhanced_metadata=list(chunk_metadata),
            original_count=len(texts),
            deduplicated_count=len(texts),
            average_quality_score=0.5,
            enhancement_stats={"fallback": True, "reason": str(e)},
        )
    except Exception as e:
        logger.error(f"Enhancement failed: {e}. Using original texts.", exc_info=True)
        return EmbeddingEnhancementResult(
            enhanced_texts=list(texts),
            enhanced_metadata=list(chunk_metadata),
            original_count=len(texts),
            deduplicated_count=len(texts),
            average_quality_score=0.5,
            enhancement_stats={"fallback": True, "error": str(e)},
        )

def build_optimal_embedding_text(
    text: str,
    section_title: str,
    section_type: str,
    document_type: str,
    domain: str = "generic",
) -> str:
    """
    Build optimal embedding text for a single chunk.

    This is a lightweight function for cases where you need to enhance
    a single chunk's embedding text without the full pipeline.

    Args:
        text: Original chunk text
        section_title: Section title/heading
        section_type: Type of section (skills, experience, etc.)
        document_type: Type of document (resume, invoice, etc.)
        domain: Document domain

    Returns:
        Optimized embedding text
    """
    # Domain-specific prefixes
    domain_prefixes = {
        "resume": {
            "skills": "Technical Skills: ",
            "experience": "Work Experience: ",
            "education": "Education: ",
            "certifications": "Certifications: ",
            "contact": "Contact Information: ",
            "summary": "Professional Summary: ",
        },
        "invoice": {
            "line_items": "Invoice Items: ",
            "totals": "Invoice Totals: ",
            "parties": "Billing Information: ",
            "terms": "Payment Terms: ",
        },
        "legal": {
            "clause": "Legal Clause: ",
            "terms": "Terms and Conditions: ",
            "definitions": "Definitions: ",
        },
    }

    # Get prefix for domain and section type
    prefixes = domain_prefixes.get(domain, {})
    prefix = prefixes.get(section_type, "")

    # If no specific prefix, use section title
    if not prefix and section_title:
        prefix = f"{section_title}: "

    return f"{prefix}{text}".strip()

def calculate_chunk_quality(
    text: str,
    section_type: str = "content",
    has_entities: bool = False,
) -> float:
    """
    Calculate a quality score for a chunk (0.0 - 1.0).

    Higher scores indicate chunks that are more likely to be useful for retrieval.

    Args:
        text: Chunk text
        section_type: Type of section
        has_entities: Whether entities were extracted

    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 0.5  # Base score

    text_len = len(text.strip())

    # Length scoring
    if 200 <= text_len <= 800:
        score += 0.2  # Ideal length
    elif 100 <= text_len <= 1000:
        score += 0.1  # Acceptable length
    elif text_len < 50:
        score -= 0.2  # Too short
    elif text_len > 2000:
        score -= 0.1  # Too long

    # Section type scoring
    if section_type in ["skills", "experience", "education", "certifications"]:
        score += 0.15  # High-value sections
    elif section_type != "content":
        score += 0.05  # Known section type

    # Entity presence
    if has_entities:
        score += 0.1

    # Sentence completeness
    if text.strip() and text.strip()[-1] in ".!?":
        score += 0.05

    # Bullet point detection (often high-value)
    if any(text.strip().startswith(b) for b in ["•", "-", "*", "1.", "2."]):
        score += 0.05

    return min(1.0, max(0.0, score))

def deduplicate_chunks(
    texts: List[str],
    metadata: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Remove near-duplicate chunks based on content similarity.

    Args:
        texts: List of chunk texts
        metadata: List of chunk metadata
        similarity_threshold: Jaccard similarity threshold

    Returns:
        Deduplicated (texts, metadata) tuple
    """
    if len(texts) <= 1:
        return texts, metadata

    kept_texts, kept_metadata, _ = deduplicate_chunks_with_info(
        texts=texts,
        metadata=metadata,
        chunks=None,
        similarity_threshold=similarity_threshold,
    )

    return kept_texts, kept_metadata

def deduplicate_chunks_with_info(
    texts: List[str],
    metadata: List[Dict[str, Any]],
    chunks: Optional[List[Dict[str, Any]]],
    similarity_threshold: float = 0.85,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Remove near-duplicate chunks with full information tracking.

    Args:
        texts: List of chunk texts
        metadata: List of chunk metadata
        chunks: Optional list of chunk dictionaries (for universal enhancer)
        similarity_threshold: Jaccard similarity threshold

    Returns:
        Tuple of (deduplicated texts, metadata, chunks)
    """
    if len(texts) <= 1:
        return texts, metadata, chunks or [{}] * len(texts)

    def _normalize(text: str) -> set:
        """Normalize text to word set for comparison."""
        return set(text.lower().split())

    def _jaccard(set_a: set, set_b: set) -> float:
        """Calculate Jaccard similarity."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    kept_texts = []
    kept_metadata = []
    kept_chunks = []
    seen_sets = []

    # Ensure chunks list exists
    if chunks is None:
        chunks = [{} for _ in texts]

    for text, meta, chunk in zip(texts, metadata, chunks):
        word_set = _normalize(text)

        # Check against all kept chunks
        is_duplicate = False
        for seen in seen_sets:
            if _jaccard(word_set, seen) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_texts.append(text)
            kept_metadata.append(meta)
            kept_chunks.append(chunk)
            seen_sets.append(word_set)

    duplicates_removed = len(texts) - len(kept_texts)
    if duplicates_removed > 0:
        logger.info(f"Deduplicated {duplicates_removed} similar chunks")

    return kept_texts, kept_metadata, kept_chunks

def enrich_query_for_retrieval(query: str) -> Dict[str, Any]:
    """
    Enrich a query for better retrieval matching.

    This adds semantic context to queries to improve vector similarity matching
    with the enhanced embedding texts stored in Qdrant.

    Args:
        query: User query text

    Returns:
        Dict with enriched_query and query metadata
    """
    if UNIVERSAL_ENHANCER_AVAILABLE:
        from src.embedding.universal_enhancer import QueryEnricher
        enricher = QueryEnricher()
        return enricher.enrich(query)

    # Fallback: return original query with basic enrichment
    return {
        "original_query": query,
        "enriched_query": query,
        "query_type": "general",
        "expansion_terms": [],
    }

def get_enhanced_embedding_text(
    text: str,
    section_title: str = "",
    section_path: str = "",
    document_type: str = "",
) -> str:
    """
    Get context-enriched embedding text for a single chunk.

    This is useful for re-embedding individual chunks or for query-time
    enhancement of search terms.

    Args:
        text: Original chunk text
        section_title: Section heading
        section_path: Full section hierarchy
        document_type: Type of document

    Returns:
        Enhanced embedding text
    """
    if UNIVERSAL_ENHANCER_AVAILABLE:
        enhancer = UniversalEmbeddingEnhancer()
        result = enhancer.enhance_chunk(
            text=text,
            section_title=section_title,
            section_path=section_path,
            document_type=document_type,
        )
        return result.embedding_text

    # Fallback: use build_optimal_embedding_text
    return build_optimal_embedding_text(
        text=text,
        section_title=section_title,
        section_type="content",
        document_type=document_type,
        domain="generic",
    )

def infer_document_domain(
    texts: List[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Infer document domain from content and metadata.

    Args:
        texts: Chunk texts
        metadata: Document metadata

    Returns:
        Inferred domain (resume, invoice, legal, medical, generic)
    """
    # Check explicit domain in metadata
    explicit_domain = metadata.get("domain") or metadata.get("doc_domain")
    if explicit_domain and explicit_domain != "generic":
        return explicit_domain.lower()

    # Check document type
    doc_type = (metadata.get("document_type") or metadata.get("doc_type") or "").lower()
    type_to_domain = {
        "resume": "resume",
        "cv": "resume",
        "invoice": "invoice",
        "contract": "legal",
        "agreement": "legal",
        "medical": "medical",
        "clinical": "medical",
    }
    for type_key, domain in type_to_domain.items():
        if type_key in doc_type:
            return domain

    # Infer from content
    combined_text = " ".join(texts[:10]).lower()  # Sample first 10 chunks

    domain_indicators = {
        "resume": ["experience", "education", "skills", "certifications", "objective"],
        "invoice": ["invoice", "amount due", "bill to", "payment terms", "subtotal"],
        "legal": ["clause", "agreement", "liability", "warranty", "indemnify"],
        "medical": ["diagnosis", "patient", "medication", "treatment", "symptoms"],
    }

    scores = {domain: 0 for domain in domain_indicators}
    for domain, indicators in domain_indicators.items():
        scores[domain] = sum(1 for ind in indicators if ind in combined_text)

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] >= 2:
        return best_domain

    return "generic"

__all__ = [
    "EmbeddingEnhancementResult",
    "enhance_chunks_for_embedding",
    "build_optimal_embedding_text",
    "calculate_chunk_quality",
    "deduplicate_chunks",
    "deduplicate_chunks_with_info",
    "infer_document_domain",
    "enrich_query_for_retrieval",
    "get_enhanced_embedding_text",
    "UNIVERSAL_ENHANCER_AVAILABLE",
]
