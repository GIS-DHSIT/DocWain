"""
Query-Aware Embedding - Creates embeddings optimized for common query patterns.

This module generates additional embedding representations that anticipate
how users will query the document, improving retrieval accuracy.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

# Common query patterns by domain
QUERY_PATTERNS = {
    "resume": [
        # Skills queries
        ("skills", "What are {name}'s technical skills?"),
        ("skills", "What technologies does {name} know?"),
        ("skills", "List the programming languages {name} is proficient in"),

        # Experience queries
        ("experience", "What is {name}'s work experience?"),
        ("experience", "Where has {name} worked?"),
        ("experience", "How many years of experience does {name} have?"),

        # Education queries
        ("education", "What is {name}'s educational background?"),
        ("education", "Where did {name} study?"),
        ("education", "What degree does {name} have?"),

        # Contact queries
        ("contact", "What is {name}'s email address?"),
        ("contact", "How can I contact {name}?"),
        ("contact", "What is {name}'s phone number?"),

        # Certification queries
        ("certifications", "What certifications does {name} have?"),
        ("certifications", "Is {name} certified in any technology?"),

        # Summary queries
        ("summary", "Give me a summary of {name}'s profile"),
        ("summary", "What is {name}'s professional background?"),
    ],
    "invoice": [
        ("line_items", "What items are on this invoice?"),
        ("line_items", "List the products/services billed"),
        ("totals", "What is the total amount?"),
        ("totals", "What is the invoice amount due?"),
        ("parties", "Who is this invoice for?"),
        ("parties", "Who issued this invoice?"),
        ("terms", "What are the payment terms?"),
        ("terms", "When is this invoice due?"),
    ],
    "legal": [
        ("clause", "What does this clause say about liability?"),
        ("clause", "What are the termination conditions?"),
        ("terms", "What are the terms and conditions?"),
        ("definitions", "How is {term} defined in this document?"),
    ],
}

@dataclass
class QueryAwareChunk:
    """A chunk with query-pattern augmented embeddings."""
    original_chunk_id: str
    section_type: str
    original_text: str

    # Primary embedding text (context-enriched)
    primary_embedding_text: str

    # Query-pattern embeddings (text variants for common queries)
    query_embeddings: List[str]

    # Metadata
    anticipated_queries: List[str]

class QueryAwareEmbedder:
    """
    Creates query-aware embeddings that anticipate how users will search.

    Instead of just embedding the raw document text, this creates
    embeddings that include common query patterns, improving retrieval
    when users ask predictable questions.
    """

    def __init__(self, domain: str = "generic"):
        self.domain = domain.lower()
        self.patterns = QUERY_PATTERNS.get(self.domain, [])

    def create_query_aware_text(
        self,
        text: str,
        section_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Create query-aware embedding text.

        Args:
            text: Original chunk text
            section_type: Type of section (skills, experience, etc.)
            metadata: Additional metadata (e.g., candidate name)

        Returns:
            Tuple of (primary embedding text, list of query-pattern texts)
        """
        metadata = metadata or {}

        # Build primary embedding with section context
        primary = self._build_primary_embedding(text, section_type, metadata)

        # Generate query-pattern augmented texts
        query_texts = self._generate_query_patterns(text, section_type, metadata)

        return primary, query_texts

    def _build_primary_embedding(
        self,
        text: str,
        section_type: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Build the primary embedding text with context."""
        name = metadata.get("candidate_name", "the candidate")

        # Section-specific prefixes
        prefixes = {
            "skills": f"Technical Skills and Competencies of {name}: ",
            "experience": f"Work Experience and Career History of {name}: ",
            "education": f"Education and Academic Background of {name}: ",
            "certifications": f"Certifications and Credentials of {name}: ",
            "contact": f"Contact Information for {name}: ",
            "summary": f"Professional Summary and Profile of {name}: ",
            "line_items": "Invoice Items and Products/Services: ",
            "totals": "Invoice Totals and Payment Amounts: ",
            "parties": "Billing and Vendor Information: ",
            "clause": "Legal Clause and Provision: ",
        }

        prefix = prefixes.get(section_type, f"{section_type.title()}: ")
        return f"{prefix}{text}"

    def _generate_query_patterns(
        self,
        text: str,
        section_type: str,
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Generate query-pattern augmented embedding texts."""
        query_texts = []
        name = metadata.get("candidate_name", "the candidate")

        # Get patterns for this section type
        relevant_patterns = [
            (st, q) for st, q in self.patterns
            if st == section_type
        ]

        for _, query_template in relevant_patterns[:3]:  # Limit to 3 patterns
            try:
                query = query_template.format(name=name, term="the term")

                # Create "question-answer" style embedding
                # This helps match when users ask similar questions
                qa_text = f"Question: {query}\nAnswer: {text}"
                query_texts.append(qa_text)
            except (KeyError, ValueError):
                continue

        return query_texts

    def augment_chunks_for_embedding(
        self,
        chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Augment chunks with query-aware embedding texts.

        Args:
            chunks: List of chunk dictionaries
            document_metadata: Document-level metadata

        Returns:
            Chunks with added 'embedding_text' and 'query_embeddings' fields
        """
        document_metadata = document_metadata or {}
        augmented_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "") or chunk.get("content", "")
            section_type = self._infer_section_type(chunk)

            primary_text, query_texts = self.create_query_aware_text(
                text=text,
                section_type=section_type,
                metadata=document_metadata,
            )

            augmented_chunk = dict(chunk)
            augmented_chunk["embedding_text"] = primary_text
            augmented_chunk["query_embeddings"] = query_texts
            augmented_chunk["section_type_inferred"] = section_type

            augmented_chunks.append(augmented_chunk)

        return augmented_chunks

    def _infer_section_type(self, chunk: Dict[str, Any]) -> str:
        """Infer section type from chunk metadata."""
        # Check explicit metadata
        section_type = chunk.get("section_type") or chunk.get("section_kind")
        if section_type and section_type != "misc":
            return section_type

        # Infer from title
        title = (chunk.get("section_title") or chunk.get("section", "") or "").lower()
        text = (chunk.get("text") or chunk.get("content") or "").lower()

        type_patterns = {
            "skills": ["skill", "technolog", "competenc", "proficienc"],
            "experience": ["experience", "employment", "work history"],
            "education": ["education", "degree", "university", "academic"],
            "certifications": ["certif", "credential", "license"],
            "contact": ["contact", "email", "phone", "address"],
            "summary": ["summary", "objective", "profile", "overview"],
        }

        for section_type, patterns in type_patterns.items():
            if any(p in title for p in patterns):
                return section_type
            if any(p in text[:200] for p in patterns):
                return section_type

        return "content"

def create_multi_view_embeddings(
    text: str,
    section_type: str,
    embedder: Any,
    document_metadata: Optional[Dict[str, Any]] = None,
    domain: str = "generic",
) -> Dict[str, Any]:
    """
    Create multiple embedding views for a chunk.

    This generates:
    1. Primary embedding (context-enriched)
    2. Query-pattern embeddings (anticipating user questions)
    3. Entity-focused embedding (for entity extraction)

    Args:
        text: Chunk text
        section_type: Section type
        embedder: Embedding model
        document_metadata: Document metadata
        domain: Document domain

    Returns:
        Dictionary with multiple embedding vectors
    """
    qa_embedder = QueryAwareEmbedder(domain=domain)

    primary_text, query_texts = qa_embedder.create_query_aware_text(
        text=text,
        section_type=section_type,
        metadata=document_metadata or {},
    )

    # Generate embeddings
    all_texts = [primary_text] + query_texts

    # Encode all at once for efficiency
    if hasattr(embedder, "encode"):
        embeddings = embedder.encode(
            all_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return {
            "primary_embedding": embeddings[0].tolist(),
            "primary_text": primary_text,
            "query_embeddings": [e.tolist() for e in embeddings[1:]],
            "query_texts": query_texts,
        }

    return {
        "primary_text": primary_text,
        "query_texts": query_texts,
    }

__all__ = [
    "QueryAwareChunk",
    "QueryAwareEmbedder",
    "create_multi_view_embeddings",
    "QUERY_PATTERNS",
]
