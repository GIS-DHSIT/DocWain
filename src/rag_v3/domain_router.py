"""
Domain Router for DocWain RAG Pipeline.

Routes queries to appropriate domain-specific handlers based on:
- Explicit tool hints from the request
- Query content analysis
- Document metadata signals
- Retrieved chunk characteristics

Supported domains:
- resume: Resume/CV analysis and extraction
- legal: Legal document processing
- financial: Financial document analysis
- technical: Technical documentation
- generic: Default catch-all handler
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config

logger = get_logger(__name__)

@dataclass
class DomainSignal:
    """A signal contributing to domain classification."""

    domain: str
    confidence: float
    source: str  # "hint", "query", "metadata", "content"

    def __repr__(self) -> str:
        return f"DomainSignal({self.domain}, {self.confidence:.2f}, {self.source})"

@dataclass
class DomainDecision:
    """Result of domain routing decision."""

    domain: str
    confidence: float
    signals: List[DomainSignal]
    is_ambiguous: bool = False

    @property
    def primary_signal(self) -> Optional[DomainSignal]:
        """The strongest signal contributing to this decision."""
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.confidence)

# Keywords and patterns for domain detection
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "resume": [
        "resume", "cv", "curriculum vitae", "work experience", "employment",
        "qualifications", "career", "job history",
        "professional experience", "candidate",
    ],
    "medical": [
        "patient", "diagnosis", "treatment", "medical", "clinical",
        "health", "prescription", "symptom", "hospital", "doctor",
        "physician", "medication", "condition", "prognosis",
        "medical history", "patient details",
    ],
    "legal": [
        "contract", "agreement", "clause", "liability", "indemnity",
        "legal", "attorney", "court", "plaintiff", "defendant",
        "jurisdiction", "statute", "regulation", "compliance",
    ],
    "financial": [
        "revenue", "profit", "expense", "balance sheet", "income statement",
        "cash flow", "investment", "stock", "equity", "dividend",
        "financial statement", "audit", "fiscal", "budget",
    ],
    "technical": [
        "api", "documentation", "specification", "architecture",
        "implementation", "code", "function", "class", "method",
        "endpoint", "request", "response", "parameter", "schema",
    ],
    "policy": [
        "insurance", "policy", "premium", "coverage", "claim",
        "deductible", "exclusion", "underwriting", "insured",
        "beneficiary", "endorsement", "peril", "policyholder",
        "indemnity", "natural calamity", "natural disaster",
    ],
}

# Metadata fields that indicate specific domains
METADATA_DOMAIN_INDICATORS: Dict[str, Dict[str, str]] = {
    "document_type": {
        "resume": "resume",
        "cv": "resume",
        "curriculum_vitae": "resume",
        "contract": "legal",
        "agreement": "legal",
        "financial_statement": "financial",
        "balance_sheet": "financial",
        "api_docs": "technical",
        "specification": "technical",
    },
    "category": {
        "hr": "resume",
        "legal": "legal",
        "finance": "financial",
        "engineering": "technical",
    },
}

class DomainRouter:
    """
    Routes queries to appropriate domain-specific handlers.

    Uses multi-signal detection combining:
    1. Explicit tool hints (highest priority)
    2. Query content analysis
    3. Document metadata
    4. Retrieved content characteristics
    """

    # Confidence thresholds
    HINT_CONFIDENCE = 1.0
    QUERY_KEYWORD_CONFIDENCE = 0.7
    METADATA_CONFIDENCE = 0.8
    CONTENT_CONFIDENCE = 0.5
    AMBIGUITY_THRESHOLD = 0.3  # Difference below which we consider ambiguous

    @staticmethod
    def resolve(
        query: str,
        tool_hint: Optional[str],
        retrieved_metadata: Optional[dict],
    ) -> str:
        """
        Resolve the domain for a query.

        This is the main entry point, providing backward compatibility.

        Args:
            query: The user's query.
            tool_hint: Optional explicit domain hint.
            retrieved_metadata: Optional metadata from retrieved documents.

        Returns:
            The resolved domain name.
        """
        decision = DomainRouter.route(query, tool_hint, retrieved_metadata)
        return decision.domain

    @classmethod
    def route(
        cls,
        query: str,
        tool_hint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_samples: Optional[List[str]] = None,
    ) -> DomainDecision:
        """
        Make a domain routing decision with full context.

        Args:
            query: The user's query.
            tool_hint: Optional explicit domain hint.
            metadata: Optional document/chunk metadata.
            content_samples: Optional content excerpts for analysis.

        Returns:
            DomainDecision with domain, confidence, and signals.
        """
        signals: List[DomainSignal] = []

        # Check if domain-specific routing is enabled
        if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
            return DomainDecision(
                domain="generic",
                confidence=1.0,
                signals=[DomainSignal("generic", 1.0, "config_disabled")],
            )

        # 1. Check explicit tool hint (highest priority)
        if tool_hint:
            hint_domain = cls._normalize_domain(tool_hint)
            if hint_domain != "generic":
                signals.append(DomainSignal(hint_domain, cls.HINT_CONFIDENCE, "hint"))

        # 2. Analyze query content
        query_signals = cls._analyze_query(query)
        signals.extend(query_signals)

        # 3. Check metadata
        if metadata:
            metadata_signals = cls._analyze_metadata(metadata)
            signals.extend(metadata_signals)

        # 4. Check content samples
        if content_samples:
            content_signals = cls._analyze_content(content_samples)
            signals.extend(content_signals)

        # Aggregate signals and make decision
        return cls._aggregate_signals(signals)

    @classmethod
    def _normalize_domain(cls, domain: str) -> str:
        """Normalize a domain name to a known domain."""
        if not domain:
            return "generic"

        normalized = domain.strip().lower()

        # Direct matches
        known_domains = {"resume", "legal", "financial", "technical", "generic", "medical", "policy"}
        if normalized in known_domains:
            return normalized

        # Aliases
        aliases = {
            "cv": "resume",
            "curriculum_vitae": "resume",
            "hr": "resume",
            "contract": "legal",
            "law": "legal",
            "finance": "financial",
            "accounting": "financial",
            "tech": "technical",
            "api": "technical",
            "docs": "technical",
            "healthcare": "medical",
            "clinical": "medical",
            "health": "medical",
            "hospital": "medical",
            "insurance": "policy",
            "underwriting": "policy",
        }
        return aliases.get(normalized, "generic")

    @classmethod
    def _analyze_query(cls, query: str) -> List[DomainSignal]:
        """Analyze query content for domain signals."""
        signals = []
        query_lower = query.lower()

        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                # Scale confidence by number of matches
                confidence = min(cls.QUERY_KEYWORD_CONFIDENCE, 0.3 + (matches * 0.1))
                signals.append(DomainSignal(domain, confidence, "query"))

        return signals

    @classmethod
    def _analyze_metadata(cls, metadata: Dict[str, Any]) -> List[DomainSignal]:
        """Analyze document metadata for domain signals."""
        signals = []

        for field, domain_map in METADATA_DOMAIN_INDICATORS.items():
            value = metadata.get(field, "")
            if isinstance(value, str):
                value_lower = value.lower().replace(" ", "_")
                if value_lower in domain_map:
                    domain = domain_map[value_lower]
                    signals.append(DomainSignal(domain, cls.METADATA_CONFIDENCE, "metadata"))

        # Check for domain field directly in metadata
        explicit_domain = metadata.get("domain") or metadata.get("doc_domain")
        if explicit_domain:
            normalized = cls._normalize_domain(str(explicit_domain))
            if normalized != "generic":
                signals.append(DomainSignal(normalized, cls.METADATA_CONFIDENCE, "metadata"))

        return signals

    @classmethod
    def _analyze_content(cls, content_samples: List[str]) -> List[DomainSignal]:
        """Analyze content samples for domain signals."""
        signals = []
        combined_content = " ".join(content_samples).lower()

        domain_scores: Dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined_content)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            max_score = max(domain_scores.values())
            for domain, score in domain_scores.items():
                if score >= max_score * 0.5:  # Include domains with at least 50% of max score
                    confidence = min(cls.CONTENT_CONFIDENCE, 0.2 + (score * 0.05))
                    signals.append(DomainSignal(domain, confidence, "content"))

        return signals

    @classmethod
    def _aggregate_signals(cls, signals: List[DomainSignal]) -> DomainDecision:
        """Aggregate signals into a final domain decision."""
        if not signals:
            return DomainDecision(
                domain="generic",
                confidence=0.5,
                signals=[],
            )

        # Aggregate confidence by domain
        domain_confidence: Dict[str, float] = {}
        domain_signals: Dict[str, List[DomainSignal]] = {}

        for signal in signals:
            if signal.domain not in domain_confidence:
                domain_confidence[signal.domain] = 0.0
                domain_signals[signal.domain] = []

            # Use max confidence for same source, otherwise add with decay
            domain_signals[signal.domain].append(signal)

            # Weight by source priority
            source_weight = {
                "hint": 1.0,
                "metadata": 0.8,
                "query": 0.6,
                "content": 0.4,
            }.get(signal.source, 0.5)

            weighted_confidence = signal.confidence * source_weight
            domain_confidence[signal.domain] = max(
                domain_confidence[signal.domain],
                weighted_confidence,
            )

        # Find best domain
        sorted_domains = sorted(
            domain_confidence.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        best_domain, best_confidence = sorted_domains[0]

        # Check for ambiguity
        is_ambiguous = False
        if len(sorted_domains) > 1:
            second_confidence = sorted_domains[1][1]
            if best_confidence - second_confidence < cls.AMBIGUITY_THRESHOLD:
                is_ambiguous = True
                logger.debug(
                    "Ambiguous domain routing: %s (%.2f) vs %s (%.2f)",
                    best_domain, best_confidence,
                    sorted_domains[1][0], second_confidence,
                )

        return DomainDecision(
            domain=best_domain,
            confidence=best_confidence,
            signals=domain_signals.get(best_domain, []),
            is_ambiguous=is_ambiguous,
        )

# ── Embedding-based zero-shot classification ─────────────────────────

_DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    "resume": "Resume or CV with work experience, education, skills, career history, candidate profile",
    "invoice": "Invoice or bill with line items, amounts, totals, payment terms, billing information",
    "legal": "Legal contract or agreement with clauses, terms, liabilities, governing law",
    "medical": "Medical record with patient info, diagnoses, treatments, prescriptions, clinical notes",
    "financial": "Financial report with revenue, expenses, balance sheet, investment data, fiscal year",
    "technical": "Technical documentation with API specs, code, architecture, implementation details",
    "policy": "Insurance policy document with coverage terms, premiums, exclusions, claims, deductibles, beneficiaries",
}

_CACHED_DOMAIN_EMBEDDINGS: Dict[str, Any] = {}

def classify_by_embedding(text_sample: str, embedder: Any) -> Tuple[str, float]:
    """Zero-shot domain classification using embedding cosine similarity.

    Uses the already-loaded sentence-transformer model to compare document text
    against domain description templates.  Domain description embeddings are
    computed once and cached for the lifetime of the process.

    Args:
        text_sample: A representative text sample from the document (first ~500 chars)
        embedder: A sentence-transformers model with ``.encode()`` method

    Returns:
        (domain, confidence) tuple.
    """
    import numpy as np

    if not text_sample or not embedder or not hasattr(embedder, "encode"):
        return "generic", 0.5

    # Cache domain description embeddings on first call
    if not _CACHED_DOMAIN_EMBEDDINGS:
        for domain, desc in _DOMAIN_DESCRIPTIONS.items():
            _CACHED_DOMAIN_EMBEDDINGS[domain] = embedder.encode(
                [desc], normalize_embeddings=True,
            )[0]

    try:
        doc_vec = embedder.encode(
            [text_sample[:500]], normalize_embeddings=True,
        )[0]
    except Exception:
        return "generic", 0.5

    best_domain, best_score = "generic", 0.0
    for domain, desc_vec in _CACHED_DOMAIN_EMBEDDINGS.items():
        sim = float(np.dot(doc_vec, desc_vec))
        if sim > best_score:
            best_score, best_domain = sim, domain

    return best_domain, best_score

__all__ = ["DomainRouter", "DomainDecision", "DomainSignal", "classify_by_embedding"]
