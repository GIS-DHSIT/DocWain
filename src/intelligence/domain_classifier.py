"""Semantic domain classification using embedding similarity.

Replaces keyword-counting with cosine similarity against domain prototype
vectors.  The sentence-transformer model (BGE-large) is already loaded at
startup — we piggyback on it for zero additional memory cost.

Falls back to a lightweight keyword scorer only when the embedder is
unavailable (e.g., during bare-metal unit tests).
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = get_logger(__name__)

DOMAIN_LABELS = [
    "resume",
    "tax",
    "invoice",
    "purchase_order",
    "bank_statement",
    "medical",
    "legal",
    "policy",
    "generic",
]

# ── Domain prototype sentences ──────────────────────────────────────────
# Each domain gets 3-5 representative sentences.  The embedder encodes
# them once; at classification time we compare the document embedding
# against each prototype centroid.

_DOMAIN_PROTOTYPES: Dict[str, List[str]] = {
    "resume": [
        "This is a professional resume listing work experience and education.",
        "Candidate has 5 years of software engineering experience at Google.",
        "Skills include Python, Java, project management, and team leadership.",
        "Career objective: seeking a senior developer role in fintech.",
        "Bachelor of Science in Computer Science from MIT, graduated 2018.",
        "Professional certifications: AWS Solutions Architect, PMP, Scrum Master.",
        "References available upon request. Languages: English, Spanish fluent.",
    ],
    "tax": [
        "Federal income tax return for the fiscal year 2024.",
        "W-2 wage and tax statement showing gross income and withholding.",
        "Schedule C reporting self-employment business income and deductions.",
        "IRS Form 1099 reporting non-employee compensation.",
        "Total taxable income after adjustments and standard deduction.",
    ],
    "invoice": [
        "Commercial invoice number INV-2024-0042 dated January 15 2024.",
        "Bill to: Acme Corp. Amount due: $4,250.00. Payment terms: Net 30.",
        "Line item: 100 units of Widget-A at $42.50 each, subtotal $4,250.",
        "Purchase order PO-9876. Remittance address: 123 Business Ave.",
        "Invoice total including tax: $4,572.50. Balance due upon receipt.",
    ],
    "purchase_order": [
        "Purchase order number PO-2024-1234 for office supplies.",
        "Requisition approved by procurement department on January 10.",
        "Delivery date: February 1, 2024. Ship to warehouse B.",
        "Order line: 500 reams of paper, unit price $5.00, total $2,500.",
    ],
    "bank_statement": [
        "Bank account statement for period January 1 to January 31 2024.",
        "Opening balance: $12,450.00. Closing balance: $15,230.00.",
        "Transaction: direct deposit salary $5,000 on January 15.",
        "ATM withdrawal $200.00 on January 20 at Main Street branch.",
    ],
    "medical": [
        "Patient presented with chest pain and shortness of breath.",
        "Diagnosis: Type 2 diabetes mellitus with peripheral neuropathy.",
        "Prescribed metformin 500mg twice daily with meals.",
        "Lab results: HbA1c 7.2%, fasting glucose 145 mg/dL.",
        "Medical history includes hypertension and prior MI in 2019.",
    ],
    "legal": [
        "This agreement is entered into between Party A and Party B.",
        "The parties hereby agree to the following terms and conditions.",
        "In witness whereof the parties have executed this contract.",
        "Governing law: this agreement shall be governed by New York law.",
        "Indemnification clause: Party A shall indemnify Party B against claims.",
        "Non-disclosure agreement: confidential information shall not be disclosed.",
        "Force majeure: neither party shall be liable for acts beyond reasonable control.",
    ],
    "policy": [
        "Insurance policy number POL-2024-5678 effective January 1 2024.",
        "Coverage: comprehensive automobile insurance with $500 deductible.",
        "Premium: $1,200 annually. Policyholder: John Smith.",
        "Exclusions: intentional damage, racing, and commercial use.",
        "Claim procedure: notify insurer within 48 hours of incident.",
    ],
    "generic": [
        "Technical user manual describing product features and operation.",
        "Company standard operating procedure for project management.",
        "Product specification sheet with technical parameters.",
        "Meeting minutes from the quarterly business review.",
        "Research report analyzing market trends and competitive landscape.",
        "Training guide with step-by-step instructions for new employees.",
        "Project proposal outlining scope, timeline, and resource requirements.",
    ],
}

# ── Cached prototype centroids ──────────────────────────────────────────

_centroids: Optional[Dict[str, Any]] = None
_centroid_lock = threading.Lock()

def _get_embedder() -> Optional[Any]:
    """Retrieve the already-loaded sentence-transformer embedder."""
    try:
        from src.api.rag_state import get_app_state
        state = get_app_state()
        if state and state.embedding_model:
            return state.embedding_model
    except Exception:
        pass
    try:
        from src.api.dw_newron import get_model
        return get_model()
    except Exception:
        pass
    return None

def _build_centroids(embedder: Any) -> Dict[str, Any]:
    """Encode prototype sentences and compute per-domain centroids."""
    global _centroids
    if _centroids is not None:
        return _centroids

    with _centroid_lock:
        if _centroids is not None:
            return _centroids

        result: Dict[str, Any] = {}
        for domain, sentences in _DOMAIN_PROTOTYPES.items():
            vecs = embedder.encode(sentences, normalize_embeddings=True)
            centroid = np.mean(vecs, axis=0)
            # Normalize centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            result[domain] = centroid

        _centroids = result
        logger.info("Domain classifier: built %d prototype centroids", len(result))
        return result

def _semantic_classify(text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[DomainClassification]:
    """Classify document using embedding cosine similarity against prototypes."""
    embedder = _get_embedder()
    if embedder is None:
        return None

    try:
        centroids = _build_centroids(embedder)
        if not centroids:
            return None

        # Stratified sample: header + middle + tail captures both preamble and body content
        _raw = text or ""
        if len(_raw) > 3000:
            sample = _raw[:1000] + " " + _raw[len(_raw)//2 - 250:len(_raw)//2 + 250] + " " + _raw[-500:]
        else:
            sample = _raw[:2000]
        if len(sample.strip()) < 20:
            return None

        doc_vec = embedder.encode([sample], normalize_embeddings=True)[0]

        # Compute cosine similarity to each domain centroid
        scores: Dict[str, float] = {}
        for domain, centroid in centroids.items():
            sim = float(np.dot(doc_vec, centroid))
            scores[domain] = sim

        # Apply metadata bonus for doc_type/filename hints
        if metadata:
            doc_type = str(metadata.get("doc_type") or metadata.get("document_type") or "").lower().strip()
            source_name = str(metadata.get("source_name") or metadata.get("filename") or "").lower().strip()
            for hint, domain in DOC_TYPE_HINTS.items():
                if domain in scores:
                    if hint and hint in doc_type:
                        scores[domain] += 0.15  # Strong metadata signal
                    if hint and hint in source_name:
                        scores[domain] += 0.10

        # Pick winner
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_domain, best_score = ranked[0]
        second_domain, second_score = ranked[1] if len(ranked) > 1 else ("generic", 0.0)

        # Confidence: relative gap ratio between top-1 and top-2
        # Using ratio instead of absolute gap for consistency with keyword fallback
        gap = best_score - second_score
        gap_ratio = gap / (best_score + second_score + 1e-6)
        # High absolute scores (>= 0.70) with clear leads (gap >= 0.05) are confident
        # even if the gap_ratio is narrow (common when related domains score similarly)
        confident = (gap_ratio >= 0.10 and best_score >= 0.35) or (best_score >= 0.70 and gap >= 0.05)

        # If top domain is generic and second is close, prefer generic
        # (avoids forcing a specific domain on truly generic documents)
        if best_domain != "generic" and not confident:
            generic_score = scores.get("generic", 0.0)
            if generic_score >= best_score - 0.02:
                best_domain = "generic"
                best_score = generic_score
                confident = True

        confidence = min(1.0, best_score)
        uncertain = not confident

        return DomainClassification(
            domain=best_domain,
            confidence=confidence,
            scores=scores,
            method="semantic",
            uncertain=uncertain,
        )
    except Exception as exc:
        logger.debug("Semantic domain classification failed: %s", exc)
        return None

# ── Lightweight keyword fallback (no regex, only substring match) ────────
# Used when embedder is unavailable (testing, cold start).

# Strong indicator phrases — multi-word, nearly unambiguous
_STRONG_INDICATORS: Dict[str, Tuple[str, ...]] = {
    "invoice": (
        "invoice number", "invoice date", "purchase order number",
        "amount due", "bill to", "remittance", "payment terms",
        "invoice",
    ),
    "resume": (
        "work experience", "professional experience", "career objective",
        "curriculum vitae", "professional summary", "key skills",
        "resume", "candidate",
    ),
    "legal": (
        "party of the first part", "hereinafter referred to",
        "in witness whereof", "governing law", "indemnification",
    ),
    "medical": (
        "chief complaint", "medical history", "review of systems",
        "history of present illness", "physical examination",
        "patient", "diagnosis",
    ),
    "bank_statement": (
        "account statement", "available balance", "statement period",
        "opening balance", "closing balance",
    ),
    "policy": (
        "insurance policy", "policy number", "sum insured",
        "coverage period", "exclusions and conditions",
    ),
    "tax": (
        "tax return", "taxable income", "tax withholding",
        "w-2", "1099", "schedule c", "irs form",
        "tax refund", "tax deduction",
    ),
}

DOC_TYPE_HINTS = {
    "cv": "resume",
    "resume": "resume",
    "invoice": "invoice",
    "po": "purchase_order",
    "purchase order": "purchase_order",
    "bank statement": "bank_statement",
    "statement": "bank_statement",
    "medical": "medical",
    "contract": "legal",
    "agreement": "legal",
    "tax": "tax",
    "insurance": "policy",
    "policy": "policy",
}

@dataclass(frozen=True)
class DomainClassification:
    domain: str
    confidence: float
    scores: Dict[str, float]
    method: str
    uncertain: bool

def _keyword_fallback_classify(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> DomainClassification:
    """Keyword fallback when embedder is unavailable.

    Only counts strong multi-word indicator phrases (NOT single generic
    words like "total", "experience", "policy" which cause false positives).
    """
    lowered = (text or "").lower()
    scores: Dict[str, float] = {domain: 0.0 for domain in DOMAIN_LABELS}

    # Only match strong multi-word indicators
    for domain, phrases in _STRONG_INDICATORS.items():
        for phrase in phrases:
            if phrase in lowered:
                scores[domain] += 2.0

    # Metadata bonuses
    if metadata:
        doc_type = str(metadata.get("doc_type") or metadata.get("document_type") or "").lower().strip()
        source_name = str(metadata.get("source_name") or metadata.get("filename") or "").lower().strip()
        for hint, domain in DOC_TYPE_HINTS.items():
            if domain in scores:
                if hint and hint in doc_type:
                    scores[domain] += 3.0
                if hint and hint in source_name:
                    scores[domain] += 2.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score <= 0:
        return DomainClassification(
            domain="generic", confidence=0.0, scores=scores,
            method="keyword_fallback", uncertain=True,
        )

    confidence = best_score / max(best_score + second_score, 1.0)
    gap_ratio = (best_score - second_score) / (best_score + second_score + 1e-6)
    uncertain = gap_ratio < 0.10 or best_score < 2.0

    return DomainClassification(
        domain=best_domain, confidence=confidence, scores=scores,
        method="keyword_fallback", uncertain=uncertain,
    )

def classify_domain(
    document_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    llm_labeler: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
) -> DomainClassification:
    """Classify a document's domain.

    Strategy:
    1. Semantic similarity (embedding-based) — primary, accurate
    2. LLM labeler — when semantic is uncertain and LLM is available
    3. Keyword fallback — when embedder unavailable (only strong phrases)
    """
    # 1. Try semantic classification first
    result = _semantic_classify(document_text, metadata)
    if result is not None and not result.uncertain:
        return result

    # 2. If uncertain and LLM is available, try LLM
    if result is not None and result.uncertain and llm_labeler:
        try:
            llm_domain = llm_labeler(document_text, metadata or {})
            if llm_domain and llm_domain in DOMAIN_LABELS:
                return DomainClassification(
                    domain=llm_domain,
                    confidence=max(result.confidence, 0.6),
                    scores=result.scores,
                    method="llm",
                    uncertain=False,
                )
        except Exception:
            pass

    # 3. Return semantic result if available (even if uncertain)
    if result is not None:
        return result

    # 4. Keyword fallback (only when embedder unavailable)
    fallback = _keyword_fallback_classify(document_text, metadata)
    if fallback.uncertain and llm_labeler:
        try:
            llm_domain = llm_labeler(document_text, metadata or {})
            if llm_domain and llm_domain in DOMAIN_LABELS:
                return DomainClassification(
                    domain=llm_domain,
                    confidence=max(fallback.confidence, 0.6),
                    scores=fallback.scores,
                    method="llm",
                    uncertain=False,
                )
        except Exception:
            pass

    return fallback

def infer_domain(document_text: str, *, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Convenience: return just the domain string."""
    return classify_domain(document_text, metadata).domain

def reset_centroids() -> None:
    """Clear cached centroids (for testing)."""
    global _centroids
    _centroids = None

# ── Canonical domain normalization ──────────────────────────────────────
# Maps legacy/variant labels from all classifiers to a canonical set.
# Unknown labels pass through as-is (enabling generic analysis of any doc type).
_CANONICAL_DOMAIN_MAP: Dict[str, str] = {
    # Classifier 1 (document_classifier.py) enum names
    "RESUME": "resume", "CV": "resume",
    "INVOICE": "invoice", "PURCHASE_ORDER": "invoice",
    "BANK_STATEMENT": "financial", "TAX_DOCUMENT": "financial",
    "MEDICAL_RECORD": "medical",
    "LEGAL_DOCUMENT": "legal", "LEGAL_CONTRACT": "legal",
    "INSURANCE_CLAIM": "policy",
    "GENERIC": "general",
    # Classifier 3 (identify.py) labels
    "contract": "legal", "statement": "financial",
    "brochure": "general", "presentation": "general",
    "report": "general", "other": "general",
    # Synonyms
    "purchase_order": "invoice", "bank_statement": "financial",
    "tax": "financial", "insurance": "policy",
    "generic": "general",
}

def normalize_domain(label: str) -> str:
    """Normalize any domain label to a canonical form.

    Known domains map to their canonical name. Unknown labels pass through
    unchanged (lowercase, stripped) — this enables generic analysis of
    document types not in the predefined set.
    """
    cleaned = (label or "").strip()
    # Check both original and lowercased
    canonical = _CANONICAL_DOMAIN_MAP.get(cleaned)
    if canonical:
        return canonical
    canonical = _CANONICAL_DOMAIN_MAP.get(cleaned.lower())
    if canonical:
        return canonical
    return cleaned.lower() or "general"

# Keep for backward compat — some modules import these
DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {d: () for d in DOMAIN_LABELS}

__all__ = ["DOMAIN_LABELS", "DomainClassification", "classify_domain", "infer_domain", "reset_centroids"]
