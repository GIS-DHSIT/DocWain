"""Query-aware focus abstraction for chunk filtering, extraction, and rendering.

Computes a unified ``QueryFocus`` once at pipeline entry, then threads it
through chunk selection → extraction → rendering so that different queries
against the *same* document set produce *different* responses.

ML enhancements:
- **Semantic similarity**: cosine similarity via BAAI/bge-large-en-v1.5 embedder
- **Field importance classifier**: learned MLP for query→field_tag prediction
- **Section routing**: string-matching based section affinity scoring
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# QueryFocus dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryFocus:
    """Encapsulates *what the user actually wants* from a query."""
    keywords: List[str] = field(default_factory=list)
    bigrams: List[str] = field(default_factory=list)
    field_tags: Set[str] = field(default_factory=set)
    section_kinds: List[str] = field(default_factory=list)
    intent: str = "factual"
    is_exhaustive: bool = False
    # ML fields
    query_embedding: Optional[Any] = field(default=None, repr=False)
    field_probabilities: Optional[Dict[str, float]] = field(default=None, repr=False)
    _embedder: Optional[Any] = field(default=None, repr=False)
    _raw_query: str = ""


# ---------------------------------------------------------------------------
# Stop-words shared with extract._keywords()
# ---------------------------------------------------------------------------
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "from",
    "what", "how", "who", "where", "when", "which", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "this", "that", "these", "those", "me", "my", "your", "his", "her",
    "tell", "give", "show", "find", "get", "list", "all", "about",
})

# ---------------------------------------------------------------------------
# Section-kind map (mirrors retrieve._QUERY_SECTION_MAP but returns ALL)
# ---------------------------------------------------------------------------
_SECTION_MAP = [
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
    (("clause", "provision", "article", "section"), "clause"),
    (("liability", "indemnification", "indemnity"), "liability"),
    (("termination", "cancellation", "exit"), "termination"),
    (("confidentiality", "non-disclosure", "nda"), "confidentiality"),
    # Invoice sections
    (("line item", "product", "service", "description"), "items"),
    (("total", "subtotal", "amount due", "balance", "grand total"), "totals"),
    (("payment term", "due date", "net 30", "net 60"), "payment_terms"),
    # Policy sections
    (("coverage", "benefit", "protection", "insured"), "coverage"),
    (("exclusion", "exception", "limitation"), "exclusions"),
    (("claim", "filing", "procedure"), "claims"),
]

# ---------------------------------------------------------------------------
# Field-focus map — keyword fallback when ML classifier produces nothing
# ---------------------------------------------------------------------------
_FIELD_FOCUS_MAP = {
    "skill": {"skills"},
    "skills": {"skills"},
    "technical": {"skills"},
    "education": {"education"},
    "degree": {"education"},
    "university": {"education"},
    "college": {"education"},
    "qualification": {"education", "certifications"},
    "certif": {"certifications"},
    "experience": {"experience", "summary"},
    "years": {"experience"},
    "achievement": {"achievements"},
    "award": {"achievements"},
    "summary": {"summary"},
    "overview": {"summary"},
    "profile": {"summary", "experience", "skills"},
    "contact": {"contact"},
    "email": {"contact"},
    "phone": {"contact"},
    "linkedin": {"contact"},
    "address": {"contact"},
    "project": {"experience", "summary"},
    "work": {"experience", "summary"},
    "worked": {"experience", "summary"},
    # Invoice-focused
    "total": {"totals"},
    "amount": {"totals"},
    "balance": {"totals"},
    "subtotal": {"totals"},
    "tax": {"totals"},
    "discount": {"totals"},
    "price": {"totals", "items"},
    "cost": {"totals", "items"},
    "item": {"items"},
    "product": {"items"},
    "line item": {"items"},
    "quantity": {"items"},
    "vendor": {"parties"},
    "customer": {"parties"},
    "supplier": {"parties"},
    "bill": {"parties", "totals"},
    "invoice number": {"identifiers"},
    "purchase order": {"identifiers"},
    "due date": {"terms"},
    "payment": {"terms", "totals"},
    # Legal-focused
    "clause": {"clauses"},
    "liability": {"clauses"},
    "obligation": {"clauses"},
    "indemnification": {"clauses"},
    "warranty": {"clauses"},
    "termination": {"clauses"},
    "confidentiality": {"clauses"},
    "governing law": {"clauses"},
    "term": {"terms"},
    "effective date": {"terms"},
    "renewal": {"terms"},
    "notice period": {"terms"},
    "penalty": {"terms", "clauses"},
    # Medical-focused
    "diagnosis": {"diagnosis"},
    "condition": {"diagnosis"},
    "symptom": {"diagnosis"},
    "medication": {"treatment"},
    "prescription": {"treatment"},
    "drug": {"treatment"},
    "dosage": {"treatment"},
    "treatment": {"treatment"},
    "therapy": {"treatment"},
    "lab": {"lab_results"},
    "test result": {"lab_results"},
    "blood": {"lab_results"},
    "vitals": {"lab_results"},
    "allergy": {"medical_history"},
    "allergies": {"medical_history"},
    "history": {"medical_history", "experience"},
    "surgery": {"medical_history"},
    # Policy-focused
    "coverage": {"coverage"},
    "premium": {"coverage"},
    "deductible": {"coverage"},
    "exclusion": {"exclusions"},
    "rider": {"coverage"},
    "beneficiary": {"parties"},
    "claim": {"claims"},
    "policyholder": {"parties"},
}

_EXHAUSTIVE_RE = re.compile(
    r"\b(everything|all\s+details|full\s+profile|complete\s+profile|summarize\s+(?:the\s+)?(?:document|profile)|overview|entire)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Thread-local caches for ML scoring
# ---------------------------------------------------------------------------
_chunk_embed_cache = threading.local()
def clear_chunk_embed_cache() -> None:
    """Free per-request chunk embedding cache."""
    _chunk_embed_cache.cache = {}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_query_focus(
    query: str,
    intent_hint: Optional[str] = None,
    embedder: Optional[Any] = None,
) -> QueryFocus:
    """Compute a ``QueryFocus`` from a raw query string.

    When *embedder* is provided (a SentenceTransformer), the query is encoded
    once for semantic similarity scoring downstream.
    """
    if not query:
        return QueryFocus(is_exhaustive=True)

    lowered = query.lower()

    # --- keywords & bigrams (mirrors extract._keywords) ---
    tokens = re.findall(r"[A-Za-z0-9]+", lowered)
    singles = [tok for tok in tokens if tok not in _STOP and len(tok) > 2]
    bigrams: list[str] = []
    for i in range(len(tokens) - 1):
        if tokens[i] not in _STOP and tokens[i + 1] not in _STOP:
            bigrams.append(f"{tokens[i]} {tokens[i + 1]}")

    # --- ML: encode query embedding ---
    query_embedding = None
    if embedder is not None and query:
        try:
            query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        except Exception:
            log.debug("Failed to encode query for semantic similarity", exc_info=True)

    # --- ML-FIRST: field classifier predictions ---
    field_tags: set[str] = set()
    field_probabilities = None
    if query_embedding is not None:
        try:
            from .field_classifier import get_field_classifier
            clf = get_field_classifier()
            if clf is not None:
                field_probabilities = clf.predict(query_embedding)
                field_tags.update(field_probabilities.keys())
        except Exception:
            log.debug("Field classifier prediction failed", exc_info=True)

    # --- Keyword fallback: only when ML classifier produced nothing ---
    # Use word-boundary matching to prevent false positives
    # (e.g., "term" matching "terminal", "art" matching "article")
    if not field_tags:
        for keyword, tags in _FIELD_FOCUS_MAP.items():
            if " " in keyword:
                # Multi-word keywords: exact substring match is fine
                if keyword in lowered:
                    field_tags.update(tags)
            else:
                # Single-word keywords: require word boundary
                if re.search(r'\b' + re.escape(keyword) + r'\b', lowered):
                    field_tags.update(tags)

    # --- section kinds (return ALL matching, not just first) ---
    section_kinds: list[str] = []
    for keywords_tuple, kind in _SECTION_MAP:
        if any(kw in lowered for kw in keywords_tuple):
            section_kinds.append(kind)

    # --- exhaustive detection ---
    is_exhaustive = bool(_EXHAUSTIVE_RE.search(query))
    # Also exhaustive when no specific field tags detected and query is short/vague
    if not field_tags and not section_kinds and len(singles) <= 1:
        is_exhaustive = True

    # --- intent ---
    intent = intent_hint or "factual"

    return QueryFocus(
        keywords=singles,
        bigrams=bigrams,
        field_tags=field_tags,
        section_kinds=section_kinds,
        intent=intent,
        is_exhaustive=is_exhaustive,
        query_embedding=query_embedding,
        field_probabilities=field_probabilities,
        _embedder=embedder,
        _raw_query=query,
    )


# ---------------------------------------------------------------------------
# Chunk relevance scoring
# ---------------------------------------------------------------------------

def score_chunk_relevance(chunk: Any, focus: QueryFocus) -> float:
    """Score a chunk's relevance to the query focus (0.0 – 1.0)."""
    if focus.is_exhaustive:
        return max(0.3, _raw_chunk_score(chunk, focus))
    return _raw_chunk_score(chunk, focus)


def _assess_query_complexity(query: str) -> str:
    """Classify query complexity for adaptive weight selection.

    Entity-aware: queries with proper nouns (names, specific terms) are
    treated as more specific even if they're long, because the entities
    serve as strong retrieval anchors.
    """
    if not query or not query.strip():
        return "balanced"  # Default for missing/empty query
    tokens = query.split()
    n = len(tokens)

    # Detect entity-rich queries (proper nouns = specific intent)
    _entity_count = sum(1 for t in tokens if t[0:1].isupper() and len(t) > 2
                        and t.lower() not in _STOP)

    # Short specific queries (1-4 tokens with a specific term)
    if n <= 4:
        return "short_specific"
    # Entity-rich longer queries are still specific (e.g., "Compare John Smith and Sarah Chen")
    if _entity_count >= 2 and n <= 10:
        return "short_specific"
    # Long queries (8+ tokens) tend to be vague/conversational
    if n >= 8:
        return "long_vague"
    return "balanced"


# Adaptive weights keyed by complexity: (semantic, keyword, section, reranker)
_ADAPTIVE_WEIGHTS = {
    "short_specific": (0.10, 0.35, 0.25, 0.30),
    "balanced":       (0.25, 0.20, 0.20, 0.35),
    "long_vague":     (0.35, 0.10, 0.15, 0.40),
}
_ADAPTIVE_WEIGHTS_FALLBACK = {
    "short_specific": (0.00, 0.40, 0.25, 0.35),
    "balanced":       (0.00, 0.35, 0.25, 0.40),
    "long_vague":     (0.00, 0.20, 0.25, 0.55),
}


def _raw_chunk_score(chunk: Any, focus: QueryFocus) -> float:
    """Weighted combination of scoring signals.

    Weights are adaptive based on query complexity:
    - **short_specific** (1-4 tokens): keyword-heavy (exact term matching)
    - **balanced** (5-7 tokens): even spread across signals
    - **long_vague** (8+ tokens): semantic/reranker-heavy (meaning over words)

    Reranker (cross-encoder) gets highest weight because it's trained on
    relevance.  Keyword overlap is raised vs semantic to catch exact domain
    terms (invoice number, patient name) that embeddings may dilute.
    """
    complexity = _assess_query_complexity(focus._raw_query or "")

    kw_score = _keyword_overlap_score(chunk, focus)
    sect_score = _section_affinity_score(chunk, focus)
    reranker_score = _normalized_reranker_score(chunk)

    if focus.query_embedding is not None:
        sem_score = _semantic_similarity_score(chunk, focus)
        w_sem, w_kw, w_sect, w_rr = _ADAPTIVE_WEIGHTS.get(complexity, _ADAPTIVE_WEIGHTS["balanced"])
        return w_sem * sem_score + w_kw * kw_score + w_sect * sect_score + w_rr * reranker_score

    # Fallback: keyword-only weights — reranker weighted highest
    _, w_kw, w_sect, w_rr = _ADAPTIVE_WEIGHTS_FALLBACK.get(complexity, _ADAPTIVE_WEIGHTS_FALLBACK["balanced"])
    return w_kw * kw_score + w_sect * sect_score + w_rr * reranker_score


# ---------------------------------------------------------------------------
# Semantic similarity scoring (ML Enhancement 1)
# ---------------------------------------------------------------------------

def _semantic_similarity_score(chunk: Any, focus: QueryFocus) -> float:
    """Cosine similarity between query embedding and chunk text embedding."""
    if focus.query_embedding is None:
        return 0.0
    text = (getattr(chunk, "text", "") or "")[:512]  # cap at 512 chars for speed
    if not text:
        return 0.0
    chunk_emb = _get_or_encode_chunk(text, focus)
    if chunk_emb is None:
        return 0.0
    # Cosine similarity on L2-normalized vectors = dot product
    sim = float(np.dot(focus.query_embedding, chunk_emb))
    # Clamp to [0, 1] — negative similarity means irrelevant
    return max(0.0, min(sim, 1.0))


def _get_or_encode_chunk(text: str, focus: QueryFocus) -> Optional[np.ndarray]:
    """Encode chunk text with thread-local caching."""
    cache = getattr(_chunk_embed_cache, "cache", None)
    if cache is None:
        _chunk_embed_cache.cache = {}
        cache = _chunk_embed_cache.cache

    if text in cache:
        return cache[text]

    embedder = focus._embedder
    if embedder is None:
        return None

    try:
        emb = embedder.encode([text], normalize_embeddings=True)[0]
        cache[text] = emb
        return emb
    except Exception:
        log.debug("Failed to encode chunk text for semantic similarity", exc_info=True)
        return None


def _keyword_overlap_score(chunk: Any, focus: QueryFocus) -> float:
    """TF-IDF-like keyword overlap — longer keywords count more, bigrams 1.5x."""
    text = (getattr(chunk, "text", "") or "").lower()
    if not text or not focus.keywords:
        return 0.0

    total_weight = 0.0
    max_possible = 0.0

    for kw in focus.keywords:
        weight = min(len(kw) / 6.0, 1.5)  # longer keywords count more
        max_possible += weight
        if kw in text:
            total_weight += weight

    for bg in focus.bigrams:
        weight = 1.5
        max_possible += weight
        if bg in text:
            total_weight += weight

    return min(total_weight / max(max_possible, 1.0), 1.0)


# ---------------------------------------------------------------------------
# Section affinity scoring (ML Enhancement 3)
# ---------------------------------------------------------------------------

def _section_affinity_score(chunk: Any, focus: QueryFocus) -> float:
    """1.0 if chunk section_kind matches any focus kinds, 0.5 for partial, 0.0 otherwise.

    When the chunk has a missing or generic section_kind (misc/other/unknown),
    a neutral penalty score is applied.
    """
    if not focus.section_kinds:
        return 0.5  # neutral when no section focus

    chunk_kind = _get_chunk_section_kind(chunk)

    # Use a neutral score for generic/missing section_kinds.
    _is_generic = not chunk_kind or chunk_kind in ("misc", "other", "unknown")
    if _is_generic:
        return 0.3  # unknown/generic section — mild penalty

    for fk in focus.section_kinds:
        if fk == chunk_kind:
            return 1.0
        # Partial match: check if kinds share a root component
        # (e.g., "skills_technical" and "skills" share "skills")
        fk_parts = set(fk.split("_"))
        ck_parts = set(chunk_kind.split("_"))
        if fk_parts & ck_parts:
            return 0.7

    return 0.0


def _get_chunk_section_kind(chunk: Any) -> str:
    """Extract section_kind from chunk metadata."""
    meta = getattr(chunk, "meta", None) or {}
    # Support both dict and SimpleNamespace (used in tests)
    if isinstance(meta, dict):
        chunk_kind = meta.get("section_kind", "")
        if not chunk_kind:
            section = meta.get("section")
            if isinstance(section, dict):
                chunk_kind = section.get("kind", "")
    else:
        chunk_kind = getattr(meta, "section_kind", "") or ""
    return (chunk_kind or "").lower().strip()




def _normalized_reranker_score(chunk: Any) -> float:
    """Normalize existing reranker score to [0, 1]. 0.0 for unscored chunks."""
    score = getattr(chunk, "score", None)
    if score is None or score <= 0:
        return 0.0
    # Cross-encoder scores typically range -10 to +10; reranker scores 0–1
    if score > 1.0:
        return min(score / 10.0, 1.0)
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Batch encoding helper
# ---------------------------------------------------------------------------

def _batch_encode_chunks(chunks: List[Any], focus: QueryFocus) -> None:
    """Pre-encode all chunk texts in a single batch call.

    Populates the thread-local ``_chunk_embed_cache`` so that subsequent
    ``_get_or_encode_chunk()`` calls are free cache hits — avoiding 16+
    individual ``encode()`` calls that saturate the GPU.
    """
    if focus.query_embedding is None:
        return  # No semantic scoring → nothing to encode

    embedder = focus._embedder
    if embedder is None:
        return

    cache = getattr(_chunk_embed_cache, "cache", None)
    if cache is None:
        _chunk_embed_cache.cache = {}
        cache = _chunk_embed_cache.cache

    # Collect texts that are not already cached
    texts_to_encode: List[str] = []
    text_indices: List[int] = []  # maps into texts_to_encode
    chunk_texts: List[str] = []   # all chunk texts in order

    for chunk in chunks:
        text = (getattr(chunk, "text", "") or "")[:512]
        chunk_texts.append(text)
        if text and text not in cache:
            texts_to_encode.append(text)
            text_indices.append(len(chunk_texts) - 1)

    if not texts_to_encode:
        return

    try:
        embeddings = embedder.encode(texts_to_encode, normalize_embeddings=True)
        for i, text in enumerate(texts_to_encode):
            cache[text] = embeddings[i]
    except Exception:
        log.debug("Batch chunk encoding failed, falling back to per-chunk", exc_info=True)


# ---------------------------------------------------------------------------
# Chunk filtering
# ---------------------------------------------------------------------------

def filter_chunks_by_focus(
    chunks: List[Any],
    focus: QueryFocus,
    min_keep: int = 4,
    top_k: int = 12,
) -> List[Any]:
    """Score, sort, and filter chunks by query relevance.

    Returns at most *top_k* chunks.  When the focus is exhaustive the full
    list is kept (up to *top_k*).  Otherwise only chunks scoring >= 0.15
    survive, with a guaranteed minimum of *min_keep*.
    """
    if not chunks or not focus:
        return chunks

    # Pre-encode all chunk texts in ONE batch call (instead of per-chunk)
    _batch_encode_chunks(chunks, focus)

    scored = [(score_chunk_relevance(c, focus), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if focus.is_exhaustive:
        # Exhaustive queries keep all chunks, but always respect caller's top_k cap
        return [c for _, c in scored[:top_k]]

    # Keep chunks above threshold, with min_keep guarantee
    # Threshold 0.15 (documented) — not 0.25 which was too aggressive
    # and dropped legitimate evidence scoring 0.15-0.24
    above = [(s, c) for s, c in scored if s >= 0.15]
    if len(above) < min_keep:
        result = [c for _, c in scored[:min_keep]]
    else:
        result = [c for _, c in above]

    return result[:top_k]


# ---------------------------------------------------------------------------
# Field relevance scoring (for rendering) — ML Enhancement 2 integration
# ---------------------------------------------------------------------------

def score_field_relevance(field_tag: str, focus: QueryFocus) -> float:
    """Score how relevant a schema field (e.g. 'skills', 'education') is to the focus.

    When field_probabilities are available (from the field importance classifier),
    uses continuous probability scores. Otherwise falls back to binary keyword matching.
    """
    if not focus or focus.is_exhaustive:
        return 1.0

    # Use classifier probabilities when available (continuous score)
    if focus.field_probabilities and field_tag in focus.field_probabilities:
        return focus.field_probabilities[field_tag]

    if not focus.field_tags:
        return 1.0  # no specific focus → show everything

    if field_tag in focus.field_tags:
        return 1.0

    # keyword overlap with field name
    field_lower = field_tag.lower()
    for kw in focus.keywords:
        if kw in field_lower or field_lower in kw:
            return 0.5

    return 0.1  # low but non-zero — secondary fields show minimally


# ---------------------------------------------------------------------------
# Fact relevance scoring (for generic / document intelligence rendering)
# ---------------------------------------------------------------------------

def score_fact_relevance(label: str, value: str, focus: QueryFocus) -> float:
    """Score how relevant a single (label, value) fact is to the focus."""
    if not focus or focus.is_exhaustive:
        return 1.0

    if not focus.keywords and not focus.field_tags:
        return 1.0

    score = 0.0
    combined = f"{label or ''} {value or ''}".lower()

    # Keyword overlap
    if focus.keywords:
        hits = sum(1 for kw in focus.keywords if kw in combined)
        score += min(hits * 0.3, 0.6)

    # Bigram overlap
    if focus.bigrams:
        hits = sum(1 for bg in focus.bigrams if bg in combined)
        score += min(hits * 0.4, 0.6)

    # Label matches field_tags
    if focus.field_tags and label:
        label_lower = label.lower()
        for tag in focus.field_tags:
            if tag in label_lower or label_lower in tag:
                score += 0.4
                break

    return min(score, 1.0)
