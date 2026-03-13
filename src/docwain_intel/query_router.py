"""NLP-based query routing -- classifies queries using spaCy features, not regex."""
from __future__ import annotations

import enum
from src.utils.logging_utils import get_logger
import re
import threading
from typing import List, Optional

from pydantic import BaseModel, Field

logger = get_logger(__name__)

_spacy_lock = threading.Lock()
_spacy_nlp = None

# Matches null bytes and C0/C1 control characters (except common whitespace).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_MAX_QUERY_LENGTH = 10_000

def _get_spacy():
    """Load spaCy model (lazy singleton, thread-safe). Prefers en_core_web_lg."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    with _spacy_lock:
        if _spacy_nlp is not None:  # double-check after acquiring lock
            return _spacy_nlp
        import spacy

        try:
            _spacy_nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.debug("en_core_web_lg not found, falling back to en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

class QueryRoute(str, enum.Enum):
    GRAPH_DIRECT = "GRAPH_DIRECT"
    HYBRID_SEARCH = "HYBRID_SEARCH"
    FULL_SEARCH = "FULL_SEARCH"
    LLM_GENERATION = "LLM_GENERATION"

class QueryAnalysis(BaseModel):
    query: str
    route: QueryRoute
    entities: List[str] = Field(default_factory=list)
    specificity: float = 0.5
    is_comparison: bool = False
    is_aggregation: bool = False
    is_conversational: bool = False
    has_numeric_constraint: bool = False
    reasoning_required: bool = False
    confidence: float = 0.5

# Lemmas that signal broad scope / summarization (imperative verbs).
_BROAD_SCOPE_LEMMAS = frozenset({
    "summarize", "overview", "outline", "recap", "review", "digest",
})

# Lemmas that signal reasoning / LLM generation need.
_REASONING_LEMMAS = frozenset({
    "recommend", "evaluate", "assess", "justify", "explain",
    "analyze", "analyse", "argue", "determine", "suggest", "advise",
    "interpret", "infer",
})

# Lemmas for simple retrieval verbs.
_RETRIEVAL_LEMMAS = frozenset({
    "show", "list", "get", "find", "fetch", "display", "give", "tell",
    "retrieve", "return", "provide", "look",
})

# Lemmas for comparison.
_COMPARISON_LEMMAS = frozenset({
    "compare", "contrast", "differentiate", "distinguish", "versus",
    "difference", "differ", "similarity", "similar",
})

# Lemmas for aggregation.
_AGGREGATION_LEMMAS = frozenset({
    "count", "total", "average", "sum", "tally",
})

# Conversational greetings / filler (lowercased).
_GREETING_LEMMAS = frozenset({
    "hello", "hi", "hey", "greetings", "thanks", "thank", "bye",
    "goodbye", "okay", "ok", "yes", "no", "sure", "please",
})

# Field-specific tokens that boost specificity.
_SPECIFIC_FIELD_TOKENS = frozenset({
    "email", "phone", "address", "salary", "name", "number", "date",
    "title", "role", "position", "price", "cost", "amount", "total",
    "revenue", "profit", "age", "dob", "ssn", "id",
})

def _extract_entities(doc) -> List[str]:
    """Extract named entities from spaCy doc, deduped."""
    seen = set()
    entities = []
    for ent in doc.ents:
        # Skip purely numeric entities for the entity list (dates, money kept).
        if ent.label_ in ("CARDINAL", "ORDINAL", "QUANTITY"):
            continue
        text = ent.text.strip()
        key = text.lower()
        if key and key not in seen:
            seen.add(key)
            entities.append(text)
    return entities

def _has_possessive(doc) -> bool:
    """Check if the query contains a possessive structure (X's Y)."""
    for token in doc:
        if token.dep_ == "case" and token.text == "'s":
            return True
        if token.tag_ == "POS":
            return True
    return False

def _detect_comparison(doc) -> bool:
    """Detect comparison intent using dependency parse and lemmas."""
    for token in doc:
        if token.lemma_.lower() in _COMPARISON_LEMMAS:
            return True
    # Check for coordinating conjunction joining two+ entities.
    ent_labels = [ent.label_ for ent in doc.ents
                  if ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY", "DATE", "MONEY")]
    if len(ent_labels) >= 2:
        for token in doc:
            if token.dep_ == "cc" and token.lemma_.lower() in ("and", "or", "versus", "vs"):
                return True
    return False

def _detect_aggregation(doc) -> bool:
    """Detect aggregation queries using dependency parse."""
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & _AGGREGATION_LEMMAS:
        return True
    # "how many" pattern via dependency parse.
    for token in doc:
        if token.lemma_.lower() == "how":
            for child in token.children:
                if child.lemma_.lower() == "many" or child.lemma_.lower() == "much":
                    return True
            # Also check next token.
            if token.i + 1 < len(doc):
                nxt = doc[token.i + 1]
                if nxt.lemma_.lower() in ("many", "much"):
                    return True
    return False

def _detect_reasoning(doc) -> bool:
    """Detect if query requires reasoning / LLM generation."""
    lemmas = {t.lemma_.lower() for t in doc}
    # Direct reasoning lemmas.
    if lemmas & _REASONING_LEMMAS:
        return True
    # "why" as adverb.
    for token in doc:
        if token.lemma_.lower() == "why":
            return True
    # "best" + "who/what" pattern.
    has_best = any(t.lemma_.lower() in ("best", "ideal", "optimal") for t in doc)
    has_wh = any(t.lemma_.lower() in ("who", "what", "which") for t in doc)
    if has_best and has_wh:
        return True
    # "based on" pattern.
    for token in doc:
        if token.lemma_.lower() == "base" and token.dep_ == "case":
            return True
        if token.text.lower() == "based":
            if token.i + 1 < len(doc) and doc[token.i + 1].text.lower() == "on":
                return True
    # "should" modal.
    for token in doc:
        if token.lemma_.lower() == "should" and token.dep_ in ("aux", "ROOT"):
            return True
    return False

def _detect_broad_scope(doc) -> bool:
    """Detect broad scope queries (summarize, overview, etc.)."""
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & _BROAD_SCOPE_LEMMAS:
        return True
    # "all" + broad noun.
    has_all = any(t.lemma_.lower() in ("all", "every", "each", "entire") for t in doc)
    has_broad_noun = any(t.lemma_.lower() in ("document", "file", "finding", "result", "report")
                         for t in doc)
    if has_all and has_broad_noun:
        return True
    return False

def _detect_numeric_constraint(doc) -> bool:
    """Detect numeric constraints using NER and token features."""
    for ent in doc.ents:
        if ent.label_ in ("MONEY", "PERCENT", "QUANTITY", "CARDINAL"):
            return True
    for token in doc:
        if token.like_num:
            return True
        if token.text.startswith("$") or token.text.endswith("%"):
            return True
    return False

def _detect_conversational(doc) -> bool:
    """Detect conversational / greeting queries."""
    non_punct = [t for t in doc if not t.is_punct and not t.is_space]
    if len(non_punct) == 0:
        return True
    if len(non_punct) <= 2:
        lemmas = {t.lemma_.lower() for t in non_punct}
        if lemmas & _GREETING_LEMMAS:
            return True
        # Single word, no verb, no noun -- likely conversational.
        if len(non_punct) == 1:
            tok = non_punct[0]
            if tok.pos_ in ("INTJ", "X"):
                return True
            if tok.pos_ not in ("NOUN", "PROPN", "VERB"):
                return True
    return False

def _compute_specificity(
    doc,
    entities: List[str],
    has_possessive: bool,
    has_numeric: bool,
) -> float:
    """Compute query specificity from 0.0 (vague) to 1.0 (very specific)."""
    score = 0.0
    non_punct = [t for t in doc if not t.is_punct and not t.is_space]
    n_tokens = len(non_punct)

    # Entity contribution: each entity adds specificity.
    score += min(len(entities) * 0.2, 0.4)

    # Possessive structure adds specificity.
    if has_possessive:
        score += 0.15

    # Specific field token present.
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & _SPECIFIC_FIELD_TOKENS:
        score += 0.15

    # Numeric constraint adds specificity.
    if has_numeric:
        score += 0.1

    # Query length: very short or very long reduce specificity.
    if 4 <= n_tokens <= 12:
        score += 0.1
    elif n_tokens > 20:
        score -= 0.1

    # Broad scope reduces specificity.
    if lemmas & _BROAD_SCOPE_LEMMAS:
        score -= 0.15
    if lemmas & {"document", "documents", "file", "files"}:
        if not entities:
            score -= 0.1

    return max(0.0, min(1.0, score))

def _sanitize_query(query: str) -> str:
    """Strip, remove control characters, and truncate query."""
    query = query.strip()
    query = _CONTROL_CHAR_RE.sub("", query)
    if len(query) > _MAX_QUERY_LENGTH:
        query = query[:_MAX_QUERY_LENGTH]
    return query

def route_query(query: str) -> QueryAnalysis:
    """Classify a query and determine the optimal retrieval route."""
    if not query or not query.strip():
        return QueryAnalysis(
            query=query or "",
            route=QueryRoute.GRAPH_DIRECT,
            entities=[],
            specificity=0.0,
            is_conversational=True,
            confidence=1.0,
        )

    query = _sanitize_query(query)

    if not query:
        return QueryAnalysis(
            query=query,
            route=QueryRoute.GRAPH_DIRECT,
            entities=[],
            specificity=0.0,
            is_conversational=True,
            confidence=1.0,
        )

    nlp = _get_spacy()
    doc = nlp(query)

    # --- Feature extraction ---
    entities = _extract_entities(doc)
    is_comparison = _detect_comparison(doc)
    is_aggregation = _detect_aggregation(doc)
    reasoning_required = _detect_reasoning(doc)
    broad_scope = _detect_broad_scope(doc)
    has_numeric = _detect_numeric_constraint(doc)
    is_conversational = _detect_conversational(doc)
    has_possessive = _has_possessive(doc)

    specificity = _compute_specificity(doc, entities, has_possessive, has_numeric)

    non_punct = [t for t in doc if not t.is_punct and not t.is_space]
    n_tokens = len(non_punct)
    lemmas = {t.lemma_.lower() for t in doc}

    # --- Routing decision (priority order) ---
    route = QueryRoute.HYBRID_SEARCH  # default
    confidence = 0.5

    # 1. Conversational / empty.
    if is_conversational:
        route = QueryRoute.GRAPH_DIRECT
        confidence = 0.9

    # 2. Reasoning required (highest complexity).
    elif reasoning_required:
        route = QueryRoute.LLM_GENERATION
        confidence = 0.8
        # Boost confidence if query is long + has reasoning signals.
        if n_tokens > 15:
            confidence = 0.9

    # 3. Broad scope / summarization.
    elif broad_scope:
        route = QueryRoute.FULL_SEARCH
        confidence = 0.8

    # 4. Comparison.
    elif is_comparison:
        route = QueryRoute.HYBRID_SEARCH
        confidence = 0.75

    # 5. Aggregation.
    elif is_aggregation:
        route = QueryRoute.HYBRID_SEARCH
        confidence = 0.7

    # 6. Direct entity / fact lookup.
    elif len(entities) >= 1 and (has_possessive or specificity >= 0.4):
        route = QueryRoute.GRAPH_DIRECT
        confidence = 0.8

    # 7. Short specific query with numeric constraint.
    elif has_numeric and n_tokens <= 10:
        route = QueryRoute.GRAPH_DIRECT
        confidence = 0.7

    # 8. Retrieval verb + short query.
    elif lemmas & _RETRIEVAL_LEMMAS and n_tokens <= 10:
        route = QueryRoute.HYBRID_SEARCH
        confidence = 0.6

    # 9. Fallback for long queries.
    elif n_tokens > 20:
        route = QueryRoute.FULL_SEARCH
        confidence = 0.4

    logger.debug(
        "route_query: route=%s entities=%s specificity=%.2f comparison=%s "
        "aggregation=%s reasoning=%s broad=%s numeric=%s conversational=%s",
        route, entities, specificity, is_comparison, is_aggregation,
        reasoning_required, broad_scope, has_numeric, is_conversational,
    )

    return QueryAnalysis(
        query=query,
        route=route,
        entities=entities,
        specificity=round(specificity, 3),
        is_comparison=is_comparison,
        is_aggregation=is_aggregation,
        is_conversational=is_conversational,
        has_numeric_constraint=has_numeric,
        reasoning_required=reasoning_required,
        confidence=round(confidence, 3),
    )
