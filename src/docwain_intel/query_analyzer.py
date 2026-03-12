"""Algorithmic query geometry analyzer -- derives structural properties from NLP features."""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import List, Optional

from pydantic import BaseModel, Field

from .query_router import QueryAnalysis, _get_spacy

logger = get_logger(__name__)

# Temporal lemmas that signal chronological intent.
_TEMPORAL_LEMMAS = frozenset({
    "before", "after", "during", "first", "last", "recent", "recently",
    "history", "timeline", "earlier", "later", "previously", "since",
    "until", "meanwhile", "chronological", "sequence",
})

# Lemmas that signal enumeration intent.
_ENUMERATIVE_LEMMAS = frozenset({
    "list", "all", "every", "each", "enumerate", "catalog",
})

# Lemmas that push granularity toward detailed (0.0).
_DETAIL_LEMMAS = frozenset({
    "tell", "describe", "explain", "elaborate", "detail", "comprehensive",
    "thorough", "overview", "summarize", "summary",
})

# Entity labels to skip when counting "real" entities.
_SKIP_ENT_LABELS = frozenset({"CARDINAL", "ORDINAL", "QUANTITY"})

# WH-words recognized as question words.
_WH_WORDS = frozenset({"who", "whom", "what", "when", "where", "why", "how", "which"})

# Copula lemmas.
_COPULA_LEMMAS = frozenset({"be"})

class QueryGeometry(BaseModel):
    query: str
    intent_type: str = "narrative"
    expected_entity_count: int = 0
    granularity: float = 0.5
    temporal_ordering: bool = False
    is_comparison: bool = False
    is_aggregation: bool = False
    focus_type: str = "process_centric"
    question_word: Optional[str] = None
    requested_attributes: List[str] = Field(default_factory=list)

def _find_question_word(doc) -> Optional[str]:
    """Extract the WH-word from the query, if present."""
    for token in doc:
        if token.lemma_.lower() in _WH_WORDS:
            return token.lemma_.lower()
    return None

def _detect_temporal_ordering(doc, question_word: Optional[str]) -> bool:
    """Detect whether the query expects chronologically ordered results."""
    if question_word == "when":
        return True
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            return True
    for token in doc:
        if token.lemma_.lower() in _TEMPORAL_LEMMAS:
            return True
    return False

def _derive_intent_type(
    doc,
    analysis: QueryAnalysis,
    question_word: Optional[str],
    temporal: bool,
) -> str:
    """Algorithmically derive intent type from NLP features."""
    # Priority order mirrors the spec.
    if analysis.is_comparison:
        return "comparative"
    if analysis.is_aggregation:
        return "quantitative"
    if analysis.reasoning_required:
        return "causal_inquiry"
    if temporal:
        return "temporal_inquiry"
    # "how" + verb → process inquiry
    if question_word == "how":
        # Check it's not "how many/much" (that's aggregation, caught above).
        has_verb_child = False
        for token in doc:
            if token.lemma_.lower() == "how":
                for child in token.children:
                    if child.pos_ == "VERB":
                        has_verb_child = True
                # Also check: if next non-punct is a verb
                if token.i + 1 < len(doc):
                    nxt = doc[token.i + 1]
                    if nxt.pos_ == "VERB" or nxt.pos_ == "AUX":
                        has_verb_child = True
                break
        if has_verb_child:
            return "process_inquiry"
        # "how" without verb child but with a noun → still process
        return "process_inquiry"

    # Enumerative
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & _ENUMERATIVE_LEMMAS:
        return "enumerative"

    # Entity lookup: entities present and high specificity
    if analysis.entities and analysis.specificity >= 0.3:
        return "entity_lookup"

    return "narrative"

def _derive_granularity(
    doc,
    analysis: QueryAnalysis,
    intent_type: str,
    requested_attributes: List[str],
) -> float:
    """Derive granularity: 0.0 = detailed/comprehensive, 1.0 = concise/single-value."""
    score = analysis.specificity

    non_punct = [t for t in doc if not t.is_punct and not t.is_space]
    n_tokens = len(non_punct)

    # Short queries → concise
    if n_tokens <= 5 and n_tokens > 0:
        score += 0.25

    # Detail / broad lemmas → detailed
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & _DETAIL_LEMMAS:
        score -= 0.3

    # Single specific attribute → concise
    if len(requested_attributes) == 1:
        score += 0.2

    # Enumerative → detailed (low granularity)
    if intent_type == "enumerative":
        score -= 0.2

    # Long queries → push toward detailed
    if n_tokens > 15:
        score -= 0.15

    return max(0.0, min(1.0, round(score, 3)))

def _derive_focus_type(
    doc,
    question_word: Optional[str],
    entities: List[str],
) -> str:
    """Derive focus type from question word and dependency parse."""
    if question_word in ("who", "whom"):
        return "entity_centric"

    if question_word == "how":
        return "process_centric"

    if question_word == "what":
        # "what" + copula → attribute_centric; "what" + action verb → relationship_centric
        for token in doc:
            if token.lemma_.lower() == "what":
                # Look for a verb in the sentence
                for other in doc:
                    if other.pos_ == "AUX" and other.lemma_.lower() in _COPULA_LEMMAS:
                        return "attribute_centric"
                    if other.pos_ == "VERB" and other.lemma_.lower() not in _COPULA_LEMMAS:
                        return "relationship_centric"
                return "attribute_centric"

    # Possessive ("X's Y") → attribute_centric
    for token in doc:
        if token.dep_ == "case" and token.text == "'s":
            return "attribute_centric"
        if token.tag_ == "POS":
            return "attribute_centric"

    # Default
    if entities:
        return "entity_centric"
    return "process_centric"

def _extract_requested_attributes(doc, entities: List[str]) -> List[str]:
    """Extract requested attributes -- nouns/noun-phrases that are NOT named entities."""
    entity_tokens: set[int] = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            entity_tokens.add(i)

    # Lowercase entity texts for matching.
    entity_lower = {e.lower() for e in entities}

    attributes: list[str] = []
    seen: set[str] = set()

    for token in doc:
        if token.i in entity_tokens:
            continue
        if token.pos_ not in ("NOUN", "PROPN"):
            continue
        # Skip WH-words, determiners, etc.
        if token.lemma_.lower() in _WH_WORDS:
            continue
        # Skip very common/stopword nouns
        if token.is_stop:
            continue
        lemma = token.lemma_.lower()
        # Skip if it matches an entity name
        if lemma in entity_lower:
            continue
        # Accept: direct objects, objects of prepositions, subjects of copulas,
        # or any noun that could be an attribute.
        if token.dep_ in (
            "dobj", "pobj", "attr", "nsubj", "nsubjpass", "conj",
            "appos", "compound", "ROOT",
        ) or token.head.pos_ in ("ADP", "AUX", "VERB"):
            if lemma not in seen and len(lemma) > 1:
                seen.add(lemma)
                attributes.append(lemma)

    return attributes

def analyze_query(query: str, analysis: QueryAnalysis) -> QueryGeometry:
    """Enrich a QueryAnalysis with geometric properties derived from NLP features."""
    if not query or not query.strip():
        return QueryGeometry(
            query=query or "",
            intent_type="narrative",
            expected_entity_count=0,
            granularity=0.5,
            temporal_ordering=False,
            is_comparison=False,
            is_aggregation=False,
            focus_type="process_centric",
            question_word=None,
            requested_attributes=[],
        )

    nlp = _get_spacy()
    doc = nlp(query)

    # Count entities (excluding purely numeric ones).
    entity_count = len([
        ent for ent in doc.ents if ent.label_ not in _SKIP_ENT_LABELS
    ])

    question_word = _find_question_word(doc)
    temporal = _detect_temporal_ordering(doc, question_word)
    requested_attrs = _extract_requested_attributes(doc, analysis.entities)
    intent_type = _derive_intent_type(doc, analysis, question_word, temporal)
    granularity = _derive_granularity(doc, analysis, intent_type, requested_attrs)
    focus_type = _derive_focus_type(doc, question_word, analysis.entities)

    return QueryGeometry(
        query=query,
        intent_type=intent_type,
        expected_entity_count=entity_count,
        granularity=granularity,
        temporal_ordering=temporal,
        is_comparison=analysis.is_comparison,
        is_aggregation=analysis.is_aggregation,
        focus_type=focus_type,
        question_word=question_word,
        requested_attributes=requested_attrs,
    )
