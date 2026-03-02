"""NLP-based entity extraction from user queries.

Uses spaCy dependency parsing for entity extraction.
No regex patterns for entity name detection.

The key NLP signals used:
- ``pobj`` of prepositions (of / for / about / from / by)
- ``poss`` — possessive modifier ("gokul's skills")
- ``nsubj`` / ``nsubjpass`` — subject of question verbs
- ``dobj`` of action verbs (summarize / show / get / find / contact)
- Tokens tagged as ``PROPN`` (proper noun) are always candidates.
- Tokens tagged as ``NOUN`` are candidates only if they are **not** common
  English vocabulary words (checked via spaCy + domain stopword set).
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── spaCy model (thread-safe lazy-loaded singleton) ──────────────────

_nlp = None
_nlp_load_attempted = False
_nlp_lock = threading.Lock()


def _get_nlp():
    """Lazy-load spaCy English model. Thread-safe; returns None if unavailable."""
    global _nlp, _nlp_load_attempted
    if _nlp is not None:
        return _nlp
    with _nlp_lock:
        # Double-check after acquiring lock (another thread may have loaded it)
        if _nlp is not None:
            return _nlp
        if _nlp_load_attempted:
            return None
        _nlp_load_attempted = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded for NLP entity extraction")
            return _nlp
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy model load failed (NLP entity extraction degraded): %s", exc)
            return None


def preload_spacy():
    """Pre-load spaCy model at startup to avoid first-request latency."""
    return _get_nlp() is not None


# ── Domain stopwords (never extracted as entity names) ───────────────

_DOMAIN_STOPWORDS = frozenset({
    # Query verbs / function words
    "give", "get", "show", "tell", "find", "list", "fetch", "describe",
    "summarize", "summarise", "review", "extract", "identify", "compare",
    "rank", "contact", "reach", "please", "help",
    # Document domain terms
    "document", "resume", "cv", "invoice", "profile", "report", "details",
    "information", "candidate", "candidates", "position", "role", "history",
    "summary", "overview", "background", "designation",
    # Field names
    "skills", "experience", "education", "certifications", "certificates",
    "qualification", "qualifications", "achievements", "career", "projects",
    "technologies", "tools", "degrees", "degree", "work",
    # Contact fields
    "email", "phone", "linkedin", "contact",
    # Common nouns / determiners
    "details", "data", "info", "list", "table", "output", "result", "results",
    "plan", "plans", "treatment", "medications", "diagnosis", "findings",
    "format", "type", "name", "number", "date", "year", "years",
    "all", "each", "every", "many", "some", "other", "most",
    "best", "top", "total", "average", "count",
    "machine", "learning",  # too generic as standalone
    # Abstract domain terms (too generic as entity names)
    "conditions", "terms", "clauses", "requirements", "provisions",
    "procedures", "policy", "coverage", "exclusions",
    # Language names (translation queries — never valid entity hints)
    "french", "spanish", "german", "italian", "portuguese", "dutch",
    "russian", "chinese", "japanese", "korean", "arabic", "hindi",
    "turkish", "polish", "swedish", "norwegian", "danish", "finnish",
    "greek", "czech", "romanian", "hungarian", "thai", "vietnamese",
    "indonesian", "malay", "hebrew", "ukrainian", "tamil", "telugu",
    "bengali", "urdu", "persian", "swahili", "catalan", "english",
    "translate", "translation", "translator",
    # Tool-action words that should never be entities
    "draft", "compose", "generate", "create", "write", "build",
    "search", "internet", "online", "web",
    # Adjectives/fragments that should never be entities
    "non", "available", "highlevel", "high", "level", "brief",
    # Generic document/file nouns — never valid entity hints
    "file", "files", "product", "products", "manual", "manuals",
    "version", "versions", "page", "pages", "section", "sections",
    "item", "items", "record", "records", "entry", "entries",
    "content", "contents", "attachment", "attachments",
    "convert", "conversion", "highlight", "highlevel",
    # Plural forms of existing stopwords (spaCy treats them as different tokens)
    "documents", "reports", "profiles", "invoices", "resumes",
    "candidates", "positions", "roles", "summaries", "overviews",
    # Medical document types and terms — NOT person names
    "patient", "patients", "pathology", "radiology", "clinical",
    "progress", "note", "notes", "lab", "laboratory", "medication",
    "medications", "prescription", "prescriptions", "vital", "signs",
    "authorized", "signature", "signatures", "dosage", "prognosis",
    "discharge", "admission", "consultation", "referral", "imaging",
    "procedure", "procedures", "surgical", "operative", "specimen",
    # Legal document types
    "clause", "clauses", "agreement", "agreements", "contract", "contracts",
    "liability", "indemnification", "arbitration", "jurisdiction",
    # Invoice/financial terms
    "subtotal", "balance", "remittance", "payable", "receivable",
    "vendor", "vendors", "supplier", "suppliers",
})

# Dependency labels that indicate entity position
_ENTITY_DEP_LABELS = frozenset({"pobj", "poss", "nsubj", "nsubjpass", "dobj", "attr"})

# Prepositions whose pobj is likely an entity name
_ENTITY_PREPOSITIONS = frozenset({"of", "for", "about", "from", "by"})

# Verbs whose dobj is likely an entity name
_ENTITY_VERBS = frozenset({
    "summarize", "summarise", "show", "get", "fetch", "find",
    "describe", "review", "contact", "reach",
})


def _is_likely_entity(token) -> bool:
    """Check if a spaCy token is a likely entity name (not a common word)."""
    text_lower = token.text.lower()

    # Proper nouns are always candidates
    if token.pos_ == "PROPN":
        return text_lower not in _DOMAIN_STOPWORDS

    # Must be at least 3 chars
    if len(text_lower) < 3:
        return False

    # Domain stopwords are never entities
    if text_lower in _DOMAIN_STOPWORDS:
        return False

    # spaCy stop words are never entities
    if token.is_stop:
        return False

    # Pronouns, determiners, conjunctions, adpositions are never entities
    if token.pos_ in ("PRON", "DET", "CCONJ", "SCONJ", "ADP", "AUX", "PART", "PUNCT", "SPACE"):
        return False

    # Common verbs are not entities
    if token.pos_ == "VERB":
        return False

    # If the word is in spaCy's vocab with high frequency, likely common English
    if token.is_oov is False and token.prob > -8.0:
        return False

    return True


def _extract_compound_noun_phrase(token) -> Optional[str]:
    """When a pobj is a stopword, look for prepositional children.

    Handles patterns like "conditions FOR natural calamities" where "conditions"
    is a stopword pobj but the real entity is "natural calamities" (the prep
    phrase hanging off "conditions").
    """
    # Look for prepositional children of this token (prep → pobj chain)
    for child in token.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    # Collect the full noun phrase from this grandchild
                    parts = []
                    for left in grandchild.lefts:
                        if left.dep_ in ("compound", "amod", "nn") and left.pos_ not in ("DET", "PRON"):
                            parts.append(left.text)
                    parts.append(grandchild.text)
                    for right in grandchild.rights:
                        if right.dep_ in ("compound", "flat", "flat:name", "amod"):
                            parts.append(right.text)
                    phrase = " ".join(parts)
                    # Reject if the entire phrase is stopwords
                    if phrase.lower() not in _DOMAIN_STOPWORDS and len(phrase) > 2:
                        return phrase
    return None


def _extract_via_dependency_parse(query: str) -> Optional[str]:
    """Extract entity name from query using spaCy dependency parsing.

    Looks for tokens in entity-indicating syntactic positions (pobj, poss,
    nsubj, dobj) that are not common English words.
    """
    nlp = _get_nlp()
    if nlp is None:
        return None

    doc = nlp(query)
    candidates: List[str] = []

    for token in doc:
        # Skip punctuation and whitespace
        if token.pos_ in ("PUNCT", "SPACE"):
            continue

        is_entity_position = False

        # pobj of entity-indicating prepositions
        if token.dep_ == "pobj" and token.head.text.lower() in _ENTITY_PREPOSITIONS:
            is_entity_position = True

        # Possessive modifier (e.g., "gokul's skills")
        elif token.dep_ == "poss":
            is_entity_position = True

        # Subject of a question/action verb
        elif token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ in ("VERB", "AUX"):
            is_entity_position = True

        # Direct object of entity-seeking verbs
        elif token.dep_ == "dobj" and token.head.lemma_.lower() in _ENTITY_VERBS:
            is_entity_position = True

        # Fallback: token is a direct child of an action verb and is OOV
        # (spaCy sometimes misclassifies unknown names as advmod/amod)
        elif (
            token.dep_ in ("advmod", "amod", "npadvmod", "oprd")
            and token.head.pos_ == "VERB"
            and token.head.lemma_.lower() in _ENTITY_VERBS
            and token.is_oov
        ):
            is_entity_position = True

        if is_entity_position:
            # If token is a domain stopword, try compound noun phrase extraction
            if token.text.lower() in _DOMAIN_STOPWORDS:
                compound = _extract_compound_noun_phrase(token)
                if compound:
                    candidates.append(compound)
                continue

            if _is_likely_entity(token):
                # Collect multi-word entity (token + following compound/flat tokens)
                entity_parts = [token.text]
                for right in token.rights:
                    if right.dep_ in ("compound", "flat", "flat:name") and _is_likely_entity(right):
                        entity_parts.append(right.text)
                # Also check left compounds
                for left in token.lefts:
                    if left.dep_ == "compound" and _is_likely_entity(left):
                        entity_parts.insert(0, left.text)
                candidates.append(" ".join(entity_parts))

    # Also check for PROPN tokens not caught by dependency rules
    if not candidates:
        for token in doc:
            if token.pos_ == "PROPN" and _is_likely_entity(token):
                entity_parts = [token.text]
                for right in token.rights:
                    if right.dep_ in ("compound", "flat", "flat:name"):
                        entity_parts.append(right.text)
                for left in token.lefts:
                    if left.dep_ == "compound":
                        entity_parts.insert(0, left.text)
                candidates.append(" ".join(entity_parts))

    # Also check NER results from spaCy
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE"):
            text_lower = ent.text.lower()
            if text_lower not in _DOMAIN_STOPWORDS and len(text_lower) > 2:
                candidates.append(ent.text)

    if not candidates:
        return None

    # Return the first candidate (priority: dependency parse > PROPN > NER)
    return candidates[0]


def extract_entity_from_query(query: str) -> Optional[str]:
    """Extract the target entity name from a user query.

    Uses spaCy dependency parsing.

    Returns the entity name string or None if no entity detected.
    """
    if not query or not query.strip():
        return None

    entity = _extract_via_dependency_parse(query)
    if entity:
        return entity

    return None


def extract_all_entities(query: str) -> List[str]:
    """Extract all entity mentions from a query (for multi-entity queries).

    Returns a list of entity name strings. Used for comparison/ranking queries
    that mention multiple people.
    """
    if not query or not query.strip():
        return []

    entities: List[str] = []

    # spaCy
    nlp = _get_nlp()
    if nlp is None:
        return entities

    doc = nlp(query)

    # Collect all entity-like tokens
    for token in doc:
        if token.pos_ in ("PUNCT", "SPACE"):
            continue
        if _is_likely_entity(token):
            dep = token.dep_
            if dep in _ENTITY_DEP_LABELS or token.pos_ == "PROPN":
                entities.append(token.text)

    # Also NER
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG") and ent.text not in entities:
            entities.append(ent.text)

    return entities
