from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = get_logger(__name__)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "in", "on", "at", "by",
    "for", "to", "from", "with", "without", "as", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "i", "you", "we", "they", "he", "she", "them",
    "my", "your", "our", "their", "what", "which", "who", "where", "when", "how", "why",
    "does", "do", "did", "can", "could", "should", "would", "may", "might", "will", "shall",
}

def _nlu_detect_intent(query: str) -> Optional[str]:
    """Detect query intent using NLU engine."""
    try:
        from src.nlp.nlu_engine import classify_intent
        intent = classify_intent(query)
        # Map NLU engine intents to query analyzer intents
        _NLU_TO_ANALYZER = {
            "comparison": "comparison",
            "ranking": "comparison",  # ranking is a type of comparison
            "summary": "summarization",
            "cross_document": "deep_analysis",
            "analytics": "deep_analysis",
            "reasoning": "deep_analysis",
            "factual": None,  # factual doesn't map to any special intent
            "timeline": None,
            "multi_field": None,
        }
        return _NLU_TO_ANALYZER.get(intent)
    except Exception:
        logger.debug("_nlu_detect_intent: NLU engine unavailable", exc_info=True)
        return None

_NUMERIC_HINTS = [
    "how many",
    "number of",
    "total",
    "sum",
    "amount",
    "count",
    "average",
    "median",
    "mean",
    "percent",
    "percentage",
    "ratio",
    "years",
    "months",
    "days",
    "cost",
    "price",
    "budget",
]

_FIELD_HINTS = [
    "email",
    "phone",
    "address",
    "title",
    "role",
    "company",
    "id",
    "total",
    "order",
    "account",
    "contact",
]

_SECTION_HINTS = [
    "summary",
    "overview",
    "details",
    "analysis",
    "results",
    "conclusion",
    "requirements",
]

_UNITS_PATTERN = re.compile(
    r"\b(usd|eur|gbp|inr|\$|%|percent|percentage|years?|months?|days?|hours?|mins?|minutes?)\b",
    re.IGNORECASE,
)

_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r"\b\d+[\d,\.]*\b")

_ERROR_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,}[\w-]*\d+[\w-]*\b")

_QUOTED_PATTERN = re.compile(r"['\"]([^'\"]{2,})['\"]")

@dataclass
class QueryAnalysis:
    intent_type: str
    required_evidence: Dict[str, object]
    output_format_preference: str
    explicitness_score: float
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    comparison_entities: List[str] = field(default_factory=list)
    high_stakes: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "intent_type": self.intent_type,
            "required_evidence": self.required_evidence,
            "output_format_preference": self.output_format_preference,
            "explicitness_score": self.explicitness_score,
            "entities": self.entities,
            "keywords": self.keywords,
            "comparison_entities": self.comparison_entities,
            "high_stakes": self.high_stakes,
        }

class QueryAnalyzer:
    """Fast, rule-based query analyzer for intent and evidence constraints."""

    def analyze(self, query: str) -> QueryAnalysis:
        logger.debug("analyze: query_len=%d", len(query or ""))
        query = (query or "").strip()
        lowered = query.lower()
        intent = self._detect_intent(lowered)
        comparison_entities = self._extract_comparison_entities(query) if intent == "comparison" else []
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        quoted = self._extract_quoted_phrases(query)
        requires_number = self._requires_numeric(lowered)
        requires_date = self._requires_date(lowered)
        requires_units = bool(_UNITS_PATTERN.search(lowered)) or (requires_number and "percent" in lowered)
        error_tokens = self._extract_error_tokens(query) if intent == "troubleshooting" else []

        required_evidence: Dict[str, object] = {
            "contains_number": requires_number,
            "contains_date_range": requires_date,
            "contains_units": requires_units,
            "contains_named_entities": entities,
            "must_include_keywords": keywords,
            "must_include_phrases": quoted,
            "error_tokens": error_tokens,
            "comparison_entities": comparison_entities,
            "section_hints": self._section_hints(lowered),
        }

        output_pref = self._output_format_preference(intent, lowered)
        explicitness = self._explicitness_score(query, entities, keywords, quoted)
        high_stakes = self._is_high_stakes(lowered)

        result = QueryAnalysis(
            intent_type=intent,
            required_evidence=required_evidence,
            output_format_preference=output_pref,
            explicitness_score=explicitness,
            entities=entities,
            keywords=keywords,
            comparison_entities=comparison_entities,
            high_stakes=high_stakes,
        )
        logger.debug("analyze: intent=%s, entities=%d, keywords=%d, high_stakes=%s", intent, len(entities), len(keywords), high_stakes)
        return result

    @staticmethod
    def _detect_intent(lowered: str) -> str:
        nlu_intent = _nlu_detect_intent(lowered)
        if nlu_intent is not None:
            return nlu_intent
        if any(hint in lowered for hint in _NUMERIC_HINTS) or _NUMBER_PATTERN.search(lowered):
            return "numeric_lookup"
        if any(hint in lowered for hint in _FIELD_HINTS):
            return "field_extraction"
        return "factual"

    @staticmethod
    def _requires_numeric(lowered: str) -> bool:
        return bool(_NUMBER_PATTERN.search(lowered) or any(hint in lowered for hint in _NUMERIC_HINTS))

    @staticmethod
    def _requires_date(lowered: str) -> bool:
        if "between" in lowered or "from" in lowered or "during" in lowered:
            return True
        return bool(_DATE_PATTERN.search(lowered))

    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        """Extract named entities from the query using spaCy NER."""
        entities: List[str] = []
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(query)
            for ent in doc.ents:
                cleaned = ent.text.strip()
                if cleaned and cleaned.lower() not in _STOPWORDS and cleaned not in entities:
                    entities.append(cleaned)
            # Also include proper nouns not caught by NER
            for token in doc:
                if token.pos_ == "PROPN" and len(token.text) > 1:
                    word = token.text.strip()
                    if word and word.lower() not in _STOPWORDS and word not in entities:
                        entities.append(word)
        except Exception:
            logger.debug("_extract_entities: spaCy unavailable, using fallback", exc_info=True)
            # Fallback: find capitalized words
            for word in query.split():
                word = word.strip(".,!?;:\"'()[]{}")
                if word and word[0].isupper() and len(word) > 1:
                    if word.lower() not in _STOPWORDS and word not in entities:
                        entities.append(word)
        return entities[:6]

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract meaningful keywords using spaCy lemmatization."""
        try:
            from src.nlp.nlu_engine import parse_query
            sem = parse_query(query)
            keywords = sem.action_verbs + sem.target_nouns + sem.context_words
            return list(dict.fromkeys(keywords))[:6]
        except Exception:
            logger.debug("_extract_keywords: NLU engine unavailable, using fallback", exc_info=True)
        # Fallback: simple tokenization
        tokens = [w for w in query.lower().split() if len(w) >= 3]
        keywords = [tok.strip(".,!?;:\"'()[]{}") for tok in tokens if tok.strip(".,!?;:\"'()[]{}") not in _STOPWORDS]
        return list(dict.fromkeys(keywords))[:6]

    @staticmethod
    def _extract_quoted_phrases(query: str) -> List[str]:
        phrases = []
        for match in _QUOTED_PATTERN.findall(query):
            cleaned = match.strip()
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)
        return phrases

    @staticmethod
    def _extract_error_tokens(query: str) -> List[str]:
        tokens = []
        for match in _ERROR_TOKEN_PATTERN.findall(query):
            if match not in tokens:
                tokens.append(match)
        return tokens

    @staticmethod
    def _extract_comparison_entities(query: str) -> List[str]:
        """Extract entities being compared using spaCy dependency parsing."""
        try:
            from src.nlp.nlu_engine import _get_nlp
            nlp = _get_nlp()
            if nlp is not None:
                doc = nlp(query)
                # Find comparison conjunctions (X vs Y, X versus Y, compare X and Y)
                for token in doc:
                    if token.text.lower() in ("vs", "vs.", "versus"):
                        # Get left and right sides
                        left_tokens = [t.text for t in doc[:token.i] if not t.is_punct and not t.is_space]
                        right_tokens = [t.text for t in doc[token.i + 1:] if not t.is_punct and not t.is_space]
                        if left_tokens and right_tokens:
                            return [" ".join(left_tokens).strip(" -:,."),
                                    " ".join(right_tokens).strip(" -:,.")]
                    elif token.lemma_ == "compare" and token.pos_ == "VERB":
                        # Extract objects of "compare"
                        entities = []
                        for child in token.subtree:
                            if child.pos_ in ("NOUN", "PROPN") and child.dep_ in ("dobj", "pobj", "conj"):
                                entities.append(child.text)
                        if len(entities) >= 2:
                            return entities[:2]
        except Exception:
            logger.debug("_extract_comparison_entities: spaCy unavailable, using regex fallback", exc_info=True)
        # Regex fallback when spaCy is unavailable
        import re as _re
        vs_match = _re.search(r'(?i)\b(\w+)\s+(?:vs\.?|versus)\s+(\w+)', query)
        if vs_match:
            return [vs_match.group(1), vs_match.group(2)]
        compare_match = _re.search(r'(?i)\bcompare\s+(\w+)\s+(?:and|&|with)\s+(\w+)', query)
        if compare_match:
            return [compare_match.group(1), compare_match.group(2)]
        return []

    @staticmethod
    def _output_format_preference(intent: str, lowered: str) -> str:
        if intent == "comparison" or "table" in lowered:
            return "table"
        if intent in {"instruction/how-to", "troubleshooting"}:
            return "steps"
        if intent in {"summarization", "deep_analysis"} or "list" in lowered or "bullet" in lowered:
            return "bullets"
        return "narrative"

    @staticmethod
    def _explicitness_score(query: str, entities: List[str], keywords: List[str], phrases: List[str]) -> float:
        score = 0.0
        tokens = [w for w in query.split() if len(w) >= 2]
        if len(tokens) >= 6:
            score += 0.1
        if entities:
            score += 0.25
        if keywords:
            score += 0.2
        if phrases:
            score += 0.2
        if any(c.isdigit() for c in query):
            score += 0.2
        return min(score, 1.0)

    @staticmethod
    def _section_hints(lowered: str) -> List[str]:
        hints = [hint for hint in _SECTION_HINTS if hint in lowered]
        return hints[:4]

    @staticmethod
    def _is_high_stakes(lowered: str) -> bool:
        stakes = (
            "policy",
            "compliance",
            "contract",
            "legal",
            "invoice",
            "payment",
            "salary",
            "tax",
            "medical",
            "security",
            "audit",
        )
        return any(term in lowered for term in stakes)
