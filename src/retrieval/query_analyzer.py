from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "in", "on", "at", "by",
    "for", "to", "from", "with", "without", "as", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "i", "you", "we", "they", "he", "she", "them",
    "my", "your", "our", "their", "what", "which", "who", "where", "when", "how", "why",
    "does", "do", "did", "can", "could", "should", "would", "may", "might", "will", "shall",
}

_INTENT_PATTERNS = {
    "feedback": [
        r"\bwrong answer\b",
        r"\bnot correct\b",
        r"\bdoesn'?t help\b",
        r"\bfeedback\b",
        r"\bthis is wrong\b",
    ],
    "troubleshooting": [
        r"\berror\b",
        r"\bexception\b",
        r"\bfailed\b",
        r"\bnot working\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\btimeout\b",
        r"\bstack trace\b",
    ],
    "instruction/how-to": [
        r"\bhow to\b",
        r"\bsteps\b",
        r"\bprocedure\b",
        r"\bconfigure\b",
        r"\bsetup\b",
        r"\binstall\b",
        r"\bguide\b",
        r"\bprocess\b",
    ],
    "comparison": [
        r"\bcompare\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference between\b",
        r"\bcontrast\b",
    ],
    "summarization": [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\bbrief\b",
        r"\bhigh[- ]level\b",
        r"\btl;dr\b",
    ],
    "deep_analysis": [
        r"\banaly[sz]e\b",
        r"\broot cause\b",
        r"\bimplication\b",
        r"\brisk\b",
        r"\btrade[- ]off\b",
        r"\bimpact\b",
        r"\bdeep dive\b",
        r"\bevaluate\b",
    ],
}

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
    "experience",
    "salary",
    "cost",
    "price",
    "revenue",
    "budget",
]

_FIELD_HINTS = [
    "email",
    "phone",
    "address",
    "title",
    "role",
    "company",
    "education",
    "skills",
    "certification",
    "license",
    "id",
    "policy number",
    "invoice",
    "total",
    "order",
    "account",
    "contact",
]

_SECTION_HINTS = [
    "summary",
    "overview",
    "details",
    "spec",
    "experience",
    "invoice",
    "totals",
    "financial",
    "education",
    "skills",
    "contact",
    "history",
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

_COMPARISON_PATTERN = re.compile(r"(.+?)\s+(?:vs\.?|versus|compare)\s+(.+)", re.IGNORECASE)

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

        return QueryAnalysis(
            intent_type=intent,
            required_evidence=required_evidence,
            output_format_preference=output_pref,
            explicitness_score=explicitness,
            entities=entities,
            keywords=keywords,
            comparison_entities=comparison_entities,
            high_stakes=high_stakes,
        )

    @staticmethod
    def _detect_intent(lowered: str) -> str:
        for intent, patterns in _INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lowered):
                    return intent
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
        entities: List[str] = []
        for match in re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\b", query):
            cleaned = match.strip()
            if cleaned and cleaned.lower() not in _STOPWORDS and cleaned not in entities:
                entities.append(cleaned)
        for match in re.findall(r"\b[A-Z]{2,}\b", query):
            if match not in entities:
                entities.append(match)
        return entities[:6]

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9]{3,}", query.lower())
        keywords = [tok for tok in tokens if tok not in _STOPWORDS]
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
        match = _COMPARISON_PATTERN.search(query)
        if not match:
            return []
        left = match.group(1).strip(" -:,.")
        right = match.group(2).strip(" -:,.")
        if not left or not right:
            return []
        return [left, right]

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
        tokens = re.findall(r"[A-Za-z0-9]{2,}", query)
        if len(tokens) >= 6:
            score += 0.1
        if entities:
            score += 0.25
        if keywords:
            score += 0.2
        if phrases:
            score += 0.2
        if _NUMBER_PATTERN.search(query):
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
