"""Deep document analysis during ingestion.

Performs entity extraction, temporal analysis, quality grading, complexity
scoring, domain signal detection, and relationship extraction using generic
algorithms.  No domain-specific regex is used for content extraction -- only
structured-format patterns (email, phone, URL, duration, amount, ID) are
allowed per project design principles.

All heavy external imports (spaCy, dateutil) are lazy so the module works
in environments where those packages are not installed.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

__all__ = ["deep_analyze", "extract_typed_relationships", "DeepAnalysisResult", "EntityMention"]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntityMention:
    """A single entity mention found in the document."""

    text: str
    type: str  # PERSON, ORG, LOCATION, DATE, AMOUNT, DURATION, PRODUCT, LAW, EVENT, SKILL, EMAIL, PHONE, ID
    start_char: int
    end_char: int
    confidence: float
    normalized: str  # ISO date, lowercase name, etc.
    page: Optional[int] = None
    section_title: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class DeepAnalysisResult:
    """Aggregate result of deep document analysis."""

    entities: List[EntityMention] = field(default_factory=list)
    temporal_spans: List[Dict] = field(default_factory=list)  # {start_date, end_date, description, page}
    chronological_order: List[Dict] = field(default_factory=list)  # Events sorted by date
    relationships: List[Dict] = field(default_factory=list)  # {entity1, entity2, relation_type, evidence}
    typed_relationships: List[Dict] = field(default_factory=list)  # {entity1, entity1_type, entity2, entity2_type, relation_type, evidence}
    quality_grade: str = "C"
    quality_score: float = 50.0
    complexity_score: float = 0.0  # 0-1
    domain_signals: Dict[str, float] = field(default_factory=dict)
    section_roles: Dict[str, str] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Helper: generic dict / object accessor
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Access attribute or dict key -- supports both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

# ---------------------------------------------------------------------------
# Structured-format regex patterns (allowed per design principles)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\b(\d{4})\s*[-–—]\s*(\d{4})\b"
    r"|\b(\d+)\s+(?:years?|months?|weeks?|days?)\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?:USD|EUR|GBP|INR|AUD|CAD|CHF|\$|£|€|₹)\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"|\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|INR|AUD|CAD|CHF)\b",
    re.IGNORECASE,
)
_ID_RE = re.compile(
    r"\b(?:[A-Z]{2,5}[-/#]?\d{3,12})\b"
    r"|\b\d{3}[-]\d{2}[-]\d{4}\b",  # SSN-like or generic ID
)

# ---------------------------------------------------------------------------
# spaCy label mapping
# ---------------------------------------------------------------------------

_SPACY_LABEL_MAP: Dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE",
    "TIME": "DATE",
    "MONEY": "AMOUNT",
    "PRODUCT": "PRODUCT",
    "LAW": "LAW",
    "EVENT": "EVENT",
    "WORK_OF_ART": "PRODUCT",
    "NORP": "ORG",
    "LANGUAGE": "SKILL",
}

# ---------------------------------------------------------------------------
# Lazy spaCy loading
# ---------------------------------------------------------------------------

_NLP_INSTANCE = None

def _get_spacy_nlp() -> Any:
    """Return a spaCy Language instance, or None if unavailable."""
    global _NLP_INSTANCE  # noqa: PLW0603
    if _NLP_INSTANCE is not None:
        return _NLP_INSTANCE
    try:
        import spacy  # noqa: F811

        try:
            _NLP_INSTANCE = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed -- try the transformer model as fallback.
            try:
                _NLP_INSTANCE = spacy.load("en_core_web_md")
            except OSError:
                logger.warning("No spaCy model found (en_core_web_sm / en_core_web_md). "
                               "Entity extraction will use regex-only fallback.")
                return None
        return _NLP_INSTANCE
    except ImportError:
        logger.warning("spaCy not installed. Entity extraction will use regex-only fallback.")
        return None

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _extract_entities_spacy(
    text: str,
    section_title: Optional[str] = None,
    page: Optional[int] = None,
) -> List[EntityMention]:
    """Extract entities using spaCy NER."""
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    # Limit text to avoid excessive processing time.
    max_chars = 100_000
    truncated = text[:max_chars]

    try:
        doc = nlp(truncated)
    except Exception as exc:
        logger.warning("spaCy processing failed: %s", exc)
        return []

    entities: List[EntityMention] = []
    for ent in doc.ents:
        mapped_type = _SPACY_LABEL_MAP.get(ent.label_)
        if mapped_type is None:
            continue
        # Skip very short or very long entities that are likely noise.
        stripped = ent.text.strip()
        if len(stripped) < 2 or len(stripped) > 200:
            continue

        normalized = _normalize_entity(stripped, mapped_type)

        # spaCy does not expose per-entity confidence directly; use a
        # heuristic based on label assignment consistency (default 0.80
        # for NER pipeline).
        confidence = 0.80

        entities.append(
            EntityMention(
                text=stripped,
                type=mapped_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=confidence,
                normalized=normalized,
                page=page,
                section_title=section_title,
            )
        )
    return entities

def _extract_entities_regex(
    text: str,
    section_title: Optional[str] = None,
    page: Optional[int] = None,
) -> List[EntityMention]:
    """Extract structured-format entities via regex (EMAIL, PHONE, URL, DURATION, AMOUNT, ID)."""
    entities: List[EntityMention] = []
    patterns: List[Tuple[re.Pattern, str, float]] = [
        (_EMAIL_RE, "EMAIL", 0.95),
        (_PHONE_RE, "PHONE", 0.85),
        (_URL_RE, "ID", 0.90),  # URLs treated as identifiers
        (_AMOUNT_RE, "AMOUNT", 0.90),
        (_ID_RE, "ID", 0.70),
    ]

    for pattern, ent_type, conf in patterns:
        for match in pattern.finditer(text):
            raw = match.group().strip()
            if len(raw) < 3:
                continue
            normalized = _normalize_entity(raw, ent_type)
            entities.append(
                EntityMention(
                    text=raw,
                    type=ent_type,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=conf,
                    normalized=normalized,
                    page=page,
                    section_title=section_title,
                )
            )

    # Duration patterns
    for match in _DURATION_RE.finditer(text):
        raw = match.group().strip()
        normalized = raw.lower()
        entities.append(
            EntityMention(
                text=raw,
                type="DURATION",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.85,
                normalized=normalized,
                page=page,
                section_title=section_title,
            )
        )

    return entities

def _normalize_entity(text: str, ent_type: str) -> str:
    """Normalize an entity value based on its type."""
    if ent_type == "DATE":
        return _try_parse_date_iso(text) or text.strip()
    if ent_type == "EMAIL":
        return text.strip().lower()
    if ent_type in ("PERSON", "ORG", "LOCATION"):
        return text.strip().title()
    if ent_type == "AMOUNT":
        return text.strip()
    return text.strip()

def _deduplicate_entities(entities: List[EntityMention]) -> List[EntityMention]:
    """Remove duplicate entities by (normalized, type) keeping highest confidence."""
    seen: Dict[Tuple[str, str], EntityMention] = {}
    for ent in entities:
        key = (ent.normalized.lower(), ent.type)
        if key not in seen or ent.confidence > seen[key].confidence:
            seen[key] = ent
    return list(seen.values())

# ---------------------------------------------------------------------------
# Temporal extraction
# ---------------------------------------------------------------------------

_DATE_RANGE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})"
    r"\s*[-–—to]+\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)",
    re.IGNORECASE,
)

_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s*[-–—]\s*(\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)\b")

# Full date ranges: "1st Jan 2024 to 31st Mar 2024", "January 1, 2024 - March 31, 2024"
_FULL_DATE_RANGE_RE = re.compile(
    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\s*(?:to|through|until|[-–—])\s*(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}",
    re.IGNORECASE,
)

# ISO date ranges: "2024-01-01 to 2024-03-31"
_ISO_DATE_RANGE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}\s*(?:to|through|until|[-–—])\s*\d{4}-\d{2}-\d{2}",
)

# Quarter references: "Q1 2024", "First Quarter 2024", "1st Quarter of 2024"
_QUARTER_RE = re.compile(
    r"\b(?:Q[1-4]|(?:First|Second|Third|Fourth|1st|2nd|3rd|4th)\s+Quarter(?:\s+of)?)\s+\d{4}\b",
    re.IGNORECASE,
)

# Fiscal year references: "FY2023-24", "FY 2024", "Fiscal Year 2024"
_FISCAL_YEAR_RE = re.compile(
    r"\b(?:FY\s*\d{4}(?:-\d{2,4})?|Fiscal\s+Year\s+\d{4}(?:[/-]\d{2,4})?)\b",
    re.IGNORECASE,
)

# Slash date ranges: "01/01/2024 - 03/31/2024", "1/1/2024 to 3/31/2024"
_SLASH_DATE_RANGE_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}\s*(?:to|through|until|[-–—])\s*\d{1,2}/\d{1,2}/\d{2,4}",
)

# Between dates: "between Jan 1, 2024 and Mar 31, 2024"
_BETWEEN_DATES_RE = re.compile(
    r"\bbetween\s+(?:\w+\s+\d{1,2},?\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\s+and\s+(?:\w+\s+\d{1,2},?\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

# Relative period: "last 6 months", "past 3 years", "previous quarter"
_RELATIVE_PERIOD_RE = re.compile(
    r"\b(?:last|past|previous|prior|recent)\s+(?:\d+\s+)?(?:day|week|month|quarter|year|decade)s?\b",
    re.IGNORECASE,
)

def _try_parse_date_iso(text: str) -> Optional[str]:
    """Attempt to parse *text* into an ISO date string. Returns None on failure."""
    try:
        from dateutil import parser as dateutil_parser

        dt = dateutil_parser.parse(text, fuzzy=True, dayfirst=False)
        return dt.date().isoformat()
    except Exception:  # noqa: BLE001
        return None

def _llm_extract_temporal_spans(text: str, existing_spans: List[Dict]) -> List[Dict]:
    """When regex found DATE entities but no ranges, use LLM to extract date ranges.

    Short timeout (5s), small output (256 tokens). Only called when:
    - Existing spans have dates but no ranges
    - Text is long enough to warrant LLM analysis
    """
    if len(text) < 200:
        return []

    # Only call if we have dates but no ranges
    has_ranges = any(
        s.get("type") in ("date_range", "fiscal_year", "quarter", "relative_period")
        for s in existing_spans
    )
    if has_ranges:
        return []

    try:
        from src.llm.clients import get_default_client

        client = get_default_client()
        prompt = (
            "Extract date ranges from this text. Return a JSON array of objects with "
            "'start', 'end', 'raw_text', 'type' fields. Types: date_range, period, fiscal_year. "
            "Return [] if no ranges found.\n\n"
            f"TEXT (first 1000 chars):\n{text[:1000]}\n\nJSON:"
        )
        response, _ = client.generate_with_metadata(
            prompt,
            {
                "temperature": 0.0,
                "num_predict": 256,
                "num_ctx": 2048,
            },
        )
        # Parse response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        import json

        spans = json.loads(cleaned)
        if isinstance(spans, list):
            return [s for s in spans if isinstance(s, dict) and s.get("raw_text")]
        return []
    except Exception:
        return []

def _extract_temporal_spans(
    text: str,
    entities: List[EntityMention],
    page: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Build temporal_spans and chronological_order from DATE entities and range patterns."""
    spans: List[Dict] = []

    # From explicit date-range patterns (e.g. "Jan 2020 - Dec 2021")
    for match in _DATE_RANGE_RE.finditer(text):
        raw = match.group()
        parts = re.split(r"\s*[-–—to]+\s*", raw, maxsplit=1)
        start_iso = _try_parse_date_iso(parts[0]) if parts else None
        end_str = parts[1].strip() if len(parts) > 1 else None
        if end_str and end_str.lower() in ("present", "current", "now"):
            end_iso = "present"
        else:
            end_iso = _try_parse_date_iso(end_str) if end_str else None
        spans.append({
            "start_date": start_iso or parts[0].strip(),
            "end_date": end_iso or (end_str or ""),
            "description": raw,
            "page": page,
        })

    # From year-range patterns (e.g. "2019-2023")
    for match in _YEAR_RANGE_RE.finditer(text):
        start_year = match.group(1)
        end_raw = match.group(2)
        if end_raw.lower() in ("present", "current", "now"):
            end_val = "present"
        else:
            end_val = end_raw
        # Avoid duplicates with date-range spans already captured
        dup = False
        for existing in spans:
            if start_year in str(existing.get("start_date", "")) and end_val in str(existing.get("end_date", "")):
                dup = True
                break
        if not dup:
            spans.append({
                "start_date": f"{start_year}-01-01",
                "end_date": f"{end_val}-01-01" if end_val != "present" else "present",
                "description": match.group(),
                "page": page,
            })

    # Enhanced temporal patterns
    for match in _FULL_DATE_RANGE_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "date_range", "source": "regex"})
    for match in _ISO_DATE_RANGE_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "date_range", "source": "regex"})
    for match in _QUARTER_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "quarter", "source": "regex"})
    for match in _FISCAL_YEAR_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "fiscal_year", "source": "regex"})
    for match in _SLASH_DATE_RANGE_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "date_range", "source": "regex"})
    for match in _BETWEEN_DATES_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "date_range", "source": "regex"})
    for match in _RELATIVE_PERIOD_RE.finditer(text):
        spans.append({"raw_text": match.group(), "type": "relative_period", "source": "regex"})

    # LLM fallback for complex temporal expressions
    try:
        llm_spans = _llm_extract_temporal_spans(text, spans)
        for s in llm_spans:
            s["source"] = "llm"
            spans.append(s)
    except Exception:
        pass

    # Deduplicate by raw_text
    seen: set = set()
    unique_spans: List[Dict] = []
    for s in spans:
        key = s.get("raw_text", s.get("description", "")).strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_spans.append(s)
        elif not key:
            # Keep spans without a text key (original format spans)
            unique_spans.append(s)
    spans = unique_spans

    # Chronological order from all DATE entities
    chronological: List[Dict] = []
    for ent in entities:
        if ent.type != "DATE":
            continue
        iso = _try_parse_date_iso(ent.text)
        if iso:
            chronological.append({
                "date": iso,
                "text": ent.text,
                "page": ent.page,
                "section_title": ent.section_title,
            })

    chronological.sort(key=lambda d: d.get("date", ""))

    return spans, chronological

# ---------------------------------------------------------------------------
# Quality grading
# ---------------------------------------------------------------------------

def _score_text_quality(text: str) -> float:
    """Score text quality 0-20: encoding errors, avg word length, readability."""
    if not text:
        return 0.0

    words = re.findall(r"\w+", text)
    if not words:
        return 0.0

    total_chars = len(text)

    # Encoding error ratio: count replacement characters and garbled sequences
    garbled_count = text.count("\ufffd") + len(re.findall(r"[^\x00-\x7F]{3,}", text))
    encoding_score = max(0.0, 1.0 - (garbled_count / max(1, total_chars)) * 100)

    # Average word length (reasonable range: 3-8 chars)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_len_score = 1.0 - min(1.0, abs(avg_word_len - 5.0) / 5.0)

    # Readability proxy: sentence count vs word count
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    if sentences:
        avg_sentence_len = len(words) / len(sentences)
        readability_score = 1.0 - min(1.0, abs(avg_sentence_len - 18.0) / 30.0)
    else:
        readability_score = 0.3

    return round(
        20.0 * (0.40 * min(1.0, encoding_score) + 0.30 * word_len_score + 0.30 * readability_score),
        2,
    )

def _score_structural_completeness(sections: List[Dict], text: str) -> float:
    """Score structural completeness 0-20: section count, headings, hierarchy."""
    section_count = len(sections)
    if section_count == 0:
        return 4.0  # Minimal score for flat text

    # Section count factor (more sections = more structured, up to ~15)
    count_score = min(1.0, section_count / 10.0)

    # Heading presence: how many sections have non-default titles
    titled = sum(1 for s in sections if s.get("title", "").strip()
                 and s["title"].lower() not in ("untitled section", "document", ""))
    heading_score = titled / max(1, section_count)

    # Hierarchy proxy: do titles vary in apparent depth (numbered, length variation)
    titles = [s.get("title", "") for s in sections if s.get("title")]
    unique_len_buckets = len(set(min(len(t) // 10, 5) for t in titles)) if titles else 0
    hierarchy_score = min(1.0, unique_len_buckets / 3.0)

    return round(20.0 * (0.40 * count_score + 0.35 * heading_score + 0.25 * hierarchy_score), 2)

def _score_content_density(text: str, sections: List[Dict], page_count: Optional[int]) -> float:
    """Score content density 0-20: words per page, entity density, information richness."""
    words = re.findall(r"\w+", text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # Words per page (ideal range: 200-500)
    pages = max(1, page_count or 1)
    wpp = word_count / pages
    wpp_score = min(1.0, wpp / 300.0)

    # Entity density (using structural patterns)
    entity_hits = sum(len(p.findall(text)) for p in [_EMAIL_RE, _PHONE_RE, _AMOUNT_RE, _ID_RE])
    ent_density_score = min(1.0, entity_hits / max(1, word_count) * 50)

    # Information richness: ratio of unique n-grams to total
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)] if len(words) > 1 else []
    richness = len(set(bigrams)) / max(1, len(bigrams)) if bigrams else 0.5

    return round(20.0 * (0.35 * wpp_score + 0.30 * ent_density_score + 0.35 * richness), 2)

def _score_metadata_richness(entities: List[EntityMention]) -> float:
    """Score metadata richness 0-20: dates, names, identifiers present."""
    type_counts: Dict[str, int] = Counter(e.type for e in entities)

    has_dates = min(1.0, type_counts.get("DATE", 0) / 3.0)
    has_names = min(1.0, type_counts.get("PERSON", 0) / 2.0)
    has_orgs = min(1.0, type_counts.get("ORG", 0) / 2.0)
    has_ids = min(1.0, (type_counts.get("ID", 0) + type_counts.get("EMAIL", 0) + type_counts.get("PHONE", 0)) / 2.0)
    has_amounts = min(1.0, type_counts.get("AMOUNT", 0) / 2.0)

    return round(20.0 * (0.25 * has_dates + 0.25 * has_names + 0.20 * has_orgs + 0.15 * has_ids + 0.15 * has_amounts), 2)

def _score_consistency(entities: List[EntityMention], sections: List[Dict]) -> float:
    """Score consistency 0-20: no contradictions, uniform formatting."""
    if not entities and not sections:
        return 10.0  # Neutral

    # Check for duplicate names with different normalizations (inconsistency signal)
    person_variants: Dict[str, set] = defaultdict(set)
    for ent in entities:
        if ent.type == "PERSON":
            key = ent.normalized.lower().split()[0] if ent.normalized else ""
            if key and len(key) > 2:
                person_variants[key].add(ent.normalized)

    # Fewer variant spellings per first-name = more consistent
    variant_counts = [len(v) for v in person_variants.values()]
    avg_variants = sum(variant_counts) / max(1, len(variant_counts)) if variant_counts else 1.0
    name_consistency = max(0.0, 1.0 - (avg_variants - 1.0) * 0.3)

    # Section length uniformity (lower variance = more consistent)
    section_lengths = [len(s.get("text", "")) for s in sections if s.get("text")]
    if len(section_lengths) > 1:
        mean_len = sum(section_lengths) / len(section_lengths)
        variance = sum((l - mean_len) ** 2 for l in section_lengths) / len(section_lengths)
        cv = math.sqrt(variance) / max(1, mean_len)  # Coefficient of variation
        length_consistency = max(0.0, 1.0 - cv)
    else:
        length_consistency = 0.7

    return round(20.0 * (0.50 * name_consistency + 0.50 * length_consistency), 2)

def _grade_from_score(score: float) -> str:
    """Map a 0-100 quality score to a letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"

# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

def _compute_complexity(
    entities: List[EntityMention],
    sections: List[Dict],
    text: str,
) -> float:
    """Compute document complexity on a 0-1 scale."""
    if not text:
        return 0.0

    words = re.findall(r"\w+", text.lower())
    total_words = len(words)
    if total_words == 0:
        return 0.0

    # Unique entity count factor
    unique_entities = len(set((e.normalized.lower(), e.type) for e in entities))
    entity_factor = min(1.0, unique_entities / 50.0)

    # Section count factor
    section_factor = min(1.0, len(sections) / 15.0)

    # Cross-reference detection: count internal references like "see section", "as above", "refer to"
    cross_ref_patterns = re.findall(
        r"\b(?:see\s+(?:section|above|below|page)|refer\s+to|as\s+(?:mentioned|noted|described)\s+(?:above|below|earlier))\b",
        text,
        re.IGNORECASE,
    )
    cross_ref_factor = min(1.0, len(cross_ref_patterns) / 10.0)

    # Vocabulary richness (type-token ratio)
    unique_words = len(set(words))
    ttr = unique_words / total_words

    complexity = (
        0.30 * entity_factor
        + 0.20 * section_factor
        + 0.15 * cross_ref_factor
        + 0.35 * ttr
    )

    return round(min(1.0, max(0.0, complexity)), 4)

# ---------------------------------------------------------------------------
# Domain signal detection (keyword overlap, no regex)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "hr": [
        "resume", "candidate", "experience", "education", "skills", "employment",
        "interview", "salary", "hiring", "position", "qualification", "recruitment",
        "applicant", "curriculum vitae", "linkedin", "reference", "competency",
        "onboarding", "job description", "performance review",
    ],
    "invoice": [
        "invoice", "bill to", "payment terms", "amount due", "total due", "total amount",
        "vat", "gst", "purchase order", "vendor", "supplier", "unit price", "subtotal",
        "discount", "remittance", "accounts payable", "net amount", "line item",
        "due date", "invoice number", "invoice date",
    ],
    "medical": [
        "patient", "diagnosis", "medication", "treatment", "prescription", "dosage",
        "symptoms", "clinical", "hospital", "physician", "laboratory", "blood pressure",
        "medical history", "allergies", "immunization", "discharge", "prognosis",
        "vital signs", "radiology", "pathology",
    ],
    "legal": [
        "agreement", "contract", "clause", "party", "liability", "indemnification",
        "governing law", "jurisdiction", "arbitration", "confidentiality", "termination",
        "warranty", "breach", "damages", "intellectual property", "amendment",
        "force majeure", "representation", "covenant", "statute",
    ],
    "policy": [
        "policy", "procedure", "compliance", "regulation", "guideline", "standard",
        "requirement", "audit", "certification", "risk management", "control",
        "framework", "governance", "reporting", "assessment", "mitigation",
        "enforcement", "documentation", "approval", "escalation",
    ],
    "generic": [
        "report", "summary", "analysis", "overview", "introduction", "conclusion",
        "recommendation", "finding", "observation", "methodology", "objective",
        "scope", "background", "appendix", "reference", "figure", "table",
        "abstract", "executive summary", "discussion",
    ],
}

def _detect_domain_signals(text: str) -> Dict[str, float]:
    """Score each domain based on keyword overlap with the document text."""
    if not text:
        return {domain: 0.0 for domain in _DOMAIN_KEYWORDS}

    lowered = text.lower()
    words_in_text = set(re.findall(r"\w+", lowered))
    signals: Dict[str, float] = {}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            # Support multi-word keywords
            if " " in kw:
                if kw in lowered:
                    hits += 1
            else:
                if kw in words_in_text:
                    hits += 1
        signals[domain] = round(min(1.0, hits / max(1, len(keywords))), 4)

    return signals

# ---------------------------------------------------------------------------
# Section role detection
# ---------------------------------------------------------------------------

def _detect_section_roles(sections: List[Dict]) -> Dict[str, str]:
    """Assign a semantic role to each section based on title and content heuristics."""
    roles: Dict[str, str] = {}
    for sec in sections:
        title = (sec.get("title") or "").strip()
        text = (sec.get("text") or "").strip()
        if not title:
            continue

        lowered_title = title.lower()
        lowered_text = text[:500].lower() if text else ""

        if any(w in lowered_title for w in ("summary", "abstract", "overview", "executive")):
            roles[title] = "summary"
        elif any(w in lowered_title for w in ("introduction", "background", "purpose", "objective")):
            roles[title] = "introduction"
        elif any(w in lowered_title for w in ("conclusion", "closing", "recommendation")):
            roles[title] = "conclusion"
        elif any(w in lowered_title for w in ("appendix", "annex", "attachment", "exhibit")):
            roles[title] = "appendix"
        elif any(w in lowered_title for w in ("reference", "bibliography", "citation")):
            roles[title] = "references"
        elif any(w in lowered_title for w in ("method", "procedure", "approach")):
            roles[title] = "methodology"
        elif any(w in lowered_title for w in ("result", "finding", "outcome", "analysis")):
            roles[title] = "findings"
        elif any(w in lowered_title for w in ("table", "figure", "chart", "graph")):
            roles[title] = "visual_element"
        elif any(w in lowered_text for w in ("invoice", "amount due", "total due", "subtotal", "unit price", "balance due")) and \
                len(re.findall(r"\d", text[:500])) > 10:
            roles[title] = "transactional"
        else:
            roles[title] = "content"

    return roles

# ---------------------------------------------------------------------------
# Relationship extraction (co-occurrence)
# ---------------------------------------------------------------------------

def _extract_relationships(
    entities: List[EntityMention],
) -> List[Dict]:
    """Extract co-occurrence relationships between entities in the same section.

    Only entities of relevant types (PERSON, ORG, LOCATION, PRODUCT, LAW)
    are considered.  Relationships require frequency >= 2 co-occurrences.
    """
    relevant_types = {"PERSON", "ORG", "LOCATION", "PRODUCT", "LAW"}

    # Group entities by section
    section_groups: Dict[Optional[str], List[EntityMention]] = defaultdict(list)
    for ent in entities:
        if ent.type in relevant_types:
            section_groups[ent.section_title].append(ent)

    # Count co-occurrences
    pair_counter: Counter = Counter()
    pair_evidence: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for section_title, group in section_groups.items():
        # Deduplicate within this section
        unique_in_section = list({(e.normalized.lower(), e.type): e for e in group}.values())
        for i, e1 in enumerate(unique_in_section):
            for e2 in unique_in_section[i + 1:]:
                # Create a canonical pair key
                key = tuple(sorted([(e1.normalized, e1.type), (e2.normalized, e2.type)]))
                pair_counter[key] += 1
                evidence_str = section_title or "same section"
                if evidence_str not in pair_evidence[key]:
                    pair_evidence[key].append(evidence_str)

    relationships: List[Dict] = []
    for pair, count in pair_counter.items():
        if count < 2:
            continue
        (name1, type1), (name2, type2) = pair
        relationships.append({
            "entity1": {"text": name1, "type": type1},
            "entity2": {"text": name2, "type": type2},
            "relation_type": "RELATED_TO",
            "evidence": f"Co-occurred in {count} section(s): {', '.join(pair_evidence[pair][:5])}",
        })

    return relationships

# ---------------------------------------------------------------------------
# Typed relationship extraction (sentence-level, keyword matching)
# ---------------------------------------------------------------------------

# Verb patterns for each typed relationship
_WORKED_AT_VERBS = re.compile(
    r"\b(?:work(?:s|ed|ing)?|join(?:s|ed|ing)?|manag(?:es|ed|ing)?|employ(?:s|ed|ing)?|"
    r"lead(?:s|ing)?|led|head(?:s|ed|ing)?|serv(?:es|ed|ing)?|serv(?:ing)?)\b",
    re.IGNORECASE,
)
_STUDIED_AT_VERBS = re.compile(
    r"\b(?:studi(?:es|ed|ing)?|graduat(?:es|ed|ing)?|attend(?:s|ed|ing)?|"
    r"enroll(?:s|ed|ing)?|complet(?:es|ed|ing)?)\b",
    re.IGNORECASE,
)
_OWES_TO_VERBS = re.compile(
    r"\b(?:ow(?:es|ed|ing)?|due|pay(?:s|ing)?|paid|invoic(?:es|ed|ing)?|bill(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)

def extract_typed_relationships(
    entities: List[EntityMention],
    full_text: str,
) -> List[Dict]:
    """Extract typed relationships between entities using sentence-level keyword matching.

    For each sentence in *full_text* that contains two or more entities, the
    verb patterns present in the sentence determine the relation type:

    - PERSON + ORG + work/join/manage/employ/lead verbs  → ``WORKED_AT``
    - PERSON + ORG + study/graduate/attend/enroll verbs  → ``STUDIED_AT``
    - SKILL  + PERSON (same sentence)                    → ``HAS_SKILL``
    - AMOUNT + ORG + owe/due/pay/invoice/bill verbs       → ``OWES_TO``
    - PERSON + (DATE | DURATION) in same sentence         → ``ACTIVE_DURING``
    - Any other co-occurring pair                         → ``RELATED_TO``

    spaCy dependency parse is intentionally *not* used — simple string
    membership checks keep this function fast and dependency-free.

    Parameters
    ----------
    entities:
        The :class:`EntityMention` objects produced by :func:`_deep_analyze_impl`.
    full_text:
        The full document text, used for sentence splitting.

    Returns
    -------
    list of dicts
        Each dict has keys: ``entity1``, ``entity1_type``, ``entity2``,
        ``entity2_type``, ``relation_type``, ``evidence``.
    """
    if not entities or not full_text:
        return []

    # Split into sentences on ". " boundary (fast, no NLTK required)
    sentences: List[str] = re.split(r"(?<=[.!?])\s+", full_text)

    # Build a lookup: entity_text (lowercased) → EntityMention
    entity_lookup: Dict[str, EntityMention] = {}
    for ent in entities:
        key = ent.text.strip().lower()
        if key and len(key) >= 2:
            entity_lookup[key] = ent

    typed_rels: List[Dict] = []
    seen_pairs: set = set()

    for sentence in sentences:
        if not sentence.strip():
            continue
        sent_lower = sentence.lower()

        # Collect all entities whose text appears in this sentence
        present: List[EntityMention] = [
            ent for key, ent in entity_lookup.items() if key in sent_lower
        ]
        if len(present) < 2:
            continue

        # Compare every pair in this sentence
        for i, e1 in enumerate(present):
            for e2 in present[i + 1:]:
                # Canonical pair key to avoid duplicate edges
                pair_key = tuple(sorted([
                    (e1.normalized.lower(), e1.type),
                    (e2.normalized.lower(), e2.type),
                ]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                t1, t2 = e1.type, e2.type

                # Determine relation type based on entity types + verb patterns
                relation_type = "RELATED_TO"

                if "PERSON" in (t1, t2) and "ORG" in (t1, t2):
                    if _WORKED_AT_VERBS.search(sentence):
                        relation_type = "WORKED_AT"
                    elif _STUDIED_AT_VERBS.search(sentence):
                        relation_type = "STUDIED_AT"

                elif "SKILL" in (t1, t2) and "PERSON" in (t1, t2):
                    relation_type = "HAS_SKILL"

                elif "AMOUNT" in (t1, t2) and "ORG" in (t1, t2):
                    if _OWES_TO_VERBS.search(sentence):
                        relation_type = "OWES_TO"

                elif "PERSON" in (t1, t2) and t1 in ("DATE", "DURATION") or t2 in ("DATE", "DURATION"):
                    if "PERSON" in (t1, t2):
                        relation_type = "ACTIVE_DURING"

                # Truncate evidence to keep payloads small
                evidence = sentence.strip()
                if len(evidence) > 200:
                    evidence = evidence[:197] + "..."

                typed_rels.append({
                    "entity1": e1.text,
                    "entity1_type": t1,
                    "entity2": e2.text,
                    "entity2_type": t2,
                    "relation_type": relation_type,
                    "evidence": evidence,
                })

    return typed_rels

# ---------------------------------------------------------------------------
# Text extraction from document objects
# ---------------------------------------------------------------------------

def _extract_full_text(extracted: Any) -> str:
    """Extract the full text from an extracted document object or dict."""
    full_text = _get(extracted, "full_text", "") or ""
    if full_text:
        return full_text

    sections = _get(extracted, "sections") or []
    if sections:
        parts = []
        for sec in sections:
            text = _get(sec, "text", "") or ""
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    return ""

def _extract_sections_list(extracted: Any) -> List[Dict]:
    """Extract sections as a list of dicts with title, text, page info."""
    sections = _get(extracted, "sections") or []
    result: List[Dict] = []
    for sec in sections:
        result.append({
            "title": (_get(sec, "title") or "Untitled Section"),
            "text": (_get(sec, "text") or ""),
            "start_page": _get(sec, "start_page"),
            "end_page": _get(sec, "end_page"),
        })

    if not result:
        full_text = _get(extracted, "full_text", "") or ""
        if full_text:
            result.append({
                "title": "Document",
                "text": full_text,
                "start_page": None,
                "end_page": None,
            })

    return result

def _get_page_count(extracted: Any) -> Optional[int]:
    """Try to determine the page count from the extracted document."""
    try:
        sections = _get(extracted, "sections") or []
        pages = [_get(sec, "end_page") for sec in sections]
        pages = [p for p in pages if isinstance(p, int)]
        if pages:
            return max(pages)
    except Exception:  # noqa: BLE001
        pass
    return None

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def deep_analyze(
    extracted: Any,
    identification: Optional[Dict] = None,
    content_map: Optional[Dict] = None,
    structure: Optional[Dict] = None,
) -> DeepAnalysisResult:
    """Perform deep document analysis on an extracted document.

    This function orchestrates entity extraction, temporal analysis, quality
    grading, complexity scoring, domain signal detection, section role
    assignment, and relationship extraction.

    Parameters
    ----------
    extracted:
        The extracted document -- either a dict with "full_text"/"sections"
        keys, or an object with corresponding attributes.
    identification:
        Optional output from ``identify_document()``.
    content_map:
        Optional output from ``build_content_map()``.
    structure:
        Optional output from ``infer_structure()``.

    Returns
    -------
    DeepAnalysisResult
        Comprehensive analysis result.  On any unrecoverable error a
        minimal empty result is returned.
    """
    try:
        return _deep_analyze_impl(extracted, identification, content_map, structure)
    except Exception as exc:
        logger.warning("Deep analysis failed, returning minimal result: %s", exc, exc_info=True)
        return DeepAnalysisResult(
            domain_signals={d: 0.0 for d in _DOMAIN_KEYWORDS},
        )

def _deep_analyze_impl(
    extracted: Any,
    identification: Optional[Dict],
    content_map: Optional[Dict],
    structure: Optional[Dict],
) -> DeepAnalysisResult:
    """Internal implementation of deep_analyze -- may raise."""
    full_text = _extract_full_text(extracted)
    sections = _extract_sections_list(extracted)
    page_count = _get_page_count(extracted)

    if not full_text and not sections:
        logger.debug("Deep analysis received empty document -- returning minimal result.")
        return DeepAnalysisResult(domain_signals={d: 0.0 for d in _DOMAIN_KEYWORDS})

    # -----------------------------------------------------------------------
    # 1. Entity extraction
    # -----------------------------------------------------------------------
    all_entities: List[EntityMention] = []

    for sec in sections:
        sec_title = sec.get("title")
        sec_page = sec.get("start_page")
        sec_text = sec.get("text", "")
        if not sec_text:
            continue

        # spaCy NER
        spacy_ents = _extract_entities_spacy(sec_text, section_title=sec_title, page=sec_page)
        all_entities.extend(spacy_ents)

        # Regex structured formats
        regex_ents = _extract_entities_regex(sec_text, section_title=sec_title, page=sec_page)
        all_entities.extend(regex_ents)

    all_entities = _deduplicate_entities(all_entities)

    # -----------------------------------------------------------------------
    # 2. Temporal extraction
    # -----------------------------------------------------------------------
    temporal_spans, chronological_order = _extract_temporal_spans(
        full_text, all_entities, page=None,
    )

    # -----------------------------------------------------------------------
    # 3. Quality grading (5 dimensions, 0-20 each)
    # -----------------------------------------------------------------------
    q_text = _score_text_quality(full_text)
    q_structure = _score_structural_completeness(sections, full_text)
    q_density = _score_content_density(full_text, sections, page_count)
    q_metadata = _score_metadata_richness(all_entities)
    q_consistency = _score_consistency(all_entities, sections)

    quality_score = round(q_text + q_structure + q_density + q_metadata + q_consistency, 2)
    quality_score = max(0.0, min(100.0, quality_score))
    quality_grade = _grade_from_score(quality_score)

    # -----------------------------------------------------------------------
    # 4. Complexity scoring
    # -----------------------------------------------------------------------
    complexity_score = _compute_complexity(all_entities, sections, full_text)

    # -----------------------------------------------------------------------
    # 5. Domain signals
    # -----------------------------------------------------------------------
    domain_signals = _detect_domain_signals(full_text)

    # -----------------------------------------------------------------------
    # 6. Section roles
    # -----------------------------------------------------------------------
    section_roles = _detect_section_roles(sections)

    # -----------------------------------------------------------------------
    # 7. Relationship extraction (co-occurrence baseline)
    # -----------------------------------------------------------------------
    relationships = _extract_relationships(all_entities)

    # -----------------------------------------------------------------------
    # 8. Typed relationship extraction (sentence-level, keyword matching)
    # -----------------------------------------------------------------------
    typed_relationships = extract_typed_relationships(all_entities, full_text)

    return DeepAnalysisResult(
        entities=all_entities,
        temporal_spans=temporal_spans,
        chronological_order=chronological_order,
        relationships=relationships,
        typed_relationships=typed_relationships,
        quality_grade=quality_grade,
        quality_score=quality_score,
        complexity_score=complexity_score,
        domain_signals=domain_signals,
        section_roles=section_roles,
    )
