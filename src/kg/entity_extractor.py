import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{6,}\d)")
DURATION_RE = re.compile(r"\b(\d{1,2})\s*(years?|yrs?)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.]{2,}")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(r"(?i)\b(?:USD|EUR|GBP|INR|AUD|CAD|SGD|JPY)\s*\d+(?:[,\d]*)(?:\.\d{2})?\b")
CURRENCY_RE = re.compile(r"[$€£]\s?\d+(?:[,\d]*)(?:\.\d{2})?")
ID_RE = re.compile(
    r"\b(?:invoice|inv|po|purchase\s+order|contract|case|ticket|id|ref)\s*#?:?\s*([A-Za-z0-9\-]{3,})\b",
    re.IGNORECASE,
)
UPPER_ID_RE = re.compile(r"\b[A-Z]{2,}-\d{3,}\b")
TITLECASE_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
ORG_SUFFIX_RE = re.compile(
    r"\b(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co|PLC|LLP|Group|Holdings|Systems|Technologies|University|Institute|Bank)\b"
)
LOCATION_SUFFIX_RE = re.compile(r"\b(?:City|County|State|Province|Region|District)\b")

DEFAULT_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "go",
    "golang",
    "rust",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "neo4j",
    "qdrant",
    "fastapi",
    "django",
    "flask",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "spark",
    "hadoop",
    "kafka",
    "airflow",
    "linux",
    "supply chain",
    "supply chain management",
]

DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "will",
    "your",
    "into",
    "have",
    "has",
    "are",
    "was",
    "were",
    "you",
    "our",
    "but",
    "not",
    "all",
    "any",
    "can",
    "use",
    "using",
    "used",
    "about",
    "more",
    "less",
    "than",
    "then",
    "also",
    "via",
}


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    type: str


@dataclass(frozen=True)
class ExtractedEntity:
    entity_id: str
    name: str
    type: str
    normalized_name: str
    confidence: float

    @staticmethod
    def build(entity_type: str, name: str, confidence: float) -> "ExtractedEntity":
        normalized = normalize_entity_name(name)
        entity_id = f"{entity_type}::{normalized}"
        return ExtractedEntity(
            entity_id=entity_id,
            name=name.strip(),
            type=entity_type,
            normalized_name=normalized,
            confidence=confidence,
        )


def normalize_entity_name(name: str) -> str:
    normalized = " ".join((name or "").strip().lower().split())
    return normalized



def _nlp_enrich(text: str, add_fn) -> bool:
    """Use spaCy NER to extract PERSON, ORG, GPE/LOC entities.

    Returns True if NLP produced at least one entity, False otherwise.
    The *add_fn* callable has signature ``(type, name, confidence) -> None``.
    """
    try:
        from src.nlp.query_entity_extractor import _get_nlp
        nlp = _get_nlp()
    except Exception:  # noqa: BLE001
        return False
    if nlp is None:
        return False

    doc = nlp(text[:5000])  # cap for performance
    found = False

    # spaCy NER entities
    _SPACY_TYPE_MAP = {
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "LOCATION",
        "PRODUCT": "PRODUCT",
        "LAW": "LAW",
        "EVENT": "EVENT",
    }
    for ent in doc.ents:
        mapped = _SPACY_TYPE_MAP.get(ent.label_)
        if mapped:
            name = ent.text.strip()
            if len(name) > 1:
                add_fn(mapped, name, 0.75)
                found = True

    return found


class EntityExtractor:
    def __init__(
        self,
        *,
        skills: Optional[Iterable[str]] = None,
        stopwords: Optional[Set[str]] = None,
        max_keywords: int = 50,
        use_nlp: bool = True,
    ):
        self.skills = [s.strip().lower() for s in (skills or DEFAULT_SKILLS) if s.strip()]
        self.skill_set = set(self.skills)
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self.max_keywords = max_keywords
        self.use_nlp = use_nlp
        self._skill_patterns = self._compile_skill_patterns(self.skills)

    def _compile_skill_patterns(self, skills: List[str]) -> List[tuple[str, re.Pattern]]:
        patterns: List[tuple[str, re.Pattern]] = []
        for skill in sorted(set(skills), key=len, reverse=True):
            if " " in skill:
                pat = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
            else:
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])", re.IGNORECASE)
            patterns.append((skill, pat))
        return patterns

    def _add_entity(self, entities: Dict[str, Entity], entity_type: str, name: str) -> None:
        normalized = name.strip().lower()
        if not normalized:
            return
        entity_id = f"{entity_type}::{normalized}"
        if entity_id in entities:
            return
        entities[entity_id] = Entity(entity_id=entity_id, name=name.strip(), type=entity_type)

    def extract(self, text: str) -> List[Entity]:
        text = text or ""
        entities: Dict[str, Entity] = {}

        for match in EMAIL_RE.findall(text):
            self._add_entity(entities, "email", match.lower())

        for match in URL_RE.findall(text):
            cleaned = match.rstrip(".,;:)]}")
            if cleaned.startswith("www."):
                cleaned = f"http://{cleaned}"
            self._add_entity(entities, "url", cleaned.lower())

        for match in PHONE_RE.findall(text):
            digits = re.sub(r"\D", "", match)
            if not digits:
                continue
            if len(digits) < 7 or len(digits) > 15:
                continue
            self._add_entity(entities, "phone", digits)

        for match in DURATION_RE.finditer(text):
            years = match.group(1)
            label = f"{years} years"
            self._add_entity(entities, "duration_years", label)

        for match in YEAR_RE.findall(text):
            self._add_entity(entities, "year", match)

        for skill, pattern in self._skill_patterns:
            if pattern.search(text):
                self._add_entity(entities, "skill", skill)

        keywords_added = 0
        for token in TOKEN_RE.findall(text):
            norm = token.lower()
            if norm in self.stopwords:
                continue
            if norm in self.skill_set:
                continue
            if len(norm) < 4:
                continue
            self._add_entity(entities, "keyword", norm)
            keywords_added += 1
            if keywords_added >= self.max_keywords:
                break

        return [entities[key] for key in sorted(entities)]

    def extract_with_metadata(self, text: str) -> List[ExtractedEntity]:
        text = text or ""
        entities: Dict[str, ExtractedEntity] = {}

        def add(entity_type: str, name: str, confidence: float) -> None:
            normalized = normalize_entity_name(name)
            if not normalized:
                return
            entity_id = f"{entity_type}::{normalized}"
            if entity_id in entities:
                return
            entities[entity_id] = ExtractedEntity.build(entity_type, name, confidence)

        # ── Regex-first for structured formats ───────────────────────
        for match in EMAIL_RE.findall(text):
            add("EMAIL", match.lower(), 0.95)

        for match in URL_RE.findall(text):
            cleaned = match.rstrip(".,;:)]}")
            if cleaned.startswith("www."):
                cleaned = f"http://{cleaned}"
            add("URL", cleaned.lower(), 0.85)

        for match in PHONE_RE.findall(text):
            digits = re.sub(r"\D", "", match)
            if 7 <= len(digits) <= 15:
                add("PHONE", digits, 0.9)

        for match in DATE_RE.findall(text):
            add("DATE", match, 0.7)

        for match in YEAR_RE.findall(text):
            add("DATE", match, 0.65)

        for match in AMOUNT_RE.findall(text):
            add("AMOUNT", match, 0.8)

        for match in CURRENCY_RE.findall(text):
            add("AMOUNT", match, 0.85)

        for match in ID_RE.findall(text):
            add("ID", match, 0.7)

        for match in UPPER_ID_RE.findall(text):
            add("ID", match, 0.75)

        for skill, pattern in self._skill_patterns:
            if pattern.search(text):
                add("SKILL", skill, 0.8)

        # ── NLP-first for semantic entities (PERSON, ORG, LOCATION) ──
        nlp_enriched = False
        if self.use_nlp:
            try:
                nlp_enriched = _nlp_enrich(text, add)
            except Exception:  # noqa: BLE001
                pass  # graceful degradation to regex

        # ── Regex fallback for PERSON/ORG/LOCATION if NLP unavailable ─
        if not nlp_enriched:
            for phrase in TITLECASE_ENTITY_RE.findall(text):
                if ORG_SUFFIX_RE.search(phrase):
                    add("ORGANIZATION", phrase, 0.65)
                elif LOCATION_SUFFIX_RE.search(phrase):
                    add("LOCATION", phrase, 0.55)
                else:
                    add("PERSON", phrase, 0.55)

        keywords_added = 0
        for token in TOKEN_RE.findall(text):
            norm = token.lower()
            if norm in self.stopwords or norm in self.skill_set or len(norm) < 4:
                continue
            add("KEYWORD", norm, 0.4)
            keywords_added += 1
            if keywords_added >= self.max_keywords:
                break

        return [entities[key] for key in sorted(entities)]
