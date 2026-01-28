import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{6,}\d)")
DURATION_RE = re.compile(r"\b(\d{1,2})\s*(years?|yrs?)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.]{2,}")

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


class EntityExtractor:
    def __init__(
        self,
        *,
        skills: Optional[Iterable[str]] = None,
        stopwords: Optional[Set[str]] = None,
        max_keywords: int = 50,
    ):
        self.skills = [s.strip().lower() for s in (skills or DEFAULT_SKILLS) if s.strip()]
        self.skill_set = set(self.skills)
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self.max_keywords = max_keywords
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
