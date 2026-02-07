from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


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
    "tensorflow",
    "pytorch",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
]

ORG_SUFFIXES = {
    "inc",
    "llc",
    "ltd",
    "corp",
    "corporation",
    "company",
    "co",
    "group",
    "holdings",
    "university",
    "institute",
    "bank",
}

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

CURRENCY_SYMBOLS = {"$", "€", "£"}
CURRENCY_CODES = {"usd", "eur", "gbp", "inr", "aud", "cad", "sgd", "jpy"}


@dataclass(frozen=True)
class ExtractedEntity:
    entity_norm: str
    entity_type: str
    surface_form: str


class EntityExtractor:
    def __init__(self, skills: Optional[Iterable[str]] = None) -> None:
        skill_list = [s.strip().lower() for s in (skills or DEFAULT_SKILLS) if s.strip()]
        self.skills = set(skill_list)

    def extract(self, text: str) -> List[ExtractedEntity]:
        tokens = _tokenize(text)
        entities: Dict[tuple[str, str], ExtractedEntity] = {}

        for token in tokens:
            lowered = token.lower()
            if _is_email(token):
                _add_entity(entities, "email", lowered, token)
                continue
            if _is_phone(token):
                _add_entity(entities, "phone", _digits_only(token), token)
                continue
            if _is_money(token):
                _add_entity(entities, "money", lowered, token)
                continue
            if _is_date_token(lowered):
                _add_entity(entities, "date", lowered, token)
                continue
            if lowered in self.skills:
                _add_entity(entities, "skill", lowered, token)
                continue
            if _looks_like_id(token):
                _add_entity(entities, "id", lowered, token)
                continue
            if _is_org_token(token):
                _add_entity(entities, "organization", lowered, token)

        for phrase in _titlecase_phrases(tokens):
            _add_entity(entities, "person", _normalize_phrase(phrase), phrase)

        return list(entities.values())


def _add_entity(entities: Dict[tuple[str, str], ExtractedEntity], entity_type: str, norm: str, surface: str) -> None:
    key = (entity_type, norm)
    if not norm:
        return
    if key in entities:
        return
    entities[key] = ExtractedEntity(entity_norm=norm, entity_type=entity_type, surface_form=surface)


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    for ch in text or "":
        if ch.isalnum() or ch in {"@", ".", "+", "-", "_", "#"}:
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _normalize_phrase(text: str) -> str:
    cleaned = text.strip().lower()
    return " ".join(cleaned.split())


def _titlecase_phrases(tokens: List[str]) -> List[str]:
    phrases: List[str] = []
    current: List[str] = []
    for token in tokens:
        if _is_titlecase(token):
            current.append(token)
        else:
            if len(current) >= 2:
                phrases.append(" ".join(current))
            current = []
    if len(current) >= 2:
        phrases.append(" ".join(current))
    return phrases


def _is_titlecase(token: str) -> bool:
    if not token:
        return False
    if not token[0].isupper():
        return False
    if any(ch.isdigit() for ch in token):
        return False
    rest = token[1:]
    if not rest:
        return False
    return rest.islower() or rest.istitle()


def _is_email(token: str) -> bool:
    if token.count("@") != 1:
        return False
    name, domain = token.split("@", 1)
    if not name or not domain or "." not in domain:
        return False
    return True


def _digits_only(token: str) -> str:
    return "".join(ch for ch in token if ch.isdigit())


def _is_phone(token: str) -> bool:
    digits = _digits_only(token)
    if len(digits) < 7 or len(digits) > 15:
        return False
    return True


def _is_money(token: str) -> bool:
    if not token:
        return False
    lowered = token.lower()
    if token[0] in CURRENCY_SYMBOLS and any(ch.isdigit() for ch in token):
        return True
    if lowered.endswith(tuple(CURRENCY_CODES)) and any(ch.isdigit() for ch in token):
        return True
    return False


def _is_date_token(token: str) -> bool:
    if token in MONTHS:
        return True
    if token.isdigit() and len(token) == 4:
        return True
    if "/" in token or "-" in token:
        parts = [p for p in token.replace("-", "/").split("/") if p]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return True
    return False


def _looks_like_id(token: str) -> bool:
    if len(token) < 4:
        return False
    has_alpha = any(ch.isalpha() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    return has_alpha and has_digit


def _is_org_token(token: str) -> bool:
    cleaned = token.strip(string.punctuation).lower()
    if cleaned in ORG_SUFFIXES:
        return True
    return False


__all__ = ["EntityExtractor", "ExtractedEntity"]
