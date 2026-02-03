from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


_SECTION_MAP = {
    "summary": {"summary", "professional summary", "profile", "overview"},
    "experience": {"experience", "work experience", "work history", "employment", "professional experience"},
    "skills": {"skills", "technical skills", "core competencies", "technologies", "tools"},
    "certifications": {"certifications", "certification", "credentials", "licenses"},
    "education": {"education", "academic", "academics", "degree", "degrees"},
    "awards": {"awards", "achievements", "honors", "recognition"},
    "projects": {"projects", "project experience"},
}

_TECH_KEYWORDS = {
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "sql",
    "nosql",
    "mysql",
    "postgres",
    "postgresql",
    "mongodb",
    "redis",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "react",
    "node",
    "nodejs",
    "linux",
    "git",
    "spark",
    "hadoop",
    "kafka",
    "airflow",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "ml",
    "ai",
    "nlp",
    "rest",
    "graphql",
    "snowflake",
    "tableau",
    "power bi",
    "powerbi",
    "etl",
    "ci/cd",
    "ci-cd",
    "jenkins",
    "terraform",
    "ansible",
    "s3",
    "lambda",
    "kotlin",
    "go",
    "golang",
    "php",
    "ruby",
    "swift",
    "objective-c",
    "android",
    "ios",
}

_FUNCTIONAL_KEYWORDS = {
    "agile",
    "scrum",
    "stakeholder",
    "stakeholder management",
    "leadership",
    "estimation",
    "architecture",
    "mentoring",
    "roadmap",
    "delivery",
    "planning",
    "project management",
    "program management",
    "requirements",
    "analysis",
    "communication",
    "coordination",
    "process improvement",
    "change management",
    "attention to detail",
    "supply chain management",
    "requirement analysis",
}

_STOP_SKILL_PREFIXES = {"other", "additional", "misc"}


def normalize_section_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def classify_section(title: str, text: str | None = None) -> Optional[str]:
    normalized = normalize_section_title(title)
    if normalized:
        for key, aliases in _SECTION_MAP.items():
            for alias in aliases:
                if alias in normalized:
                    return key
    if text:
        lowered = text.lower()
        for key, aliases in _SECTION_MAP.items():
            if any(alias in lowered for alias in aliases):
                return key
    return None


def _split_skill_tokens(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n,|/•\u2022]", text)
    output = []
    for part in parts:
        cleaned = part.strip(" \t-:;")
        if not cleaned:
            continue
        if cleaned.lower().startswith(tuple(_STOP_SKILL_PREFIXES)):
            continue
        if len(cleaned) < 2 or len(cleaned) > 30:
            continue
        if len(cleaned.split()) > 5:
            continue
        output.append(cleaned)
    return output


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
    return output


def classify_skills(text: str) -> Tuple[List[str], List[str]]:
    tokens = _split_skill_tokens(text)
    technical: List[str] = []
    functional: List[str] = []
    for token in tokens:
        lowered = token.lower()
        if any(key in lowered for key in _TECH_KEYWORDS):
            technical.append(token)
        elif any(key in lowered for key in _FUNCTIONAL_KEYWORDS):
            functional.append(token)
    return _dedupe(technical), _dedupe(functional)


__all__ = [
    "classify_section",
    "normalize_section_title",
    "classify_skills",
]
