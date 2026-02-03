from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_ROLE_KEYWORDS = {
    "engineer",
    "developer",
    "manager",
    "analyst",
    "consultant",
    "architect",
    "designer",
    "lead",
    "director",
    "specialist",
    "intern",
    "scientist",
    "administrator",
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
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "react",
    "node",
    "linux",
    "git",
    "spark",
    "pandas",
    "tensorflow",
    "pytorch",
    "nlp",
    "ml",
    "ai",
    "rest",
    "graphql",
    "snowflake",
    "tableau",
}

_FUNC_KEYWORDS = {
    "project management",
    "stakeholder",
    "communication",
    "leadership",
    "agile",
    "scrum",
    "requirements",
    "analysis",
    "product",
    "strategy",
    "planning",
    "mentoring",
    "presentation",
    "coordination",
    "roadmap",
    "delivery",
}

_NAME_STOPWORDS = {"resume", "curriculum", "vitae", "profile", "summary", "experience"}

_YEAR_PATTERN = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s*\+?\s*years?\b", re.IGNORECASE)
_YEAR_TOKEN_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class CandidateProfile:
    candidate_name: Optional[str]
    total_years_experience: Optional[float]
    experience_summary: str
    technical_skills: List[str]
    functional_skills: List[str]
    certifications: List[str]
    education: List[str]
    achievements_awards: List[str]
    source_type: str
    source_document: str
    evidence_chunk_ids: List[str]


def _chunk_text(chunk: Any) -> str:
    return getattr(chunk, "text", "") or ""


def _chunk_meta(chunk: Any) -> Dict[str, Any]:
    return getattr(chunk, "metadata", {}) or {}


def _chunk_id(chunk: Any) -> str:
    meta = _chunk_meta(chunk)
    return str(meta.get("chunk_id") or getattr(chunk, "id", "") or "")


def _lines_from_chunks(chunks: Sequence[Any], max_lines: int = 400) -> List[str]:
    lines: List[str] = []
    for chunk in chunks:
        for raw in _chunk_text(chunk).splitlines():
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                return lines
    return lines


def _looks_like_name(line: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z\s.-]", "", line).strip()
    if not cleaned:
        return False
    tokens = cleaned.split()
    if not (2 <= len(tokens) <= 4):
        return False
    if any(token.lower() in _NAME_STOPWORDS for token in tokens):
        return False
    letters = [c for c in cleaned if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    return upper_ratio >= 0.6


def _extract_name(lines: Sequence[str]) -> Optional[str]:
    for line in lines[:30]:
        match = re.search(r"\bname\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and len(candidate.split()) <= 4:
                return candidate
    for line in lines[:20]:
        if _looks_like_name(line):
            return line.strip()
    return None


def _extract_years(lines: Sequence[str]) -> Optional[float]:
    years = []
    for line in lines:
        for match in _YEAR_PATTERN.finditer(line):
            try:
                years.append(float(match.group(1)))
            except Exception:
                continue
    if years:
        return max(years)
    return None


def _estimate_years_from_ranges(lines: Sequence[str]) -> Optional[float]:
    ranges: List[Tuple[int, int]] = []
    current_year = datetime.utcnow().year
    for line in lines:
        years = [int(y) for y in _YEAR_TOKEN_PATTERN.findall(line)]
        if not years:
            continue
        lower = line.lower()
        if "present" in lower or "current" in lower:
            ranges.append((min(years), current_year))
        elif len(years) >= 2:
            ranges.append((min(years), max(years)))
    if not ranges:
        return None
    start_year = min(start for start, _ in ranges)
    end_year = max(end for _, end in ranges)
    if end_year <= start_year:
        return None
    return float(end_year - start_year)


def _extract_list_items(lines: Iterable[str], cue_words: Iterable[str]) -> List[str]:
    items: List[str] = []
    for line in lines:
        lower = line.lower()
        if any(cue in lower for cue in cue_words):
            parts = re.split(r"[;|,•]", line)
            for part in parts:
                cleaned = part.strip(" -:\t")
                if len(cleaned) < 2:
                    continue
                items.append(cleaned)
    return items


def _classify_skills(items: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    technical: List[str] = []
    functional: List[str] = []
    unknown: List[str] = []
    for item in items:
        token = item.strip()
        if not token:
            continue
        token_lower = token.lower()
        if any(key in token_lower for key in _TECH_KEYWORDS):
            technical.append(token)
        elif any(key in token_lower for key in _FUNC_KEYWORDS):
            functional.append(token)
        else:
            unknown.append(token)
    return technical, functional, unknown


def _unique(items: Iterable[str], limit: int = 12) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
        if len(output) >= limit:
            break
    return output


def _extract_roles(lines: Sequence[str]) -> List[str]:
    roles = []
    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in _ROLE_KEYWORDS):
            roles.append(line.strip())
    return _unique(roles, limit=3)


def _infer_source_type(chunks: Sequence[Any]) -> str:
    for chunk in chunks:
        meta = _chunk_meta(chunk)
        value = (meta.get("doc_type") or meta.get("document_type") or meta.get("source_type") or "").lower()
        if value in {"resume", "linkedin"}:
            return value
    for chunk in chunks:
        text = _chunk_text(chunk).lower()
        if "linkedin" in text:
            return "linkedin"
        if "resume" in text or "curriculum vitae" in text:
            return "resume"
    return "unknown"


def extract_candidate_profile(
    *,
    chunks: Sequence[Any],
    source_document: str,
) -> CandidateProfile:
    lines = _lines_from_chunks(chunks)
    candidate_name = _extract_name(lines)
    total_years = _extract_years(lines)
    if total_years is None:
        total_years = _estimate_years_from_ranges(lines)

    skill_lines = _extract_list_items(lines, cue_words=["skills", "technologies", "tools", "expertise"])
    technical, functional, unknown = _classify_skills(skill_lines)
    technical = _unique(technical)
    functional = _unique(functional)
    if unknown:
        functional.extend([f"Other: {item}" for item in _unique(unknown, limit=6)])

    certifications = _unique(_extract_list_items(lines, cue_words=["certification", "certified", "certificate"]))
    education = _unique(_extract_list_items(lines, cue_words=["education", "university", "college", "degree", "bachelor", "master", "phd"]))
    achievements = _unique(_extract_list_items(lines, cue_words=["award", "achievement", "honor", "recognition"]))

    roles = _extract_roles(lines)
    if roles:
        experience_summary = (
            f"Roles referenced include {roles[0]}."
            + (f" Additional roles mentioned: {', '.join(roles[1:])}." if len(roles) > 1 else "")
            + " Summary is based on the retrieved sections."
        )
    else:
        experience_summary = "The retrieved sections reference experience details but do not list clear role titles."

    evidence_ids: List[str] = []
    for chunk in chunks:
        text = _chunk_text(chunk)
        if not text:
            continue
        if candidate_name and candidate_name in text:
            evidence_ids.append(_chunk_id(chunk))
        if total_years is not None and re.search(r"\byears?\b", text, re.IGNORECASE):
            evidence_ids.append(_chunk_id(chunk))
        if any(skill.lower() in text.lower() for skill in technical[:5]):
            evidence_ids.append(_chunk_id(chunk))
        if education and any(term.lower() in text.lower() for term in education[:3]):
            evidence_ids.append(_chunk_id(chunk))

    return CandidateProfile(
        candidate_name=candidate_name,
        total_years_experience=total_years,
        experience_summary=experience_summary,
        technical_skills=technical,
        functional_skills=functional,
        certifications=certifications,
        education=education,
        achievements_awards=achievements,
        source_type=_infer_source_type(chunks),
        source_document=source_document,
        evidence_chunk_ids=_unique([e for e in evidence_ids if e], limit=6),
    )


__all__ = ["CandidateProfile", "extract_candidate_profile"]
