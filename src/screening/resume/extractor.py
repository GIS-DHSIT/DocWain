from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .models import CertificationItem, EducationItem, ExperienceItem, ResumeProfile


CERT_HEADERS = {"certifications", "licenses", "certs", "certification"}
EDU_KEYWORDS = {"university", "college", "school", "institute", "academy"}
URL_RE = re.compile(r"https?://\S+")
CRED_ID_RE = re.compile(
    r"\b(?:credential(?:\s*id)?|cert(?:ification)?(?:\s*id)?|license|id|number|no\.?)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\-]+)",
    re.IGNORECASE,
)
DATE_TOKEN_RE = re.compile(
    r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+)?(19|20)\d{2}", re.IGNORECASE
)
DATE_RANGE_RE = re.compile(
    r"(?P<start>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+)?(?P<start_year>(19|20)\d{2})"
    r"\s*[-–]\s*"
    r"(?P<end>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+)?(?P<end_year>(19|20)\d{2}|present|current)",
    re.IGNORECASE,
)


def _sectionize(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = "body"
    sections[current] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = line.rstrip(":").lower()
        if len(line) < 60 and header.replace(" ", "_") in {
            "experience",
            "work_experience",
            "employment",
            "education",
            "summary",
            "objective",
            "skills",
            "certifications",
            "certification",
            "certs",
            "licenses",
        }:
            current = header.replace(" ", "_")
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _extract_links(text: str) -> List[str]:
    links = URL_RE.findall(text)
    normalized: List[str] = []
    for link in links:
        cleaned = link.rstrip(").,;")
        normalized.append(cleaned)
    return list(dict.fromkeys(normalized))


def _parse_cert_line(line: str) -> CertificationItem:
    url_match = URL_RE.search(line)
    cred_match = CRED_ID_RE.search(line)
    date_tokens = DATE_TOKEN_RE.findall(line)

    issue_date = None
    expiry_date = None
    if date_tokens:
        issue_date = " ".join(date_tokens[0]).strip()
        if len(date_tokens) > 1:
            expiry_date = " ".join(date_tokens[1]).strip()

    name_part = line
    issuer = None
    if "-" in line:
        first, _, rest = line.partition("-")
        name_part = first.strip()
        issuer = rest.strip() or None
    elif "|" in line:
        first, _, rest = line.partition("|")
        name_part = first.strip()
        issuer = rest.strip() or None
    elif "(" in line and ")" in line:
        inside = line[line.find("(") + 1 : line.find(")")]
        if len(inside.split()) > 1:
            issuer = inside.strip()
            name_part = line[: line.find("(")].strip()

    credential_id = cred_match.group(1) if cred_match else None
    verification_url = url_match.group(0) if url_match else None

    return CertificationItem(
        name=name_part or line,
        issuer=issuer,
        credential_id=credential_id,
        issue_date=issue_date,
        expiry_date=expiry_date,
        verification_url=verification_url,
    )


def _normalize_certifications(raw_items: Iterable[CertificationItem]) -> List[CertificationItem]:
    normalized: List[CertificationItem] = []
    seen = set()
    for item in raw_items:
        key = f"{(item.name or '').lower()}::{(item.issuer or '').lower()}"
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _extract_certifications(section_lines: List[str], fallback_lines: List[str]) -> List[CertificationItem]:
    candidates: List[str] = []
    for line in section_lines:
        if line and len(line) > 2:
            candidates.append(line)

    if not candidates:
        for line in fallback_lines:
            lower = line.lower()
            if "cert" in lower or "license" in lower:
                candidates.append(line)

    parsed: List[CertificationItem] = []
    for line in candidates:
        if line.lower().startswith("credential"):
            extra = _parse_cert_line(line)
            if parsed:
                target = parsed[-1]
                if extra.credential_id:
                    target.credential_id = extra.credential_id
                if extra.verification_url and not target.verification_url:
                    target.verification_url = extra.verification_url
                if extra.issue_date and not target.issue_date:
                    target.issue_date = extra.issue_date
                if extra.expiry_date and not target.expiry_date:
                    target.expiry_date = extra.expiry_date
                continue
        parsed.append(_parse_cert_line(line))
    return _normalize_certifications(parsed)


def _extract_experience(lines: List[str]) -> List[ExperienceItem]:
    experiences: List[ExperienceItem] = []
    for line in lines:
        range_match = DATE_RANGE_RE.search(line)
        start_date = end_date = None
        if range_match:
            start_prefix = (range_match.group("start") or "").strip()
            end_prefix = (range_match.group("end") or "").strip()
            start_date = f"{start_prefix} {range_match.group('start_year')}".strip()
            end_year = range_match.group("end_year")
            end_date = f"{end_prefix} {end_year}".strip() if end_year.lower() not in {"present", "current"} else "Present"

        title = None
        company = None
        if " at " in line:
            title, _, company = line.partition(" at ")
        elif "-" in line and not title:
            first, _, rest = line.partition("-")
            title = first.strip()
            company = rest.strip()

        experiences.append(
            ExperienceItem(
                title=title or None,
                company=company or None,
                start_date=start_date,
                end_date=end_date,
                description=line,
            )
        )
    return experiences


def _extract_education(lines: List[str]) -> List[EducationItem]:
    education: List[EducationItem] = []
    for line in lines:
        lower = line.lower()
        if not any(token in lower for token in EDU_KEYWORDS):
            continue
        start_year = end_year = None
        date_tokens = DATE_TOKEN_RE.findall(line)
        if date_tokens:
            start_year = " ".join(date_tokens[0]).strip()
            if len(date_tokens) > 1:
                end_year = " ".join(date_tokens[1]).strip()
        degree = None
        institution = None
        if "-" in line:
            institution, _, degree = line.partition("-")
        education.append(
            EducationItem(
                institution=institution or line,
                degree=degree,
                start_year=start_year,
                end_year=end_year,
            )
        )
    return education


def _extract_skills(lines: List[str]) -> List[str]:
    skills: List[str] = []
    for line in lines:
        if ":" in line:
            _, _, remainder = line.partition(":")
            parts = remainder.replace("•", "").split(",")
        else:
            parts = line.replace("•", "").split(",")
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                skills.append(cleaned)
    return list(dict.fromkeys(skills))


class ResumeExtractor:
    """Deterministic resume parser with optional LLM hook."""

    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm

    def extract(self, text: str) -> ResumeProfile:
        sections = _sectionize(text)
        summary_lines = sections.get("summary", []) or sections.get("objective", [])
        experience_lines = sections.get("experience", []) or sections.get("work_experience", []) or []
        education_lines = sections.get("education", []) or []
        skills_lines = sections.get("skills", []) or []
        cert_lines = []
        for header in CERT_HEADERS:
            cert_lines.extend(sections.get(header, []))

        profile = ResumeProfile(
            summary=" ".join(summary_lines[:3]).strip(),
            experience=_extract_experience(experience_lines),
            education=_extract_education(education_lines),
            skills=_extract_skills(skills_lines),
            certifications=_extract_certifications(cert_lines, list(text.splitlines())),
            links=_extract_links(text),
        )

        if self.use_llm:
            llm_profile = self._llm_extract(text)
            if llm_profile:
                profile = llm_profile

        return profile

    def _llm_extract(self, text: str) -> Optional[ResumeProfile]:
        # Placeholder for future structured LLM extraction. Deterministic parsing remains the default.
        return None
