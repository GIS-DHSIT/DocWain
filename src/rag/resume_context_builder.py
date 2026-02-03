from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.rag.section_classifier import classify_section


_CONTACT_RE = re.compile(r"\b(email|e-mail|phone|tel|mobile|fax|address|street|zip|postal)\b", re.IGNORECASE)
_PERSONAL_RE = re.compile(r"\b(personal data|place|date|marital status|dob|date of birth)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{6,}\b")
_URL_RE = re.compile(r"https?://|www\.")

_SECTION_LABELS = {"summary", "experience", "work experience", "skills", "certifications", "education", "awards", "projects"}


@dataclass(frozen=True)
class ResumePacket:
    sections: Dict[str, str]
    used_chunk_ids: List[str]


def _chunk_text(chunk: Any) -> str:
    return getattr(chunk, "text", "") or ""


def _chunk_meta(chunk: Any) -> Dict[str, Any]:
    return getattr(chunk, "metadata", {}) or {}


def _chunk_id(chunk: Any) -> str:
    meta = _chunk_meta(chunk)
    return str(meta.get("chunk_id") or getattr(chunk, "id", "") or "")


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip().lower())


def _is_noise_line(line: str) -> bool:
    if not line:
        return True
    stripped = line.strip()
    lowered = stripped.lower()
    if _EMAIL_RE.search(stripped) or _PHONE_RE.search(stripped) or _URL_RE.search(stripped):
        return True
    if _CONTACT_RE.search(lowered) and len(stripped.split()) < 8:
        return True
    if _PERSONAL_RE.search(lowered):
        return True
    if len(stripped.split()) == 1 and lowered in _SECTION_LABELS:
        return True
    return False


def _dedupe_lines(lines: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for line in lines:
        key = _normalize_line(line)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(line)
    return output


def _dedupe_ids(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def build_resume_packet(chunks: Sequence[Any]) -> ResumePacket:
    sections: Dict[str, List[str]] = {
        "summary": [],
        "experience": [],
        "skills": [],
        "certifications": [],
        "education": [],
        "awards": [],
    }
    used_chunk_ids: List[str] = []

    line_counts: Dict[str, int] = {}
    for chunk in chunks:
        for raw in _chunk_text(chunk).splitlines():
            normalized = _normalize_line(raw)
            if not normalized:
                continue
            line_counts[normalized] = line_counts.get(normalized, 0) + 1

    for chunk in chunks:
        text = _chunk_text(chunk)
        if not text:
            continue
        meta = _chunk_meta(chunk)
        section_title = meta.get("section_title") or meta.get("section_path") or ""
        section_key = classify_section(section_title, text)
        chunk_lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            normalized = _normalize_line(line)
            if line_counts.get(normalized, 0) >= 3 and len(normalized) < 60:
                continue
            if _is_noise_line(line):
                continue
            chunk_lines.append(line)

        if not chunk_lines:
            continue

        if section_key and section_key in sections:
            sections[section_key].extend(chunk_lines)
            cid = _chunk_id(chunk)
            if cid:
                used_chunk_ids.append(cid)
            continue

        # Fallback: only attach when explicit heading is present in the first lines.
        heading_window = [ln.strip().lower() for ln in text.splitlines()[:5] if ln.strip()]
        for key in sections.keys():
            header_pattern = re.compile(rf"^{re.escape(key)}\\s*[:\\-]?$", re.IGNORECASE)
            if any(header_pattern.match(line) for line in heading_window):
                sections[key].extend(chunk_lines)
                cid = _chunk_id(chunk)
                if cid:
                    used_chunk_ids.append(cid)
                break

    cleaned_sections = {key: "\n".join(_dedupe_lines(lines)) for key, lines in sections.items()}
    used_chunk_ids = _dedupe_ids([cid for cid in used_chunk_ids if cid])

    return ResumePacket(sections=cleaned_sections, used_chunk_ids=used_chunk_ids)


__all__ = ["ResumePacket", "build_resume_packet"]
