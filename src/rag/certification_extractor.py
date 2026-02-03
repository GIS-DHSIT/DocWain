from __future__ import annotations

import re
from typing import Iterable, List


_CERT_KEYWORDS = ("certified", "certification", "certificate", "credential")
_STOPWORDS = {"contact", "skills", "education", "summary", "certifications"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9+.#-]+")
_TOKEN_STOP = {
    "education",
    "certification",
    "certifications",
    "certificate",
    "certified",
    "processing",
    "integration",
    "package",
    "project",
    "development",
    "responsibilities",
    "with",
    "and",
    "to",
    "of",
    "in",
    "for",
    "the",
    "a",
}

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{6,}\b")
_URL_RE = re.compile(r"https?://|www\.")


def _clean_item(item: str) -> str:
    return re.sub(r"\s+", " ", item or "").strip(" \t-:;")


def _valid_cert(item: str) -> bool:
    if not item:
        return False
    cleaned = _clean_item(item)
    if len(cleaned) < 4 or len(cleaned) > 80:
        return False
    if cleaned.lower() in _STOPWORDS:
        return False
    if _EMAIL_RE.search(cleaned) or _PHONE_RE.search(cleaned) or _URL_RE.search(cleaned):
        return False
    if not re.search(r"[A-Za-z]", cleaned):
        return False
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(cleaned)]
    if not tokens:
        return False
    if len(tokens) > 12:
        return False
    if all(token in _TOKEN_STOP for token in tokens):
        return False
    return True


def _split_candidates(line: str) -> List[str]:
    if not line:
        return []
    line = re.sub(
        r"^\s*(?:[A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{4})\s*[-–—]\s*(?:present|current|[A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{4})\s+",
        "",
        line,
        flags=re.IGNORECASE,
    )
    if line.strip().startswith(("-", "*", "•", "\u2022")):
        return [_clean_item(line.strip("-*•\u2022 "))]
    if "," in line and len(line.split(",")) >= 2:
        return [_clean_item(part) for part in line.split(",")]
    if ";" in line:
        return [_clean_item(part) for part in line.split(";")]
    if "|" in line:
        return [_clean_item(part) for part in line.split("|")]
    if ":" in line:
        left, right = line.split(":", 1)
        if left.strip().lower() in {"certifications", "certification", "certificates"}:
            return [_clean_item(part) for part in re.split(r"[,;|]", right)]
    return [_clean_item(line)]


def _extract_from_lines(lines: Iterable[str]) -> List[str]:
    items: List[str] = []
    for line in lines:
        for candidate in _split_candidates(line):
            if _valid_cert(candidate):
                items.append(candidate)
    return items


def extract_certifications(cert_section: str, *, fallback_text: str | None = None) -> List[str]:
    candidates: List[str] = []
    if cert_section:
        candidates.extend(_extract_from_lines(cert_section.splitlines()))

    if fallback_text:
        for line in fallback_text.splitlines():
            lowered = line.lower()
            if any(keyword in lowered for keyword in _CERT_KEYWORDS) and len(line.split()) <= 10:
                candidates.extend(_extract_from_lines([line]))

    cleaned: List[str] = []
    seen = set()
    for item in candidates:
        value = _clean_item(item)
        key = value.lower()
        if not value or key in seen:
            continue
        if key in _STOPWORDS:
            continue
        seen.add(key)
        cleaned.append(value)

    return cleaned


__all__ = ["extract_certifications"]
