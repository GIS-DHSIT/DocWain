from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional, Tuple


_EXPLICIT_YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(years|yrs)\s*(of)?\s*(experience|exp)", re.IGNORECASE)

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DASH = r"[-\u2013\u2014]"
_DATE_RANGE_RE = re.compile(
    rf"(?P<start_month>[A-Za-z]{{3,9}})\s+(?P<start_year>\d{{4}})\s*{_DASH}\s*"
    rf"(?P<end_month>Present|Current|[A-Za-z]{{3,9}})\s*(?P<end_year>\d{{4}})?",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(rf"(?P<start_year>\d{{4}})\s*{_DASH}\s*(?P<end_year>\d{{4}}|Present|Current)", re.IGNORECASE)
_MONTH_YEAR_RANGE_RE = re.compile(
    rf"(?P<start_month>\d{{1,2}})/(?P<start_year>\d{{4}})\s*{_DASH}\s*"
    rf"(?P<end_month>\d{{1,2}})/(?P<end_year>\d{{4}}|Present|Current)",
    re.IGNORECASE,
)

_EXCLUDE_LINE_RE = re.compile(r"\b(dob|birth|born)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExperienceResult:
    total_years_experience: Optional[int]
    experience_confidence: str
    experience_basis: str
    details: Optional[str] = None


def _parse_month(value: str) -> Optional[int]:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered.isdigit():
        month_val = int(lowered)
        if 1 <= month_val <= 12:
            return month_val
        return None
    return _MONTHS.get(lowered)


def _to_month_index(year: int, month: int) -> int:
    return year * 12 + month


def _range_months(start: date, end: date) -> int:
    return max(0, _to_month_index(end.year, end.month) - _to_month_index(start.year, start.month) + 1)


def _normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def extract_explicit_years(text: str) -> List[int]:
    years: List[int] = []
    for match in _EXPLICIT_YEARS_RE.finditer(text or ""):
        try:
            years.append(int(match.group(1)))
        except Exception:
            continue
    return years


def _parse_date_range(match: re.Match, *, as_of: date) -> Optional[Tuple[date, date]]:
    start_year = match.group("start_year")
    if not start_year:
        return None
    try:
        start_year_val = int(start_year)
    except Exception:
        return None

    start_month_val = 1
    if match.groupdict().get("start_month"):
        start_month_val = _parse_month(match.group("start_month")) or 1

    end_year_val: Optional[int] = None
    end_month_val: Optional[int] = None

    end_year_raw = match.groupdict().get("end_year")
    end_month_raw = match.groupdict().get("end_month")

    if end_month_raw and end_month_raw.lower() in {"present", "current"}:
        end_year_val = as_of.year
        end_month_val = as_of.month
    elif end_year_raw and end_year_raw.lower() in {"present", "current"}:
        end_year_val = as_of.year
        end_month_val = as_of.month
    else:
        if end_year_raw:
            try:
                end_year_val = int(end_year_raw)
            except Exception:
                end_year_val = None
        elif end_month_raw and end_month_raw.isdigit():
            end_year_val = None
        if end_month_raw and end_month_raw.isdigit():
            end_month_val = _parse_month(end_month_raw) or 12
        elif end_month_raw:
            end_month_val = _parse_month(end_month_raw) or 12

    if end_year_val is None:
        return None
    if end_month_val is None:
        end_month_val = 12

    start = date(start_year_val, start_month_val, 1)
    end = date(end_year_val, end_month_val, 1)
    if end <= start:
        return None
    return (start, end)


def extract_date_ranges(text: str, *, as_of: Optional[date] = None) -> List[Tuple[date, date]]:
    as_of_date = as_of or datetime.utcnow().date()
    ranges: List[Tuple[date, date]] = []
    for line in _normalize_lines(text):
        if _EXCLUDE_LINE_RE.search(line):
            continue
        for match in _DATE_RANGE_RE.finditer(line):
            parsed = _parse_date_range(match, as_of=as_of_date)
            if parsed:
                ranges.append(parsed)
        for match in _MONTH_YEAR_RANGE_RE.finditer(line):
            parsed = _parse_date_range(match, as_of=as_of_date)
            if parsed:
                ranges.append(parsed)
        for match in _YEAR_RANGE_RE.finditer(line):
            parsed = _parse_date_range(match, as_of=as_of_date)
            if parsed:
                ranges.append(parsed)
    return ranges


def _merge_ranges(ranges: List[Tuple[date, date]]) -> List[Tuple[date, date]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: r[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _total_months(ranges: Iterable[Tuple[date, date]]) -> int:
    total = 0
    for start, end in ranges:
        total += _range_months(start, end)
    return total


def extract_experience(text: str, *, as_of: Optional[date] = None) -> ExperienceResult:
    explicit_years = extract_explicit_years(text)
    if explicit_years:
        unique = sorted(set(explicit_years))
        if len(unique) > 1:
            detail = f"Conflicting in retrieved sections ({', '.join(str(v) for v in unique)} years)."
            return ExperienceResult(
                total_years_experience=None,
                experience_confidence="low",
                experience_basis="conflicting",
                details=detail,
            )
        return ExperienceResult(
            total_years_experience=unique[0],
            experience_confidence="high",
            experience_basis="explicit_years",
        )

    ranges = extract_date_ranges(text, as_of=as_of)
    if not ranges:
        return ExperienceResult(
            total_years_experience=None,
            experience_confidence="low",
            experience_basis="not_stated",
        )

    merged = _merge_ranges(ranges)
    months = _total_months(merged)
    if months <= 0:
        return ExperienceResult(
            total_years_experience=None,
            experience_confidence="low",
            experience_basis="not_stated",
        )

    years = int(months // 12)
    if years <= 0:
        years = 1

    confidence = "low"
    if len(merged) >= 2:
        confidence = "high"
    elif months >= 24:
        confidence = "medium"

    return ExperienceResult(
        total_years_experience=years,
        experience_confidence=confidence,
        experience_basis="computed_date_ranges",
    )


__all__ = ["ExperienceResult", "extract_experience", "extract_explicit_years", "extract_date_ranges"]
