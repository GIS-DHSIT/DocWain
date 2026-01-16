from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional, Tuple

from .models import AuthenticityReport, AuthenticitySignal, CertificationItem, ExperienceItem, ResumeProfile

DATE_EXTRACT_RE = re.compile(r"(19|20)\d{2}")
STYLE_SPLIT_RE = re.compile(r"[.!?]\s+")
BUZZWORDS = {
    "synergy",
    "innovative",
    "dynamic",
    "results-driven",
    "thought leader",
    "proven track record",
    "strategic thinker",
    "leveraging",
}


def _year_from_text(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = DATE_EXTRACT_RE.search(value)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


def _overlaps(ranges: List[Tuple[int, int]]) -> bool:
    ranges.sort(key=lambda r: r[0])
    for idx in range(1, len(ranges)):
        prev_start, prev_end = ranges[idx - 1]
        start, end = ranges[idx]
        if start < prev_end:
            return True
    return False


class AuthenticityAnalyzer:
    """Privacy-safe authenticity heuristics for resumes."""

    def __init__(self, ai_authorship_hint: Optional[float] = None) -> None:
        self.ai_authorship_hint = ai_authorship_hint

    def _timeline_signals(self, experience: Iterable[ExperienceItem]) -> List[AuthenticitySignal]:
        signals: List[AuthenticitySignal] = []
        ranges: List[Tuple[int, int]] = []
        for exp in experience:
            start = _year_from_text(exp.start_date)
            end = _year_from_text(exp.end_date) or start
            if start and end:
                ranges.append((start, end if end != -1 else start))
                if start > end:
                    signals.append(
                        AuthenticitySignal(
                            type="implausible_dates",
                            severity="high",
                            message=f"Experience at {exp.company or 'unknown company'} has end date before start date.",
                            evidence={"start_date": exp.start_date, "end_date": exp.end_date},
                        )
                    )
        if len(ranges) >= 2 and _overlaps(ranges):
            signals.append(
                AuthenticitySignal(
                    type="timeline_overlap",
                    severity="medium",
                    message="Experience date ranges overlap in a way that may be implausible.",
                    evidence={"ranges": ranges},
                )
            )
        return signals

    def _cert_signals(self, certs: Iterable[CertificationItem]) -> List[AuthenticitySignal]:
        signals: List[AuthenticitySignal] = []
        for cert in certs:
            if not cert.issuer:
                signals.append(
                    AuthenticitySignal(
                        type="missing_cert_issuer",
                        severity="low",
                        message=f"Certification '{cert.name}' is missing issuer information.",
                        evidence=None,
                    )
                )
            if not cert.issue_date and not cert.expiry_date:
                signals.append(
                    AuthenticitySignal(
                        type="missing_cert_dates",
                        severity="low",
                        message=f"Certification '{cert.name}' has no issue/expiry dates.",
                        evidence=None,
                    )
                )
        return signals

    def _content_integrity(self, profile: ResumeProfile, full_text: str) -> List[AuthenticitySignal]:
        signals: List[AuthenticitySignal] = []
        sentences = STYLE_SPLIT_RE.split(full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences if s]
            if lengths:
                span = max(lengths) - min(lengths)
                if span > 40:
                    signals.append(
                        AuthenticitySignal(
                            type="style_shift",
                            severity="medium",
                            message="Abrupt writing style shifts detected across sections.",
                            evidence={"sentence_lengths": lengths[:10]},
                        )
                    )

        words = re.findall(r"[A-Za-z']+", full_text.lower())
        if words:
            freq = Counter(words)
            most_common = freq.most_common(1)[0][1]
            if most_common / max(len(words), 1) > 0.08:
                signals.append(
                    AuthenticitySignal(
                        type="repetitive_language",
                        severity="low",
                        message="Text contains repetitive or templated language.",
                        evidence={"top_term": freq.most_common(3)},
                    )
                )
            buzz_count = sum(full_text.lower().count(term) for term in BUZZWORDS)
            if buzz_count / max(len(words), 1) > 0.06:
                signals.append(
                    AuthenticitySignal(
                        type="buzzword_density",
                        severity="low",
                        message="High density of buzzwords relative to substantive content.",
                        evidence={"buzzword_hits": buzz_count},
                    )
                )

        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        line_freq = Counter(lines)
        repeated_lines = [line for line, count in line_freq.items() if count >= 3 and len(line) > 20]
        if repeated_lines:
            signals.append(
                AuthenticitySignal(
                    type="templated_repetition",
                    severity="medium",
                    message="Repeated templated lines detected across the document.",
                    evidence={"lines": repeated_lines[:3]},
                )
            )

        if self.ai_authorship_hint and self.ai_authorship_hint >= 0.75:
            signals.append(
                AuthenticitySignal(
                    type="ai_authorship_likelihood",
                    severity="medium",
                    message="High likelihood of machine-assisted writing.",
                    evidence={"confidence": self.ai_authorship_hint},
                )
            )
        return signals

    def _consistency(self, profile: ResumeProfile, full_text: str) -> List[AuthenticitySignal]:
        signals: List[AuthenticitySignal] = []
        all_experience_text = " ".join([exp.description or "" for exp in profile.experience]).lower()
        unused_skills = [skill for skill in profile.skills if skill.lower() not in all_experience_text]
        if unused_skills and profile.experience:
            signals.append(
                AuthenticitySignal(
                    type="skill_alignment",
                    severity="low",
                    message="Some skills are listed but not demonstrated in experience.",
                    evidence={"skills": unused_skills[:8]},
                )
            )

        if profile.certifications and all_experience_text:
            misaligned = []
            for cert in profile.certifications:
                cert_tokens = (cert.name or "").lower().split()
                if cert_tokens and not any(token in all_experience_text for token in cert_tokens[:3]):
                    misaligned.append(cert.name)
            if misaligned:
                signals.append(
                    AuthenticitySignal(
                        type="cert_role_alignment",
                        severity="low",
                        message="Certifications do not clearly align with listed roles.",
                        evidence={"certifications": misaligned[:5]},
                    )
                )

        return signals

    def analyze(self, profile: ResumeProfile, *, full_text: str) -> AuthenticityReport:
        signals: List[AuthenticitySignal] = []
        signals.extend(self._timeline_signals(profile.experience))
        signals.extend(self._cert_signals(profile.certifications))
        signals.extend(self._content_integrity(profile, full_text))
        signals.extend(self._consistency(profile, full_text))

        severity_score = 0
        for sig in signals:
            if sig.severity == "high":
                severity_score += 3
            elif sig.severity == "medium":
                severity_score += 2
            else:
                severity_score += 1

        risk_level = "low"
        if severity_score >= 6:
            risk_level = "high"
        elif severity_score >= 3:
            risk_level = "medium"

        confidence = min(100.0, 60.0 + 5.0 * len(signals))

        recommendations: List[str] = []
        if any(sig.type in {"timeline_overlap", "implausible_dates"} for sig in signals):
            recommendations.append("Request clarification on employment dates and overlapping roles.")
        if any(sig.type.startswith("missing_cert") for sig in signals):
            recommendations.append("Ask for certification issuer details or proof where missing.")

        return AuthenticityReport(
            signals=signals,
            risk_level=risk_level,
            confidence_0_100=confidence,
            recommendations=list(dict.fromkeys(recommendations)),
        )
