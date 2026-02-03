from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


def _ensure_period(text: str) -> str:
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else text + "."


def _list_or_missing(items: Sequence[str]) -> str:
    if not items:
        return "Not Mentioned"
    return ", ".join(items)


def _sentence_count(text: str) -> int:
    return len([part for part in (text or "").replace("!", ".").replace("?", ".").split(".") if part.strip()])


def _doc_list(doc_names: Sequence[str]) -> str:
    names = [d for d in doc_names if d]
    if not names:
        return "the retrieved resume sections"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:2])}, and others"


@dataclass(frozen=True)
class ResumeProfileView:
    candidate_name: str
    source_document: str
    source_type: str
    total_years_experience: Optional[int]
    experience_confidence: str
    experience_basis: str
    experience_details: Optional[str]
    experience_summary: str
    technical_skills: List[str]
    functional_skills: List[str]
    certifications: List[str]
    education: List[str]
    awards: List[str]


def _experience_line(profile: ResumeProfileView) -> str:
    if profile.experience_basis == "conflicting" and profile.experience_details:
        return profile.experience_details
    if profile.total_years_experience is None:
        return "Not Mentioned"
    basis = "explicit statement" if profile.experience_basis == "explicit_years" else "date ranges"
    return f"{profile.total_years_experience} years ({basis}, confidence: {profile.experience_confidence})."


def format_resume_response(
    *,
    profiles: Sequence[ResumeProfileView],
    assumption_line: Optional[str] = None,
    wants_table: bool = False,
    ranking_lines: Optional[Sequence[str]] = None,
) -> str:
    doc_names = [p.source_document for p in profiles]
    intro: List[str] = []
    if assumption_line:
        intro.append(_ensure_period(assumption_line))
    intro.append(
        _ensure_period(
            f"I analyzed {len(profiles)} resume document(s) and built structured summaries for each candidate."
        )
    )
    intro.append(_ensure_period(f"Documents used: {_doc_list(doc_names)}"))
    intro_block = " ".join(intro[:3])

    blocks: List[str] = [intro_block]

    if ranking_lines:
        blocks.append("Ranking:\n" + "\n".join(ranking_lines))

    if len(profiles) == 1:
        profile = profiles[0]
        summary_text = profile.experience_summary or "Not Mentioned"
        details = [
            f"Name: {profile.candidate_name}.",
            f"Experience Summary: {summary_text}.",
            f"Total Experience: {_experience_line(profile)}",
            f"Technical Skills: {_list_or_missing(profile.technical_skills)}.",
            f"Functional Skills: {_list_or_missing(profile.functional_skills)}.",
            f"Certifications: {_list_or_missing(profile.certifications)}.",
            f"Education: {_list_or_missing(profile.education)}.",
            f"Achievements/Awards: {_list_or_missing(profile.awards)}.",
            f"Source: {profile.source_type}.",
        ]
        blocks.append(f"Candidate (Source: {profile.source_document})\n" + "\n".join(details))
    else:
        if wants_table:
            header = "| Candidate | Experience | Technical Skills | Certifications | Document |"
            separator = "| --- | --- | --- | --- | --- |"
            rows = []
            for profile in profiles:
                rows.append(
                    "| {name} | {exp} | {tech} | {certs} | {doc} |".format(
                        name=profile.candidate_name,
                        exp=_experience_line(profile),
                        tech=_list_or_missing(profile.technical_skills),
                        certs=_list_or_missing(profile.certifications),
                        doc=profile.source_document,
                    )
                )
            blocks.append("\n".join([header, separator] + rows))
        else:
            for profile in profiles:
                summary_text = profile.experience_summary or "Not Mentioned"
                details = [
                    f"Name: {profile.candidate_name}.",
                    f"Experience Summary: {summary_text}.",
                    f"Total Experience: {_experience_line(profile)}",
                    f"Technical Skills: {_list_or_missing(profile.technical_skills)}.",
                    f"Functional Skills: {_list_or_missing(profile.functional_skills)}.",
                    f"Certifications: {_list_or_missing(profile.certifications)}.",
                    f"Education: {_list_or_missing(profile.education)}.",
                    f"Achievements/Awards: {_list_or_missing(profile.awards)}.",
                    f"Source: {profile.source_type}.",
                ]
                blocks.append(
                    f"Candidate (Source: {profile.source_document})\n" + "\n".join(details)
                )

    takeaways: List[str] = []
    missing_experience = any(p.total_years_experience is None for p in profiles)
    if missing_experience:
        takeaways.append("Some candidates do not state total experience explicitly; confirm during screening.")
    if len(profiles) > 1:
        takeaways.append("Compare skills and certifications across candidates for final shortlisting.")

    if takeaways:
        blocks.append("Takeaways:\n" + "\n".join(f"- {_ensure_period(t)}" for t in takeaways[:3]))

    assembled = "\n\n".join(blocks).strip()
    if _sentence_count(assembled) < 5:
        assembled = assembled + " " + _ensure_period("Let me know if you want a deeper breakdown by section")
    return assembled


__all__ = ["ResumeProfileView", "format_resume_response"]
