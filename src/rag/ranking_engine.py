from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from src.rag.candidate_profile_extractor import CandidateProfile


_SKILL_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9+._-]{2,}")


def _extract_query_skills(query: str) -> List[str]:
    if not query:
        return []
    tokens = _SKILL_TOKEN.findall(query.lower())
    stop = {"rank", "candidate", "candidates", "experience", "skills", "years", "top", "best"}
    skills = [t for t in tokens if t not in stop]
    return list(dict.fromkeys(skills))


@dataclass(frozen=True)
class RankedCandidate:
    profile: CandidateProfile
    score: float
    rationale: str


def _skill_coverage(profile: CandidateProfile, query_skills: Iterable[str]) -> float:
    all_skills = " ".join(profile.technical_skills + profile.functional_skills).lower()
    if not query_skills:
        return min(len(profile.technical_skills) / 8.0, 1.0)
    hits = sum(1 for skill in query_skills if skill in all_skills)
    return hits / max(len(list(query_skills)), 1)


def rank_candidates(profiles: List[CandidateProfile], query: str) -> List[RankedCandidate]:
    query_skills = _extract_query_skills(query)
    ranked: List[RankedCandidate] = []
    for profile in profiles:
        exp_years = profile.total_years_experience or 0.0
        exp_score = min(exp_years / 10.0, 1.0)
        skill_score = _skill_coverage(profile, query_skills)
        role_score = 0.2 if "lead" in profile.experience_summary.lower() else 0.0
        score = 0.5 * exp_score + 0.4 * skill_score + 0.1 * role_score
        rationale = (
            f"Experience score {exp_score:.2f}, skill match {skill_score:.2f}, role signal {role_score:.2f}."
        )
        ranked.append(RankedCandidate(profile=profile, score=round(score, 3), rationale=rationale))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


__all__ = ["RankedCandidate", "rank_candidates"]
