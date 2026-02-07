from __future__ import annotations

from ..types import HRSchema, MISSING_REASON


def render_hr(schema: HRSchema, intent: str, strict: bool = False) -> str:
    candidates = (schema.candidates.items if schema.candidates else None) or []
    if not candidates:
        if strict:
            return ""
        if schema.candidates and schema.candidates.missing_reason:
            return schema.candidates.missing_reason
        return MISSING_REASON
    if intent in {"rank", "compare"} and len(candidates) > 1:
        ranked = _rank_candidates(candidates)
        lines = ["Based on the skills mentioned, here is a ranking of the candidates:"]
        for idx, cand in enumerate(ranked, start=1):
            lines.append(_format_rank_line(idx, cand))
        return "\n".join(lines)
    if len(candidates) > 1:
        sections = []
        for cand in candidates:
            sections.append(_format_candidate_detail(cand))
        return "\n\n".join(sections)

    cand = candidates[0]
    parts = []
    name = cand.name or "Candidate"
    parts.append(f"**Candidate:** {name}")

    parts.append(f"- Total experience: {cand.total_years_experience or MISSING_REASON}")
    summary = cand.experience_summary or MISSING_REASON
    parts.append(f"- Summary: {summary}")

    parts.append(f"- Technical skills: {', '.join(cand.technical_skills or []) or MISSING_REASON}")
    parts.append(f"- Functional skills: {', '.join(cand.functional_skills or []) or MISSING_REASON}")
    parts.append(f"- Certifications: {', '.join(cand.certifications or []) or MISSING_REASON}")
    parts.append(f"- Education: {', '.join(cand.education or []) or MISSING_REASON}")
    parts.append(f"- Achievements/Awards: {', '.join(cand.achievements or []) or MISSING_REASON}")
    parts.append(f"- Source type: {cand.source_type or MISSING_REASON}")

    return "\n".join(parts).strip()


def _rank_candidates(candidates):
    scored = []
    for cand in candidates:
        score = 0.0
        score += 1.5 * len(cand.technical_skills or [])
        score += 1.0 * len(cand.functional_skills or [])
        score += 0.7 * len(cand.certifications or [])
        score += 0.4 * len(cand.achievements or [])
        if cand.total_years_experience:
            years = _parse_years(cand.total_years_experience)
            if years is not None:
                score += min(10.0, years) * 0.2
        scored.append((score, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cand for _, cand in scored]


def _parse_years(value: str):
    try:
        number = float(value.split()[0])
        return number
    except Exception:
        return None


def _format_rank_line(idx: int, cand) -> str:
    name = cand.name or "Candidate"
    tech = ", ".join((cand.technical_skills or [])[:4]) or MISSING_REASON
    func = ", ".join((cand.functional_skills or [])[:4]) or MISSING_REASON
    certs = ", ".join((cand.certifications or [])[:3]) or MISSING_REASON
    summary = cand.experience_summary or MISSING_REASON
    label = "Top pick" if idx == 1 else "Next"
    return (
        f"- {label}: {name} — key technical skills: {tech}; "
        f"functional skills: {func}; certifications: {certs}; summary: {summary}"
    )


def _format_candidate_detail(cand) -> str:
    name = cand.name or "Candidate"
    summary = cand.experience_summary or MISSING_REASON
    tech = ", ".join((cand.technical_skills or [])[:6]) or MISSING_REASON
    func = ", ".join((cand.functional_skills or [])[:6]) or MISSING_REASON
    certs = ", ".join((cand.certifications or [])[:4]) or MISSING_REASON
    edu = ", ".join((cand.education or [])[:3]) or MISSING_REASON
    awards = ", ".join((cand.achievements or [])[:3]) or MISSING_REASON
    years = cand.total_years_experience or MISSING_REASON
    source = cand.source_type or MISSING_REASON
    lines = [
        f"**Candidate:** {name}",
        f"- Total experience: {years}",
        f"- Summary: {summary}",
        f"- Technical skills: {tech}",
        f"- Functional skills: {func}",
        f"- Certifications: {certs}",
        f"- Education: {edu}",
        f"- Achievements/Awards: {awards}",
        f"- Source type: {source}",
    ]
    return "\n".join(lines)
