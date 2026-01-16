from __future__ import annotations

from typing import List

from .models import ResumeScreeningDetailedResponse


def _format_list(items: List[str], label: str) -> List[str]:
    if not items:
        return [f"- {label}: None noted"]
    return [f"- {label}: {item}" for item in items]


def build_narrative_report(result: ResumeScreeningDetailedResponse) -> str:
    lines: List[str] = []
    profile = result.candidate_profile
    lines.append(f"Candidate Summary: {profile.summary or 'Not provided.'}")

    lines.append("\nExperience:")
    if profile.extracted.experience:
        for exp in profile.extracted.experience:
            role = exp.title or "Role not specified"
            company = exp.company or "Company not specified"
            dates = " - ".join(filter(None, [exp.start_date, exp.end_date])) or "Dates not provided"
            lines.append(f"- {role} at {company} ({dates})")
    else:
        lines.append("- Not provided")

    lines.append("\nEducation:")
    if profile.extracted.education:
        for edu in profile.extracted.education:
            details = edu.degree or "Program not specified"
            institution = edu.institution or "Institution not specified"
            years = " - ".join(filter(None, [edu.start_year, edu.end_year])) or "Years not provided"
            lines.append(f"- {institution}: {details} ({years})")
    else:
        lines.append("- Not provided")

    lines.append("\nCertifications:")
    if result.validations.certifications:
        for cert in result.validations.certifications:
            name = cert.certification.name
            issuer = cert.certification.issuer or "Issuer not provided"
            status = cert.status
            confidence = f"{cert.confidence_0_100:.0f}%"
            lines.append(f"- {name} ({issuer}) — {status} ({confidence})")
    elif profile.extracted.certifications:
        for cert in profile.extracted.certifications:
            lines.append(f"- {cert.name} ({cert.issuer or 'Issuer not provided'}) — not validated")
    else:
        lines.append("- None listed")

    lines.append("\nValidation Evidence:")
    top_sources = []
    for bundle in result.validations.companies[:2] + result.validations.institutions[:2]:
        if bundle.sources:
            src = bundle.sources[0]
            top_sources.append(f"- {bundle.type.title()}: {bundle.name} -> {src.title or src.url}")
    if top_sources:
        lines.extend(top_sources)
    else:
        lines.append("- Not collected (internet disabled or no data)")

    lines.append("\nAuthenticity Signals:")
    if result.authenticity.signals:
        for sig in result.authenticity.signals:
            lines.append(f"- ({sig.severity}) {sig.message}")
    else:
        lines.append("- None detected")

    lines.append(
        f"\nOverall Risk: {result.risk_level.upper()} | Overall Confidence: {result.overall_confidence_0_100:.0f}%"
    )

    lines.append("\nRecommendations:")
    recs = result.recommendations.next_steps or ["No immediate follow-up recommended."]
    for rec in recs:
        lines.append(f"- {rec}")
    if result.recommendations.documents_to_request:
        lines.append("- Documents to request: " + ", ".join(result.recommendations.documents_to_request))

    return "\n".join(lines)

