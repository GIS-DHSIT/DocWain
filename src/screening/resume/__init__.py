from __future__ import annotations

from statistics import mean
from typing import Dict, Optional

from .authenticity import AuthenticityAnalyzer
from .certifications import CertificationVerifier
from .extractor import ResumeExtractor
from .models import (
    ResumeCandidateProfile,
    ResumeFindings,
    ResumeProfile,
    ResumeRecommendations,
    ResumeScreeningDetailedResponse,
    ResumeScoringComponents,
    ResumeValidations,
)
from .report import build_narrative_report
from .validators import CachedSearchClient, OrganizationValidator


def _candidate_summary(profile: ResumeProfile) -> str:
    parts = []
    if profile.summary:
        parts.append(profile.summary)
    if profile.experience:
        first_role = profile.experience[0]
        role = first_role.title or "professional"
        company = first_role.company or ""
        parts.append(f"Recent role: {role}{' at ' + company if company else ''}.")
    if profile.skills:
        parts.append(f"Key skills: {', '.join(profile.skills[:6])}.")
    return " ".join(parts) or "No summary available."


def run_resume_analysis(
    *,
    text: str,
    doc_id: Optional[str],
    metadata: Optional[Dict],
    search_client,
    internet_enabled: bool,
) -> ResumeScreeningDetailedResponse:
    extractor = ResumeExtractor()
    profile = extractor.extract(text)
    metadata = metadata or {}
    candidate_name = metadata.get("candidate_name")
    if candidate_name and not profile.name:
        profile.name = candidate_name
    linkedin_links = [link for link in profile.links if "linkedin.com/in" in link]
    org_search = CachedSearchClient(search_client, internet_enabled=internet_enabled)
    org_validator = OrganizationValidator(org_search)

    company_names = [exp.company for exp in profile.experience if exp.company]
    institution_names = [edu.institution for edu in profile.education if edu.institution]

    companies = org_validator.validate_many(company_names, "company")
    institutions = org_validator.validate_many(institution_names, "institution")

    cert_verifier = CertificationVerifier(org_search)
    certifications = [cert_verifier.verify(cert) for cert in profile.certifications]

    authenticity = AuthenticityAnalyzer().analyze(profile, full_text=text)

    extraction_quality = 0.0
    if profile.summary:
        extraction_quality += 20
    if profile.experience:
        extraction_quality += 30
    if profile.education:
        extraction_quality += 20
    if profile.skills:
        extraction_quality += 15
    if profile.certifications:
        extraction_quality += 15

    org_confidence = mean([bundle.confidence_0_100 for bundle in companies + institutions]) if companies or institutions else 0.0
    cert_confidence = mean([cert.confidence_0_100 for cert in certifications]) if certifications else 0.0

    scoring = ResumeScoringComponents(
        extraction_quality=float(min(extraction_quality, 100.0)),
        org_validation_confidence=float(org_confidence),
        cert_validation_confidence=float(cert_confidence),
        authenticity_confidence=float(authenticity.confidence_0_100),
    )

    overall_confidence = mean(
        [
            scoring.extraction_quality or 0.0,
            scoring.org_validation_confidence,
            scoring.cert_validation_confidence,
            scoring.authenticity_confidence,
        ]
    )

    risk_level = authenticity.risk_level
    if any(cert.status == "suspicious" for cert in certifications):
        risk_level = "high"
    elif any(bundle.status == "uncertain" for bundle in companies + institutions) and risk_level == "low":
        risk_level = "medium"

    strengths = []
    concerns = []
    missing = []
    if profile.experience:
        strengths.append("Relevant experience provided.")
    else:
        missing.append("Work experience details")
    if profile.education:
        strengths.append("Education history listed.")
    else:
        missing.append("Education history")
    if profile.skills:
        strengths.append("Skills enumerated.")
    else:
        missing.append("Skills section")
    if profile.certifications:
        strengths.append("Certifications listed.")
        if any(not cert.issuer for cert in profile.certifications):
            concerns.append("Some certifications missing issuer details.")
    else:
        missing.append("Certifications (if any)")

    for cert in certifications:
        if cert.status in {"unverified", "suspicious"}:
            concerns.append(f"Certification {cert.certification.name} is {cert.status}.")

    findings = ResumeFindings(
        strengths=list(dict.fromkeys(strengths)),
        concerns=list(dict.fromkeys(concerns)),
        missing_information=list(dict.fromkeys(missing)),
    )

    rec_candidates = [
        rec for rec in authenticity.recommendations if rec
    ] + ["Request proof of certifications with issuer/credential links."]
    if companies:
        rec_candidates.append("Ask for references or verification for recent roles.")
    recommendations = ResumeRecommendations(
        next_steps=list(dict.fromkeys(rec_candidates)),
        documents_to_request=["Certification PDFs", "Transcripts"] if profile.certifications or profile.education else [],
    )

    warnings = []
    if not internet_enabled:
        warnings.append("Internet validation disabled; organization and certification verification limited.")
    if org_search.stats["skipped_due_to_rate_limit"]:
        warnings.append("Some web checks skipped due to rate limiting.")
    if linkedin_links:
        warnings.append("LinkedIn profile link present; not auto-validated to preserve privacy.")

    candidate_profile = ResumeCandidateProfile(summary=_candidate_summary(profile), extracted=profile)
    validations = ResumeValidations(companies=companies, institutions=institutions, certifications=certifications)

    result = ResumeScreeningDetailedResponse(
        doc_id=doc_id,
        candidate_profile=candidate_profile,
        validations=validations,
        authenticity=authenticity,
        scoring=scoring,
        overall_confidence_0_100=float(round(overall_confidence, 2)),
        risk_level=risk_level,
        findings=findings,
        recommendations=recommendations,
        warnings=list(dict.fromkeys(warnings)),
        errors=[],
        narrative_report="",  # filled after structuring
    )
    result.narrative_report = build_narrative_report(result)
    return result
