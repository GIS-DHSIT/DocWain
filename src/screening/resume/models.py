from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _strip(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None

    @field_validator("title", "company", "start_date", "end_date", "location", "description", mode="before")
    def _strip_experience(cls, value):
        return _strip(value)


class EducationItem(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("institution", "degree", "field", "start_year", "end_year", "notes", mode="before")
    def _strip_education(cls, value):
        return _strip(value)


class CertificationItem(BaseModel):
    name: str
    issuer: Optional[str] = None
    credential_id: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    verification_url: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "name",
        "issuer",
        "credential_id",
        "issue_date",
        "expiry_date",
        "verification_url",
        "notes",
        mode="before",
    )
    def _strip_cert(cls, value):
        return _strip(value)


class ResumeProfile(BaseModel):
    name: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[CertificationItem] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    def _normalize_skills(cls, values: List[str]) -> List[str]:
        if values is None:
            return []
        normalized = []
        for val in values:
            cleaned = (val or "").strip()
            if cleaned:
                normalized.append(cleaned)
        deduped = list(dict.fromkeys(normalized))
        return deduped

    @field_validator("links", mode="before")
    def _normalize_links(cls, values: List[str]) -> List[str]:
        if values is None:
            return []
        normalized = []
        for val in values:
            cleaned = (val or "").strip()
            if cleaned:
                normalized.append(cleaned)
        return list(dict.fromkeys(normalized))


class EvidenceSource(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: Optional[str] = None
    score: Optional[float] = None


class EvidenceBundle(BaseModel):
    name: str
    type: str
    exists: bool
    confidence_0_100: float
    sources: List[EvidenceSource] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    status: str = "unknown"


class CertificationEvidence(BaseModel):
    certification: CertificationItem
    exists: bool
    issuer_verified: bool
    credential_verifiable: bool
    credential_verified: Optional[bool] = None
    confidence_0_100: float = 0.0
    sources: List[EvidenceSource] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    status: str = "unverified"


class AuthenticitySignal(BaseModel):
    type: str
    severity: str
    message: str
    evidence: Optional[Dict[str, Any]] = None


class AuthenticityReport(BaseModel):
    signals: List[AuthenticitySignal] = Field(default_factory=list)
    risk_level: str = "low"
    confidence_0_100: float = 0.0
    recommendations: List[str] = Field(default_factory=list)


class ResumeScoringComponents(BaseModel):
    extraction_quality: float = 0.0
    org_validation_confidence: float = 0.0
    cert_validation_confidence: float = 0.0
    authenticity_confidence: float = 0.0


class ResumeCandidateProfile(BaseModel):
    summary: str
    extracted: ResumeProfile


class ResumeValidations(BaseModel):
    companies: List[EvidenceBundle] = Field(default_factory=list)
    institutions: List[EvidenceBundle] = Field(default_factory=list)
    certifications: List[CertificationEvidence] = Field(default_factory=list)


class ResumeFindings(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)


class ResumeRecommendations(BaseModel):
    next_steps: List[str] = Field(default_factory=list)
    documents_to_request: List[str] = Field(default_factory=list)


class ResumeScreeningDetailedResponse(BaseModel):
    doc_id: Optional[str] = None
    category: str = "resume"
    candidate_profile: ResumeCandidateProfile
    validations: ResumeValidations
    authenticity: AuthenticityReport
    scoring: ResumeScoringComponents
    overall_confidence_0_100: float
    risk_level: str
    findings: ResumeFindings
    recommendations: ResumeRecommendations
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    narrative_report: str = ""
