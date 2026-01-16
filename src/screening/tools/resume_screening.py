from __future__ import annotations

from typing import List

from ..models import ScreeningContext
from ..resume import run_resume_analysis
from ..resume.models import ResumeScreeningDetailedResponse
from ..search import NullSearchClient
from .base import ScreeningTool, clamp


def _get_analysis(ctx: ScreeningContext) -> ResumeScreeningDetailedResponse:
    cache_key = "_resume_analysis"
    cached = ctx.metadata.get(cache_key)
    if isinstance(cached, ResumeScreeningDetailedResponse):
        return cached

    internet_enabled = bool(ctx.config and ctx.config.internet_enabled)
    analysis = run_resume_analysis(
        text=ctx.text or "",
        doc_id=ctx.doc_id,
        metadata=ctx.metadata,
        search_client=ctx.search_client or NullSearchClient(),
        internet_enabled=internet_enabled,
    )
    ctx.metadata[cache_key] = analysis
    return analysis


class ResumeExtractorTool(ScreeningTool):
    name = "resume_extractor_tool"
    category = "Resume Screening"
    default_weight = 0.08
    requires_internet = False
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.1"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        missing = analysis.findings.missing_information
        score = clamp(len(missing) / 5.0)
        reasons: List[str] = []
        if missing:
            reasons.append(f"Missing sections: {', '.join(missing)}.")
        if analysis.findings.strengths:
            reasons.append(f"Found strengths: {', '.join(analysis.findings.strengths[:3])}.")
        return self.result(
            ctx,
            score,
            reasons or ["Resume sections extracted successfully."],
            raw_features={"profile": analysis.candidate_profile.extracted.model_dump(), "missing": missing},
            actions=["tag"],
        )


class ResumeCompanyValidatorTool(ScreeningTool):
    name = "company_validator_tool"
    category = "Resume Screening"
    default_weight = 0.08
    requires_internet = True
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.1"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        companies = analysis.validations.companies
        unverified = [c for c in companies if not c.exists]
        uncertain = [c for c in companies if c.status == "uncertain"]
        score = clamp(((len(unverified) * 0.7) + (len(uncertain) * 0.4)) / max(len(companies), 1))
        reasons: List[str] = []
        if unverified:
            reasons.append(f"Companies without existence evidence: {', '.join(c.name for c in unverified)}.")
        if uncertain and not (ctx.config and ctx.config.internet_enabled):
            reasons.append("Internet validation disabled; company checks marked uncertain.")
        if not reasons:
            reasons.append("Companies appear valid based on existence-only checks.")
        return self.result(
            ctx,
            score,
            reasons,
            raw_features={"companies": [c.model_dump() for c in companies]},
            actions=["tag"],
        )


class ResumeInstitutionValidatorTool(ScreeningTool):
    name = "institution_validator_tool"
    category = "Resume Screening"
    default_weight = 0.06
    requires_internet = True
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.1"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        institutions = analysis.validations.institutions
        unverified = [c for c in institutions if not c.exists]
        uncertain = [c for c in institutions if c.status == "uncertain"]
        score = clamp(((len(unverified) * 0.7) + (len(uncertain) * 0.4)) / max(len(institutions), 1))
        reasons: List[str] = []
        if unverified:
            reasons.append(f"Institutions without existence evidence: {', '.join(c.name for c in unverified)}.")
        if uncertain and not (ctx.config and ctx.config.internet_enabled):
            reasons.append("Internet validation disabled; institution checks marked uncertain.")
        if not reasons:
            reasons.append("Institutions appear valid based on existence-only checks.")
        return self.result(
            ctx,
            score,
            reasons,
            raw_features={"institutions": [c.model_dump() for c in institutions]},
            actions=["tag"],
        )


class CertificationVerifierTool(ScreeningTool):
    name = "certification_verifier_tool"
    category = "Resume Screening"
    default_weight = 0.1
    requires_internet = True
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.1"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        certs = analysis.validations.certifications
        suspicious = [c for c in certs if c.status == "suspicious"]
        unverified = [c for c in certs if c.status == "unverified"]
        score = clamp(((len(suspicious) * 0.8) + (len(unverified) * 0.5)) / max(len(certs), 1))
        reasons: List[str] = []
        if suspicious:
            reasons.append(f"Suspicious certifications: {', '.join(c.certification.name for c in suspicious)}.")
        if unverified and not suspicious:
            reasons.append(f"Certifications could not be verified: {', '.join(c.certification.name for c in unverified)}.")
        if not reasons:
            reasons.append("Certifications validated with available open-web evidence.")
        return self.result(
            ctx,
            score,
            reasons,
            raw_features={"certifications": [c.model_dump() for c in certs]},
            actions=["tag"],
        )


class ResumeAuthenticityTool(ScreeningTool):
    name = "authenticity_analyzer_tool"
    category = "Resume Screening"
    default_weight = 0.12
    requires_internet = False
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.1"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        signals = analysis.authenticity.signals
        score = clamp(len(signals) / 6.0)
        reasons = [sig.message for sig in signals[:4]] or ["No authenticity risks detected."]
        return self.result(
            ctx,
            score,
            reasons,
            raw_features={
                "signals": [s.model_dump() for s in signals],
                "risk_level": analysis.authenticity.risk_level,
            },
            actions=["tag"],
        )


class ResumeScreeningTool(ScreeningTool):
    name = "resume_screening"
    category = "Resume Screening"
    default_weight = 0.12
    requires_internet = False
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "2.0"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        risk_map = {"low": 0.2, "medium": 0.55, "high": 0.8}
        base_score = risk_map.get(analysis.risk_level.lower(), 0.5) if analysis.risk_level else 0.5
        score = clamp(base_score)
        reasons: List[str] = []
        if analysis.findings.concerns:
            reasons.extend(analysis.findings.concerns[:3])
        if analysis.warnings:
            reasons.extend(analysis.warnings)
        if not reasons:
            reasons.append("Resume appears consistent with provided evidence and authenticity signals are low.")
        return self.result(
            ctx,
            score,
            reasons,
            raw_features={"analysis": analysis.model_dump()},
            actions=["tag"],
        )
