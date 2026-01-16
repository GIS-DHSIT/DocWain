from __future__ import annotations

from typing import Dict, List

from .models import Finding, LegalityScores

SEVERITY_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.35}


def _risk_level(score: float) -> str:
    if score >= 0.66:
        return "HIGH"
    if score >= 0.33:
        return "MEDIUM"
    return "LOW"


def score_legality(findings: List[Finding], missing_clause_count: int) -> LegalityScores:
    if not findings:
        return LegalityScores(
            risk_score_0_1=0.1,
            risk_level="LOW",
            overall_confidence_0_100=92.0,
            breakdown={"missing_clause": 0.0, "risky_term": 0.0, "region_compliance": 0.0, "inconsistency": 0.0},
        )

    severity_total = sum(SEVERITY_WEIGHTS.get(f.severity, 0.5) for f in findings)
    severity_avg = severity_total / max(len(findings), 1)
    coverage_penalty = min(0.35, missing_clause_count * 0.05)
    risk_score = min(1.0, severity_avg * 0.7 + coverage_penalty)

    confidence = sum(f.confidence_0_100 for f in findings) / max(len(findings), 1)

    breakdown: Dict[str, float] = {"missing_clause": 0.0, "risky_term": 0.0, "region_compliance": 0.0, "inconsistency": 0.0}
    for f in findings:
        breakdown[f.category] = breakdown.get(f.category, 0.0) + SEVERITY_WEIGHTS.get(f.severity, 0.5)
    for key in list(breakdown.keys()):
        breakdown[key] = breakdown[key] / max(len(findings), 1)

    return LegalityScores(
        risk_score_0_1=round(risk_score, 4),
        risk_level=_risk_level(risk_score),
        overall_confidence_0_100=round(confidence, 2),
        breakdown=breakdown,
    )
