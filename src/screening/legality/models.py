from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

DISCLAIMER_TEXT = "This is an automated screening report, not legal advice."


@dataclass
class Clause:
    id: str
    heading: str
    text: str
    clause_type: Optional[str] = None
    confidence_0_100: float = 75.0


@dataclass
class Finding:
    id: str
    title: str
    severity: str
    category: str
    clause_refs: List[str] = field(default_factory=list)
    description: str = ""
    recommendation: str = ""
    evidence_snippets: List[str] = field(default_factory=list)
    confidence_0_100: float = 75.0


@dataclass
class LegalityScores:
    risk_score_0_1: float
    risk_level: str
    overall_confidence_0_100: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "risk_score_0_1": round(self.risk_score_0_1, 4),
            "risk_level": self.risk_level,
            "overall_confidence_0_100": round(self.overall_confidence_0_100, 2),
            "breakdown": {k: round(v, 4) for k, v in (self.breakdown or {}).items()},
        }


@dataclass
class LegalityScreeningResult:
    doc_id: Optional[str]
    category: str
    region: str
    doc_type: str
    clauses: List[Clause] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    missing_clauses: List[str] = field(default_factory=list)
    risky_terms: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    scores: Optional[LegalityScores] = None
    narrative_report: str = ""
    disclaimer: str = DISCLAIMER_TEXT
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    region_compliance_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "category": self.category,
            "region": self.region,
            "doc_type": self.doc_type,
            "clauses": [asdict(clause) for clause in self.clauses],
            "findings": [asdict(finding) for finding in self.findings],
            "missing_clauses": self.missing_clauses,
            "risky_terms": self.risky_terms,
            "recommendations": self.recommendations,
            "scores": self.scores.to_dict() if self.scores else {},
            "narrative_report": self.narrative_report,
            "disclaimer": self.disclaimer,
            "warnings": self.warnings,
            "errors": self.errors,
            "region_compliance_notes": self.region_compliance_notes,
        }
