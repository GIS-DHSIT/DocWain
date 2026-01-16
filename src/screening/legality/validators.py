from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .models import Clause, Finding

SEVERITY_DEFAULTS = {"high": "high", "medium": "medium", "low": "low"}


def _normalize(value: str) -> str:
    return value.lower().strip()


def _clause_matches(clause: Clause, target: str) -> bool:
    normalized = _normalize(target)
    return normalized in clause.heading.lower() or normalized in clause.text.lower()


def _evidence_snippet(text: str, keyword: str, window: int = 80) -> str:
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return text[:window]
    start = max(idx - 40, 0)
    end = min(idx + window, len(text))
    return text[start:end].strip()


@dataclass
class ValidatorContext:
    doc_type: str
    region: str
    jurisdiction: Optional[str]
    rulepack: Dict[str, Any]
    full_text: str

    def doc_rules(self) -> Dict[str, Any]:
        rules = (self.rulepack.get("document_types") or {}).get(self.doc_type, {})
        default_rules = (self.rulepack.get("document_types") or {}).get("DEFAULT", {})
        merged: Dict[str, Any] = {}
        merged.update(default_rules)
        merged.update(rules)
        return merged


class BaseValidator:
    name = "base"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:  # pragma: no cover - interface
        raise NotImplementedError


class RequiredClauseValidator(BaseValidator):
    name = "required_clauses"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        rules = ctx.doc_rules()
        required = rules.get("required_clauses") or []
        findings: List[Finding] = []
        for idx, clause_name in enumerate(required, start=1):
            if any(_clause_matches(clause, clause_name) for clause in clauses):
                continue
            findings.append(
                Finding(
                    id=f"{self.name}_{idx}",
                    title=f"Missing clause: {clause_name}",
                    severity="high",
                    category="missing_clause",
                    description=f"Expected clause '{clause_name}' not found for {ctx.doc_type}.",
                    recommendation=f"Add a clear section titled '{clause_name}' covering obligations and scope.",
                    confidence_0_100=82.0,
                )
            )
        return findings


class RiskyClausePatternValidator(BaseValidator):
    name = "risky_clauses"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        rules = ctx.doc_rules()
        patterns: Iterable[Dict[str, Any]] = rules.get("risky_clauses") or []
        findings: List[Finding] = []
        for idx, pattern_cfg in enumerate(patterns, start=1):
            pattern = pattern_cfg.get("pattern")
            term_name = pattern_cfg.get("name") or pattern
            severity = SEVERITY_DEFAULTS.get(str(pattern_cfg.get("severity", "medium")).lower(), "medium")
            if not pattern:
                continue
            regex = re.compile(pattern, re.IGNORECASE)
            for clause in clauses:
                match = regex.search(clause.text)
                if not match:
                    continue
                findings.append(
                    Finding(
                        id=f"{self.name}_{idx}_{clause.id}",
                        title=f"Risky clause: {term_name}",
                        severity=severity,
                        category="risky_term",
                        clause_refs=[clause.id],
                        description=pattern_cfg.get("note", f"Clause contains risky term '{term_name}'."),
                        recommendation="Flag for legal review and consider narrowing scope.",
                        evidence_snippets=[_evidence_snippet(clause.text, match.group(0))],
                        confidence_0_100=76.0,
                    )
                )
        return findings


class DefinedTermsValidator(BaseValidator):
    name = "defined_terms"
    TERM_DEF_RE = re.compile(r'"([^"]+)"\s+(?:means|shall mean)', re.IGNORECASE)

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        findings: List[Finding] = []
        defined_terms = {m.group(1).strip() for m in self.TERM_DEF_RE.finditer(ctx.full_text)}
        used_terms = {word.strip('"') for word in re.findall(r"\b[A-Z][A-Za-z]+\b", ctx.full_text) if len(word) > 3}

        missing_definitions = sorted(term for term in used_terms if term not in defined_terms)[:5]
        if not defined_terms and ctx.doc_type in {"NDA", "CONTRACT", "TERMS", "EMPLOYMENT_OFFER"}:
            findings.append(
                Finding(
                    id=f"{self.name}_definitions_missing",
                    title="No defined terms section",
                    severity="medium",
                    category="inconsistency",
                    description="Document lacks explicit defined terms, which can create ambiguity.",
                    recommendation="Add a Definitions section to clarify parties, scope, and key terms.",
                    confidence_0_100=72.0,
                )
            )

        if missing_definitions:
            findings.append(
                Finding(
                    id=f"{self.name}_usage",
                    title="Undefined capitalized terms",
                    severity="medium",
                    category="inconsistency",
                    description="Some capitalized terms are used without definitions.",
                    recommendation=f"Define the following terms or lower-case them for plain meaning: {', '.join(missing_definitions)}.",
                    confidence_0_100=70.0,
                )
            )
        return findings


class GoverningLawValidator(BaseValidator):
    name = "governing_law"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        findings: List[Finding] = []
        matches = [cl for cl in clauses if _clause_matches(cl, "governing law")]
        if not matches:
            findings.append(
                Finding(
                    id=f"{self.name}_missing",
                    title="Governing law not specified",
                    severity="medium",
                    category="region_compliance",
                    description="Document should state governing law aligned to the operating region.",
                    recommendation="Add a Governing Law clause referencing the intended jurisdiction and rationale.",
                    confidence_0_100=78.0,
                )
            )
            return findings

        clause = matches[0]
        overrides = ctx.doc_rules().get("region_overrides") or []
        for override in overrides:
            if override.get("clause", "").lower() != "governing law":
                continue
            requirement = override.get("requirement")
            if ctx.region and ctx.region.lower() not in clause.text.lower():
                findings.append(
                    Finding(
                        id=f"{self.name}_mismatch",
                        title="Governing law does not match region",
                        severity="medium",
                        category="region_compliance",
                        clause_refs=[clause.id],
                        description=requirement or f"Governing law should reference {ctx.region}.",
                        recommendation="Align governing law to the contracting region or explain the deviation.",
                        evidence_snippets=[_evidence_snippet(clause.text, "law")],
                        confidence_0_100=74.0,
                    )
                )
        return findings


class JurisdictionVenueValidator(BaseValidator):
    name = "jurisdiction"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        findings: List[Finding] = []
        targets = [cl for cl in clauses if any(_clause_matches(cl, kw) for kw in ("venue", "jurisdiction", "court"))]
        if targets:
            return findings
        findings.append(
            Finding(
                id=f"{self.name}_missing",
                title="Forum/venue not specified",
                severity="medium",
                category="region_compliance",
                description="Dispute resolution venue is not specified; this is needed for enforceability.",
                recommendation="Add a Jurisdiction or Venue clause to clarify where disputes will be handled.",
                confidence_0_100=73.0,
            )
        )
        return findings


class LiabilityIndemnityValidator(BaseValidator):
    name = "liability_indemnity"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        if ctx.doc_type not in {"CONTRACT", "NDA", "TERMS", "EMPLOYMENT_OFFER", "POLICY"}:
            return []

        findings: List[Finding] = []
        text = ctx.full_text.lower()
        has_indemnity = "indemn" in text
        has_liability = "liability" in text

        if not has_indemnity or not has_liability:
            missing = []
            if not has_indemnity:
                missing.append("Indemnity")
            if not has_liability:
                missing.append("Limitation of Liability")
            findings.append(
                Finding(
                    id=f"{self.name}_missing",
                    title=f"Missing {' and '.join(missing)} clause",
                    severity="high",
                    category="missing_clause",
                    description="Critical risk-shifting clauses are absent.",
                    recommendation="Add indemnity and liability limitation clauses with clear caps and exclusions.",
                    confidence_0_100=79.0,
                )
            )

        risky_language = [
            ("unlimited liability", "high"),
            ("hold harmless", "medium"),
            ("consequential damages", "medium"),
        ]
        for phrase, severity in risky_language:
            if phrase in text:
                findings.append(
                    Finding(
                        id=f"{self.name}_{phrase.replace(' ', '_')}",
                        title=f"Risky liability term: {phrase}",
                        severity=severity,
                        category="risky_term",
                        description=f"Found '{phrase}' which may expand exposure.",
                        recommendation="Review liability allocation and consider negotiated caps/carve-outs.",
                        evidence_snippets=[_evidence_snippet(ctx.full_text, phrase)],
                        confidence_0_100=76.0,
                    )
                )
        return findings


class TerminationSurvivalValidator(BaseValidator):
    name = "termination_survival"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        if ctx.doc_type not in {"CONTRACT", "TERMS", "NDA", "EMPLOYMENT_OFFER", "POLICY"}:
            return []

        findings: List[Finding] = []
        has_termination = any(_clause_matches(cl, "termination") for cl in clauses)
        mentions_survival = "survival" in ctx.full_text.lower()

        if not has_termination:
            findings.append(
                Finding(
                    id=f"{self.name}_termination_missing",
                    title="Termination rights unclear",
                    severity="medium",
                    category="missing_clause",
                    description="Termination conditions are not described.",
                    recommendation="Add a Termination clause covering notice, breach, and convenience scenarios.",
                    confidence_0_100=71.0,
                )
            )
        if not mentions_survival:
            findings.append(
                Finding(
                    id=f"{self.name}_survival_missing",
                    title="Survival terms absent",
                    severity="low",
                    category="inconsistency",
                    description="Document does not state which obligations survive termination.",
                    recommendation="List confidentiality, payment, IP, and dispute resolution as surviving obligations.",
                    confidence_0_100=70.0,
                )
            )
        return findings


class IPAssignmentValidator(BaseValidator):
    name = "ip_assignment"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        if ctx.doc_type not in {"EMPLOYMENT_OFFER", "CONTRACT", "NDA"}:
            return []

        has_ip = any(_clause_matches(cl, "intellectual property") or _clause_matches(cl, "ownership") for cl in clauses)
        findings: List[Finding] = []
        if not has_ip:
            findings.append(
                Finding(
                    id=f"{self.name}_missing",
                    title="IP ownership not addressed",
                    severity="medium",
                    category="missing_clause",
                    description="Document does not clarify ownership or assignment of intellectual property.",
                    recommendation="Add IP ownership/assignment terms and moral rights waivers where applicable.",
                    confidence_0_100=74.0,
                )
            )
        return findings


class DataProtectionValidator(BaseValidator):
    name = "data_protection"

    def validate(self, ctx: ValidatorContext, clauses: List[Clause]) -> List[Finding]:
        rules = ctx.doc_rules()
        region_checks = rules.get("region_checks", {})
        region_specific: List[str] = []
        if ctx.region and isinstance(region_checks, dict):
            region_specific = region_checks.get(ctx.region.upper(), []) or []

        findings: List[Finding] = []
        if ctx.doc_type not in {"PRIVACY_NOTICE", "POLICY", "TERMS"}:
            return findings

        required = region_specific or region_checks.get("GLOBAL") or []
        for idx, item in enumerate(required, start=1):
            if any(_normalize(item) in clause.text.lower() for clause in clauses):
                continue
            findings.append(
                Finding(
                    id=f"{self.name}_{ctx.region}_{idx}",
                    title=f"Region-specific privacy item missing: {item}",
                    severity="high",
                    category="region_compliance",
                    description=f"Privacy obligations for {ctx.region} are incomplete: {item}.",
                    recommendation="Add concise language covering the missing privacy obligation; tailor to the region.",
                    confidence_0_100=83.0,
                )
            )
        return findings


VALIDATORS: List[BaseValidator] = [
    RequiredClauseValidator(),
    RiskyClausePatternValidator(),
    DefinedTermsValidator(),
    GoverningLawValidator(),
    JurisdictionVenueValidator(),
    LiabilityIndemnityValidator(),
    TerminationSurvivalValidator(),
    IPAssignmentValidator(),
    DataProtectionValidator(),
]
