from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..models import ScreeningContext
from .clause_extractor import ClauseExtractor
from .models import DISCLAIMER_TEXT, Clause, LegalityScreeningResult
from .report import build_report
from .scorer import score_legality
from .validators import VALIDATORS, ValidatorContext

SUPPORTED_REGIONS = {"US", "UK", "EU", "IN", "CA", "AU", "SG"}


class RulepackLoader:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path(__file__).parent / "rulepacks"

    def _load_yaml(self, name: str) -> Dict[str, Any]:
        path = self.base_dir / name
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text()) or {}
        except Exception:
            return {}

    def _merge_rules(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_rules(merged[key], value)
            else:
                merged[key] = value
        return merged

    def load(self, region: str) -> Dict[str, Any]:
        base_rules = self._load_yaml("base.yaml")
        region_key = region.lower()
        region_rules = self._load_yaml(f"{region_key}.yaml")
        return self._merge_rules(base_rules, region_rules)


class LegalityAgent:
    """Multi-step legality screening agent."""

    def __init__(self, rulepack_loader: Optional[RulepackLoader] = None, clause_extractor: Optional[ClauseExtractor] = None) -> None:
        self.rulepack_loader = rulepack_loader or RulepackLoader()
        self.clause_extractor = clause_extractor or ClauseExtractor()

    def _normalize_doc_type(self, doc_type: Optional[str]) -> Optional[str]:
        if not doc_type:
            return None
        lookup = {
            "nda": "NDA",
            "non disclosure": "NDA",
            "privacy": "PRIVACY_NOTICE",
            "privacy_notice": "PRIVACY_NOTICE",
            "policy": "POLICY",
            "contract": "CONTRACT",
            "agreement": "CONTRACT",
            "invoice": "INVOICE",
            "terms": "TERMS",
            "t&c": "TERMS",
            "offer": "EMPLOYMENT_OFFER",
            "employment": "EMPLOYMENT_OFFER",
        }
        lowered = doc_type.lower()
        for key, value in lookup.items():
            if key in lowered:
                return value
        return doc_type.upper()

    def _infer_doc_type(self, ctx: ScreeningContext) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        if ctx.doc_type:
            normalized = self._normalize_doc_type(ctx.doc_type)
            if normalized:
                return normalized, warnings
        text = ctx.text.lower()
        heuristics = [
            ("NDA", ["non-disclosure", "confidential information", "nda"]),
            ("PRIVACY_NOTICE", ["privacy policy", "personal data", "data subject"]),
            ("TERMS", ["terms of service", "terms and conditions", "t&c"]),
            ("EMPLOYMENT_OFFER", ["offer letter", "at-will", "position title", "compensation"]),
            ("INVOICE", ["invoice", "bill to", "invoice number"]),
            ("POLICY", ["policy", "procedure", "scope", "responsibilities"]),
            ("CONTRACT", ["agreement", "contract", "hereby"]),
        ]
        for doc_type, keywords in heuristics:
            if any(keyword in text for keyword in keywords):
                return doc_type, warnings
        warnings.append("Document type inferred as CONTRACT by default.")
        return "CONTRACT", warnings

    def _resolve_region(self, ctx: ScreeningContext, explicit_region: Optional[str]) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        region = explicit_region or ctx.region
        if region:
            region = region.upper()
        if region and region not in SUPPORTED_REGIONS:
            warnings.append(f"Region '{region}' not recognized; using GLOBAL rules.")
            region = "GLOBAL"
        if not region:
            warnings.append("Region not provided; using GLOBAL base rules.")
            region = "GLOBAL"
        return region, warnings

    def _collect_recommendations(self, findings: List[Any]) -> List[str]:
        recs: List[str] = []
        for finding in findings:
            if finding.recommendation:
                recs.append(finding.recommendation)
        return recs

    def _region_notes(self, rulepack: Dict[str, Any], region: str) -> List[str]:
        notes: List[str] = []
        region_notes = rulepack.get("region_notes", {})
        if isinstance(region_notes, dict):
            notes.extend(region_notes.get(region, []) or [])
            notes.extend(region_notes.get("GLOBAL", []) or [])
        return notes

    def analyze(
        self,
        ctx: ScreeningContext,
        *,
        region: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> LegalityScreeningResult:
        resolved_region, region_warnings = self._resolve_region(ctx, region)
        resolved_doc_type, doc_type_warnings = self._infer_doc_type(ctx)
        if doc_type:
            resolved_doc_type = self._normalize_doc_type(doc_type) or resolved_doc_type
        clauses: List[Clause] = self.clause_extractor.extract(ctx.text or "")

        rulepack = self.rulepack_loader.load(resolved_region.lower())
        validator_ctx = ValidatorContext(
            doc_type=resolved_doc_type,
            region=resolved_region,
            jurisdiction=jurisdiction or ctx.jurisdiction,
            rulepack=rulepack,
            full_text=ctx.text or "",
        )

        findings: List[Any] = []
        for validator in VALIDATORS:
            findings.extend(validator.validate(validator_ctx, clauses))

        missing_clauses = sorted({f.title.replace("Missing clause: ", "") for f in findings if f.category == "missing_clause"})
        risky_terms = sorted({f.title for f in findings if f.category == "risky_term"})
        recommendations = sorted(set(self._collect_recommendations(findings)))
        scores = score_legality(findings, len(missing_clauses))
        warnings = region_warnings + doc_type_warnings
        if ctx.config and not ctx.config.internet_enabled:
            warnings.append("Internet validation disabled; external checks skipped.")

        region_compliance_notes = self._region_notes(rulepack, resolved_region)
        result = LegalityScreeningResult(
            doc_id=ctx.doc_id,
            category="legality",
            region=resolved_region,
            doc_type=resolved_doc_type,
            clauses=clauses,
            findings=findings,
            missing_clauses=missing_clauses,
            risky_terms=risky_terms,
            recommendations=recommendations,
            scores=scores,
            warnings=warnings,
            errors=[],
            region_compliance_notes=region_compliance_notes,
        )
        result.narrative_report = build_report(result)
        result.disclaimer = DISCLAIMER_TEXT
        return result


def serialize_legality_result(result: LegalityScreeningResult) -> Dict[str, Any]:
    """Serialize to json-friendly structure with deterministic formatting."""
    payload = result.to_dict()
    # Keep narrative deterministic by ensuring stable ordering of findings and recommendations.
    payload["recommendations"] = sorted(set(result.recommendations))
    payload["missing_clauses"] = sorted(set(result.missing_clauses))
    payload["risky_terms"] = sorted(set(result.risky_terms))
    payload["warnings"] = result.warnings
    payload["errors"] = result.errors
    payload["disclaimer"] = DISCLAIMER_TEXT
    return payload
