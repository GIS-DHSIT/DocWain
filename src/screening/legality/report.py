from __future__ import annotations

from typing import List

from .models import LegalityScreeningResult

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _format_findings(result: LegalityScreeningResult) -> List[str]:
    ordered = sorted(result.findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), f.title))
    lines: List[str] = []
    for finding in ordered:
        lines.append(f"- [{finding.severity.upper()}] {finding.title}: {finding.description}")
    return lines


def build_report(result: LegalityScreeningResult) -> str:
    lines: List[str] = []
    lines.append(f"Legality screening for doc_type={result.doc_type} region={result.region}")
    lines.append("Key findings:")
    finding_lines = _format_findings(result)
    if finding_lines:
        lines.extend(finding_lines)
    else:
        lines.append("- No critical legal gaps detected in the current pass.")

    if result.missing_clauses:
        lines.append(f"Missing/weak clauses: {', '.join(sorted(set(result.missing_clauses)))}")
    if result.risky_terms:
        lines.append(f"Risk flags: {', '.join(sorted(set(result.risky_terms)))}")
    if result.recommendations:
        lines.append("Recommended edits:")
        for rec in sorted(set(result.recommendations)):
            lines.append(f"- {rec}")
    if result.region_compliance_notes:
        lines.append("Region-specific compliance notes:")
        for note in result.region_compliance_notes:
            lines.append(f"- {note}")
    lines.append(result.disclaimer)
    return "\n".join(lines)
