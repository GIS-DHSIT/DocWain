"""
Document insights and anomaly detection tool for DocWain.

Provides proactive intelligence: anomaly detection, pattern recognition,
risk identification across document domains.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.tools.base import register_tool, standard_response

logger = get_logger(__name__)

@dataclass
class Insight:
    """A single insight or anomaly detected in documents."""

    category: str  # "anomaly", "pattern", "risk", "observation"
    title: str
    description: str
    severity: str = "info"  # "critical", "warning", "info"
    domain: str = "generic"
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "domain": self.domain,
        }
        if self.evidence:
            result["evidence"] = self.evidence
        return result

# ── Number extraction ─────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(
    r"[\$\u20ac\u00a3\u20b9]\s*[\d,]+\.?\d*|\d[\d,]*\.?\d*\s*(?:USD|EUR|GBP|INR)",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*\b")

_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{2,4}|"
    r"\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
    re.IGNORECASE,
)

def _parse_currency(text: str) -> float:
    """Parse a currency string to float."""
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0

# ── Statistical helpers ───────────────────────────────────────────────────

def _find_outliers(values: List[float], multiplier: float = 1.5) -> List[tuple[int, float]]:
    """Find statistical outliers using IQR method."""
    if len(values) < 3:
        return []

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    outliers = []
    for i, v in enumerate(values):
        if v < lower or v > upper:
            outliers.append((i, v))
    return outliers

# ── Domain-specific insight generators ────────────────────────────────────

def _hr_insights(text: str, chunks: List[str]) -> List[Insight]:
    """Generate HR/resume-specific insights."""
    insights: List[Insight] = []
    text_lower = text.lower()

    # Skill detection — always report found skills
    common_skills = {
        "python", "java", "javascript", "sql", "aws", "docker", "kubernetes",
        "react", "node", "typescript", "git", "linux", "ci/cd", "agile",
    }
    found_skills = {s for s in common_skills if s in text_lower}

    # Always produce a skill summary
    if found_skills:
        sorted_skills = sorted(found_skills)
        insights.append(Insight(
            category="pattern",
            title="Skill Distribution",
            description=f"Found {len(found_skills)} common technical skill(s) across candidates: "
                        f"{', '.join(sorted_skills)}. "
                        f"{'Broad skill coverage.' if len(found_skills) >= 7 else 'Focused skill set.'}",
            severity="info",
            domain="hr",
        ))
    else:
        insights.append(Insight(
            category="observation",
            title="Skill Profile",
            description="No standard technical skill keywords detected. "
                        "Candidates may have domain-specific or non-technical skill sets.",
            severity="info",
            domain="hr",
        ))

    # Experience pattern — always report
    exp_re = re.compile(r"(\d+)\s*(?:\+\s*)?(?:years?|yrs?)\s*(?:of\s+)?(?:experience)?", re.IGNORECASE)
    years = [int(m.group(1)) for m in exp_re.finditer(text)]
    if years:
        unique_years = sorted(set(years))
        avg_years = sum(years) / len(years)
        insights.append(Insight(
            category="pattern",
            title="Experience Distribution",
            description=f"Experience range: {min(years)}-{max(years)} years "
                        f"(average {avg_years:.1f} years, {len(years)} data points). "
                        f"Levels represented: {', '.join(str(y) for y in unique_years[:8])} years.",
            severity="info",
            domain="hr",
        ))
        if max(years) > 20:
            insights.append(Insight(
                category="anomaly",
                title="Experience Outlier",
                description=f"Very high experience claim detected: {max(years)} years. Verify accuracy.",
                severity="warning",
                domain="hr",
                evidence=f"Found experience claims: {years}",
            ))
    else:
        insights.append(Insight(
            category="observation",
            title="Experience Data",
            description="No explicit years-of-experience values detected in the documents.",
            severity="info",
            domain="hr",
        ))

    # Education detection
    degree_re = re.compile(r"\b(?:Ph\.?D|Master|Bachelor|B\.?(?:Tech|E|Sc|A)|M\.?(?:Tech|E|Sc|A|BA)|MBA)\b", re.IGNORECASE)
    degrees = degree_re.findall(text)
    if degrees:
        from collections import Counter
        degree_counts = Counter(d.lower() for d in degrees)
        top_degrees = degree_counts.most_common(5)
        deg_summary = ", ".join(f"{d.title()} ({c}x)" for d, c in top_degrees)
        insights.append(Insight(
            category="observation",
            title="Education Profile",
            description=f"Found {len(degrees)} educational qualifications: {deg_summary}.",
            severity="info",
            domain="hr",
        ))

    return insights

def _invoice_insights(text: str, chunks: List[str]) -> List[Insight]:
    """Generate invoice-specific insights."""
    insights: List[Insight] = []

    # Amount analysis
    amounts = [_parse_currency(m.group()) for m in _CURRENCY_RE.finditer(text)]
    amounts = [a for a in amounts if a > 0]

    if amounts:
        outliers = _find_outliers(amounts)
        if outliers:
            outlier_values = [f"${v:,.2f}" for _, v in outliers]
            insights.append(Insight(
                category="anomaly",
                title="Amount Outliers",
                description=f"Unusual amounts detected: {', '.join(outlier_values)}. "
                            f"Average is ${statistics.mean(amounts):,.2f}.",
                severity="warning",
                domain="invoice",
                evidence=f"All amounts: {[f'${a:,.2f}' for a in amounts[:10]]}",
            ))

        total_candidates = [a for a in amounts if a == max(amounts)]
        if len(total_candidates) == 1 and len(amounts) > 2:
            subtotal = sum(a for a in amounts if a != max(amounts))
            if abs(subtotal - max(amounts)) / max(max(amounts), 1) > 0.1:
                insights.append(Insight(
                    category="anomaly",
                    title="Total Mismatch",
                    description=f"Line items sum to ${subtotal:,.2f} but highest amount "
                                f"is ${max(amounts):,.2f}. Check for missing items or tax.",
                    severity="warning",
                    domain="invoice",
                ))

    # Missing field detection
    required_fields = ["invoice", "date", "total", "amount"]
    text_lower = text.lower()
    missing = [f for f in required_fields if f not in text_lower]
    if missing:
        insights.append(Insight(
            category="risk",
            title="Missing Invoice Fields",
            description=f"Potentially missing standard fields: {', '.join(missing)}.",
            severity="info",
            domain="invoice",
        ))

    # Date anomaly
    dates = _DATE_RE.findall(text)
    if not dates:
        insights.append(Insight(
            category="risk",
            title="No Dates Found",
            description="No dates detected in the invoice. Check for issue and due dates.",
            severity="warning",
            domain="invoice",
        ))

    return insights

def _legal_insights(text: str, chunks: List[str]) -> List[Insight]:
    """Generate legal/contract-specific insights."""
    insights: List[Insight] = []

    # Risky clause detection
    risky_patterns = [
        (re.compile(r"\b(?:unlimited\s+liability|sole\s+discretion|without\s+limitation)\b", re.IGNORECASE),
         "Unlimited Liability/Discretion", "critical"),
        (re.compile(r"\b(?:indemnif\w+\s+(?:and\s+)?hold\s+harmless)\b", re.IGNORECASE),
         "Indemnification Clause", "warning"),
        (re.compile(r"\b(?:auto[\s-]?renew|automatic(?:ally)?\s+renew)\b", re.IGNORECASE),
         "Auto-Renewal", "info"),
        (re.compile(r"\b(?:non[\s-]?compete|non[\s-]?solicitation|restrictive\s+covenant)\b", re.IGNORECASE),
         "Restrictive Covenant", "warning"),
        (re.compile(r"\b(?:liquidated\s+damages|penalty|penalti)\b", re.IGNORECASE),
         "Penalty Clause", "warning"),
        (re.compile(r"\b(?:unilateral\w*\s+(?:modify|amend|change|terminate))\b", re.IGNORECASE),
         "Unilateral Modification Rights", "critical"),
    ]

    for pattern, title, severity in risky_patterns:
        m = pattern.search(text)
        if m:
            context_start = max(0, m.start() - 50)
            context_end = min(len(text), m.end() + 100)
            insights.append(Insight(
                category="risk",
                title=title,
                description=f"Detected: '{m.group()}'. Review this clause carefully.",
                severity=severity,
                domain="legal",
                evidence=text[context_start:context_end].strip(),
            ))

    # Missing standard clauses
    standard_clauses = {
        "governing law": re.compile(r"\b(?:govern\w*\s+law|jurisdiction|applicable\s+law)\b", re.IGNORECASE),
        "termination": re.compile(r"\b(?:terminat\w+|expir\w+)\b", re.IGNORECASE),
        "dispute resolution": re.compile(r"\b(?:dispute|arbitrat|mediat)\b", re.IGNORECASE),
        "force majeure": re.compile(r"\b(?:force\s+majeure|act\s+of\s+god)\b", re.IGNORECASE),
        "confidentiality": re.compile(r"\b(?:confidential|non[\s-]?disclosure)\b", re.IGNORECASE),
    }

    missing_clauses = [name for name, pat in standard_clauses.items() if not pat.search(text)]
    if missing_clauses:
        insights.append(Insight(
            category="risk",
            title="Missing Standard Clauses",
            description=f"Standard clauses not found: {', '.join(missing_clauses)}. "
                        "These are typically expected in contracts.",
            severity="warning" if len(missing_clauses) >= 2 else "info",
            domain="legal",
        ))

    return insights

def _medical_insights(text: str, chunks: List[str]) -> List[Insight]:
    """Generate medical-specific insights."""
    insights: List[Insight] = []

    # Medication interaction check
    med_re = re.compile(
        r"\b(?:aspirin|warfarin|metformin|lisinopril|atorvastatin|omeprazole|"
        r"amlodipine|metoprolol|amoxicillin|ibuprofen|acetaminophen|prednisone|"
        r"insulin|levothyroxine|hydrochlorothiazide)\b",
        re.IGNORECASE,
    )
    medications = list({m.group().lower() for m in med_re.finditer(text)})

    known_interactions = {
        frozenset({"warfarin", "aspirin"}): "Increased bleeding risk",
        frozenset({"warfarin", "ibuprofen"}): "Increased bleeding risk",
        frozenset({"metformin", "prednisone"}): "May increase blood sugar",
        frozenset({"lisinopril", "ibuprofen"}): "May reduce antihypertensive effect",
    }

    if len(medications) >= 2:
        for pair, risk in known_interactions.items():
            if pair.issubset(set(medications)):
                meds = sorted(pair)
                insights.append(Insight(
                    category="risk",
                    title="Potential Drug Interaction",
                    description=f"{meds[0].title()} + {meds[1].title()}: {risk}.",
                    severity="critical",
                    domain="medical",
                    evidence=f"Medications found: {', '.join(medications)}",
                ))

    # Abnormal lab values
    lab_patterns = [
        (re.compile(r"(?:glucose|blood\s+sugar)\s*[:\-]?\s*(\d+)", re.IGNORECASE), "Glucose", 70, 200),
        (re.compile(r"(?:hemoglobin|hgb|hb)\s*[:\-]?\s*([\d.]+)", re.IGNORECASE), "Hemoglobin", 7, 18),
        (re.compile(r"(?:creatinine)\s*[:\-]?\s*([\d.]+)", re.IGNORECASE), "Creatinine", 0.5, 3.0),
        (re.compile(r"(?:temperature|temp)\s*[:\-]?\s*([\d.]+)", re.IGNORECASE), "Temperature", 95, 104),
    ]

    for pattern, name, low, high in lab_patterns:
        m = pattern.search(text)
        if m:
            try:
                value = float(m.group(1))
                if value < low or value > high:
                    insights.append(Insight(
                        category="anomaly",
                        title=f"Abnormal {name} Value",
                        description=f"{name} value of {value} is outside normal range ({low}-{high}).",
                        severity="warning" if low <= value <= high * 1.2 else "critical",
                        domain="medical",
                        evidence=m.group(),
                    ))
            except (ValueError, TypeError):
                pass

    # Missing follow-up
    followup_re = re.compile(r"\b(?:follow[\s-]?up|return\s+visit|schedule|appointment)\b", re.IGNORECASE)
    if not followup_re.search(text) and len(text) > 200:
        insights.append(Insight(
            category="observation",
            title="No Follow-Up Mentioned",
            description="No follow-up appointment or visit mentioned in the document.",
            severity="info",
            domain="medical",
        ))

    return insights

def _generic_insights(text: str, chunks: List[str]) -> List[Insight]:
    """Generate domain-agnostic insights."""
    insights: List[Insight] = []

    # Topic distribution
    text_lower = text.lower()
    topic_keywords = {
        "financial": ["payment", "amount", "cost", "price", "fee", "total", "tax"],
        "personnel": ["employee", "candidate", "staff", "team", "manager"],
        "technical": ["system", "software", "data", "server", "api", "code"],
        "compliance": ["policy", "regulation", "compliance", "requirement", "standard"],
        "operational": ["process", "procedure", "workflow", "schedule", "timeline"],
    }

    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count >= 2:
            topic_scores[topic] = count

    if topic_scores:
        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        topics_str = ", ".join(f"{t[0]} ({t[1]} signals)" for t in top_topics)
        insights.append(Insight(
            category="pattern",
            title="Topic Distribution",
            description=f"Primary topics detected: {topics_str}.",
            severity="info",
            domain="generic",
        ))

    # Entity frequency
    entity_re = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")
    entities = [m.group() for m in entity_re.finditer(text)]
    if entities:
        from collections import Counter
        entity_counts = Counter(entities)
        most_common = entity_counts.most_common(5)
        if most_common:
            entities_str = ", ".join(f"{e} ({c}x)" for e, c in most_common)
            insights.append(Insight(
                category="pattern",
                title="Key Entities",
                description=f"Most frequently mentioned entities: {entities_str}.",
                severity="info",
                domain="generic",
            ))

    # Field completeness
    numbers = _NUMBER_RE.findall(text)
    dates = _DATE_RE.findall(text)
    currencies = _CURRENCY_RE.findall(text)

    data_richness = []
    if numbers:
        data_richness.append(f"{len(numbers)} numeric values")
    if dates:
        data_richness.append(f"{len(dates)} date references")
    if currencies:
        data_richness.append(f"{len(currencies)} monetary amounts")

    if data_richness:
        insights.append(Insight(
            category="observation",
            title="Data Richness",
            description=f"Document contains: {', '.join(data_richness)}.",
            severity="info",
            domain="generic",
        ))

    return insights

# ── Main insight generation ───────────────────────────────────────────────

def generate_insights(
    text: str,
    domain: Optional[str] = None,
    chunks: Optional[List[str]] = None,
) -> List[Insight]:
    """
    Generate insights and anomalies from document text.

    Uses domain-specific generators for HR, invoice, legal, medical.
    Falls back to generic pattern detection.
    """
    if not text or len(text.strip()) < 20:
        return []

    chunk_list = chunks or []
    insights: List[Insight] = []

    domain_lower = (domain or "").lower()
    domain_map = {
        "hr": _hr_insights,
        "resume": _hr_insights,
        "invoice": _invoice_insights,
        "legal": _legal_insights,
        "contract": _legal_insights,
        "medical": _medical_insights,
        "policy": _medical_insights,
    }

    generator = domain_map.get(domain_lower)
    if generator:
        try:
            insights.extend(generator(text, chunk_list))
        except Exception as exc:
            logger.debug("Domain insight generation failed: %s", exc)

    try:
        insights.extend(_generic_insights(text, chunk_list))
    except Exception as exc:
        logger.debug("Generic insight generation failed: %s", exc)

    # Deduplicate by title
    seen: Set[str] = set()
    unique: List[Insight] = []
    for ins in insights:
        if ins.title not in seen:
            seen.add(ins.title)
            unique.append(ins)

    # Sort by severity: critical > warning > info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    unique.sort(key=lambda x: severity_order.get(x.severity, 2))

    return unique

def render_insights(insights: List[Insight], domain: Optional[str] = None) -> str:
    """Render insights into a readable format."""
    if not insights:
        return "No notable insights or anomalies found."

    severity_icons = {"critical": "[!]", "warning": "[*]", "info": "[-]"}

    # Include domain in header for context
    domain_label = (domain or "").strip().title()
    if domain_label:
        lines = [f"**{len(insights)} {domain_label} Insight(s) Found**\n"]
    else:
        # Derive domain from insights themselves
        domains = {ins.domain for ins in insights if ins.domain and ins.domain != "generic"}
        if domains:
            domain_label = ", ".join(d.title() for d in sorted(domains))
            lines = [f"**{len(insights)} {domain_label} Insight(s) Found**\n"]
        else:
            lines = [f"**{len(insights)} Insight(s) Found**\n"]

    for ins in insights:
        icon = severity_icons.get(ins.severity, "[-]")
        lines.append(f"{icon} **{ins.title}** ({ins.category})")
        lines.append(f"   {ins.description}")
        if ins.evidence:
            lines.append(f"   Evidence: _{ins.evidence[:100]}_")
        lines.append("")

    return "\n".join(lines)

@register_tool("insights")
async def insights_handler(
    payload: Dict[str, Any], correlation_id: str | None = None
) -> Dict[str, Any]:
    """Handle document insight and anomaly detection requests."""
    input_data = payload.get("input") or payload
    text = input_data.get("text", "")
    domain = input_data.get("domain")
    chunks = input_data.get("chunks", [])

    if not text:
        return standard_response(
            "insights",
            status="error",
            context_found=False,
            grounded=False,
            error={"message": "No text provided for insight analysis"},
            correlation_id=correlation_id,
        )

    insights = generate_insights(text, domain=domain, chunks=chunks)
    rendered = render_insights(insights, domain=domain)

    return standard_response(
        "insights",
        result={
            "insights": [ins.to_dict() for ins in insights],
            "count": len(insights),
            "rendered": rendered,
            "has_critical": any(i.severity == "critical" for i in insights),
        },
        grounded=True,
        context_found=bool(insights),
        correlation_id=correlation_id,
    )
