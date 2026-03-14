"""Content verifier — post-generation grounding checks.

Validates that generated content is supported by source evidence.
Detects hallucinated facts, unsupported claims, and entity inconsistencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class VerificationResult:
    """Result of content verification against source evidence."""

    grounded: bool
    score: float  # 0.0–1.0, proportion of claims verified
    total_claims: int
    verified_claims: int
    unverified_claims: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded": self.grounded,
            "score": round(self.score, 3),
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Claim extraction from generated text
# ---------------------------------------------------------------------------

# Patterns for extractable factual claims
_NUMBER_RE = re.compile(r"\b(\d[\d,.]*)\b")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Words that indicate a factual claim when combined with specific data
_CLAIM_INDICATORS = re.compile(
    r"\b(?:has|have|earned|achieved|completed|holds?|managed|led|"
    r"worked|served|developed|created|increased|decreased|generated|"
    r"certified|licensed|graduated|totaling|amounting)\b",
    re.IGNORECASE,
)

# Compound number patterns: "1.5 million", "2.3 billion", etc.
_COMPOUND_NUM_RE = re.compile(
    r"(\d[\d,.]*)\s*(thousand|million|billion|trillion|[kKmMbB])\b",
    re.IGNORECASE,
)
_MAGNITUDE_MAP = {
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

# Common abbreviation expansions
_ABBREVIATION_MAP: Dict[str, str] = {
    "ceo": "chief executive officer",
    "cfo": "chief financial officer",
    "cto": "chief technology officer",
    "coo": "chief operating officer",
    "cmo": "chief marketing officer",
    "vp": "vice president",
    "svp": "senior vice president",
    "evp": "executive vice president",
    "hr": "human resources",
    "it": "information technology",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "mba": "master of business administration",
    "md": "doctor of medicine",
    "phd": "doctor of philosophy",
    "bs": "bachelor of science",
    "ba": "bachelor of arts",
    "ms": "master of science",
    "llb": "bachelor of laws",
    "llm": "master of laws",
    "roi": "return on investment",
    "kpi": "key performance indicator",
    "saas": "software as a service",
    "erp": "enterprise resource planning",
    "crm": "customer relationship management",
    "r&d": "research and development",
    "p&l": "profit and loss",
    "gdpr": "general data protection regulation",
    "hipaa": "health insurance portability and accountability act",
    "osha": "occupational safety and health administration",
}


def _normalize_compound_number(text: str) -> Optional[float]:
    """Parse compound numbers like '1.5 million' into 1500000.0."""
    m = _COMPOUND_NUM_RE.search(text)
    if not m:
        return None
    base = float(m.group(1).replace(",", ""))
    suffix = m.group(2).lower()
    multiplier = _MAGNITUDE_MAP.get(suffix, 1)
    return base * multiplier


def _numbers_match(a: str, b_text: str) -> bool:
    """Check if two numeric representations refer to the same value.

    Handles: '1,500,000' vs '1.5 million', '2.3M' vs '2,300,000', etc.
    """
    # Try parsing a as compound
    a_val = _normalize_compound_number(a)
    if a_val is None:
        try:
            a_val = float(a.replace(",", ""))
        except ValueError:
            return False

    # Search b_text for compound numbers and plain numbers
    for m in _COMPOUND_NUM_RE.finditer(b_text):
        base = float(m.group(1).replace(",", ""))
        suffix = m.group(2).lower()
        multiplier = _MAGNITUDE_MAP.get(suffix, 1)
        b_val = base * multiplier
        if abs(a_val - b_val) / max(abs(a_val), 1) < 0.01:
            return True

    # Also check plain numbers in b_text
    for m in _NUMBER_RE.finditer(b_text):
        try:
            b_val = float(m.group().replace(",", ""))
            if abs(a_val - b_val) / max(abs(a_val), 1) < 0.01:
                return True
        except ValueError:
            continue

    return False


def _extract_numbers(text: str) -> List[str]:
    """Extract numeric values from text."""
    return _NUMBER_RE.findall(text)


def _extract_dates(text: str) -> List[str]:
    """Extract date patterns from text."""
    return [m.group() for m in _DATE_RE.finditer(text)]


def _extract_named_entities(text: str) -> List[str]:
    """Extract capitalized multi-word entities (simple heuristic)."""
    # 2-4 consecutive capitalized words
    entities = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text)
    return list(set(entities))


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Content verifier
# ---------------------------------------------------------------------------


class ContentVerifier:
    """Verifies generated content against source evidence."""

    def __init__(self, *, strict: bool = False):
        """Initialize verifier.

        Args:
            strict: If True, mark as ungrounded when any claim is unverified.
                    If False (default), allow up to 20% unverified claims.
        """
        self._strict = strict

    def verify(
        self,
        generated_text: str,
        evidence_chunks: Sequence[Any],
        facts: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify generated content against source evidence.

        Args:
            generated_text: The LLM-generated text to verify.
            evidence_chunks: Source chunks used for generation.
            facts: Extracted facts dict (used as additional evidence).

        Returns:
            VerificationResult with grounding assessment.
        """
        if not generated_text or not generated_text.strip():
            return VerificationResult(
                grounded=False, score=0.0, total_claims=0, verified_claims=0,
                warnings=["Generated text is empty"],
            )

        # Build combined evidence text
        evidence_text = self._build_evidence_text(evidence_chunks, facts)
        evidence_normalized = _normalize_for_comparison(evidence_text)

        # Extract verifiable claims from generated text
        claims = self._extract_claims(generated_text)

        if not claims:
            # No specific claims to verify — check general consistency
            return VerificationResult(
                grounded=True, score=1.0, total_claims=0, verified_claims=0,
            )

        # Verify each claim
        verified = 0
        unverified: List[str] = []
        warnings: List[str] = []

        for claim in claims:
            if self._verify_claim(claim, evidence_normalized, evidence_text):
                verified += 1
            else:
                unverified.append(claim)

        total = len(claims)
        score = verified / total if total > 0 else 1.0
        threshold = 1.0 if self._strict else 0.8
        grounded = score >= threshold

        if not grounded:
            warnings.append(
                f"Grounding score {score:.1%} below threshold. "
                f"{len(unverified)} of {total} claims could not be verified."
            )

        # Check for hallucination indicators
        hallucination_warnings = self._check_hallucination_indicators(
            generated_text, evidence_text,
        )
        warnings.extend(hallucination_warnings)

        return VerificationResult(
            grounded=grounded,
            score=score,
            total_claims=total,
            verified_claims=verified,
            unverified_claims=unverified[:10],  # Cap to avoid huge lists
            warnings=warnings,
        )

    def _build_evidence_text(
        self,
        chunks: Sequence[Any],
        facts: Optional[Dict[str, Any]],
    ) -> str:
        """Combine all evidence sources into a single text."""
        parts: List[str] = []
        for chunk in chunks:
            text = ""
            if hasattr(chunk, "text"):
                text = chunk.text or ""
            elif isinstance(chunk, dict):
                text = chunk.get("text", "") or chunk.get("canonical_text", "") or ""
            if text:
                parts.append(text)

        if facts:
            for key, value in facts.items():
                if isinstance(value, list):
                    parts.extend(str(v) for v in value if v)
                elif value:
                    parts.append(str(value))

        return " ".join(parts)

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable factual claims from generated text."""
        claims: List[str] = []

        # Numbers (amounts, quantities, years)
        for num in _extract_numbers(text):
            claims.append(num)

        # Dates
        for date in _extract_dates(text):
            claims.append(date)

        # Emails and phones
        for email in _EMAIL_RE.findall(text):
            claims.append(email)
        for phone in _PHONE_RE.findall(text):
            claims.append(phone)

        # Named entities (check they appear in evidence)
        for entity in _extract_named_entities(text):
            claims.append(entity)

        return list(set(claims))

    def _verify_claim(
        self,
        claim: str,
        evidence_normalized: str,
        evidence_raw: str,
    ) -> bool:
        """Check if a single claim is supported by evidence."""
        claim_lower = claim.lower().strip()

        # Direct match in normalized evidence
        if claim_lower in evidence_normalized:
            return True

        # Direct match in raw evidence
        if claim.lower() in evidence_raw.lower():
            return True

        # For numbers, check if the number appears in evidence (including
        # compound forms like "1.5 million" ≈ "1,500,000")
        if claim.replace(",", "").replace(".", "").isdigit():
            plain_num = claim.replace(",", "")
            if plain_num in evidence_raw:
                return True
            # Compound number matching
            if _numbers_match(claim, evidence_raw):
                return True

        # Check compound numbers in the claim itself (e.g. "1.5 million")
        if _COMPOUND_NUM_RE.search(claim):
            if _numbers_match(claim, evidence_raw):
                return True

        # Date normalization: "March 15, 2024" ≈ "03/15/2024" ≈ "2024-03-15"
        if _DATE_RE.search(claim):
            if self._dates_match(claim, evidence_raw):
                return True

        # Percentage/fraction matching: "50%" ≈ "0.50" ≈ "half"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", claim)
        if pct_match:
            pct_val = float(pct_match.group(1))
            if self._percentage_in_evidence(pct_val, evidence_raw):
                return True

        # Abbreviation expansion: check if abbreviation or its expansion exists
        if claim_lower in _ABBREVIATION_MAP:
            expansion = _ABBREVIATION_MAP[claim_lower]
            if expansion in evidence_normalized:
                return True
        else:
            # Reverse: claim is an expansion, check if abbreviation is in evidence
            for abbrev, expansion in _ABBREVIATION_MAP.items():
                if claim_lower == expansion or expansion in claim_lower:
                    if abbrev in evidence_normalized:
                        return True

        # For named entities, require at least 2 words to match (not just any 1)
        # This prevents "Alice Cooper" matching evidence about "Bob Cooper"
        words = [w for w in claim.split() if len(w) >= 2]
        if len(words) >= 2:
            matched = sum(1 for w in words if w.lower() in evidence_normalized)
            if matched >= min(2, len(words)):
                return True

        return False

    @staticmethod
    def _dates_match(claim_date: str, evidence_text: str) -> bool:
        """Check if a date claim matches any date in evidence (format-agnostic).

        Handles: "March 15, 2024" ↔ "03/15/2024" ↔ "2024-03-15"
        """
        _MONTH_MAP = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }

        def _extract_date_parts(text: str) -> List[tuple]:
            """Extract (year, month, day) tuples from text."""
            parts = []
            # ISO/numeric: 2024-03-15, 03/15/2024, 15-03-2024
            for m in re.finditer(r"(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})", text):
                a, b, c = m.group(1), m.group(2), m.group(3)
                if len(a) == 4:
                    parts.append((a, b.zfill(2), c.zfill(2)))
                elif len(c) == 4:
                    parts.append((c, a.zfill(2), b.zfill(2)))
            # Named: March 15, 2024 / 15 March 2024
            for m in re.finditer(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
                r"\s+(\d{1,2}),?\s+(\d{4})", text, re.I
            ):
                month = _MONTH_MAP.get(m.group(1)[:3].lower(), "00")
                parts.append((m.group(3), month, m.group(2).zfill(2)))
            return parts

        claim_parts = _extract_date_parts(claim_date)
        evidence_parts = _extract_date_parts(evidence_text)
        for cp in claim_parts:
            if cp in evidence_parts:
                return True
        return False

    @staticmethod
    def _percentage_in_evidence(pct_val: float, evidence_text: str) -> bool:
        """Check if a percentage value exists in evidence in any form.

        50% ↔ 0.50 ↔ 0.5 ↔ "half"
        """
        _PCT_WORDS = {"half": 50.0, "quarter": 25.0, "third": 33.33}

        # Direct percentage match
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", evidence_text):
            if abs(float(m.group(1)) - pct_val) < 0.1:
                return True
        # Decimal fraction: 0.50 for 50%
        decimal = pct_val / 100.0
        for m in re.finditer(r"\b0\.(\d{1,3})\b", evidence_text):
            ev_decimal = float(f"0.{m.group(1)}")
            if abs(ev_decimal - decimal) < 0.005:
                return True
        # Word equivalents
        for word, val in _PCT_WORDS.items():
            if word in evidence_text.lower() and abs(val - pct_val) < 1.0:
                return True
        return False

    def _check_hallucination_indicators(
        self,
        generated: str,
        evidence: str,
    ) -> List[str]:
        """Check for common hallucination patterns."""
        warnings: List[str] = []
        generated_lower = generated.lower()
        evidence_lower = evidence.lower()

        # Check for specific factual phrases not in evidence
        specific_phrases = [
            r"according to (?:the|our) records",
            r"as stated in the document",
            r"the document (?:clearly |explicitly )?(?:states|mentions|indicates)",
        ]
        for pattern in specific_phrases:
            for match in re.finditer(pattern, generated_lower):
                # These attribution phrases are fine — don't flag them
                pass

        # Check for invented email addresses
        gen_emails = set(_EMAIL_RE.findall(generated))
        ev_emails = set(_EMAIL_RE.findall(evidence))
        fabricated_emails = gen_emails - ev_emails
        if fabricated_emails:
            warnings.append(
                f"Email(s) not found in evidence: {', '.join(fabricated_emails)}"
            )

        # Check for invented phone numbers
        gen_phones = set(_PHONE_RE.findall(generated))
        ev_phones = set(_PHONE_RE.findall(evidence))
        fabricated_phones = gen_phones - ev_phones
        if fabricated_phones:
            warnings.append(
                f"Phone number(s) not found in evidence: {', '.join(fabricated_phones)}"
            )

        return warnings
