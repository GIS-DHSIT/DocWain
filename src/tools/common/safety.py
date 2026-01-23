from __future__ import annotations

from typing import Dict, List

MEDICAL_DISCLAIMER = (
    "This summary is for informational purposes only and is not a diagnosis, treatment recommendation, "
    "or medical advice. Always consult a licensed clinician for care decisions."
)

LEGAL_DISCLAIMER = (
    "This analysis is for general information only and is not legal advice. Consult a qualified attorney "
    "for advice on your situation."
)


def add_disclaimer(text: str, *, domain: str) -> str:
    disclaimer = MEDICAL_DISCLAIMER if domain == "medical" else LEGAL_DISCLAIMER
    if disclaimer not in text:
        return f"{text}\n\n{disclaimer}"
    return text


def refusal_response(domain: str) -> Dict[str, str]:
    if domain == "medical":
        return {
            "message": "I can only provide general health information and cannot offer diagnosis or treatment.",
            "policy": MEDICAL_DISCLAIMER,
        }
    return {
        "message": "I can summarize information but cannot provide legal advice.",
        "policy": LEGAL_DISCLAIMER,
    }


def collect_warnings(domain: str, extra: List[str] | None = None) -> List[str]:
    warnings = [MEDICAL_DISCLAIMER if domain == "medical" else LEGAL_DISCLAIMER]
    if extra:
        warnings.extend(extra)
    return warnings

