import re
from typing import Dict, List, Tuple

_DOMAIN_KEYWORDS = {
    "hr": [
        "resume", "curriculum vitae", "cv", "experience", "skills", "education",
        "certification", "linkedin", "employment", "role", "candidate",
    ],
    "finance": [
        "invoice", "bill to", "due date", "subtotal", "amount due", "purchase order",
        "qty", "unit price", "balance", "payment terms",
    ],
    "medical": [
        "patient", "diagnosis", "medication", "rx", "icd", "vitals", "bp",
        "symptoms", "treatment", "clinic", "hospital",
    ],
    "legal": [
        "agreement", "contract", "party", "whereas", "hereby", "governing law",
        "liability", "indemn", "term", "effective date",
    ],
    "general": [],
}

_KIND_KEYWORDS = {
    "resume": ["resume", "curriculum vitae", "cv", "professional experience", "skills"],
    "linkedin_profile": ["linkedin", "connections", "linkedin.com", "endorsements"],
    "invoice": ["invoice", "bill to", "amount due", "subtotal", "line item"],
    "medical_document": ["patient", "diagnosis", "medication", "rx", "icd"],
    "contract": ["agreement", "contract", "party", "hereby", "governing law"],
    "general_doc": [],
}

_STRUCTURAL_CUES = {
    "resume": [r"\bexperience\b", r"\beducation\b", r"\bskills\b"],
    "linkedin_profile": [r"linkedin\.com", r"\bconnections\b"],
    "invoice": [r"\btotal\b", r"\bsubtotal\b", r"\bamount\s+due\b"],
    "medical_document": [r"\bdiagnos", r"\bmedicat", r"\bvitals\b"],
    "contract": [r"\bwhereas\b", r"\bhereby\b", r"\bparty\b"],
}


def _score_text(text: str, keywords: List[str], cues: List[str]) -> Tuple[float, List[str]]:
    score = 0.0
    signals: List[str] = []
    for kw in keywords:
        if kw in text:
            score += 1.0
            signals.append(kw)
    for cue in cues:
        if re.search(cue, text):
            score += 0.75
            signals.append(cue)
    return score, signals


def infer_doc_type(text: str) -> Dict[str, object]:
    normalized = (text or "").lower()
    domain_scores: Dict[str, float] = {}
    domain_signals: Dict[str, List[str]] = {}
    for domain, kws in _DOMAIN_KEYWORDS.items():
        score, signals = _score_text(normalized, kws, [])
        domain_scores[domain] = score
        domain_signals[domain] = signals

    kind_scores: Dict[str, float] = {}
    kind_signals: Dict[str, List[str]] = {}
    for kind, kws in _KIND_KEYWORDS.items():
        cues = _STRUCTURAL_CUES.get(kind, [])
        score, signals = _score_text(normalized, kws, cues)
        kind_scores[kind] = score
        kind_signals[kind] = signals

    best_domain = max(domain_scores.items(), key=lambda kv: kv[1])[0] if domain_scores else "general"
    best_kind = max(kind_scores.items(), key=lambda kv: kv[1])[0] if kind_scores else "general_doc"

    if kind_scores.get(best_kind, 0.0) <= 0.5 and domain_scores.get(best_domain, 0.0) <= 0.5:
        best_domain = "general"
        best_kind = "general_doc"

    combined_score = (domain_scores.get(best_domain, 0.0) + kind_scores.get(best_kind, 0.0))
    confidence = min(1.0, max(0.1, combined_score / 6.0))

    signals = sorted(set(domain_signals.get(best_domain, []) + kind_signals.get(best_kind, [])))

    return {
        "doc_domain": best_domain,
        "doc_kind": best_kind,
        "confidence": round(confidence, 2),
        "signals": signals,
    }


def infer_doc_domain_from_prompt(prompt: str) -> str:
    text = (prompt or "").lower()
    for domain, kws in _DOMAIN_KEYWORDS.items():
        if domain == "general":
            continue
        if any(kw in text for kw in kws):
            return domain
    return "general"


__all__ = ["infer_doc_type", "infer_doc_domain_from_prompt"]
