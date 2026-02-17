from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


DOMAIN_LABELS = [
    "resume",
    "tax",
    "invoice",
    "purchase_order",
    "bank_statement",
    "medical",
    "legal",
    "generic",
]

DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "resume": (
        "resume",
        "curriculum vitae",
        "cv",
        "experience",
        "work history",
        "employment",
        "education",
        "skills",
        "projects",
        "certification",
        "summary",
        "objective",
        "technical skills",
        "professional experience",
        "career objective",
        "work experience",
        "key skills",
        "professional summary",
        "career",
        "achievements",
        "accomplishments",
        "qualifications",
        "soft skills",
        "functional skills",
    ),
    "tax": (
        "tax",
        "irs",
        "w-2",
        "w2",
        "1099",
        "schedule c",
        "filing",
        "return",
        "taxable",
        "refund",
        "withholding",
    ),
    "invoice": (
        "invoice",
        "bill to",
        "amount due",
        "subtotal",
        "total",
        "balance due",
        "due date",
        "unit price",
        "line item",
    ),
    "purchase_order": (
        "purchase order",
        "order date",
        "qty",
        "delivery date",
        "requisition",
        "purchase requisition",
        "po number",
    ),
    "bank_statement": (
        "bank statement",
        "account number",
        "routing number",
        "statement period",
        "opening balance",
        "closing balance",
        "transaction",
        "debit",
        "credit",
        "withdrawal",
        "deposit",
    ),
    "medical": (
        "patient",
        "diagnosis",
        "medical",
        "medication",
        "prescription",
        "procedure",
        "lab results",
        "symptom",
        "clinical",
    ),
    "legal": (
        "agreement",
        "contract",
        "clause",
        "indemnify",
        "liability",
        "governing law",
        "arbitration",
        "whereas",
        "hereby",
    ),
    "generic": (),
}

DOC_TYPE_HINTS = {
    "cv": "resume",
    "resume": "resume",
    "invoice": "invoice",
    "po": "purchase_order",
    "purchase order": "purchase_order",
    "bank statement": "bank_statement",
    "statement": "bank_statement",
    "medical": "medical",
    "contract": "legal",
    "agreement": "legal",
    "tax": "tax",
}


@dataclass(frozen=True)
class DomainClassification:
    domain: str
    confidence: float
    scores: Dict[str, float]
    method: str
    uncertain: bool


# Strong indicator phrases that are nearly unambiguous for a domain (weighted 2.0x)
_STRONG_INDICATORS: Dict[str, Tuple[str, ...]] = {
    "invoice": (
        "invoice number", "invoice date", "purchase order number",
        "amount due", "bill to", "remittance",
    ),
    "legal": (
        "party of the first part", "hereinafter referred to",
        "in witness whereof", "governing law", "indemnification",
    ),
    "medical": (
        "chief complaint", "medical history", "review of systems",
        "history of present illness", "physical examination",
    ),
    "bank_statement": (
        "account statement", "available balance", "statement period",
        "opening balance", "closing balance",
    ),
}


def _score_keywords(text: str) -> Dict[str, float]:
    import re as _re
    lowered = (text or "").lower()
    scores = {domain: 0.0 for domain in DOMAIN_LABELS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if not keyword:
                continue
            # Short keywords (<=3 chars) need word boundaries to avoid false matches
            if len(keyword) <= 3:
                if _re.search(r'\b' + _re.escape(keyword) + r'\b', lowered):
                    scores[domain] += 1.0
            elif keyword in lowered:
                scores[domain] += 1.0
    # Apply strong indicator bonuses (2.0x weight)
    for domain, phrases in _STRONG_INDICATORS.items():
        for phrase in phrases:
            if phrase in lowered:
                scores[domain] += 2.0
    return scores


def _score_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, float]:
    scores = {domain: 0.0 for domain in DOMAIN_LABELS}
    if not metadata:
        return scores
    doc_type = str(metadata.get("doc_type") or metadata.get("document_type") or "").lower().strip()
    source_name = str(metadata.get("source_name") or metadata.get("filename") or "").lower().strip()
    for hint, domain in DOC_TYPE_HINTS.items():
        if hint and hint in doc_type:
            scores[domain] += 2.0
        if hint and hint in source_name:
            scores[domain] += 1.0
    return scores


def _merge_scores(*score_maps: Dict[str, float]) -> Dict[str, float]:
    merged = {domain: 0.0 for domain in DOMAIN_LABELS}
    for scores in score_maps:
        for domain, score in scores.items():
            merged[domain] = merged.get(domain, 0.0) + float(score or 0.0)
    return merged


def _pick_domain(scores: Dict[str, float]) -> Tuple[str, float, bool]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score <= 0:
        return "generic", 0.0, True
    confidence = best_score / max(best_score + second_score, 1.0)
    # Lower threshold when a strong indicator contributed (score includes 2.0x bonuses)
    min_score_threshold = 1.5 if best_score >= 4.0 else 2.0
    uncertain = confidence < 0.55 or best_score < min_score_threshold
    return best_domain, confidence, uncertain


def classify_domain(
    document_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    llm_labeler: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
) -> DomainClassification:
    keyword_scores = _score_keywords(document_text)
    metadata_scores = _score_metadata(metadata)
    scores = _merge_scores(keyword_scores, metadata_scores)

    domain, confidence, uncertain = _pick_domain(scores)
    method = "rules"

    if uncertain and llm_labeler:
        try:
            llm_domain = llm_labeler(document_text, metadata or {})
            if llm_domain and llm_domain in DOMAIN_LABELS:
                domain = llm_domain
                confidence = max(confidence, 0.6)
                method = "llm"
                uncertain = False
        except Exception:
            pass

    if domain not in DOMAIN_LABELS:
        domain = "generic"
        confidence = min(confidence, 0.4)
        uncertain = True

    return DomainClassification(
        domain=domain,
        confidence=confidence,
        scores=scores,
        method=method,
        uncertain=uncertain,
    )


def infer_domain(document_text: str, *, metadata: Optional[Dict[str, Any]] = None) -> str:
    return classify_domain(document_text, metadata).domain


__all__ = ["DOMAIN_LABELS", "DomainClassification", "classify_domain", "infer_domain"]
