from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = get_logger(__name__)

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Access attribute or dict key — supports both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

DOCUMENT_TAXONOMY = [
    "resume",
    "invoice",
    "purchase_order",
    "contract",
    "policy",
    "brochure",
    "report",
    "statement",
    "presentation",
    "other",
]

@dataclass(frozen=True)
class DocumentIdentification:
    doc_name: str
    document_type: str
    confidence: float
    language: Optional[str]
    page_count: Optional[int]
    file_format: Optional[str]
    created_date: Optional[str]

# Multi-word phrases preferred to avoid single-word false positives (e.g. "total" in resumes)
_INVOICE_RE = re.compile(r"\b(invoice\s+number|invoice\s+date|bill\s+to|amount\s+due|total\s+due|payment\s+terms|vat|gst|remittance|subtotal|invoice)\b", re.IGNORECASE)
_RESUME_RE = re.compile(r"\b(resume|curriculum vitae|work\s+experience|professional\s+experience|education|skills|linkedin|career\s+objective|professional\s+summary)\b", re.IGNORECASE)
_PO_RE = re.compile(r"\b(purchase order|po number|ship to|vendor)\b", re.IGNORECASE)
_CONTRACT_RE = re.compile(r"\b(agreement|terms and conditions|liability|confidentiality|governing law)\b", re.IGNORECASE)
_POLICY_RE = re.compile(r"\b(policy|procedure|compliance|guideline)\b", re.IGNORECASE)
_BROCHURE_RE = re.compile(r"\b(brochure|catalog|features|overview)\b", re.IGNORECASE)
_REPORT_RE = re.compile(r"\b(report|analysis|findings|executive summary)\b", re.IGNORECASE)
_STATEMENT_RE = re.compile(r"\b(statement|account summary|balance forward)\b", re.IGNORECASE)
_PRESENTATION_RE = re.compile(r"\b(slide|agenda|presentation)\b", re.IGNORECASE)

def _guess_from_filename(filename: str) -> Optional[str]:
    lower = (filename or "").lower()
    if not lower:
        return None
    if "invoice" in lower:
        return "invoice"
    if "resume" in lower or re.search(r"\bcv\b", lower):
        return "resume"
    if "purchase" in lower or re.search(r"\bpo\b", lower):
        return "purchase_order"
    if "contract" in lower or "agreement" in lower:
        return "contract"
    if "policy" in lower:
        return "policy"
    if "brochure" in lower:
        return "brochure"
    if "report" in lower:
        return "report"
    if "statement" in lower:
        return "statement"
    if "ppt" in lower or "presentation" in lower:
        return "presentation"
    return None

def _heuristic_classify(text: str, tables: str, filename: str) -> Optional[tuple[str, float]]:
    candidate = _guess_from_filename(filename)
    if candidate:
        return candidate, 0.72

    sample = "\n".join([text or "", tables or ""]).lower()

    # Score-based: count matches per type, pick the highest
    _PATTERNS = [
        ("invoice", _INVOICE_RE, 0.78),
        ("resume", _RESUME_RE, 0.78),
        ("purchase_order", _PO_RE, 0.76),
        ("contract", _CONTRACT_RE, 0.72),
        ("policy", _POLICY_RE, 0.68),
        ("brochure", _BROCHURE_RE, 0.66),
        ("report", _REPORT_RE, 0.64),
        ("statement", _STATEMENT_RE, 0.64),
        ("presentation", _PRESENTATION_RE, 0.62),
    ]
    best_type: Optional[str] = None
    best_count = 0
    best_conf = 0.0
    for doc_type, pattern, base_conf in _PATTERNS:
        matches = pattern.findall(sample)
        count = len(set(m.lower() if isinstance(m, str) else m for m in matches))
        if count > best_count:
            best_count = count
            best_type = doc_type
            best_conf = base_conf
    if best_type and best_count >= 1:
        return best_type, best_conf
    return None

def _ollama_classify(text: str, tables: str, filename: str, model_name: Optional[str] = None, llm_client=None) -> Optional[tuple[str, float]]:
    prompt = (
        "You are a document classification expert. Classify this document into exactly one type.\n"
        f"Allowed types: {', '.join(DOCUMENT_TAXONOMY)}.\n\n"
        "Key rules:\n"
        "- 'resume' = CV, candidate profile, job application, career summary\n"
        "- 'invoice' = bill, payment request with line items, amounts due\n"
        "- Do NOT classify a resume as invoice just because it mentions 'total experience'\n"
        "- Use filename as a strong signal when available\n\n"
        "Return strict JSON only: {\"document_type\": \"<type>\", \"confidence\": 0.0-1.0}\n\n"
        f"Filename: {filename}\n\nText sample:\n{text[:3000]}\n\nTables sample:\n{tables[:1000]}\n"
    )
    try:
        if llm_client is not None:
            content = llm_client.generate(prompt)
        else:
            from src.llm.clients import get_local_client
            client = get_local_client()
            content = client.generate(prompt)
        content = (content or "").strip()
        # Extract JSON from possible markdown code block
        if "```" in content:
            import re as _re
            json_match = _re.search(r"\{[^}]+\}", content)
            if json_match:
                content = json_match.group()
        payload = json.loads(content)
        doc_type = str(payload.get("document_type", "other")).strip().lower()
        confidence = float(payload.get("confidence", 0.0))
        if doc_type not in DOCUMENT_TAXONOMY:
            doc_type = "other"
        return doc_type, max(0.0, min(confidence, 1.0))
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM document classification failed: %s", exc)
        return None

def _extract_page_count(extracted: Any) -> Optional[int]:
    try:
        pages = [_get(sec, "end_page") for sec in (_get(extracted, "sections") or [])]
        pages = [p for p in pages if isinstance(p, int)]
        if pages:
            return max(pages)
    except Exception:  # noqa: BLE001
        return None
    return None

def classify_document_type(
    text_sample: str,
    tables_sample: str,
    filename: str,
    model_name: Optional[str] = None,
    llm_client=None,
) -> tuple[str, float]:
    # LLM-first: DocWain-Agent provides accurate classification with full context understanding
    llm = _ollama_classify(text_sample, tables_sample, filename, model_name, llm_client=llm_client)
    if llm:
        doc_type, conf = llm
        logger.info("LLM classified document '%s' as %s (confidence=%.2f)", filename, doc_type, conf)
        return llm

    # Heuristic fallback when LLM is unavailable
    heuristic = _heuristic_classify(text_sample, tables_sample, filename)
    if heuristic:
        logger.info("Heuristic classified document '%s' as %s", filename, heuristic[0])
        return heuristic

    return "other", 0.5

def identify_document(
    *,
    extracted: Any,
    filename: str,
    profile_name: Optional[str] = None,
    model_name: Optional[str] = None,
    llm_client=None,
) -> DocumentIdentification:
    text_sample = ""
    tables_sample = ""
    try:
        ft = _get(extracted, "full_text") if extracted else None
        if ft:
            text_sample = ft
        else:
            secs = (_get(extracted, "sections") or []) if extracted else []
            text_sample = "\n".join([_get(s, "text", "") for s in secs[:5] if _get(s, "text")])
        tbls = (_get(extracted, "tables") or []) if extracted else []
        if tbls:
            tables_sample = "\n".join([_get(t, "text", "") for t in tbls[:3] if _get(t, "text")])
    except Exception:  # noqa: BLE001
        text_sample = text_sample or ""

    doc_type, confidence = classify_document_type(text_sample, tables_sample, filename, model_name=model_name, llm_client=llm_client)

    # Normalize to canonical domain labels for consistency across all classifiers
    try:
        from src.intelligence.domain_classifier import normalize_domain
        doc_type = normalize_domain(doc_type)
    except ImportError:
        pass

    file_format = filename.split(".")[-1].lower() if filename and "." in filename else None

    return DocumentIdentification(
        doc_name=filename or "Untitled Document",
        document_type=doc_type,
        confidence=confidence,
        language=None,
        page_count=_extract_page_count(extracted),
        file_format=file_format,
        created_date=None,
    )

__all__ = [
    "DocumentIdentification",
    "classify_document_type",
    "identify_document",
    "DOCUMENT_TAXONOMY",
]
