from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import ollama

logger = logging.getLogger(__name__)

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


_INVOICE_RE = re.compile(r"\b(invoice|bill to|total|vat|gst|due date|amount due)\b", re.IGNORECASE)
_RESUME_RE = re.compile(r"\b(resume|curriculum vitae|experience|education|skills|linkedin)\b", re.IGNORECASE)
_PO_RE = re.compile(r"\b(purchase order|po number|ship to|vendor)\b", re.IGNORECASE)
_CONTRACT_RE = re.compile(r"\b(agreement|terms|liability|confidentiality|governing law)\b", re.IGNORECASE)
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
    if "resume" in lower or "cv" in lower:
        return "resume"
    if "purchase" in lower or "po" in lower:
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
    if _INVOICE_RE.search(sample):
        return "invoice", 0.78
    if _RESUME_RE.search(sample):
        return "resume", 0.78
    if _PO_RE.search(sample):
        return "purchase_order", 0.76
    if _CONTRACT_RE.search(sample):
        return "contract", 0.72
    if _POLICY_RE.search(sample):
        return "policy", 0.68
    if _BROCHURE_RE.search(sample):
        return "brochure", 0.66
    if _REPORT_RE.search(sample):
        return "report", 0.64
    if _STATEMENT_RE.search(sample):
        return "statement", 0.64
    if _PRESENTATION_RE.search(sample):
        return "presentation", 0.62
    return None


def _ollama_classify(text: str, tables: str, filename: str, model_name: Optional[str]) -> Optional[tuple[str, float]]:
    if not model_name:
        return None
    prompt = (
        "Classify the document into one of the following types: "
        f"{', '.join(DOCUMENT_TAXONOMY)}. "
        "Use the provided text/tables/filename sample. "
        "Return strict JSON: {\"document_type\": "
        "\"invoice|resume|purchase_order|contract|policy|brochure|report|statement|presentation|other\", "
        "\"confidence\": 0.0-1.0}.\n\n"
        f"Filename: {filename}\n\nText sample:\n{text[:2000]}\n\nTables sample:\n{tables[:2000]}\n"
    )
    try:
        response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0})
        content = response.get("response", "").strip()
        payload = json.loads(content)
        doc_type = str(payload.get("document_type", "other")).strip()
        confidence = float(payload.get("confidence", 0.0))
        if doc_type not in DOCUMENT_TAXONOMY:
            doc_type = "other"
        return doc_type, max(0.0, min(confidence, 1.0))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama document classification failed: %s", exc)
        return None


def _extract_page_count(extracted: Any) -> Optional[int]:
    try:
        pages = [getattr(sec, "end_page", None) for sec in getattr(extracted, "sections", [])]
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
) -> tuple[str, float]:
    heuristic = _heuristic_classify(text_sample, tables_sample, filename)
    if heuristic:
        return heuristic

    llm = _ollama_classify(text_sample, tables_sample, filename, model_name)
    if llm:
        return llm

    return "other", 0.5


def identify_document(
    *,
    extracted: Any,
    filename: str,
    profile_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> DocumentIdentification:
    text_sample = ""
    tables_sample = ""
    try:
        if extracted and getattr(extracted, "full_text", None):
            text_sample = extracted.full_text
        elif extracted and getattr(extracted, "sections", None):
            text_sample = "\n".join([sec.text for sec in extracted.sections[:5] if sec.text])
        if extracted and getattr(extracted, "tables", None):
            tables_sample = "\n".join([tbl.text for tbl in extracted.tables[:3] if tbl.text])
    except Exception:  # noqa: BLE001
        text_sample = text_sample or ""

    doc_type, confidence = classify_document_type(text_sample, tables_sample, filename, model_name=model_name)

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
