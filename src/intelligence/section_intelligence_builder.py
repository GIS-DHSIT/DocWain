from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.embedding.chunking.section_chunker import normalize_text
from src.intelligence.domain_classifier import classify_domain
from src.kg.entity_extractor import EntityExtractor

logger = get_logger(__name__)

SECTION_KIND_TAXONOMY = [
    "identity_contact",
    "taxpayer_identity",
    "account_identity",
    "summary_objective",
    "experience",
    "projects",
    "education",
    "certifications",
    "skills_technical",
    "skills_functional",
    "tools_technologies",
    "achievements_awards",
    "publications_patents",
    "leadership_management",
    "compliance_regulatory",
    "invoice_metadata",
    "financial_summary",
    "transactions",
    "balances",
    "fees",
    "line_items",
    "parties_addresses",
    "terms_conditions",
    "diagnoses_procedures",
    "medications",
    "lab_results",
    "notes",
    "ids",
    "totals",
    "deductions",
    "ay_fy",
    "payments",
    "attachments",
    "tables",
    "misc",
]

SECTION_KIND_LEXICON: Dict[str, Tuple[str, ...]] = {
    "identity_contact": ("contact", "address", "phone", "email", "profile", "candidate"),
    "taxpayer_identity": ("taxpayer", "taxpayer information", "taxpayer name", "taxpayer address", "ssn", "ein", "tin"),
    "account_identity": ("account", "account number", "routing", "holder", "customer", "bank"),
    "summary_objective": ("summary", "objective", "overview", "profile"),
    "experience": ("experience", "employment", "work history", "professional experience"),
    "projects": ("projects", "project", "portfolio"),
    "education": ("education", "university", "college", "degree", "school"),
    "certifications": ("certification", "certified", "license"),
    "skills_technical": ("technical skills", "skills", "programming", "software", "languages"),
    "skills_functional": ("functional skills", "competencies", "expertise"),
    "tools_technologies": ("tools", "technologies", "platforms", "systems"),
    "achievements_awards": ("awards", "achievements", "honors"),
    "publications_patents": ("publications", "patents", "research"),
    "leadership_management": ("leadership", "management", "managed", "team"),
    "compliance_regulatory": ("compliance", "regulatory", "policy", "audit"),
    "invoice_metadata": ("invoice", "invoice number", "invoice date", "due date", "bill date"),
    "financial_summary": ("total", "subtotal", "balance", "amount due", "summary", "financial"),
    "transactions": ("transactions", "transaction", "debit", "credit", "statement"),
    "balances": ("balance", "opening balance", "closing balance", "available balance"),
    "fees": ("fee", "fees", "service charge", "charges", "interest", "commission"),
    "line_items": ("line items", "items", "qty", "quantity", "unit price"),
    "parties_addresses": ("bill to", "ship to", "vendor", "supplier", "customer", "buyer"),
    "terms_conditions": ("terms", "conditions", "payment", "due date", "net"),
    "diagnoses_procedures": ("diagnosis", "procedure", "assessment", "treatment"),
    "medications": ("medications", "prescription", "rx", "dose"),
    "lab_results": ("lab results", "laboratory", "test results"),
    "notes": ("notes", "remarks", "comments"),
    "ids": ("id", "identifier", "ssn", "ein", "pan", "tax id", "tin"),
    "totals": ("total tax", "tax liability", "amount due", "total", "balance due"),
    "deductions": ("deduction", "deductions", "exemption", "allowance", "write-off"),
    "ay_fy": ("assessment year", "ay", "fy", "fiscal year", "tax year"),
    "payments": ("payment", "payments", "paid", "refund", "withholding"),
    "attachments": ("attachments", "appendix", "annex"),
    "tables": ("table", "tabular"),
}

_HEADING_RE = re.compile(
    r"^(?:chapter\b|section\b|appendix\b|\d+(?:\.\d+)+|\d+\.|[ivxlcdm]+\.)\s+.+",
    re.IGNORECASE,
)
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s,:\-]{4,}$")
_COLON_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\s\-/]{2,}:\s*$")

_AMOUNT_RE = re.compile(r"(?i)\b(?:USD|EUR|GBP|INR|AUD|CAD|SGD|JPY)?\s*[$€£]?\s*\d+(?:[,\d]*)(?:\.\d{2})?\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
    re.IGNORECASE,
)
_INVOICE_RE = re.compile(r"(?i)\b(?:invoice|inv)\s*#?:?\s*([A-Za-z0-9\-]{3,})\b")
_PO_RE = re.compile(r"(?i)\b(?:purchase\s+order|po)\s*#?:?\s*([A-Za-z0-9\-]{3,})\b")
_ACCOUNT_RE = re.compile(r"(?i)\b(?:account|acct)\s*(?:number|no\.?)\s*#?:?\s*([A-Za-z0-9\-]{4,})\b")
_TOTAL_RE = re.compile(r"(?i)\b(?:total|amount due|balance due|grand total)\s*[:\-]?\s*([$€£]?\s?\d[\d,]*\.?\d{0,2})")
_DUE_DATE_RE = re.compile(r"(?i)\b(?:due date|payment due)\s*[:\-]?\s*([A-Za-z0-9,/\- ]{4,})")
_AGE_RE = re.compile(r"(?i)\b(?:age|aged)\s*[:\-]?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:yo|yrs?|years?\s*old)\b")
_SEX_RE = re.compile(r"(?i)\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b")

_GARBAGE_TOKENS = {
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "nil",
    "-",
    "--",
    "not applicable",
}

_INVOICE_LINE_ITEM_RE = re.compile(r"(?i)\b(?:qty|quantity|unit price|amount)\b")

_EXTRACTOR_THRESHOLDS = {
    "invoice_number": 0.7,
    "purchase_order_number": 0.7,
    "account_number": 0.65,
}

DOMAIN_SECTION_KINDS: Dict[str, Tuple[str, ...]] = {
    "resume": (
        "identity_contact",
        "experience",
        "education",
        "skills_technical",
        "skills_functional",
        "tools_technologies",
        "certifications",
        "projects",
        "achievements_awards",
    ),
    "medical": ("identity_contact", "diagnoses_procedures", "medications", "lab_results", "notes"),
    "invoice": ("parties_addresses", "invoice_metadata", "line_items", "financial_summary", "terms_conditions"),
    "purchase_order": ("parties_addresses", "invoice_metadata", "line_items", "financial_summary", "terms_conditions"),
    "tax": ("taxpayer_identity", "ids", "totals", "deductions", "ay_fy", "payments"),
    "bank_statement": ("account_identity", "transactions", "balances", "fees"),
}

@dataclass
class SectionDescriptor:
    section_id: str
    section_title: str
    section_kind: str
    section_path: str
    page_range: Optional[Tuple[int, int]]
    raw_text: str
    confidence: float
    salience: float = 0.5
    start_index: int = 0
    end_index: int = 0

@dataclass
class SectionIntelligenceResult:
    doc_domain: str
    sections: List[SectionDescriptor]
    section_facts: List[Dict[str, Any]]
    section_summaries: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_domain": self.doc_domain,
            "sections": [section.__dict__ for section in self.sections],
            "section_facts": self.section_facts,
            "section_summaries": self.section_summaries,
        }

def _is_heading_line(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _HEADING_RE.match(clean):
        return True
    if _ALL_CAPS_RE.match(clean):
        return True
    if _COLON_RE.match(clean):
        return True
    if len(clean.split()) <= 6 and clean.endswith(":"):
        return True
    return False

def _hash_section_id(document_id: str, title: str, index: int) -> str:
    seed = f"{document_id}|{title}|{index}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]

def _section_kind_score(section_title: str, text: str, allowed_kinds: Optional[Iterable[str]] = None) -> Dict[str, float]:
    kinds = list(allowed_kinds) if allowed_kinds else SECTION_KIND_TAXONOMY
    scores = {kind: 0.0 for kind in kinds}
    lowered_title = (section_title or "").lower()
    lowered_text = (text or "").lower()
    for kind, keywords in SECTION_KIND_LEXICON.items():
        if kind not in scores:
            continue
        for keyword in keywords:
            if keyword in lowered_title:
                scores[kind] += 2.5
            if keyword in lowered_text:
                scores[kind] += 1.0
    return scores

def _infer_section_kind(section_title: str, text: str, doc_domain: str) -> str:
    allowed = DOMAIN_SECTION_KINDS.get(doc_domain)
    scores = _section_kind_score(section_title, text, allowed_kinds=allowed)
    best_kind, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score <= 0:
        # Heuristic fallback based on domain cues
        if doc_domain in {"invoice", "purchase_order"}:
            if "invoice" in (text or "").lower():
                return "invoice_metadata"
            if _TOTAL_RE.search(text):
                return "financial_summary"
            if "bill to" in (text or "").lower() or "ship to" in (text or "").lower():
                return "parties_addresses"
            if _is_line_item_line(text):
                return "line_items"
        if doc_domain == "bank_statement":
            lowered = (text or "").lower()
            if "transaction" in lowered:
                return "transactions"
            if "balance" in lowered:
                return "balances"
            if "fee" in lowered or "charge" in lowered:
                return "fees"
        if doc_domain == "tax":
            lowered = (text or "").lower()
            if "deduction" in lowered:
                return "deductions"
            if "tax year" in lowered or "assessment year" in lowered or "fiscal year" in lowered:
                return "ay_fy"
            if "payment" in lowered or "refund" in lowered:
                return "payments"
            if "total" in lowered or "amount due" in lowered:
                return "totals"
        if doc_domain == "medical":
            if "medication" in (text or "").lower():
                return "medications"
        fallback = "misc"
        if allowed and fallback not in allowed:
            return allowed[0]
        return fallback
    if doc_domain == "resume" and best_kind == "misc":
        refined = _refine_resume_misc(section_title, text)
        if refined:
            return refined
    return best_kind

def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    return paragraphs

def _segment_by_headings(text: str, fallback_title: str = "Section") -> List[Tuple[str, str]]:
    normalized = normalize_text(text or "")
    if not normalized:
        return []
    sections: List[Tuple[str, str]] = []
    current_title = fallback_title
    current_lines: List[str] = []

    def _flush(title: str, lines: List[str]) -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((title, body))

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            current_lines.append("")
            continue
        if _is_heading_line(line):
            if current_lines:
                _flush(current_title, current_lines)
                current_lines = []
            current_title = line.strip(": ").strip() or fallback_title
            continue
        current_lines.append(raw_line)

    _flush(current_title, current_lines)
    return sections

def _segment_by_paragraphs(text: str, *, group_size: int = 3) -> List[Tuple[str, str]]:
    paragraphs = _split_paragraphs(text or "")
    if not paragraphs:
        return []
    sections: List[Tuple[str, str]] = []
    for idx in range(0, len(paragraphs), max(1, group_size)):
        block = paragraphs[idx : idx + group_size]
        body = "\n\n".join(block).strip()
        if not body:
            continue
        title_hint = block[0].splitlines()[0].strip() if block else ""
        title = title_hint[:48] if title_hint else f"Section {len(sections) + 1}"
        sections.append((title, body))
    return sections

def _is_line_item_line(line: str) -> bool:
    if not line:
        return False
    if _INVOICE_LINE_ITEM_RE.search(line):
        return True
    numeric_hits = len(re.findall(r"\d+(?:\.\d+)?", line))
    if numeric_hits >= 2 and len(line.split()) >= 4:
        return True
    if re.search(r"\s{2,}\S+\s{2,}\S+", line):
        return True
    return False

def _derive_invoice_sections(document_text: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in (document_text or "").splitlines() if ln.strip()]
    buckets: Dict[str, List[str]] = {
        "invoice_metadata": [],
        "parties_addresses": [],
        "line_items": [],
        "financial_summary": [],
        "terms_conditions": [],
    }
    for line in lines:
        lowered = line.lower()
        if "invoice" in lowered or "inv" in lowered or "bill date" in lowered:
            buckets["invoice_metadata"].append(line)
        if any(token in lowered for token in ["bill to", "ship to", "vendor", "supplier", "customer"]):
            buckets["parties_addresses"].append(line)
        if _is_line_item_line(line):
            buckets["line_items"].append(line)
        if any(token in lowered for token in ["total", "amount due", "balance due", "subtotal"]):
            buckets["financial_summary"].append(line)
        if any(token in lowered for token in ["terms", "payment", "due date", "net"]):
            buckets["terms_conditions"].append(line)
    return {k: "\n".join(v) for k, v in buckets.items() if v}

def _refine_resume_misc(section_title: str, text: str) -> Optional[str]:
    lowered = f"{section_title} {text}".lower()
    if any(token in lowered for token in ["president", "director", "lead", "manager", "supervisor", "leadership"]):
        return "leadership_management"
    if any(token in lowered for token in ["award", "honor", "achievement"]):
        return "achievements_awards"
    if any(token in lowered for token in ["certification", "certified", "license"]):
        return "certifications"
    if any(token in lowered for token in ["skill", "tools", "technologies", "stack"]):
        return "skills_technical"
    return None

def _inject_invoice_sections(
    sections: List[SectionDescriptor],
    *,
    document_text: str,
    document_id: str,
) -> List[SectionDescriptor]:
    derived = _derive_invoice_sections(document_text)
    if not derived:
        return sections
    existing = {sec.section_kind for sec in sections}
    for kind, body in derived.items():
        if kind in existing:
            continue
        title = kind.replace("_", " ").title()
        section_id = _hash_section_id(document_id, title, len(sections))
        sections.append(
            SectionDescriptor(
                section_id=section_id,
                section_title=title,
                section_kind=kind,
                section_path=title,
                page_range=None,
                raw_text=body[:4000],
                confidence=0.55,
                start_index=0,
                end_index=0,
            )
        )
    return sections

def _group_chunks_by_section(
    chunk_texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
) -> List[Tuple[str, int, int, str, Optional[Tuple[int, int]]]]:
    if not chunk_texts:
        return []
    groups: List[Tuple[str, int, int, str, Optional[Tuple[int, int]]]] = []
    current_title = None
    start_idx = 0
    buffer: List[str] = []
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    def _flush(title: str, start: int, end: int, texts: List[str], pages: Tuple[Optional[int], Optional[int]]) -> None:
        body = "\n".join([t for t in texts if t]).strip()
        if not body:
            return
        page_range = None
        if pages[0] is not None or pages[1] is not None:
            page_range = (pages[0] or pages[1] or 0, pages[1] or pages[0] or 0)
        groups.append((title, start, end, body, page_range))

    for idx, (text, meta) in enumerate(zip(chunk_texts, chunk_metadata)):
        title = (meta.get("section_title") or meta.get("section") or "Untitled Section").strip() or "Untitled Section"
        if current_title is None:
            current_title = title
            start_idx = idx
        if title != current_title:
            _flush(current_title, start_idx, idx - 1, buffer, (page_start, page_end))
            current_title = title
            start_idx = idx
            buffer = []
            page_start = None
            page_end = None
        buffer.append(text or "")
        page_val = meta.get("page_start") or meta.get("page") or meta.get("page_number")
        if page_val is not None:
            page_start = page_val if page_start is None else min(page_start, page_val)
            page_end = page_val if page_end is None else max(page_end, page_val)

    if current_title is not None:
        _flush(current_title, start_idx, len(chunk_texts) - 1, buffer, (page_start, page_end))
    return groups

def _ensure_min_sections(
    sections: List[SectionDescriptor],
    *,
    chunk_texts: List[str],
    doc_domain: str,
    document_id: str,
    min_sections: int,
) -> List[SectionDescriptor]:
    if len(sections) >= min_sections:
        return sections
    if not chunk_texts:
        return sections
    if len(chunk_texts) >= min_sections:
        step = max(1, len(chunk_texts) // min_sections)
        new_sections: List[SectionDescriptor] = []
        for idx in range(min_sections):
            start = idx * step
            end = min(len(chunk_texts) - 1, (idx + 1) * step - 1)
            if idx == min_sections - 1:
                end = len(chunk_texts) - 1
            raw_text = "\n".join(chunk_texts[start : end + 1])
            title = f"Section {idx + 1}"
            section_kind = _infer_section_kind(title, raw_text, doc_domain)
            section_id = _hash_section_id(document_id, title, idx)
            new_sections.append(
                SectionDescriptor(
                    section_id=section_id,
                    section_title=title,
                    section_kind=section_kind,
                    section_path=title,
                    page_range=None,
                    raw_text=raw_text,
                    confidence=0.35,
                    start_index=start,
                    end_index=end,
                )
            )
        return new_sections
    return sections

def _apply_section_salience(sections: List[SectionDescriptor], total_chars: int) -> None:
    if not sections:
        return
    denom = max(total_chars, 1)
    for section in sections:
        length = len(section.raw_text or "")
        section.salience = max(0.05, min(1.0, length / denom))

def _build_chunk_map(
    chunk_texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chunk_map: List[Dict[str, Any]] = []
    for text, meta in zip(chunk_texts, chunk_metadata):
        chunk_map.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "page": meta.get("page_start") or meta.get("page") or meta.get("page_number"),
                "text": text or "",
            }
        )
    return chunk_map

def _find_evidence_span(value: str, chunk_map: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    needle = str(value)
    if not needle.strip():
        return None
    for entry in chunk_map:
        text = entry.get("text") or ""
        idx = text.lower().find(needle.lower())
        if idx != -1:
            return {
                "chunk_id": entry.get("chunk_id"),
                "page": entry.get("page"),
                "start_char": idx,
                "end_char": idx + len(needle),
            }
    return None

def _normalize_token(value: str) -> str:
    return re.sub(r"[^\w\-]", "", str(value or "")).strip()

def _score_identifier(value: str, field: str) -> float:
    token = _normalize_token(value)
    if not token:
        return 0.0
    if token.lower() in _GARBAGE_TOKENS:
        return 0.0
    if len(token) < 6:
        return 0.0
    digit_count = sum(1 for ch in token if ch.isdigit())
    if digit_count == 0:
        return 0.0
    digit_ratio = digit_count / max(len(token), 1)
    if digit_ratio < 0.2:
        return 0.0

    score = 0.6 + min(0.25, digit_ratio)
    upper = token.upper()
    if field == "invoice_number" and upper.startswith(("INV", "INVOICE")):
        score += 0.1
    if field == "purchase_order_number" and upper.startswith(("PO", "P.O", "PURCHASE")):
        score += 0.1
    if field == "account_number" and ("ACCT" in upper or "ACC" in upper):
        score += 0.05
    if "-" in token:
        score += 0.05
    return min(score, 1.0)

class SectionIntelligenceBuilder:
    def __init__(
        self,
        *,
        min_sections: int = 3,
        max_section_chars: int = 8000,
        entity_extractor: Optional[EntityExtractor] = None,
    ) -> None:
        self.min_sections = max(1, int(min_sections))
        self.max_section_chars = max(500, int(max_section_chars))
        self.entity_extractor = entity_extractor or EntityExtractor()

    def build(
        self,
        *,
        document_id: str,
        document_text: str,
        chunk_texts: List[str],
        chunk_metadata: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SectionIntelligenceResult:
        classification = classify_domain(document_text, metadata)
        doc_domain = classification.domain

        sections = self._segment_sections(
            document_id=document_id,
            document_text=document_text,
            chunk_texts=chunk_texts,
            chunk_metadata=chunk_metadata,
            doc_domain=doc_domain,
        )
        if doc_domain in {"invoice", "purchase_order"}:
            sections = _inject_invoice_sections(
                sections,
                document_text=document_text,
                document_id=document_id,
            )
        sections = _ensure_min_sections(
            sections,
            chunk_texts=chunk_texts,
            doc_domain=doc_domain,
            document_id=document_id,
            min_sections=self.min_sections,
        )
        _apply_section_salience(sections, len(document_text or ""))
        self._apply_sections_to_chunks(sections, chunk_metadata)

        chunk_map = _build_chunk_map(chunk_texts, chunk_metadata)
        section_facts = self._extract_facts(
            sections=sections,
            chunk_map=chunk_map,
            document_id=document_id,
            doc_domain=doc_domain,
        )
        section_summaries = self._summarize_sections(sections)
        return SectionIntelligenceResult(
            doc_domain=doc_domain,
            sections=sections,
            section_facts=section_facts,
            section_summaries=section_summaries,
        )

    def _segment_sections(
        self,
        *,
        document_id: str,
        document_text: str,
        chunk_texts: List[str],
        chunk_metadata: List[Dict[str, Any]],
        doc_domain: str,
    ) -> List[SectionDescriptor]:
        sections: List[SectionDescriptor] = []
        group_candidates = _group_chunks_by_section(chunk_texts, chunk_metadata)
        if len(group_candidates) <= 1 and document_text:
            by_heading = _segment_by_headings(document_text, fallback_title="Section")
            if not by_heading:
                by_heading = _segment_by_paragraphs(document_text)
            if by_heading:
                total_chunks = max(len(chunk_texts), 1)
                for idx, (title, body) in enumerate(by_heading):
                    kind = _infer_section_kind(title, body, doc_domain)
                    if doc_domain == "resume" and kind == "misc":
                        refined = _refine_resume_misc(title, body)
                        if refined:
                            kind = refined
                    section_id = _hash_section_id(document_id, title, idx)
                    start_index = int(idx * total_chunks / max(len(by_heading), 1))
                    end_index = int((idx + 1) * total_chunks / max(len(by_heading), 1)) - 1
                    if idx == len(by_heading) - 1:
                        end_index = total_chunks - 1
                    sections.append(
                        SectionDescriptor(
                            section_id=section_id,
                            section_title=title,
                            section_kind=kind,
                            section_path=title,
                            page_range=None,
                            raw_text=body[: self.max_section_chars],
                            confidence=0.55,
                            start_index=start_index,
                            end_index=end_index,
                        )
                    )
                return sections

        for idx, (title, start, end, body, page_range) in enumerate(group_candidates):
            kind = _infer_section_kind(title, body, doc_domain)
            if doc_domain == "resume" and kind == "misc":
                refined = _refine_resume_misc(title, body)
                if refined:
                    kind = refined
            section_id = _hash_section_id(document_id, title, idx)
            sections.append(
                SectionDescriptor(
                    section_id=section_id,
                    section_title=title,
                    section_kind=kind,
                    section_path=title,
                    page_range=page_range,
                    raw_text=body[: self.max_section_chars],
                    confidence=0.7 if len(group_candidates) > 1 else 0.4,
                    start_index=start,
                    end_index=end,
                )
            )
        return sections

    @staticmethod
    def _apply_sections_to_chunks(sections: List[SectionDescriptor], chunk_metadata: List[Dict[str, Any]]) -> None:
        if not sections:
            return
        for section in sections:
            for idx in range(section.start_index, section.end_index + 1):
                if idx >= len(chunk_metadata):
                    continue
                meta = chunk_metadata[idx]
                meta["section_id"] = section.section_id
                meta["section_title"] = section.section_title
                meta["section_path"] = section.section_path
                meta["section_kind"] = section.section_kind
                meta["section_confidence"] = section.confidence
                meta["section_salience"] = section.salience

    def _extract_facts(
        self,
        *,
        sections: List[SectionDescriptor],
        chunk_map: List[Dict[str, Any]],
        document_id: str,
        doc_domain: str,
    ) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        for section in sections:
            section_entities = []
            evidence_spans: List[Dict[str, Any]] = []
            attributes: Dict[str, Any] = {}
            attribute_confidence: Dict[str, float] = {}

            entities = self.entity_extractor.extract_with_metadata(section.raw_text)
            for ent in entities:
                evidence = _find_evidence_span(ent.name, chunk_map)
                if not evidence:
                    continue
                entity_obj = {
                    "type": ent.type,
                    "value": ent.name,
                    "normalized": ent.normalized_name,
                    "confidence": ent.confidence,
                }
                section_entities.append(entity_obj)
                evidence_spans.append(
                    {**evidence, "field": "entity", "value": ent.name, "confidence": ent.confidence}
                )

            if section.section_kind in {"identity_contact", "taxpayer_identity", "account_identity"}:
                names = [ent["value"] for ent in section_entities if str(ent.get("type")).upper() == "PERSON"]
                emails = [ent["value"] for ent in section_entities if str(ent.get("type")).upper() == "EMAIL"]
                phones = [ent["value"] for ent in section_entities if str(ent.get("type")).upper() == "PHONE"]
                locations = [ent["value"] for ent in section_entities if str(ent.get("type")).upper() == "LOCATION"]
                ids = [ent["value"] for ent in section_entities if str(ent.get("type")).upper() == "ID"]
                if names:
                    attributes["names"] = names[:5]
                if emails:
                    attributes["emails"] = emails[:5]
                if phones:
                    attributes["phones"] = phones[:5]
                if locations:
                    attributes["locations"] = locations[:5]
                if ids:
                    attributes["ids"] = ids[:5]
                if match := _AGE_RE.search(section.raw_text or ""):
                    age_val = match.group(1) or match.group(2)
                    if age_val:
                        attributes["age"] = str(age_val)
                        attribute_confidence["age"] = 0.6
                if match := _SEX_RE.search(section.raw_text or ""):
                    sex_val = match.group(1)
                    if sex_val:
                        attributes["sex"] = sex_val.upper()
                        attribute_confidence["sex"] = 0.6

            if section.section_kind in {"financial_summary", "line_items", "transactions", "terms_conditions"}:
                if match := _TOTAL_RE.search(section.raw_text):
                    attributes["total_amount"] = match.group(1).strip()
                    attribute_confidence["total_amount"] = 0.6
                if match := _DUE_DATE_RE.search(section.raw_text):
                    attributes["due_date"] = match.group(1).strip()
                    attribute_confidence["due_date"] = 0.55

            if doc_domain in {"invoice", "purchase_order"}:
                if match := _INVOICE_RE.search(section.raw_text):
                    candidate = match.group(1).strip()
                    confidence = _score_identifier(candidate, "invoice_number")
                    if confidence >= _EXTRACTOR_THRESHOLDS.get("invoice_number", 0.0):
                        attributes.setdefault("invoice_number", candidate)
                        attribute_confidence["invoice_number"] = confidence
                if match := _PO_RE.search(section.raw_text):
                    candidate = match.group(1).strip()
                    confidence = _score_identifier(candidate, "purchase_order_number")
                    if confidence >= _EXTRACTOR_THRESHOLDS.get("purchase_order_number", 0.0):
                        attributes.setdefault("purchase_order_number", candidate)
                        attribute_confidence["purchase_order_number"] = confidence

            if doc_domain == "bank_statement":
                if match := _ACCOUNT_RE.search(section.raw_text):
                    candidate = match.group(1).strip()
                    confidence = _score_identifier(candidate, "account_number")
                    if confidence >= _EXTRACTOR_THRESHOLDS.get("account_number", 0.0):
                        attributes.setdefault("account_number", candidate)
                        attribute_confidence["account_number"] = confidence

            if section.section_kind in {"certifications", "skills_technical", "skills_functional"}:
                skills = []
                for line in section.raw_text.splitlines():
                    candidate = line.strip("•- \t")
                    if len(candidate) < 2:
                        continue
                    if len(candidate.split()) <= 6:
                        skills.append(candidate)
                if skills:
                    attributes["items"] = skills[:20]

            if section.section_kind in {"diagnoses_procedures", "medications"}:
                medical_terms = []
                for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9\-]{3,}\b", section.raw_text):
                    token = match.group(0)
                    if token.lower() in {"patient", "diagnosis", "medication", "procedure", "treatment"}:
                        continue
                    if token[0].isupper():
                        medical_terms.append(token)
                if medical_terms:
                    attributes["terms"] = list(dict.fromkeys(medical_terms))[:20]

            # Date/amount attributes from raw text
            dates = list(dict.fromkeys([m.group(0) for m in _DATE_RE.finditer(section.raw_text)]))
            amounts = list(dict.fromkeys([m.group(0) for m in _AMOUNT_RE.finditer(section.raw_text)]))
            if dates:
                attributes.setdefault("dates", dates[:10])
            if amounts:
                attributes.setdefault("amounts", amounts[:10])

            filtered_attributes: Dict[str, Any] = {}
            for key, value in list(attributes.items()):
                confidence = float(attribute_confidence.get(key, 0.5))
                threshold = float(_EXTRACTOR_THRESHOLDS.get(key, 0.0))
                if confidence < threshold:
                    continue
                if isinstance(value, list):
                    valid_items = []
                    for item in value:
                        evidence = _find_evidence_span(str(item), chunk_map)
                        if evidence:
                            evidence_spans.append(
                                {**evidence, "field": key, "value": item, "confidence": confidence}
                            )
                            valid_items.append(item)
                    if valid_items:
                        filtered_attributes[key] = valid_items
                else:
                    evidence = _find_evidence_span(str(value), chunk_map)
                    if evidence:
                        evidence_spans.append(
                            {**evidence, "field": key, "value": value, "confidence": confidence}
                        )
                        filtered_attributes[key] = value
            attributes = filtered_attributes

            facts.append(
                {
                    "section_kind": section.section_kind,
                    "entities": section_entities,
                    "attributes": attributes,
                    "evidence_spans": evidence_spans,
                    "provenance": {"document_id": document_id, "section_id": section.section_id},
                }
            )
        return facts

    @staticmethod
    def _summarize_sections(sections: Iterable[SectionDescriptor]) -> Dict[str, str]:
        summaries: Dict[str, str] = {}
        for section in sections:
            tokens = (section.raw_text or "").split()
            if not tokens:
                continue
            summary = " ".join(tokens[:40])
            if len(tokens) > 40:
                summary += "..."
            summaries[section.section_id] = summary
        return summaries

__all__ = [
    "SECTION_KIND_TAXONOMY",
    "SectionDescriptor",
    "SectionIntelligenceResult",
    "SectionIntelligenceBuilder",
]
