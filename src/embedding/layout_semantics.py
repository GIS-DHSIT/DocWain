from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.embedding.chunking.section_chunker import normalize_text
from src.embedding.layout_graph import build_layout_graph

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ID_RE = re.compile(r"\b(?=[A-Za-z0-9]{6,})(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9\-]+\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{6,}\d)")
_MONEY_RE = re.compile(r"[$€£]\s?\d+(?:[,\d]*)(?:\.\d{2})?")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")

@dataclass
class SemanticBuildResult:
    chunks: List[str]
    chunk_metadata: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    section_summaries: Dict[str, str]
    doc_summary: Optional[str]
    entity_facts: List[Dict[str, Any]]
    doc_domain: str

class GenericEntityExtractor:
    def __init__(self, *, model: str = "en_core_web_sm") -> None:
        self._nlp = None
        self._skills: List[str] = []
        self._skill_patterns: List[Tuple[str, re.Pattern]] = []
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load(model) if model else None
        except Exception:  # noqa: BLE001
            self._nlp = None
        try:
            from src.kg.entity_extractor import DEFAULT_SKILLS

            self._skills = [s.strip().lower() for s in DEFAULT_SKILLS if s.strip()]
        except Exception:
            self._skills = ["python", "sql", "java", "aws", "excel"]
        for skill in sorted(set(self._skills), key=len, reverse=True):
            if " " in skill:
                pat = re.compile(rf"\\b{re.escape(skill)}\\b", re.IGNORECASE)
            else:
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])", re.IGNORECASE)
            self._skill_patterns.append((skill, pat))

    def extract(self, text: str) -> List[Dict[str, str]]:
        text = text or ""
        entities: Dict[str, Dict[str, str]] = {}

        def add(entity: str, etype: str) -> None:
            key = f"{etype}::{entity.lower().strip()}"
            if not entity:
                return
            if key in entities:
                return
            entities[key] = {"entity": entity, "type": etype}

        if self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                label = ent.label_
                mapped = {
                    "PERSON": "person",
                    "ORG": "organization",
                    "GPE": "location",
                    "LOC": "location",
                    "DATE": "date",
                    "MONEY": "money",
                    "PRODUCT": "product",
                }.get(label, "other")
                add(ent.text.strip(), mapped)

        for match in _EMAIL_RE.findall(text):
            add(match, "email")
        for match in _PHONE_RE.findall(text):
            digits = re.sub(r"\D", "", match)
            if 7 <= len(digits) <= 15:
                add(digits, "phone")
        for match in _MONEY_RE.findall(text):
            add(match, "money")
        for match in _DATE_RE.findall(text):
            add(match, "date")
        for match in _ID_RE.findall(text):
            add(match, "id")
        for skill, pat in self._skill_patterns:
            if pat.search(text):
                add(skill, "skill")
        for match in re.findall(r"\b\w+(?:itis|osis|emia|algia|ectomy|otomy|opathy)\b", text, flags=re.IGNORECASE):
            add(match, "medical_term")

        return list(entities.values())

def _sentence_summary(text: str, *, max_sentences: int = 3, max_chars: int = 700) -> str:
    text = normalize_text(text or "")
    if not text:
        return ""
    sentences = _SENTENCE_RE.split(text)
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return summary

def _section_kind(block_types: List[str]) -> str:
    types = {t for t in block_types if t}
    if not types:
        return "body"
    if types == {"header"}:
        return "header"
    if types == {"footer"}:
        return "footer"
    if "table" in types:
        return "table_region"
    if "list" in types:
        return "list_region" if len(types) == 1 else "mixed"
    return "mixed" if len(types) > 1 else "body"

def build_layout_sections(layout_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    pages = layout_graph.get("pages") or []
    if not pages:
        return sections

    def _new_section(idx: int, page: int, blocks: List[str], title: Optional[str]) -> Dict[str, Any]:
        seed = f"{layout_graph.get('doc_id')}|{page}|{idx}|{title or ''}"
        section_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return {
            "section_id": section_id,
            "title": title,
            "page_start": page,
            "page_end": page,
            "blocks": list(blocks),
            "kind": "body",
        }

    current: Optional[Dict[str, Any]] = None
    section_index = 0

    for page in pages:
        page_num = int(page.get("page") or 1)
        blocks = page.get("blocks") or []
        last_bbox = None
        last_column = None
        for block in blocks:
            bbox = block.get("bbox") or [0, 0, 0, 0]
            column = block.get("_column", 0)
            gap = 0.0
            if last_bbox is not None:
                gap = float(bbox[1]) - float(last_bbox[3])
            heading_level = (block.get("structure") or {}).get("heading_level")
            boundary = False
            if current is None:
                boundary = True
            elif heading_level:
                boundary = True
            elif block.get("type") == "table":
                boundary = True
            elif last_column is not None and column != last_column and gap < 0:
                boundary = True
            elif gap > max(24.0, (page.get("height") or 800) * 0.08):
                boundary = True

            if boundary:
                if current:
                    sections.append(current)
                section_index += 1
                title = block.get("text") if heading_level else None
                current = _new_section(section_index, page_num, [], title)

            if current:
                current["blocks"].append(block.get("block_id"))
                current["page_end"] = page_num

            last_bbox = bbox
            last_column = column

    if current:
        sections.append(current)

    block_lookup = {}
    for page in pages:
        for block in page.get("blocks") or []:
            block_lookup[block.get("block_id")] = block

    for idx, section in enumerate(sections, start=1):
        block_types = [block_lookup.get(bid, {}).get("type") for bid in section.get("blocks") or []]
        section["kind"] = _section_kind(block_types)
        title = section.get("title")
        if not title:
            title = f"Untitled Section {idx}"
            section["title"] = None
        section["section_path"] = [title]

    return sections

def _chunk_blocks(
    blocks: List[Dict[str, Any]],
    *,
    target_chars: int,
    max_chars: int,
    section_title: str,
    section_path: List[str],
    section_id: str,
    section_kind: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []
    buffer_texts: List[str] = []
    buffer_blocks: List[Dict[str, Any]] = []
    buffer_len = 0

    def _flush(chunk_kind: str = "section_text") -> None:
        nonlocal buffer_texts, buffer_blocks, buffer_len
        if not buffer_texts:
            return
        raw_text = "\n\n".join(buffer_texts).strip()
        canonical = normalize_text(raw_text)
        if canonical:
            page_numbers = [blk.get("page") for blk in buffer_blocks if blk.get("page") is not None]
            anchors = [{"page": blk.get("page"), "block_ids": [blk.get("block_id")]} for blk in buffer_blocks]
            chunks.append(raw_text)
            metadata.append(
                {
                    "section_id": section_id,
                    "section_title": section_title,
                    "section_path": " > ".join(section_path),
                    "section_kind": section_kind,
                    "page_start": min(page_numbers) if page_numbers else None,
                    "page_end": max(page_numbers) if page_numbers else None,
                    "page_number": page_numbers[0] if page_numbers else None,
                    "chunk_type": chunk_kind,
                    "chunk_kind": chunk_kind,
                    "anchors": anchors,
                }
            )
        buffer_texts = []
        buffer_blocks = []
        buffer_len = 0

    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        block_kind = block.get("type") or "text"
        # Tables become standalone chunks
        if block_kind == "table":
            _flush()
            buffer_texts = [text]
            buffer_blocks = [block]
            buffer_len = len(text)
            _flush("table")
            continue
        # Lists stay together but can be chunked if too long
        if block_kind == "list":
            if buffer_len + len(text) > max_chars and buffer_texts:
                _flush()
            buffer_texts.append(text)
            buffer_blocks.append(block)
            buffer_len += len(text)
            if buffer_len >= max_chars:
                _flush("list")
            continue
        # Normal text
        if buffer_len + len(text) > max_chars and buffer_texts:
            _flush()
        buffer_texts.append(text)
        buffer_blocks.append(block)
        buffer_len += len(text)
        if buffer_len >= target_chars:
            _flush()

    _flush()
    return chunks, metadata

def _infer_doc_domain(entity_counts: Dict[str, int], layout_graph: Dict[str, Any], texts: List[str]) -> str:
    scores: Dict[str, float] = {}
    doc_signals = layout_graph.get("doc_signals") or {}
    has_tables = bool(doc_signals.get("has_tables"))
    has_lists = bool(doc_signals.get("has_lists"))
    money = entity_counts.get("money", 0)
    dates = entity_counts.get("date", 0)
    people = entity_counts.get("person", 0)
    skills = entity_counts.get("skill", 0)
    ids = entity_counts.get("id", 0)
    medical = entity_counts.get("medical_term", 0)

    if people and skills >= 3:
        scores["resume"] = 1.5 + (0.5 if has_lists else 0.0)
    if has_tables and money >= 3:
        scores["invoice"] = scores.get("invoice", 0.0) + 1.2
        scores["purchase_order"] = scores.get("purchase_order", 0.0) + 0.8
    if has_tables and money >= 3 and dates >= 3:
        scores["bank_statement"] = scores.get("bank_statement", 0.0) + 1.4
    if medical >= 2:
        scores["medical"] = scores.get("medical", 0.0) + 1.5
    if ids >= 3 and money >= 2:
        scores["tax"] = scores.get("tax", 0.0) + 1.0

    # Numeric density boost for invoice/bank hints
    joined = " ".join(texts[:6])
    if joined:
        digit_ratio = sum(ch.isdigit() for ch in joined) / max(len(joined), 1)
        if digit_ratio > 0.14 and has_tables:
            scores["invoice"] = scores.get("invoice", 0.0) + 0.4
            scores["bank_statement"] = scores.get("bank_statement", 0.0) + 0.4

    if not scores:
        return "generic"
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= 1.2 else "generic"

def _semantic_kind_from_entities(counts: Dict[str, int], fallback: str) -> str:
    if counts.get("person") or counts.get("email") or counts.get("phone"):
        return "identity_contact"
    if counts.get("medical_term", 0) >= 1:
        return "diagnoses_procedures"
    if counts.get("product", 0) >= 2 and counts.get("medical_term", 0) >= 1:
        return "medications"
    if counts.get("skill", 0) >= 2:
        return "skills_technical"
    if counts.get("money", 0) >= 3 and counts.get("date", 0) >= 2:
        return "transactions"
    if counts.get("money", 0) >= 3 and counts.get("product", 0) >= 1:
        return "line_items"
    if counts.get("money", 0) >= 3:
        return "financial_summary"
    return fallback or "body"

def build_semantic_payloads(
    *,
    layout_graph: Optional[Dict[str, Any]] = None,
    extracted: Optional[Any] = None,
    document_id: str,
    source_name: str,
) -> SemanticBuildResult:
    if layout_graph is None:
        layout_graph = build_layout_graph(extracted, document_id=document_id, file_name=source_name)
    sections = build_layout_sections(layout_graph)
    pages = layout_graph.get("pages") or []
    block_lookup = {}
    for page in pages:
        for block in page.get("blocks") or []:
            block_lookup[block.get("block_id")] = block

    target_chars = int(getattr(Config.Retrieval, "CHUNK_SIZE", 800))
    max_chars = int(getattr(Config.Retrieval, "MAX_CHUNK_SIZE", max(target_chars + target_chars // 2, 1200)))
    chunks: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []
    section_payloads: List[Dict[str, Any]] = []
    section_summaries: Dict[str, str] = {}

    for idx, section in enumerate(sections, start=1):
        section_title = section.get("title") or f"Untitled Section {idx}"
        section_path = section.get("section_path") or [section_title]
        section_id = section.get("section_id")
        section_kind = section.get("kind") or "body"
        block_ids = section.get("blocks") or []
        blocks = [block_lookup[bid] for bid in block_ids if bid in block_lookup]
        section_text = "\n\n".join([blk.get("text") or "" for blk in blocks if blk.get("text")]).strip()
        section_payloads.append(
            {
                "section_id": section_id,
                "section_title": section_title,
                "section_path": " > ".join(section_path),
                "section_kind": section_kind,
                "page_range": (section.get("page_start"), section.get("page_end")),
                "raw_text": section_text,
            }
        )

        section_chunks, section_meta = _chunk_blocks(
            blocks,
            target_chars=target_chars,
            max_chars=max_chars,
            section_title=section_title,
            section_path=section_path,
            section_id=section_id,
            section_kind=section_kind,
        )
        chunks.extend(section_chunks)
        chunk_meta.extend(section_meta)

        summary = _sentence_summary(section_text, max_sentences=2, max_chars=int(getattr(Config.Intelligence, "SECTION_SUMMARY_MAX_CHARS", 700)))
        if summary:
            section_summaries[section_id] = summary

    doc_text = "\n\n".join([c for c in chunks if c]).strip()
    doc_summary = _sentence_summary(doc_text, max_sentences=4, max_chars=1200) if doc_text else None

    # Entity extraction
    extractor = GenericEntityExtractor()
    entity_facts: List[Dict[str, Any]] = []
    entity_counts: Dict[str, int] = {}
    section_entity_counts: Dict[str, Dict[str, int]] = {}
    for text, meta in zip(chunks, chunk_meta):
        canonical = normalize_text(text)
        for ent in extractor.extract(canonical):
            etype = ent.get("type") or "other"
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            sec_id = meta.get("section_id") or "unknown"
            sec_counts = section_entity_counts.setdefault(sec_id, {})
            sec_counts[etype] = sec_counts.get(etype, 0) + 1
            evidence = {
                "file": source_name,
                "page": meta.get("page_number"),
                "snippet": canonical[:160],
            }
            section_kind = meta.get("section_kind") or "body"
            attributes: Dict[str, Any] = {}
            entity_facts.append(
                {
                    "entity": ent.get("entity"),
                    "type": etype,
                    "value_normalized": (ent.get("entity") or "").strip().lower(),
                    "section_kind": section_kind,
                    "attributes": attributes,
                    "entities": [{"type": etype.upper(), "value": ent.get("entity")}],
                    "evidence": [evidence],
                    "evidence_spans": [evidence],
                    "provenance": {
                        "section_id": meta.get("section_id"),
                        "page_range": (meta.get("page_start"), meta.get("page_end")),
                    },
                }
            )

    doc_domain = _infer_doc_domain(entity_counts, layout_graph, chunks)

    # Update section kinds based on entity distribution (non-binding semantic hinting).
    if section_entity_counts:
        section_kind_map: Dict[str, str] = {}
        for section in section_payloads:
            sec_id = section.get("section_id")
            counts = section_entity_counts.get(sec_id or "")
            if not counts:
                continue
            section["section_kind"] = _semantic_kind_from_entities(counts, section.get("section_kind"))
            section_kind_map[str(sec_id)] = section["section_kind"]
        for meta in chunk_meta:
            sec_id = meta.get("section_id")
            counts = section_entity_counts.get(sec_id or "")
            if counts:
                meta["section_kind"] = _semantic_kind_from_entities(counts, meta.get("section_kind"))
        for fact in entity_facts:
            sec_id = (fact.get("provenance") or {}).get("section_id")
            if not sec_id:
                continue
            mapped = section_kind_map.get(str(sec_id))
            if mapped:
                fact["section_kind"] = mapped
                etype = str(fact.get("type") or "").lower()
                attributes = fact.get("attributes") or {}
                if mapped == "identity_contact":
                    if etype == "person":
                        attributes.setdefault("names", []).append(fact.get("entity"))
                    if etype == "id":
                        attributes.setdefault("ids", []).append(fact.get("entity"))
                    if etype == "date":
                        attributes.setdefault("dates", []).append(fact.get("entity"))
                if mapped in {"diagnoses_procedures", "medications"} and etype in {"medical_term", "product"}:
                    attributes.setdefault("terms", []).append(fact.get("entity"))
                if mapped == "skills_technical" and etype == "skill":
                    attributes.setdefault("skills", []).append(fact.get("entity"))
                if mapped == "financial_summary" and etype == "money":
                    attributes.setdefault("amounts", []).append(fact.get("entity"))
                if mapped == "line_items" and etype in {"product", "money"}:
                    attributes.setdefault("items", []).append(fact.get("entity"))
                fact["attributes"] = attributes

    # Add summaries + entity facts as chunks
    extra_chunks: List[str] = []
    extra_meta: List[Dict[str, Any]] = []

    if doc_summary:
        extra_chunks.append(doc_summary)
        extra_meta.append(
            {
                "section_id": "doc_summary",
                "section_title": "Document Summary",
                "section_path": "Document Summary",
                "section_kind": "summary",
                "page_start": None,
                "page_end": None,
                "page_number": None,
                "chunk_type": "doc_summary",
                "chunk_kind": "doc_summary",
            }
        )

    for sec in section_payloads:
        sec_id = sec.get("section_id")
        summary = section_summaries.get(sec_id)
        if not summary:
            continue
        extra_chunks.append(summary)
        extra_meta.append(
            {
                "section_id": sec_id,
                "section_title": sec.get("section_title"),
                "section_path": sec.get("section_path"),
                "section_kind": sec.get("section_kind"),
                "page_start": sec.get("page_range")[0] if sec.get("page_range") else None,
                "page_end": sec.get("page_range")[1] if sec.get("page_range") else None,
                "page_number": sec.get("page_range")[0] if sec.get("page_range") else None,
                "chunk_type": "section_summary",
                "chunk_kind": "section_summary",
            }
        )

    for fact in entity_facts[:120]:
        label = f"{fact.get('type')}: {fact.get('entity')}"
        extra_chunks.append(label)
        extra_meta.append(
            {
                "section_id": fact.get("provenance", {}).get("section_id"),
                "section_title": "Entity Fact",
                "section_path": "Entity Fact",
                "section_kind": "entity_fact",
                "page_start": None,
                "page_end": None,
                "page_number": None,
                "chunk_type": "entity_fact",
                "chunk_kind": "entity_fact",
            }
        )

    if extra_chunks:
        chunks.extend(extra_chunks)
        chunk_meta.extend(extra_meta)

    return SemanticBuildResult(
        chunks=chunks,
        chunk_metadata=chunk_meta,
        sections=section_payloads,
        section_summaries=section_summaries,
        doc_summary=doc_summary,
        entity_facts=entity_facts,
        doc_domain=doc_domain or "generic",
    )

__all__ = [
    "GenericEntityExtractor",
    "SemanticBuildResult",
    "build_layout_sections",
    "build_semantic_payloads",
]
