from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import Block, Entity, EntityFactBundle, Fact, ExtractedDocumentJSON

logger = get_logger(__name__)

_DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"\b(?:\$|USD|EUR|GBP)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?%\b")
_ID_RE = re.compile(r"\b[A-Z]{2,5}-?\d{3,}\b")

_CERT_HINT_RE = re.compile(r"\b(certified|certification|certificate)\b", re.IGNORECASE)
_SKILL_HINT_RE = re.compile(r"\b(skills?|technologies|tools)\b", re.IGNORECASE)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\d?\b")

try:  # pragma: no cover - optional dependency
    import spacy

    _NLP = None

    def _load_spacy():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm")  # type: ignore[no-redef]
        return _NLP

except Exception:  # noqa: BLE001
    spacy = None

    def _load_spacy():  # type: ignore[no-redef]
        return None

def _hash_entity(label: str, text: str) -> str:
    raw = f"{label}|{text.lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

def _entity(label: str, text: str, attributes: Optional[Dict[str, Any]] = None) -> Entity:
    return Entity(entity_id=_hash_entity(label, text), label=label, text=text, attributes=attributes or {})

def _block_text(block: Block) -> str:
    parts = [block.text or ""]
    if block.key and block.value:
        parts.append(f"{block.key}: {block.value}")
    return " ".join([p for p in parts if p]).strip()

def _collect_blocks(document: ExtractedDocumentJSON) -> List[Tuple[Block, List[str]]]:
    block_map: Dict[str, Block] = {}
    for page in document.pages:
        for block in page.blocks:
            block_map[block.block_id] = block

    block_section: Dict[str, List[str]] = {}
    for section in document.sections:
        for block_id in section.content_refs:
            block_section[block_id] = section.section_path

    blocks: List[Tuple[Block, List[str]]] = []
    for block_id, block in block_map.items():
        blocks.append((block, block_section.get(block_id, [])))
    return blocks

def _facts_from_block(
    block: Block,
    section_path: List[str],
    document_id: str,
) -> Tuple[List[Entity], List[Fact]]:
    entities: List[Entity] = []
    facts: List[Fact] = []
    text = _block_text(block)
    if not text:
        return entities, facts

    evidence = {
        "document_id": document_id,
        "section_path": section_path,
        "page_range": [block.page_number or 1, block.page_number or 1],
        "block_ids": [block.block_id],
    }

    # Regex-based entities
    for match in _DATE_RE.findall(text):
        ent = _entity("date", match)
        entities.append(ent)
        facts.append(Fact(subject_id=ent.entity_id, predicate="HAS_DATE", object_value=match, evidence=evidence))
    for match in _CURRENCY_RE.findall(text):
        ent = _entity("amount", match)
        entities.append(ent)
        facts.append(Fact(subject_id=ent.entity_id, predicate="HAS_AMOUNT", object_value=match, evidence=evidence))
    for match in _PERCENT_RE.findall(text):
        ent = _entity("percentage", match)
        entities.append(ent)
        facts.append(Fact(subject_id=ent.entity_id, predicate="HAS_PERCENT", object_value=match, evidence=evidence))
    for match in _ID_RE.findall(text):
        ent = _entity("identifier", match)
        entities.append(ent)
        facts.append(Fact(subject_id=ent.entity_id, predicate="HAS_ID", object_value=match, evidence=evidence))

    section_text = " > ".join(section_path).lower()
    if _CERT_HINT_RE.search(text) or "cert" in section_text:
        tokens = re.split(r"[,;/]\s*|\n", text)
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if _CERT_HINT_RE.search(token) or _ACRONYM_RE.search(token):
                ent = _entity("certification", token)
                entities.append(ent)
                facts.append(Fact(subject_id=ent.entity_id, predicate="HAS_CERTIFICATION", object_value=token, evidence=evidence))

    if _SKILL_HINT_RE.search(text) or "skill" in section_text or "tool" in section_text:
        tokens = re.split(r"[,;/]\s*|\n", text)
        for token in tokens:
            token = token.strip("-• ")
            if not token:
                continue
            if len(token) <= 1:
                continue
            ent = _entity("skill", token)
            entities.append(ent)
            facts.append(Fact(subject_id=ent.entity_id, predicate="HAS_SKILL", object_value=token, evidence=evidence))

    return entities, facts

def _spacy_entities(text: str) -> List[Entity]:
    nlp = _load_spacy()
    if not nlp:
        return []
    doc = nlp(text)
    result: List[Entity] = []
    for ent in doc.ents:
        label = ent.label_.lower()
        mapped = {
            "person": "person",
            "org": "organization",
            "gpe": "location",
            "loc": "location",
            "product": "product",
        }.get(label, label)
        result.append(_entity(mapped, ent.text))
    return result

def extract_entities_and_facts(document: ExtractedDocumentJSON) -> EntityFactBundle:
    entities: Dict[str, Entity] = {}
    facts: List[Fact] = []

    blocks = _collect_blocks(document)
    for block, section_path in blocks:
        new_entities, new_facts = _facts_from_block(block, section_path, document.document_id)
        for ent in new_entities:
            entities.setdefault(ent.entity_id, ent)
        facts.extend(new_facts)

    # spaCy pass for broader entities
    raw_text = document.raw_text or ""
    for ent in _spacy_entities(raw_text):
        entities.setdefault(ent.entity_id, ent)

    return EntityFactBundle(entities=list(entities.values()), facts=facts)

__all__ = ["extract_entities_and_facts"]
