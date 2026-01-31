from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import ollama

from src.api.context_understanding import ContextUnderstanding

logger = logging.getLogger(__name__)

_ENTITY_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE),
    "DATE": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    "AMOUNT": re.compile(r"\b\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"),
}


def _fallback_entities(text: str) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    for ent_type, pattern in _ENTITY_PATTERNS.items():
        for match in pattern.findall(text or ""):
            entities.append({"type": ent_type, "text": match})
    return entities


def _ollama_understand(text: str, doc_type: str, model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not model_name:
        return None
    prompt = (
        "You are extracting structured understanding from a document. "
        "Return strict JSON with keys: "
        "doc_summary (string), section_summaries (array of objects with title, summary), "
        "key_entities (array of {type,text}), key_facts (array of {fact, evidence_pointer}), "
        "intent_tags (array of strings). "
        "The document_type is: "
        f"{doc_type}. Only include facts that are explicitly stated in the text. "
        "Evidence pointer should be 'Section: <title>, Page: <page>'.\n\n"
        f"TEXT:\n{text[:8000]}"
    )
    try:
        resp = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0})
        content = resp.get("response", "").strip()
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama understanding failed: %s", exc)
        return None


def _extractive_section_summaries(extracted: Any) -> Dict[str, str]:
    ctx = ContextUnderstanding()
    summaries = {}
    for section in getattr(extracted, "sections", []) or []:
        summaries[getattr(section, "title", "Untitled Section")] = ctx.summarize_section(section)
    return summaries


def _verify_facts(text: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not text or not facts:
        return []
    verified: List[Dict[str, Any]] = []
    lowered = text.lower()
    for fact in facts:
        fact_text = str(fact.get("fact", "")).strip()
        if not fact_text:
            continue
        if fact_text.lower() in lowered:
            verified.append(fact)
    return verified


def _normalize_section_summaries(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v}
    summaries: Dict[str, str] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title") or entry.get("section_title") or "Section"
            summary = entry.get("summary") or entry.get("text")
            if title and summary:
                summaries[str(title)] = str(summary)
    return summaries


def understand_document(
    *,
    extracted: Any,
    doc_type: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    full_text = getattr(extracted, "full_text", "") or ""
    if not full_text and getattr(extracted, "sections", None):
        full_text = "\n".join([sec.text for sec in extracted.sections if sec.text])

    llm_payload = _ollama_understand(full_text, doc_type, model_name=model_name)

    if llm_payload:
        doc_summary = str(llm_payload.get("doc_summary") or "").strip()
        section_summaries = _normalize_section_summaries(llm_payload.get("section_summaries"))
        key_entities = llm_payload.get("key_entities") or []
        key_facts = llm_payload.get("key_facts") or []
        intent_tags = llm_payload.get("intent_tags") or []
    else:
        ctx = ContextUnderstanding()
        summary = ctx.summarize_document(extracted)
        doc_summary = summary.get("abstract", "")
        section_summaries = summary.get("section_summaries", {})
        key_entities = _fallback_entities(full_text)
        key_facts = []
        intent_tags = [doc_type] if doc_type else []

    verified_facts = _verify_facts(full_text, key_facts)

    if not section_summaries:
        section_summaries = _extractive_section_summaries(extracted)

    return {
        "document_summary": doc_summary.strip(),
        "section_summaries": section_summaries,
        "key_entities": key_entities,
        "key_facts": verified_facts,
        "intent_tags": intent_tags,
    }


__all__ = ["understand_document"]
