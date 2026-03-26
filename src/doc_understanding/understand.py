from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from src.api.context_understanding import ContextUnderstanding
from src.doc_understanding.structure_inference import infer_structure

logger = get_logger(__name__)

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Access attribute or dict key — supports both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

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

def _ollama_understand(
    text: str,
    doc_type: str,
    model_name: Optional[str],
    llm_client=None,
    use_thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    if not model_name and llm_client is None:
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
        f"TEXT:\n{text[:64000]}"
    )

    # Build generation options — expand for thinking mode
    _options: Dict[str, Any] = {
        "temperature": 0.1,
        "num_predict": 2048,
        "num_ctx": 16384,
    }
    if use_thinking:
        _options["think"] = True
        _options["num_ctx"] = max(_options.get("num_ctx", 4096), 16384)
        _options["num_predict"] = max(_options.get("num_predict", 1024), 4096)
        logger.info(
            "Document understanding thinking mode enabled: num_ctx=%d num_predict=%d",
            _options["num_ctx"], _options["num_predict"],
        )

    try:
        if llm_client is not None:
            if hasattr(llm_client, "generate_with_metadata"):
                content, _meta = llm_client.generate_with_metadata(prompt, options=_options)
            else:
                content = llm_client.generate(prompt)
        else:
            from src.llm.clients import get_local_client
            client = get_local_client()
            if hasattr(client, "generate_with_metadata"):
                content, _meta = client.generate_with_metadata(prompt, options=_options)
            else:
                content = client.generate(prompt)
        content = (content or "").strip()
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama understanding failed: %s", exc)
        return None

def _extractive_section_summaries(extracted: Any) -> Dict[str, str]:
    summaries = {}
    for section in _get(extracted, "sections", []) or []:
        title = _get(section, "title", "Untitled Section")
        text = _get(section, "text", "")
        if text:
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            summaries[title] = ". ".join(sentences[:3]) + "." if sentences else text[:200]
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
    use_thinking: bool = False,
    llm_client=None,
) -> Dict[str, Any]:
    full_text = _get(extracted, "full_text", "") or ""
    if not full_text and _get(extracted, "sections"):
        full_text = "\n".join([_get(s, "text", "") for s in _get(extracted, "sections", []) if _get(s, "text")])

    llm_payload = _ollama_understand(full_text, doc_type, model_name=model_name, llm_client=llm_client, use_thinking=use_thinking)

    if llm_payload:
        doc_summary = str(llm_payload.get("doc_summary") or "").strip()
        section_summaries = _normalize_section_summaries(llm_payload.get("section_summaries"))
        key_entities = llm_payload.get("key_entities") or []
        key_facts = llm_payload.get("key_facts") or []
        intent_tags = llm_payload.get("intent_tags") or []
    else:
        try:
            ctx = ContextUnderstanding()
            summary = ctx.summarize_document(extracted)
            doc_summary = summary.get("abstract", "")
            section_summaries = summary.get("section_summaries", {})
        except (AttributeError, TypeError):
            # extracted is a dict, not ExtractedDocument — use extractive fallback
            doc_summary = (full_text[:500] + "...") if len(full_text) > 500 else full_text
            section_summaries = {}
        key_entities = _fallback_entities(full_text)
        key_facts = []
        intent_tags = [doc_type] if doc_type else []

    verified_facts = _verify_facts(full_text, key_facts)

    if not section_summaries:
        section_summaries = _extractive_section_summaries(extracted)

    structure_signals = infer_structure(extracted)

    return {
        "document_summary": doc_summary.strip(),
        "section_summaries": section_summaries,
        "key_entities": key_entities,
        "key_facts": verified_facts,
        "intent_tags": intent_tags,
        "structure_inference": structure_signals,
    }

__all__ = ["understand_document"]
