from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record

logger = get_logger(__name__)

router = APIRouter(prefix="/translator", tags=["Tools-Translator"])

_ARGOS_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from argostranslate import translate as _argos_translate  # type: ignore

    _ARGOS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _argos_translate = None  # type: ignore

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_lang: str = Field(..., description="Target language code")
    source_lang: Optional[str] = Field(default=None, description="Optional source language code")

# ── Language code mapping ──────────────────────────────────────────

_LANG_CODE_MAP = {
    "french": "fr", "spanish": "es", "german": "de", "italian": "it",
    "portuguese": "pt", "dutch": "nl", "russian": "ru", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
    "turkish": "tr", "polish": "pl", "swedish": "sv", "norwegian": "no",
    "danish": "da", "finnish": "fi", "greek": "el", "czech": "cs",
    "romanian": "ro", "hungarian": "hu", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "malay": "ms", "hebrew": "he", "ukrainian": "uk",
    "tamil": "ta", "telugu": "te", "bengali": "bn", "urdu": "ur",
    "persian": "fa", "swahili": "sw", "catalan": "ca",
}

_KNOWN_LANGUAGES = frozenset(_LANG_CODE_MAP.keys()) | {"english"}

def _spacy_extract_translation_intent(query: str):
    """Use spaCy to detect translation intent and extract target language.

    Returns (target_language_name, remaining_text) or (None, None).
    Detects verbs like translate/convert/change and finds the target language
    as a noun/proper noun argument in the sentence structure.
    """
    try:
        from src.nlp.nlu_engine import _get_nlp
        nlp = _get_nlp()
        if nlp is None:
            return None, None

        doc = nlp(query)
        _TRANSLATE_VERBS = {"translate", "convert", "change", "transform"}
        translate_verb = None
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_.lower() in _TRANSLATE_VERBS:
                translate_verb = token
                break

        if translate_verb is None:
            # Check for noun form "translation"
            for token in doc:
                if token.lemma_.lower() == "translation" and token.pos_ == "NOUN":
                    translate_verb = token
                    break

        if translate_verb is None:
            # No translation verb — check for "to/in <language>" pattern
            for token in doc:
                if token.text.lower() in ("to", "into", "in") and token.dep_ == "prep":
                    for child in token.children:
                        lang_name = child.text.lower()
                        if lang_name in _KNOWN_LANGUAGES:
                            return lang_name, ""
            return None, None

        # Found translation verb — search for target language in its subtree
        for token in translate_verb.subtree:
            if token == translate_verb:
                continue
            lang_name = token.text.lower()
            if lang_name in _KNOWN_LANGUAGES:
                # Extract text after the language mention (if any)
                remaining = query[token.idx + len(token.text):].strip().lstrip(":-. ")
                return lang_name, remaining

        # Also check for preposition children: "translate to french"
        for token in doc:
            if token.head == translate_verb and token.dep_ == "prep":
                for child in token.children:
                    lang_name = child.text.lower()
                    if lang_name in _KNOWN_LANGUAGES:
                        remaining = query[child.idx + len(child.text):].strip().lstrip(":-. ")
                        return lang_name, remaining

    except Exception:
        pass
    return None, None

# ── LLM translation ────────────────────────────────────────────────

def _llm_detect_language(text: str) -> Optional[str]:
    """Use LLM to detect the source language. Returns ISO 639-1 code or None."""
    try:
        import concurrent.futures
        from src.tools.llm_tools import _DOMAIN_OPTIONS, _call_generate, get_tool_llm_client

        client = get_tool_llm_client()
        if client is None:
            return None

        prompt = (
            "What language is this text? Respond with ONLY the ISO 639-1 code "
            "(en, fr, de, ja, es, zh, ko, ar, hi, pt, ru, it, nl, etc.).\n\n"
            f"TEXT: {text[:500]}"
        )
        options = dict(_DOMAIN_OPTIONS.get("translation", {}))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_generate, client, prompt, options)
            raw = future.result(timeout=5.0)

        if raw:
            code = raw.strip().lower().rstrip(".")
            if 2 <= len(code) <= 3 and code.isalpha():
                return code
        return None
    except Exception:
        return None

def _llm_detect_target_language(query: str) -> Optional[str]:
    """Use LLM to detect the target translation language from a query."""
    try:
        import concurrent.futures
        from src.tools.llm_tools import _DOMAIN_OPTIONS, _call_generate, get_tool_llm_client

        client = get_tool_llm_client()
        if client is None:
            return None

        prompt = (
            "What language does the user want this text translated INTO? "
            "Respond with ONLY the ISO 639-1 code (fr, de, es, ja, etc.).\n\n"
            f"QUERY: {query[:500]}"
        )
        options = dict(_DOMAIN_OPTIONS.get("translation", {}))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_generate, client, prompt, options)
            raw = future.result(timeout=5.0)

        if raw:
            code = raw.strip().lower().rstrip(".")
            if 2 <= len(code) <= 3 and code.isalpha():
                return code
        return None
    except Exception:
        return None

def _llm_translate(text: str, source_lang: Optional[str], target_lang: str) -> Optional[Dict[str, Any]]:
    """LLM-powered translation with domain terminology preservation."""
    try:
        from src.tools.llm_tools import build_generation_prompt, tool_generate_structured

        source_label = source_lang or "auto-detect"
        instructions = (
            f"Translate from {source_label} to {target_lang}.\n"
            "Preserve formatting, domain terminology, and cultural context.\n"
            "Flag terms without direct equivalents.\n\n"
            'Return JSON: {"translated_text": "...", "detected_lang": "...", '
            '"flagged_terms": [...], "quality_notes": "..."}'
        )
        prompt = build_generation_prompt("translator", instructions, text)
        return tool_generate_structured(prompt, domain="translation")
    except Exception as exc:
        logger.debug("LLM translation failed: %s", exc)
        return None

# ── Argos translation ───────────────────────────────────────────────

def _argos_translate_text(text: str, source: Optional[str], target: str) -> str:
    if not _ARGOS_AVAILABLE:
        raise ToolError("Argos Translate is not installed", code="backend_unavailable", status_code=501)
    try:
        available = _argos_translate.get_installed_languages()  # type: ignore[attr-defined]
        source_lang = next((lang for lang in available if source and lang.code == source), available[0])
        target_lang = next((lang for lang in available if lang.code == target), None)
        if not target_lang:
            raise ToolError(f"Target language '{target}' is not installed", code="unsupported_language")
        translation = source_lang.get_translation(target_lang)  # type: ignore[call-arg]
        return translation.translate(text)  # type: ignore[attr-defined]
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Argos translation failed: %s", exc)
        raise ToolError("Translation failed", code="translation_failed") from exc

# ── Deterministic fallback ──────────────────────────────────────────

def _fallback_translate(text: str, target: str) -> str:
    """Deterministic offline fallback to keep endpoint responsive."""
    return f"[{target}] {text}"

# ── Smart request parsing ─────────────────────────────────────────

def _parse_translation_request(payload: Dict[str, Any]) -> TranslateRequest:
    """Build TranslateRequest from pipeline payload, extracting target_lang from query.

    When dispatched from the RAG pipeline, ``inp["text"]`` contains serialized
    chunk text and ``inp["query"]`` contains the user's query.  The translator
    should translate the *chunk text* into the target language detected from
    the *query*.
    """
    inp = payload.get("input") or payload

    # If target_lang is already provided, use it directly
    if inp.get("target_lang"):
        text = inp.get("text") or inp.get("query") or ""
        return TranslateRequest(text=text, target_lang=inp["target_lang"], source_lang=inp.get("source_lang"))

    query = inp.get("query") or ""
    chunk_text = inp.get("text") or ""
    # When pipeline-dispatched, prefer chunk content over query as source text
    source_text = chunk_text if chunk_text and len(chunk_text) > len(query) else ""

    # 1. spaCy-based translation intent detection
    lang_name, inline_text = _spacy_extract_translation_intent(query)
    if lang_name:
        target_lang = _LANG_CODE_MAP.get(lang_name, lang_name[:2] if len(lang_name) >= 2 else "en")
        text = source_text or (inline_text if inline_text else query)
        return TranslateRequest(text=text, target_lang=target_lang, source_lang=inp.get("source_lang"))

    # 4. Fallback: use LLM to detect target language from query
    target_lang = _llm_detect_target_language(query) or "en"
    text = source_text or query
    return TranslateRequest(text=text, target_lang=target_lang, source_lang=inp.get("source_lang"))

# ── Unified translation ────────────────────────────────────────────

def _translate_text(request: TranslateRequest) -> Dict[str, Any]:
    """Translate text using: LLM → Argos → deterministic fallback."""
    from src.tools.llm_tools import score_tool_response

    source = request.source_lang
    expected_fields = ["translated_text"]

    # Auto-detect language if not provided
    if not source:
        source = _llm_detect_language(request.text) or "unknown"

    # Tier 1: LLM translation
    llm_result = _llm_translate(request.text, source, request.target_lang)
    if llm_result and llm_result.get("translated_text"):
        iq = score_tool_response(
            {"translated_text": llm_result["translated_text"]},
            domain="general",
            expected_fields=expected_fields,
            source="llm",
        )
        detected = llm_result.get("detected_lang", source)
        return {
            "translated_text": llm_result["translated_text"],
            "target_lang": request.target_lang,
            "detected_lang": detected,
            "flagged_terms": llm_result.get("flagged_terms", []),
            "quality_notes": llm_result.get("quality_notes", ""),
            "backend": "DocWain-Agent",
            "warnings": [],
            "iq_score": iq.as_dict(),
            "rendered": (
                f"{llm_result['translated_text']}\n\n"
                f"*Translated from {detected} to {request.target_lang} (DocWain-Agent)*"
            ),
        }

    # Tier 2: Argos translation
    warnings: list[str] = []
    try:
        translated = _argos_translate_text(request.text, source, request.target_lang)
        iq = score_tool_response(
            {"translated_text": translated},
            domain="general",
            expected_fields=expected_fields,
            source="argos",
        )
        return {
            "translated_text": translated,
            "target_lang": request.target_lang,
            "detected_lang": source,
            "flagged_terms": [],
            "quality_notes": "",
            "backend": "argos",
            "warnings": [],
            "iq_score": iq.as_dict(),
            "rendered": (
                f"{translated}\n\n"
                f"*Translated from {source} to {request.target_lang} (argos)*"
            ),
        }
    except ToolError as exc:
        warnings.append(str(exc))

    # Tier 3: Deterministic fallback
    translated = _fallback_translate(request.text, request.target_lang)
    iq = score_tool_response(
        {"translated_text": translated},
        domain="general",
        expected_fields=expected_fields,
        source="template",
    )
    return {
        "translated_text": translated,
        "target_lang": request.target_lang,
        "detected_lang": source,
        "flagged_terms": [],
        "quality_notes": "",
        "backend": "fallback",
        "warnings": warnings,
        "iq_score": iq.as_dict(),
        "rendered": (
            f"{translated}\n\n"
            f"*Translated from {source} to {request.target_lang} (fallback)*"
        ),
    }

@register_tool("translator")
async def translator_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = _parse_translation_request(payload)
    result = _translate_text(req)
    sources = [build_source_record("tool", correlation_id or "translator", title="translator")]
    return {
        "result": result,
        "sources": sources,
        "context_found": True,
        "grounded": True,
        "warnings": result.get("warnings", []),
    }

@router.post("/translate")
async def translate(request: TranslateRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    try:
        result = _translate_text(request)
        sources = [build_source_record("tool", cid, title="translator")]
        return standard_response(
            "translator",
            grounded=True,
            context_found=True,
            result=result,
            sources=sources,
            warnings=result.get("warnings", []),
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "translator",
            status="error",
            grounded=False,
            context_found=False,
            result={},
            sources=[],
            warnings=[],
            error=exc.as_dict(),
            correlation_id=cid,
        )
