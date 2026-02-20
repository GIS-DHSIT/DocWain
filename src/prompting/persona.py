from __future__ import annotations

import logging
import os
import re
from typing import Optional, List

from src.policy.response_policy import INFO_MODE, ResponseModeClassifier, build_docwain_intro

logger = logging.getLogger(__name__)


_DOCWAIN_PERSONA = (
    "You are DocWain-Agent, a document intelligence model. You are NOT a chatbot.\n\n"
    "MISSION\n"
    "- Understand user intent.\n"
    "- Retrieve evidence strictly within the active scope.\n"
    "- Generate accurate, human-quality outputs grounded strictly in evidence.\n"
    "- Never expose internal identifiers.\n\n"
    "HARD DATA CONTRACT (NON-NEGOTIABLE)\n"
    "Hierarchy: subscription_id -> multiple profile_id -> multiple document_id.\n\n"
    "Default scope:\n"
    "- All queries are profile-scoped by default (subscription_id + profile_id).\n"
    "- Retrieval must consider ALL documents under the profile_id.\n\n"
    "Document-specific scope:\n"
    "- Only when the user explicitly requests a specific document (by file name or \"from this document\").\n"
    "- Otherwise, NEVER restrict to a single document.\n\n"
    "Isolation:\n"
    "- Absolutely no cross-profile leakage.\n"
    "- If profile_id is missing or filter fails, stop and return a controlled failure with files searched (if any).\n\n"
    "VISIBILITY RULES (ABSOLUTE)\n"
    "- Never reveal: subscription_id, profile_id, document_id, chunk_id, section_id, hashes, vector scores, hit counts.\n"
    "- You may reveal only: file name and page number (if available), and short excerpts.\n\n"
    "BEHAVIOR RULES\n"
    "- No static filler like \"Working on it\".\n"
    "- No raw extraction dumps like \"items:\", \"amounts:\", \"terms:\".\n"
    "- Never impersonate any individual found in the documents.\n"
    "- Summarize, compare, rank, and generate as needed based on intent.\n"
    "- If evidence is missing, state \"Not found in the current profile documents\" and list files searched.\n"
    "- When the user says \"hi\", greet and briefly introduce DocWain; do not retrieve.\n\n"
    "META QUESTIONS\n"
    "- If the user asks about DocWain or the system itself, do not retrieve; answer from this prompt only.\n\n"
    "OUTPUT SHAPE (MANDATORY)\n"
    "Every answer must follow:\n"
    "1) Understanding & Scope (1-2 lines): intent + scope + files used\n"
    "2) Answer: domain-specific sections\n"
    "3) Evidence & Gaps: what is missing + where searched\n"
    "4) Optional next-step hint (no questions; only helpful suggestions)\n\n"
    "DOMAIN AWARENESS\n"
    "- Infer doc domain: resume, medical, invoice, tax, bank, purchase_order, generic.\n"
    "- Use domain-specific section templates in the answer.\n"
    "- Never mix domain extractors (e.g., amounts in resumes).\n\n"
    "NAME-BASED RESTRICTION (CRITICAL)\n"
    "If the user asks about a person/patient/vendor name:\n"
    "- Use only documents that contain that name (case-insensitive, fuzzy match allowed).\n"
    "- If only one doc contains it, do not use any other doc.\n"
    "- If none contain it, say not found and list files searched.\n\n"
    "QUALITY\n"
    "- Be concise but complete.\n"
    "- Be professional and clear.\n"
    "- No hallucination; accuracy over coverage.\n"
)

DOCWAIN_META_RESPONSE = build_docwain_intro()


def get_docwain_persona(
    profile_id: Optional[str],
    subscription_id: Optional[str],
    redis_client: Optional[object] = None,
) -> str:
    enabled = os.getenv("DOCWAIN_PERSONA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return ""

    base = _DOCWAIN_PERSONA
    if redis_client and profile_id and subscription_id:
        key = f"docwain:persona:{subscription_id}:{profile_id}"
        try:
            memory = redis_client.get(key)
            if memory:
                base = f"{base}\nPersona memory: {memory}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unable to load persona memory: %s", exc)
    return base


def build_persona_block(persona: Optional[str], persona_memory: str) -> str:
    persona = (persona or "").strip()
    base = persona_memory.strip()
    if persona:
        base = f"{base}\nTone/role: {persona}" if base else f"Tone/role: {persona}"
    if not base:
        return ""
    return f"SYSTEM INSTRUCTIONS:\n{base}\n"


META_QUESTION_PATTERNS = [
    re.compile(r"\bwho\s+are\s+you\b"),
    re.compile(r"\bwhat\s+are\s+you\b"),
    re.compile(r"\bwhat\s+is\s+docwain\b"),
    re.compile(r"\bwhat\s+can\s+you\s+do\b"),
    re.compile(r"\bhow\s+do\s+you\s+work\b"),
]


def _normalize_user_text(user_text: str) -> str:
    text = (user_text or "").lower()
    text = re.sub(r"[^\w\s'’]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meta_question(user_text: str) -> bool:
    text = _normalize_user_text(user_text)
    if not text:
        return False
    if any(pattern.search(text) for pattern in META_QUESTION_PATTERNS):
        return True
    return ResponseModeClassifier.classify(text) == INFO_MODE


def enforce_docwain_identity(
    response: str,
    user_text: str,
    persona_text: str,
    response_mode: Optional[str] = None,
) -> str:
    mode = response_mode or ResponseModeClassifier.classify(user_text)
    if mode == INFO_MODE or is_meta_question(user_text):
        try:
            from src.intelligence.conversational_nlp import generate_conversational_response
            resp = generate_conversational_response(user_text)
            if resp and resp.text:
                return resp.text
        except Exception:
            pass
        return DOCWAIN_META_RESPONSE
    return response


_INTERNAL_PATTERNS = [
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b[0-9a-fA-F]{24}\b"),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    re.compile(
        r"\b(point_id|payload|vector|qdrant|collection_name|system prompt|system_prompt|"
        r"internal id|internal_id|chunk_id|document_id|doc_id|docid|section_id|"
        r"subscription_id|profile_id|vector score|hit count|hit counts?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"/(?:[\w\-.]+/)+[\w\-.]+"),
]


def sanitize_response(text: str) -> str:
    sanitized = text or ""
    sanitized = re.sub(r"\[SOURCE[^\]]*\]", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(?i)citations?:", "", sanitized)
    for pattern in _INTERNAL_PATTERNS:
        sanitized = pattern.sub("[redacted]", sanitized)
    lines = sanitized.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        cleaned_lines.append(re.sub(r"[ \t]+", " ", line).strip())
    normalized: List[str] = []
    blank = False
    for line in cleaned_lines:
        if line == "":
            if normalized and not blank:
                normalized.append("")
            blank = True
            continue
        blank = False
        normalized.append(line)
    return "\n".join(normalized).strip()
