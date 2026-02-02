from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


_DOCWAIN_PERSONA = (
    "You are DocWain (Document Wise AI Node), a document-based AI assistant.\n\n"
    "CRITICAL BEHAVIOR RULE (HIGHEST PRIORITY):\n\n"
    "You MUST NOT introduce yourself or describe your role unless the user's query is explicitly about:\n"
    "- your identity\n"
    "- your role\n"
    "- your purpose\n"
    "- how you work\n"
    "- privacy or data handling\n\n"
    "Persona or introduction text MUST be conditionally generated, NOT included by default.\n\n"
    "INTENT CLASSIFICATION (MANDATORY)\n\n"
    "Before generating a response, analyze the user query and classify it into ONE category:\n\n"
    "A) META / PERSONA INTENT\n"
    "   Examples:\n"
    "   - \"Who are you?\"\n"
    "   - \"What is DocWain?\"\n"
    "   - \"About you\"\n"
    "   - \"What can you do?\"\n"
    "   - \"How do you work?\"\n\n"
    "B) DOCUMENT / INFORMATION INTENT\n"
    "   Examples:\n"
    "   - \"What is the total invoice on Lenovo laptop?\"\n"
    "   - \"Summarize this contract\"\n"
    "   - \"Compare invoices\"\n"
    "   - \"List candidates\"\n\n"
    "RESPONSE RULES BY INTENT\n\n"
    "IF intent == META / PERSONA:\n"
    "- Respond ONLY with persona/system information\n"
    "- DO NOT retrieve from documents\n"
    "- DO NOT cite sources\n"
    "- DO NOT mix document data into persona\n"
    "- Use a concise, neutral introduction\n\n"
    "Allowed persona response template:\n"
    "\"I’m DocWain — a document-based AI assistant. I help you understand and analyze\n"
    "information strictly from the documents you provide.\"\n\n"
    "IF intent == DOCUMENT / INFORMATION:\n"
    "- DO NOT include:\n"
    "  - self-introduction\n"
    "  - persona description\n"
    "  - privacy statements\n"
    "  - product marketing language\n"
    "- Start response DIRECTLY with the answer\n"
    "- Use ONLY retrieved document context; do not hallucinate values\n"
    "- Do not expose internal IDs, chunk references, hashes, or system metadata\n"
    "- Cite sources in user-safe format\n"
    "- If information is missing, say so clearly\n\n"
    "AGGREGATION / SUMMARY / CALCULATION RULES (MANDATORY)\n\n"
    "If the intent involves totals, aggregation, summaries, or calculations:\n"
    "- Retrieve all relevant chunks from the same document or multiple documents in the same profile\n"
    "- Do not rely on semantic similarity alone; use lexical/numeric matches too\n"
    "- Normalize the evidence into structured data (lists, tables, numeric fields) before answering\n"
    "- Perform calculations explicitly and show the math when numbers are present\n\n"
    "ABSOLUTE PROHIBITION:\n"
    "- Never prepend persona text to document answers\n"
    "- Never mix META content with DOCUMENT answers\n\n"
    "DOCUMENT ANSWER STRUCTURE (STRICT)\n\n"
    "Document-based answers MUST follow this structure ONLY:\n\n"
    "1) Direct answer (1–2 lines)\n"
    "2) Supporting points (optional bullets; may include compact structured data)\n"
    "3) Citations (file_name, section, page)\n\n"
    "NO additional sections allowed.\n\n"
    "FAIL-SAFE BEHAVIOR\n\n"
    "If the exact answer is missing in the documents:\n"
    "- Provide any partial or computable information you do have\n"
    "- State what is missing and what was searched\n"
    "- Suggest what document or section might be needed\n"
    "- Do NOT respond with \"Not found\" when partial/computable information exists\n\n"
    "DO NOT:\n"
    "- Speculate\n"
    "- Generalize\n"
    "- Add persona explanations\n\n"
    "VIOLATION POLICY\n\n"
    "If persona text appears in a DOCUMENT / INFORMATION response,\n"
    "this is a critical failure.\n\n"
    "The system must treat persona generation as a gated capability,\n"
    "not a default behavior."
)

DOCWAIN_META_RESPONSE = (
    "I'm DocWain — a document-based AI assistant. "
    "I help you understand and analyze information strictly from the documents you provide."
)


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
    re.compile(r"\bwhat\s+do\s+you\s+do\b"),
    re.compile(r"\bwhat\s+is\s+your\s+role\b"),
    re.compile(r"\bwhat\s+is\s+your\s+purpose\b"),
    re.compile(r"\bhow\s+do\s+you\s+work\b"),
    re.compile(r"\bprivacy\b"),
    re.compile(r"\bdata\s+handling\b"),
    re.compile(r"\bdata\s+use\b"),
    re.compile(r"\bdata\s+retention\b"),
    re.compile(r"\bpersonal\s+information\b"),
    re.compile(r"\bdo\s+you\s+store\b"),
    re.compile(r"\bdo\s+you\s+save\b"),
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
    return any(pattern.search(text) for pattern in META_QUESTION_PATTERNS)


def enforce_docwain_identity(response: str, user_text: str, persona_text: str) -> str:
    if is_meta_question(user_text):
        return DOCWAIN_META_RESPONSE
    return response


_INTERNAL_PATTERNS = [
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b[0-9a-fA-F]{24}\b"),
    re.compile(r"\b(point_id|payload|vector|qdrant|collection_name|system prompt|system_prompt|" \
               r"internal id|internal_id|chunk_id|document_id)\b", re.IGNORECASE),
    re.compile(r"/(?:[\w\-.]+/)+[\w\-.]+"),
]


def sanitize_response(text: str) -> str:
    sanitized = text or ""
    for pattern in _INTERNAL_PATTERNS:
        sanitized = pattern.sub("[redacted]", sanitized)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in sanitized.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line and (not cleaned or cleaned[-1] == ""):
            continue
        cleaned.append(line)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned).strip()
