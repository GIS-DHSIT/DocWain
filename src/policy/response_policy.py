from __future__ import annotations

import datetime as dt
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.payload_utils import get_source_name

logger = get_logger(__name__)

TASK_MODE = "TASK_MODE"
INFO_MODE = "INFO_MODE"

INFO_KEYWORDS = (
    "document assistant",
    "document-based",
    "what is docwain",
    "who are you",
    "what are you",
    "what can you do",
    "how do you work",
    "how does docwain work",
    # NOTE: removed bare "privacy", "help", "docs", "support", "features",
    # "capabilities" — these caused document queries containing these words
    # (e.g. "What is the privacy policy?", "What features does X have?") to
    # be misclassified as INFO_MODE.  The _INFO_QUERY_PATTERNS regexes
    # already handle the real meta-question variants precisely.
    "documentation",
)

_INFO_QUERY_PATTERNS = (
    re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+docwain\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+(?:you|docwain)\s+work\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:else\s+|all\s+)?can\s+(?:you|docwain)\s+(?:do|help\s+with)\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?what\s+(?:you|docwain)\s+can\s+do\b", re.IGNORECASE),
    re.compile(r"\bhow\s+can\s+(?:you|docwain)\s+help(?:\s+me)?\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:start|begin|use)\s+docwain\b", re.IGNORECASE),
    re.compile(
        r"\bhow\s+(?:do\s+i|can\s+i|to)\s+"
        r"(?:compare|rank|summarize|summarise|extract|list|find|screen|generate|upload|add|import)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:your\s+)?(?:features|capabilities)\b", re.IGNORECASE),
)

TASK_VERB_CUES = (
    "summarize",
    "summarise",
    "compare",
    "rank",
    "extract",
    "analyze",
    "analyse",
    "list",
    "find",
    "identify",
    "review",
    "explain",
    "highlight",
)

DOC_CONTEXT_CUES = (
    "document",
    "file",
    "contract",
    "agreement",
    "section",
    "clause",
    "page",
    "excerpt",
    "report",
)

DOCWAIN_INTRO_SHORT = (
    "Understanding & Scope: Intent: info. Scope: no document retrieval. Files used: none.\n\n"
    "Answer:\n"
    "DocWain Summary:\n"
    "DocWain-Agent is a document intelligence model that answers questions using the documents in your current profile.\n\n"
    "Evidence & Gaps: No document retrieval performed. Files searched: none.\n\n"
    "Next step: Ask a document question or upload documents for analysis."
)

# Phrases to drop from TASK_MODE outputs even if present in model text.
PROHIBITED_TASK_PHRASES = (
    "i’m docwain",
    "i am docwain",
    "docwain",
    "document-based ai assistant",
    "document based ai assistant",
    "do not store",
    "docs section",
)

# Filler/hedge phrases to drop for precision.
BANNED_HEDGE_PHRASES = (
    "not explicitly stated",
    "project details not mentioned",
    "not mentioned",
    "not provided",
    "not specified",
    "not available",
    "cannot be determined",
    "it seems",
    "it appears",
    "likely",
    "possibly",
    "maybe",
)

def _normalize_text(text: str) -> str:
    cleaned = (text or "").lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

class ResponseModeClassifier:
    """Lightweight intent classifier with keyword rules + optional LLM fallback."""

    @staticmethod
    def classify(query: str, llm_client: Optional[Any] = None) -> str:
        normalized = _normalize_text(query)
        if not normalized:
            return TASK_MODE

        if ResponseModeClassifier._has_doc_context_cues(normalized) and "docwain" not in normalized:
            return TASK_MODE

        if ResponseModeClassifier._has_task_cues(normalized):
            if ResponseModeClassifier._explicit_info_request(normalized):
                return INFO_MODE
            return TASK_MODE

        keyword_mode = ResponseModeClassifier._keyword_classify(normalized)
        if keyword_mode:
            return keyword_mode

        if llm_client and ResponseModeClassifier._needs_fallback(normalized):
            return ResponseModeClassifier._llm_fallback(normalized, llm_client)

        return TASK_MODE

    @staticmethod
    def _has_task_cues(text: str) -> bool:
        return any(cue in text for cue in TASK_VERB_CUES)

    @staticmethod
    def _has_doc_context_cues(text: str) -> bool:
        return any(cue in text for cue in DOC_CONTEXT_CUES)

    @staticmethod
    def _keyword_classify(text: str) -> Optional[str]:
        if ResponseModeClassifier._matches_info_patterns(text):
            return INFO_MODE
        if "docwain" in text or "doc wain" in text:
            if ResponseModeClassifier._explicit_info_request(text) or len(text.split()) <= 3:
                return INFO_MODE
            return None
        if any(keyword in text for keyword in INFO_KEYWORDS):
            return INFO_MODE
        return None

    @staticmethod
    def _needs_fallback(text: str) -> bool:
        return ResponseModeClassifier._matches_info_patterns(text)

    @staticmethod
    def _explicit_info_request(text: str) -> bool:
        if ResponseModeClassifier._matches_info_patterns(text):
            return True
        # Only match full meta-question phrases, not bare keywords that can
        # appear in legitimate document content queries.
        info_cues = (
            "what is docwain",
            "who are you",
            "what are you",
            "how do you work",
            "what can you do",
            "documentation",
        )
        return any(cue in text for cue in info_cues)

    @staticmethod
    def _matches_info_patterns(text: str) -> bool:
        return any(pattern.search(text) for pattern in _INFO_QUERY_PATTERNS)

    @staticmethod
    def _llm_fallback(text: str, llm_client: Any) -> str:
        prompt = (
            "Classify the user request into exactly one label: INFO_MODE or TASK_MODE.\n"
            "INFO_MODE = questions about DocWain itself (what it is, how it works, privacy, docs/help).\n"
            "TASK_MODE = questions about document content (summarize, compare, extract, analyze).\n"
            "Return only the label.\n"
            f"User message: {text}\n"
            "Label:"
        )
        try:
            raw = str(llm_client.generate(prompt) or "").strip().upper()
            if "INFO_MODE" in raw:
                return INFO_MODE
            if "TASK_MODE" in raw:
                return TASK_MODE
        except Exception as exc:  # noqa: BLE001
            logger.debug("Response mode LLM fallback failed: %s", exc)
        return TASK_MODE

@dataclass(frozen=True)
class EvidenceSnippet:
    source_id: int
    doc_name: str
    chunk_id: Optional[str]
    exact_snippet: str
    tags: Dict[str, Any]
    tokens: Tuple[str, ...]
    numbers: Tuple[str, ...]

@dataclass
class EvidenceLedger:
    entries: List[EvidenceSnippet]
    by_source_id: Dict[int, EvidenceSnippet]
    source_names: Tuple[str, ...]
    source_name_variants: Tuple[str, ...]

def _chunk_text(chunk: Any) -> str:
    if hasattr(chunk, "text"):
        return str(chunk.text or "")
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return ""

def _chunk_meta(chunk: Any) -> Dict[str, Any]:
    if hasattr(chunk, "metadata"):
        return dict(chunk.metadata or {})
    if isinstance(chunk, dict):
        return dict(chunk.get("metadata") or {})
    return {}

def _chunk_source_name(chunk: Any) -> str:
    meta = _chunk_meta(chunk)
    return (
        getattr(chunk, "source", None)
        or get_source_name(meta)
        or meta.get("source_name")
        or ""
    )

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "as", "at", "from", "that", "this", "it",
    "their", "they", "them", "its", "these", "those", "which", "who", "whom",
    "can", "could", "may", "might", "will", "would", "should", "has", "have",
    "had", "not", "no",
}

def _content_tokens(tokens: Iterable[str]) -> List[str]:
    filtered = []
    for token in tokens:
        if any(ch.isdigit() for ch in token):
            filtered.append(token)
            continue
        if token in _STOPWORDS:
            continue
        if len(token) > 2:
            filtered.append(token)
    return filtered

def _extract_numbers(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if any(ch.isdigit() for ch in t)]

def _normalize_source_names(names: Iterable[str]) -> Tuple[str, ...]:
    variants = []
    for name in names:
        if not name:
            continue
        lowered = name.lower()
        variants.append(lowered)
        base = lowered.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        variants.append(base)
    return tuple(sorted(set(variants)))

def _match_source_to_chunk(source: Dict[str, Any], chunks: Sequence[Any]) -> Optional[Any]:
    if not chunks:
        return None
    source_name = _normalize_text(source.get("source_name") or "")
    excerpt = _normalize_text(source.get("excerpt") or "")
    best_chunk = None
    best_score = 0.0
    for chunk in chunks:
        chunk_name = _normalize_text(_chunk_source_name(chunk))
        chunk_text = _normalize_text(_chunk_text(chunk))
        score = 0.0
        if source_name and source_name == chunk_name:
            score += 2.5
        if excerpt:
            if excerpt in chunk_text or chunk_text in excerpt:
                score += 3.0
            else:
                excerpt_tokens = set(_content_tokens(_tokenize(excerpt)))
                chunk_tokens = set(_content_tokens(_tokenize(chunk_text)))
                if excerpt_tokens and chunk_tokens:
                    overlap = len(excerpt_tokens & chunk_tokens) / max(1, len(excerpt_tokens))
                    score += overlap
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk if best_score > 0 else None

def build_evidence_ledger(chunks: Sequence[Any], sources: Sequence[Dict[str, Any]]) -> EvidenceLedger:
    entries: List[EvidenceSnippet] = []
    source_names: List[str] = []
    for source in sources or []:
        try:
            source_id = int(source.get("source_id") or 0)
        except Exception:
            source_id = 0
        if source_id <= 0:
            continue
        matched_chunk = _match_source_to_chunk(source, chunks)
        meta = _chunk_meta(matched_chunk) if matched_chunk is not None else {}
        doc_name = str(source.get("source_name") or _chunk_source_name(matched_chunk) or "").strip() or "Document"
        snippet_text = _chunk_text(matched_chunk) or str(source.get("excerpt") or "")
        chunk_id = meta.get("chunk_id") or meta.get("chunk") or None
        tags = {
            "document_id": meta.get("document_id"),
            "document_type": meta.get("document_type") or meta.get("doc_type"),
            "document_category": meta.get("document_category"),
            "section_title": meta.get("section_title") or meta.get("section"),
            "page": meta.get("page"),
            "chunk_kind": meta.get("chunk_kind") or meta.get("chunk_type"),
            "detected_language": meta.get("detected_language"),
        }
        tokens = tuple(_content_tokens(_tokenize(snippet_text)))
        numbers = tuple(_extract_numbers(tokens))
        entries.append(
            EvidenceSnippet(
                source_id=source_id,
                doc_name=doc_name,
                chunk_id=str(chunk_id) if chunk_id else None,
                exact_snippet=snippet_text,
                tags=tags,
                tokens=tokens,
                numbers=numbers,
            )
        )
        if doc_name:
            source_names.append(doc_name)

    name_variants = _normalize_source_names(source_names)
    by_source_id = {entry.source_id: entry for entry in entries}
    return EvidenceLedger(entries=entries, by_source_id=by_source_id, source_names=tuple(source_names), source_name_variants=name_variants)

def build_docwain_intro(*, query: str = "") -> str:
    if query:
        try:
            from src.intelligence.conversational_nlp import generate_conversational_response
            resp = generate_conversational_response(query)
            if resp and resp.text:
                return resp.text
        except Exception:
            pass
    return DOCWAIN_INTRO_SHORT

def _extract_citation_ids(text: str) -> List[int]:
    ids = []
    for match in re.findall(r"SOURCE-(\d+)", text):
        try:
            ids.append(int(match))
        except Exception:
            continue
    return ids

def _strip_citations(text: str) -> str:
    return re.sub(r"\[SOURCE-[^\]]+\]", "", text).strip()

def _contains_unretrieved_doc_name(text: str, source_name_variants: Tuple[str, ...]) -> bool:
    matches = list(re.finditer(r"\b[\w\-.]+\.(?:pdf|docx|pptx|xlsx|csv|txt)\b", text, flags=re.IGNORECASE))
    if not matches:
        return False
    for match in matches:
        token = match.group(0)
        if token and token.lower() not in source_name_variants:
            return True
    return False

def _has_banned_phrases(text: str, phrases: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    return any(phrase in normalized for phrase in phrases)

def _extract_dates(text: str) -> List[Tuple[int, dt.date]]:
    results: List[Tuple[int, dt.date]] = []
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            year, month, day = map(int, match.groups())
            results.append((match.start(), dt.date(year, month, day)))
        except Exception:
            continue
    for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text):
        try:
            month, day, year = map(int, match.groups())
            results.append((match.start(), dt.date(year, month, day)))
        except Exception:
            continue
    month_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    month_regex = r"\b(" + "|".join(month_map.keys()) + r")\s+(\d{1,2}),?\s+(\d{4})\b"
    for match in re.finditer(month_regex, text, flags=re.IGNORECASE):
        try:
            month_key = match.group(1).lower()
            day = int(match.group(2))
            year = int(match.group(3))
            month = month_map.get(month_key)
            if month:
                results.append((match.start(), dt.date(year, month, day)))
        except Exception:
            continue
    return sorted(results, key=lambda x: x[0])

def _has_conflicting_date_range(text: str) -> bool:
    dates = _extract_dates(text)
    if len(dates) < 2:
        return False
    normalized = _normalize_text(text)
    range_markers = ("from", "to", "between", "and", "through", "until", "-", "–", "—")
    if not any(marker in normalized for marker in range_markers):
        return False
    first_date = dates[0][1]
    second_date = dates[1][1]
    return first_date > second_date

def _is_supported_by_evidence(claim: str, citation_ids: List[int], ledger: EvidenceLedger) -> bool:
    stripped = _strip_citations(claim)
    tokens = _content_tokens(_tokenize(stripped))
    if not tokens:
        return False
    numbers = _extract_numbers(tokens)
    quoted = re.findall(r'"([^"]+)"', claim) + re.findall(r"'([^']+)'", claim)
    for cid in citation_ids:
        snippet = ledger.by_source_id.get(cid)
        if not snippet:
            continue
        if numbers:
            if not all(num in snippet.numbers for num in numbers):
                continue
        if quoted:
            if not all(q.lower() in snippet.exact_snippet.lower() for q in quoted):
                continue
        overlap = len(set(tokens) & set(snippet.tokens)) / max(1, len(set(tokens)))
        if overlap >= 0.35:
            return True
    return False

def apply_evidence_gate(answer: str, ledger: EvidenceLedger, response_mode: str = TASK_MODE) -> Tuple[str, Dict[str, Any]]:
    if response_mode != TASK_MODE or not answer:
        return answer, {"used_source_ids": []}

    lines = answer.splitlines()
    cleaned_lines: List[str] = []
    used_sources: set[int] = set()

    for line in lines:
        raw = line.rstrip("\n")
        if not raw.strip():
            cleaned_lines.append(raw)
            continue

        heading = raw.strip().endswith(":") and "[" not in raw
        if heading:
            cleaned_lines.append(raw)
            continue

        bullet_match = re.match(r"^(\s*-\s+)(.+)$", raw)
        prefix = bullet_match.group(1) if bullet_match else ""
        content = bullet_match.group(2) if bullet_match else raw

        sentences = [content.strip()] if bullet_match else re.split(r"(?<=[.!?])\s+", content.strip())
        kept_sentences: List[str] = []
        for sentence in sentences:
            if not sentence:
                continue
            if _has_banned_phrases(sentence, BANNED_HEDGE_PHRASES):
                continue
            if _has_banned_phrases(sentence, PROHIBITED_TASK_PHRASES):
                continue
            if _has_conflicting_date_range(sentence):
                continue
            citation_ids = _extract_citation_ids(sentence)
            if not citation_ids:
                continue
            if any(cid not in ledger.by_source_id for cid in citation_ids):
                continue
            if _contains_unretrieved_doc_name(sentence, ledger.source_name_variants):
                continue
            if not _is_supported_by_evidence(sentence, citation_ids, ledger):
                continue
            used_sources.update(citation_ids)
            kept_sentences.append(sentence)

        if kept_sentences:
            cleaned_line = prefix + " ".join(kept_sentences)
            cleaned_lines.append(cleaned_line)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        cleaned = "No relevant information found in the retrieved documents."
    return cleaned, {"used_source_ids": sorted(used_sources)}
