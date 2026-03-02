from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.payload_utils import get_document_type, get_source_name


_REFUSAL_PHRASES = (
    "i don’t have enough information",
    "i don't have enough information",
    "i do not have enough information",
    "i cannot find",
    "i can't find",
    "cannot find",
    "can't find",
    "not in the documents",
    "not in the provided documents",
    "not provided in the documents",
    "no relevant information",
    "no information in the documents",
    "unable to answer",
    "i’m unable to answer",
    "i am unable to answer",
)


def is_effectively_empty(text: Optional[str], *, min_chars: int = 30) -> bool:
    value = (text or "").strip()
    return len(value) < min_chars


def is_bad_refusal(text: Optional[str]) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def _chunk_meta(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        meta = chunk.get("metadata") or {}
    else:
        meta = getattr(chunk, "metadata", None) or {}
    return meta if isinstance(meta, dict) else {}


def _chunk_source(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        return chunk.get("source") or chunk.get("source_name")
    return getattr(chunk, "source", None)


def _normalize_lines(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def _compute_boilerplate_lines(texts: Sequence[str]) -> set[str]:
    freq: Dict[str, int] = {}
    for text in texts:
        for line in set(_normalize_lines(text)):
            if len(line) < 24:
                continue
            freq[line] = freq.get(line, 0) + 1
    # "generic heuristics": remove lines that repeat across multiple chunks
    return {line for line, count in freq.items() if count >= 3}


def compress_context_for_retry(
    chunks: Sequence[Any],
    *,
    top_k: int = 6,
    max_excerpt_chars: int = 1200,
) -> str:
    """Compact context representation to reduce prompt tokens deterministically."""
    selected = list(chunks or [])[:top_k]
    boilerplate = _compute_boilerplate_lines([_chunk_text(c) for c in selected])

    blocks: List[str] = []
    for chunk in selected:
        meta = _chunk_meta(chunk)
        source_name = _chunk_source(chunk) or get_source_name(meta) or "Unknown"
        doc_type = get_document_type(meta) or meta.get("doc_type") or meta.get("type") or ""
        category = meta.get("document_category") or meta.get("category") or "unknown"
        language = meta.get("detected_language") or meta.get("language") or "unknown"

        lines = []
        seen = set()
        for line in _normalize_lines(_chunk_text(chunk)):
            if line in boilerplate:
                continue
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned[:max_excerpt_chars].rstrip()

        header_bits = [str(source_name).strip()]
        if doc_type:
            header_bits.append(str(doc_type).strip())
        header = " / ".join([b for b in header_bits if b])
        blocks.append(f"[{header} | {category} | {language}]\n{cleaned}".strip())

    return "\n\n".join([b for b in blocks if b]).strip()


def _score_bucket(value: str, confidence: Optional[float]) -> float:
    try:
        conf = float(confidence) if confidence is not None else 0.0
    except Exception:
        conf = 0.0
    # If confidence is missing, still count weakly to avoid "unknown" dominating.
    if conf <= 0.0:
        conf = 0.2
    return conf


def derive_manifest(chunks: Sequence[Any], *, max_sources: int = 10) -> Tuple[str, str, str, List[str]]:
    """
    Returns: (manifest_text, dominant_language, dominant_category, source_titles)
    """
    lang_scores: Dict[str, float] = {}
    cat_scores: Dict[str, float] = {}
    titles: List[str] = []

    for chunk in chunks or []:
        meta = _chunk_meta(chunk)
        lang = str(meta.get("detected_language") or meta.get("language") or "unknown").strip().lower() or "unknown"
        cat = str(meta.get("document_category") or meta.get("category") or "others").strip().lower() or "others"

        lang_scores[lang] = lang_scores.get(lang, 0.0) + _score_bucket(lang, meta.get("language_confidence"))
        cat_scores[cat] = cat_scores.get(cat, 0.0) + _score_bucket(cat, meta.get("category_confidence"))

        title = _chunk_source(chunk) or get_source_name(meta)
        if title:
            title = str(title).strip()
            if title and title not in titles:
                titles.append(title)

    dominant_language = max(lang_scores.items(), key=lambda kv: kv[1])[0] if lang_scores else "unknown"
    dominant_category = max(cat_scores.items(), key=lambda kv: kv[1])[0] if cat_scores else "others"

    source_lines: List[str] = []
    for title in titles[:max_sources]:
        doc_type = ""
        for chunk in chunks or []:
            meta = _chunk_meta(chunk)
            if (get_source_name(meta) or _chunk_source(chunk) or "") == title:
                doc_type = str(get_document_type(meta) or meta.get("doc_type") or "").strip()
                break
        if doc_type:
            source_lines.append(f"- {title} (type={doc_type})")
        else:
            source_lines.append(f"- {title}")

    manifest = (
        "CONTEXT MANIFEST\n"
        f"- dominant_detected_language: {dominant_language}\n"
        f"- dominant_document_category: {dominant_category}\n"
        "SOURCES (names only)\n"
        + ("\n".join(source_lines) if source_lines else "- (none)")
    ).strip()

    return manifest, dominant_language, dominant_category, titles[:max_sources]


@dataclass(frozen=True)
class SchemaSpec:
    schema_id: str
    title: str
    instructions: str
    required_columns: Tuple[str, ...] = ()


_CANDIDATE_TABLE_COLUMNS = (
    "name",
    "total_experience",
    "summary",
    "technical_skills",
    "functional_skills",
    "certifications",
    "education",
    "achievements",
    "source",
)


def select_schema(query: str) -> SchemaSpec:
    q = (query or "").lower()
    if "extract" in q and ("candidate" in q or "candidates" in q or "resume" in q or "cv" in q):
        return SchemaSpec(
            schema_id="S3",
            title="Candidate Comparison Table",
            instructions=(
                "Output a single table of candidates.\n"
                "Use a Markdown table.\n"
                "If a value is not found, leave it blank or write 'Missing in retrieved excerpts'.\n"
                "Do not default everything to 'Not mentioned'. If safe, infer and label as 'Inferred: ...'."
            ),
            required_columns=_CANDIDATE_TABLE_COLUMNS,
        )
    if any(token in q for token in ("compare", "comparison", "vs", "versus", "rank", "ranking")):
        return SchemaSpec(
            schema_id="S3",
            title="Comparison/Ranking Table",
            instructions=(
                "Prefer a table for comparisons/rankings.\n"
                "Explain ranking criteria briefly before the table.\n"
                "Keep claims grounded in retrieved excerpts; label inferred items."
            ),
        )
    if any(token in q for token in ("timeline", "chronology", "when did", "history")):
        return SchemaSpec(
            schema_id="S4",
            title="Timeline",
            instructions="Output a chronological list with dates (if present) and brief events; mark missing dates explicitly.",
        )
    return SchemaSpec(
        schema_id="S1",
        title="Structured Answer",
        instructions=(
            "Use clear sections and bullets.\n"
            "Include: Answer, Key facts from excerpts, Missing/unclear.\n"
            "If you must infer, label it clearly as 'Inferred: ...'."
        ),
    )


def validate_schema_output(text: str, schema: SchemaSpec) -> Tuple[bool, str]:
    if not schema.required_columns:
        return True, "no_required_columns"
    lowered = (text or "").lower()
    # Require all column names to appear at least once; tolerant of spacing.
    missing = [col for col in schema.required_columns if col not in lowered]
    if missing:
        return False, f"missing_columns:{','.join(missing)}"
    # Require a markdown-style table with at least one data row.
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if "|" in ln]
    if len(pipe_lines) < 3:
        return False, "table_too_short"
    # Heuristic: include a separator line with dashes.
    if not any(re.search(r"\|\s*:?-{3,}", ln) for ln in pipe_lines[1:3]):
        return False, "missing_table_separator"
    return True, "ok"


def build_finalizer_prompt(
    *,
    draft_thinking: str,
    query: str,
    context_titles: Sequence[str],
    manifest_text: str,
    schema: SchemaSpec,
    max_tokens_hint: str,
) -> str:
    titles = "\n".join(f"- {t}" for t in (context_titles or [])[:12]) or "- (none)"
    return (
        "You are DocWain-Agent. Convert the draft below into the FINAL user-facing answer.\n\n"
        "RULES (MANDATORY)\n"
        "- Do NOT include internal reasoning, chain-of-thought, or scratchpad.\n"
        "- Stay grounded in the retrieved document titles; do not invent new facts.\n"
        "- If information is missing, say what is missing and provide partial output.\n"
        f"- {max_tokens_hint}\n\n"
        f"{manifest_text}\n\n"
        "RESPONSE SCHEMA SELECTOR (MANDATORY)\n"
        f"- Chosen schema: {schema.schema_id} ({schema.title})\n"
        f"- Instructions: {schema.instructions}\n\n"
        "RETRIEVED CONTEXT (titles only)\n"
        f"{titles}\n\n"
        "USER QUESTION\n"
        f"{query}\n\n"
        "DRAFT (do not expose as-is)\n"
        f"{draft_thinking}\n\n"
        "FINAL ANSWER:"
    ).strip()
