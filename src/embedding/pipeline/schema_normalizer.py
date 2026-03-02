from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.embedding.pipeline.content_classifier import classify_doc_domain, classify_section_kind, classify_section_kind_with_source
from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text
from src.metadata.normalizer import normalize_chunk_kind
from src.utils.payload_utils import token_count


EMBED_PIPELINE_VERSION = "dwx-2026-02-05"


def validate_chunk_for_embedding(text: str, max_tokens: int = 2048) -> Tuple[bool, str]:
    """Validate a chunk before embedding.

    Returns (is_valid, reason).
    """
    if not text or not text.strip():
        return False, "empty_or_whitespace"
    stripped = text.strip()
    # Reject very short chunks
    if len(stripped) < 20:
        return False, "too_short"
    # Reject chunks exceeding token limit (rough estimate: 4 chars per token)
    if len(stripped) > max_tokens * 4:
        return False, "exceeds_token_limit"
    # Reject metadata garbage
    if _is_chunk_metadata_garbage(stripped):
        return False, "metadata_garbage"
    return True, "ok"


def _is_chunk_metadata_garbage(text: str) -> bool:
    """Detect chunks that are pure metadata noise."""
    # More than 60% non-alphanumeric characters
    alpha_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / max(1, len(text))
    if alpha_ratio < 0.4:
        return True
    # Very repetitive content (same line repeated)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 3:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.3:
            return True
    return False


# Matches C0/C1 control characters EXCEPT \t (0x09), \n (0x0a), \r (0x0d).
# Strips NULL, backspace, vertical tab, form feed, escape sequences, etc.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _strip_control_chars(text: str) -> str:
    """Remove non-printable control characters that corrupt embeddings."""
    return _CONTROL_CHAR_RE.sub("", text) if text else ""


_DOC_CONNECTOR_MAP = {
    "local": "LOCAL",
    "filesystem": "LOCAL",
    "file": "LOCAL",
    "s3": "S3",
    "aws_s3": "S3",
    "ftp": "FTP",
    "sftp": "FTP",
    "azure_blob": "AZURE_BLOB",
    "azure": "AZURE_BLOB",
    "blob": "AZURE_BLOB",
}


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _intify(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _floatify(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _section_path_list(section_path: Optional[Any]) -> Optional[list[str]]:
    if not section_path:
        return None
    if isinstance(section_path, list):
        parts = [str(part).strip() for part in section_path if str(part).strip()]
        return parts or None
    text = str(section_path)
    parts = [part.strip() for part in text.split(">") if part.strip()]
    return parts or None


def _infer_file_type(source_name: Optional[str]) -> Optional[str]:
    if not source_name:
        return None
    suffix = Path(source_name).suffix.lower().lstrip(".")
    return suffix or None


def _normalize_connector_type(value: Optional[str]) -> str:
    if not value:
        return "LOCAL"
    key = str(value).strip().lower()
    return _DOC_CONNECTOR_MAP.get(key, key.upper())


def _doc_version_hash(raw: Dict[str, Any], *, canonical_text: str) -> str:
    provided = _stringify(raw.get("doc_version_hash") or raw.get("docVersionHash"))
    if provided:
        return provided
    seed = _stringify(raw.get("doc_version_seed")) or _stringify(raw.get("full_text")) or canonical_text
    digest = hashlib.sha1((seed or "").encode("utf-8")).hexdigest()
    return digest[:12]


def normalize_content(text: str) -> str:
    if not text:
        return ""
    # Strip UTF-16 null byte artifacts before any other processing
    if '\x00' in text:
        text = text.replace('\x00', '')
    normalized = _strip_control_chars(str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Only safe digit-letter boundary splitting (preserves camelCase, compound words)
    normalized = re.sub(r"([A-Za-z])(\d)", r"\1 \2", normalized)
    normalized = re.sub(r"(\d)([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"\s*&\s*", " & ", normalized)
    normalized = re.sub(r"\s*—\s*", " — ", normalized)

    normalized = re.sub(r"(?<!^)(?<!\n)\s*([•●])", r"\n\1", normalized)
    lines = []
    for line in normalized.split("\n"):
        compacted = re.sub(r"[ \t]+", " ", line).strip()
        if compacted:
            lines.append(compacted)
    return "\n".join(lines).strip()


def _is_encoding_garbage(text: str) -> bool:
    """Detect text corrupted by wrong encoding (e.g. UTF-16 decoded as UTF-8).

    Signatures:
    - High density of null chars (\\x00) or replacement chars (U+FFFD)
    - Very low alphanumeric ratio (most chars are control/garbage)
    """
    if not text or len(text) < 20:
        return False
    sample = text[:500]
    # Null byte interleaving (UTF-16 artifact)
    null_count = sample.count('\x00')
    if null_count > len(sample) * 0.1:
        return True
    # Replacement char flood
    repl_count = sample.count('\ufffd')
    if repl_count > len(sample) * 0.1:
        return True
    # Very low alphanumeric ratio (less than 30% of chars are letters/digits/spaces)
    alnum_count = sum(1 for c in sample if c.isalnum() or c in ' \n\t')
    if len(sample) > 50 and alnum_count / len(sample) < 0.3:
        return True
    return False


_METADATA_GARBAGE_MARKERS = (
    "'chunk_type':", "'section_id':", "'section_title':", "'page': None",
    "section_id :", "chunk_type :", "section_title :",
    "Chunk Candidate text", "chunk_candidates Chunk",
    # Space-delimited format (no colon between key and value)
    ", section_id ", ", chunk_type ", ", section_title ",
    ", start_page ", ", end_page ", "text : ",
)
# Strong markers: a single occurrence means garbage (never in real document text)
_STRONG_GARBAGE_MARKERS = (
    "Extracted Document full_text",
    "Extracted Document (full_text=",
    "ExtractedDocument(",
    "Section section_id",
    "start_page 1, end_page",
    "Chunk Candidate text",
)


def _is_metadata_garbage(text: str) -> bool:
    """Detect text that is actually stringified chunk metadata dicts or ExtractedDocument repr."""
    if not text or len(text) < 30:
        return False
    if _is_encoding_garbage(text):
        return True
    # Definitive: text IS a Python repr of an internal object (prefix is unmistakable)
    if text.startswith("Extracted Document (") or text.startswith("ExtractedDocument("):
        return True
    # Long texts (>500 chars) are real content even if they contain strong/soft markers.
    # Markers like "Section section_id" or "start_page 1, end_page" can appear
    # incidentally in large documents (e.g. SAP technical docs discussing DB schemas).
    # Only the definitive prefix check above should flag long texts as garbage.
    if len(text) > 500:
        return False
    if any(m in text for m in _STRONG_GARBAGE_MARKERS):
        return True
    # Multiple "text :" or ", text " segments = metadata key contamination
    if text.count("text : ") >= 2 or text.count(", text ") >= 2:
        return True
    return sum(1 for marker in _METADATA_GARBAGE_MARKERS if marker in text) >= 2


_SECTION_PREFIX_RE = re.compile(r"^\[[\w\s]+\]\s*(?:Section\s*\d+\s*:\s*)?(?:[\w\s&/,-]+:\s*)?")
_METADATA_KEY_RE = re.compile(
    r"^(?:section_id|section_title|chunk_type|page|start_page|end_page"
    r"|tables|figures|level|Section section_id|title)\s",
    re.IGNORECASE,
)
_CHUNK_CANDIDATE_RE = re.compile(
    r"^(?:chunk_candidates\s+)?Chunk\s+Candidate\s+text\s*(.*)",
    re.IGNORECASE,
)
# "text <actual content>" — strip key prefix, keep value
_TEXT_KEY_RE = re.compile(r"^text\s+:?\s*", re.IGNORECASE)


def _strip_section_prefix(text: str) -> str:
    """Strip [Section Kind] Section N: prefix from embedding text."""
    return _SECTION_PREFIX_RE.sub("", text).strip() if text else ""


_EXTRACTED_DOC_PREFIX_RE = re.compile(
    r"^(?:Extracted\s+Document\s+(?:full_text|[\w.]+)\s*)", re.IGNORECASE
)


def _clean_metadata_fragments(text: str) -> str:
    """Strip metadata key-value fragments that are comma-separated with actual content."""
    if not text:
        return ""
    # Strip "Extracted Document full_text" prefix
    text = _EXTRACTED_DOC_PREFIX_RE.sub("", text).strip()
    text = text.lstrip("-").strip()
    parts = re.split(r"\s*,\s*", text)
    content_parts: list = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _METADATA_KEY_RE.match(part) and len(part) < 40:
            continue
        m = _CHUNK_CANDIDATE_RE.match(part)
        if m:
            remainder = m.group(1).strip()
            if remainder:
                content_parts.append(remainder)
            continue
        # "text <content>" — strip key prefix, keep the value
        tm = _TEXT_KEY_RE.match(part)
        if tm:
            remainder = part[tm.end():].strip()
            if remainder and len(remainder) > 3:
                content_parts.append(remainder)
            continue
        content_parts.append(part)
    result = ", ".join(content_parts) if content_parts else ""
    return re.sub(r"\s{2,}", " ", result).strip()


def build_qdrant_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    subscription_id = _stringify(raw.get("subscription_id") or raw.get("subscriptionId") or raw.get("subscription"))
    profile_id = _stringify(raw.get("profile_id") or raw.get("profileId") or raw.get("profile"))
    document_id = _stringify(raw.get("document_id") or raw.get("documentId") or raw.get("doc_id") or raw.get("docId"))
    if not subscription_id or not profile_id or not document_id:
        missing = [name for name, value in (("subscription_id", subscription_id), ("profile_id", profile_id), ("document_id", document_id)) if not value]
        raise ValueError(f"Missing required payload fields: {', '.join(missing)}")

    source_name = _stringify(raw.get("source_name") or raw.get("sourceName"))
    if not source_name:
        source_name = _stringify((raw.get("source") or {}).get("name")) or "unknown"

    connector_type = _normalize_connector_type(
        _stringify(raw.get("connector_type") or raw.get("ingestion_source") or (raw.get("document") or {}).get("ingestion_source"))
    )
    file_type = _stringify(raw.get("file_type")) or _infer_file_type(source_name) or "unknown"
    mime_type = _stringify(raw.get("mime_type") or raw.get("mimeType"))

    doc_domain = _stringify(raw.get("doc_domain") or (raw.get("document") or {}).get("domain")) or "unknown"

    # Filename-based domain override — takes precedence over pickle classification
    # because pickle classification can be wrong (e.g., SCM resume classified as purchase_order)
    if source_name:
        from src.embedding.pipeline.content_classifier import _FILENAME_HINTS
        fn_lower = source_name.lower()
        for hint, hint_domain in _FILENAME_HINTS.items():
            if hint in fn_lower:
                doc_domain = hint_domain
                break

    section_id = _stringify(raw.get("section_id") or (raw.get("section") or {}).get("id")) or "unknown"
    section_title = _stringify(raw.get("section_title") or (raw.get("section") or {}).get("title") or "Section")
    section_path = _section_path_list(raw.get("section_path") or (raw.get("section") or {}).get("path") or section_title)
    section_kind = _stringify(raw.get("section_kind") or (raw.get("section") or {}).get("kind")) or "misc"
    section_confidence = _floatify(raw.get("section_confidence") or (raw.get("section") or {}).get("confidence"), default=0.5)
    section_salience = _floatify(raw.get("section_salience") or (raw.get("section") or {}).get("salience"), default=0.5)

    page_start = _intify(raw.get("page_start") or raw.get("page") or raw.get("page_number")) or 0
    page_end = _intify(raw.get("page_end")) or page_start
    chunk_id = _stringify(raw.get("chunk_id") or (raw.get("chunk") or {}).get("id")) or "unknown"
    chunk_index = _intify(raw.get("chunk_index") or (raw.get("chunk") or {}).get("index") or 0) or 0
    chunk_count = _intify(raw.get("chunk_count") or (raw.get("chunk") or {}).get("count")) or 1
    chunk_kind = normalize_chunk_kind(
        {
            "chunk_kind": raw.get("chunk_kind") or (raw.get("chunk") or {}).get("type"),
            "chunk_type": raw.get("chunk_type") or (raw.get("chunk") or {}).get("type"),
        },
        strict=False,
    )
    chunking_mode = _stringify(raw.get("chunking_mode"))

    raw_content = raw.get("content")
    if raw_content is None:
        raw_content = raw.get("text") or raw.get("text_raw") or raw.get("text_clean") or (raw.get("text_data") or {}).get("clean")
    content = normalize_content(raw_content or "")

    canonical_text = _stringify(raw.get("canonical_text")) or content
    if _is_metadata_garbage(canonical_text):
        canonical_text = content
    canonical_text = normalize_content(canonical_text or content)
    if _is_metadata_garbage(canonical_text):
        # Try cleaning metadata fragments instead of emptying
        cleaned = _clean_metadata_fragments(canonical_text)
        if cleaned and len(cleaned) > 20 and not _is_metadata_garbage(cleaned):
            canonical_text = cleaned
        else:
            canonical_text = ""

    # Fallback: derive canonical_text from embedding_text (strip prefix + metadata)
    if not canonical_text:
        et = _stringify(raw.get("embedding_text") or "")
        if et:
            et = _strip_section_prefix(et)
            et = _clean_metadata_fragments(et)
            if et and not _is_metadata_garbage(et) and len(et) > 20:
                canonical_text = normalize_content(et)

    # ── Classify generic section_kind / doc_domain ──
    section_kind_source = _stringify(raw.get("section_kind_source")) or "content"
    # Section kinds that belong to invoice/purchase_order domain — if the doc is actually
    # a resume, these must be re-classified to get correct HR section kinds.
    _INVOICE_ONLY_KINDS = {
        "financial_summary", "line_items", "invoice_metadata",
        "parties_addresses", "terms_conditions",
    }
    needs_reclassify = (
        section_kind in ("misc", "section_text", "unknown")
        or (doc_domain == "resume" and section_kind in _INVOICE_ONLY_KINDS)
    )
    if needs_reclassify:
        section_kind, section_kind_source = classify_section_kind_with_source(canonical_text, section_title or "")
    # NOTE: Do NOT re-classify doc_domain per-chunk from chunk text.
    # Document-level classification (dataHandler.py) is the single source of truth.
    # Per-chunk classification causes inconsistent domains for the same document_id
    # (e.g., resume address chunks misclassified as "purchase_order").
    # Filename hints above (lines 229-235) are still applied as they are consistent.

    # ── Strip wrong upstream prefixes from embedding_text ──
    # The universal_enhancer may have applied a semantic prefix (e.g.,
    # "Contact Information:") based on incorrect ContentTypeDetector
    # classification (e.g., product codes matching phone regex).
    # Strip any known prefix so we can re-classify and re-prefix correctly.
    _KNOWN_PREFIXES_RE = re.compile(
        r"^(?:Contact Information|Skills and (?:Competencies|Technologies)"
        r"|Financial Information|Education and Qualifications"
        r"|Professional (?:Experience|Summary)|Medical Information"
        r"|Legal Terms|Work Experience|Technical Skills"
        r"|Certifications(?: and Credentials)?|Invoice (?:Items|Totals)"
        r"|Billing Information|Data|Payment Terms)\s*:\s*",
        re.IGNORECASE,
    )
    embedding_text = _stringify(raw.get("embedding_text") or raw.get("text_clean") or raw.get("embeddingText"))
    if embedding_text:
        embedding_text = _KNOWN_PREFIXES_RE.sub("", embedding_text, count=1).strip() or None
    if embedding_text and _is_metadata_garbage(embedding_text):
        embedding_text = None
    if not embedding_text:
        embedding_text = ensure_embedding_text(content, doc_domain, section_kind)
    if embedding_text.strip() == content.strip():
        embedding_text = ensure_embedding_text(content, doc_domain, section_kind)

    # ── Enrich embedding_text with section prefix for vector awareness ──
    # Only add prefix when section_kind was determined from a clear title match
    # (high confidence).  Content-keyword-derived kinds are lower confidence and
    # the prefix biases the embedding vector toward labels rather than content.
    if (
        section_kind not in ("misc", "section_text", "unknown", None)
        and section_kind_source == "title"
    ):
        pretty = section_kind.replace("_", " ").title()
        prefix = f"[{pretty}]"
        if section_title and section_title.lower() not in ("untitled section", "section"):
            prefix += f" {section_title}:"
        embedding_text = f"{prefix} {embedding_text}"

    canonical_text_len = len(canonical_text or "")
    canonical_token_count = token_count(canonical_text or "")

    chunk_hash = _stringify(raw.get("hash") or (raw.get("chunk") or {}).get("hash"))
    if not chunk_hash:
        chunk_hash = hashlib.sha256((embedding_text or content).encode("utf-8")).hexdigest()

    detected_language = _stringify(raw.get("detected_language") or raw.get("language"))
    language_confidence = raw.get("language_confidence")
    try:
        if language_confidence is not None:
            language_confidence = _floatify(language_confidence, default=0.0)
    except Exception:
        language_confidence = None
    languages = raw.get("languages")
    if isinstance(languages, str):
        languages = [languages]

    # ── Slim payload: ~15 core fields + embedding_text ──
    # Retrieval layer (_to_chunk) reads canonical_text, source_name, page, chunk_id
    # from flat fields with fallback to nested objects, so old data keeps working.
    payload: Dict[str, Any] = {
        # identity
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": document_id,
        # text
        "canonical_text": canonical_text,
        "embedding_text": embedding_text,
        # source
        "source_name": source_name,
        # section
        "section_id": section_id,
        "section_kind": section_kind,
        "section_title": section_title,
        # location
        "page": page_start,
        "chunk_index": chunk_index,
        # classification
        "doc_domain": doc_domain,
        "chunk_kind": chunk_kind,
        "chunk_id": chunk_id,
        # integrity
        "hash": chunk_hash,
        "embed_pipeline_version": _stringify(raw.get("embed_pipeline_version")) or EMBED_PIPELINE_VERSION,
        # multi-resolution: "doc", "section", or "chunk" (default)
        "resolution": _stringify(raw.get("resolution")) or "chunk",
    }

    # Table structure metadata (only for table chunks)
    if raw.get("chunk_type") == "table":
        table_headers = raw.get("table_headers")
        if isinstance(table_headers, list):
            payload["table_headers"] = table_headers
        table_type = raw.get("table_type")
        if table_type:
            payload["table_type"] = table_type
        table_row_count = raw.get("table_row_count")
        if isinstance(table_row_count, int):
            payload["table_row_count"] = table_row_count

    # Document understanding context (enriches retrieval without separate lookups)
    _doc_summary = raw.get("doc_summary")
    _doc_entities = raw.get("doc_key_entities")
    _doc_tags = raw.get("doc_intent_tags")
    if _doc_summary or _doc_entities or _doc_tags:
        ctx: Dict[str, Any] = {}
        if _doc_summary:
            ctx["document_summary"] = str(_doc_summary)[:500]
        if isinstance(_doc_entities, list) and _doc_entities:
            ctx["key_entities"] = _doc_entities
        if isinstance(_doc_tags, list) and _doc_tags:
            ctx["intent_tags"] = _doc_tags
        if ctx:
            payload["context"] = ctx

    # Query answerability index (enables pre-filtering at retrieval)
    _answerability = raw.get("answerability")
    if isinstance(_answerability, list) and _answerability:
        payload["answerability"] = _answerability

    # Schema completeness score
    _schema_completeness = raw.get("schema_completeness")
    if _schema_completeness is not None:
        payload["schema_completeness"] = _floatify(_schema_completeness, default=0.0)

    # Parent section ID for multi-resolution back-linking
    _parent_section_id = _stringify(raw.get("parent_section_id"))
    if _parent_section_id:
        payload["parent_section_id"] = _parent_section_id

    return {k: v for k, v in payload.items() if v is not None}


__all__ = ["EMBED_PIPELINE_VERSION", "build_qdrant_payload", "normalize_content", "validate_chunk_for_embedding"]
