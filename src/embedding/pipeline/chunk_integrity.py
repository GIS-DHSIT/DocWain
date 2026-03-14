from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.embedding.chunking.sentence_splitter import split_into_sentences

logger = get_logger(__name__)

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")
_HEADING_RE = re.compile(
    r"^(?:chapter\b|section\b|appendix\b|\d+(?:\.\d+)+|\d+\.|[ivxlcdm]+\.)\s+.+",
    re.IGNORECASE,
)
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s,:\-]{4,}$")
# Matches C0/C1 control characters EXCEPT \t (0x09), \n (0x0a), \r (0x0d).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_PAGE_MARKER_RE = re.compile(r"^\s*-+\s*page\s*\d+\s*-+\s*$", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$", re.IGNORECASE)
_ONLY_PUNCT_RE = re.compile(r"^[\W_]+$")

@dataclass(frozen=True)
class ChunkIntegrityConfig:
    target_min_tokens: int = 250
    target_max_tokens: int = 450
    hard_max_tokens: int = 520
    min_chars: int = 300
    overlap_tokens: int = 60

@dataclass
class AtomicUnit:
    text: str
    unit_type: str
    page_start: Optional[int]
    page_end: Optional[int]

def clean_text_for_embedding(text: str) -> str:
    if not text:
        return ""
    text = _CONTROL_CHAR_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _PAGE_MARKER_RE.match(stripped) or _PAGE_NUMBER_RE.match(stripped):
            continue
        lines.append(line.rstrip())
    cleaned: List[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def _is_heading_line(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _HEADING_RE.match(clean):
        return True
    if len(clean.split()) <= 10 and _ALL_CAPS_RE.match(clean):
        return True
    return False

def _token_count(text: str) -> int:
    return len(text.split())

def _lex_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))

def is_valid_chunk_text(
    text: Optional[str],
    *,
    min_chars: Optional[int] = None,
    min_tokens: Optional[int] = None,
) -> bool:
    if text is None:
        return False
    stripped = str(text).strip()
    if not stripped:
        return False
    if min_chars is None or min_tokens is None:
        try:
            from src.api.config import Config
            if min_chars is None:
                min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 50))
            if min_tokens is None:
                min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 10))
        except Exception:
            min_chars = min_chars or 50
            min_tokens = min_tokens or 10
    if len(stripped) < int(min_chars or 0):
        return False
    if _lex_token_count(stripped) < int(min_tokens or 0):
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and all(_PAGE_MARKER_RE.match(line) or _PAGE_NUMBER_RE.match(line) for line in lines):
        return False
    if _ONLY_PUNCT_RE.match(stripped):
        return False
    alnum_ratio = len(re.findall(r"[A-Za-z0-9]", stripped)) / max(1, len(stripped))
    if alnum_ratio < 0.2:
        return False
    return True

def _split_bullets(lines: List[str], page_start: Optional[int], page_end: Optional[int]) -> List[AtomicUnit]:
    units: List[AtomicUnit] = []
    non_bullet_buf: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        if _BULLET_RE.match(stripped):
            # Flush any buffered non-bullet text first
            if non_bullet_buf:
                text = "\n".join(non_bullet_buf).strip()
                if text:
                    units.append(AtomicUnit(text, "sentence", page_start, page_end))
                non_bullet_buf = []
            bullet_lines = [line]
            idx += 1
            while idx < len(lines):
                nxt = lines[idx]
                nxt_stripped = nxt.strip()
                if not nxt_stripped:
                    bullet_lines.append("")
                    idx += 1
                    continue
                if _BULLET_RE.match(nxt_stripped) or _is_heading_line(nxt_stripped):
                    break
                bullet_lines.append(nxt)
                idx += 1
            bullet_text = "\n".join(bullet_lines).strip()
            units.append(AtomicUnit(bullet_text, "bullet", page_start, page_end))
            continue
        # Non-bullet line: buffer it to preserve as content
        non_bullet_buf.append(line)
        idx += 1
    # Flush remaining non-bullet text
    if non_bullet_buf:
        text = "\n".join(non_bullet_buf).strip()
        if text:
            units.append(AtomicUnit(text, "sentence", page_start, page_end))
    return units

def build_atomic_units(text: str, page_start: Optional[int], page_end: Optional[int]) -> List[AtomicUnit]:
    if not text:
        return []
    lines = text.splitlines()
    has_bullets = any(_BULLET_RE.match(ln.strip()) for ln in lines if ln.strip())
    if has_bullets:
        return _split_bullets(lines, page_start, page_end)

    units: List[AtomicUnit] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)
        if sentences:
            for sentence in sentences:
                cleaned = sentence.strip()
                if cleaned:
                    units.append(AtomicUnit(cleaned, "sentence", page_start, page_end))
        else:
            units.append(AtomicUnit(paragraph, "paragraph", page_start, page_end))
    return units

def _dangling_heading(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith(":"):
        return True
    return _is_heading_line(stripped)

def _finalize_chunk(units: List[AtomicUnit]) -> Tuple[str, Optional[int], Optional[int], str]:
    text = "\n\n".join(unit.text for unit in units if unit.text).strip()
    page_start = None
    page_end = None
    unit_types = {unit.unit_type for unit in units}
    for unit in units:
        if unit.page_start is not None:
            page_start = unit.page_start if page_start is None else min(page_start, unit.page_start)
        if unit.page_end is not None:
            page_end = unit.page_end if page_end is None else max(page_end, unit.page_end)
    chunk_type = unit_types.pop() if len(unit_types) == 1 else "text"
    return text, page_start, page_end, chunk_type

def is_chunk_complete(text: str, chunk_type: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if chunk_type in {"bullet", "table"}:
        return True
    if stripped.endswith(":"):
        return False
    last_line = ""
    for line in reversed(stripped.splitlines()):
        if line.strip():
            last_line = line.strip()
            break
    if last_line and _BULLET_RE.match(last_line):
        return True
    return stripped.endswith((".", "?", "!"))

def _pack_units(units: List[AtomicUnit], config: ChunkIntegrityConfig) -> List[Tuple[str, Dict[str, Any]]]:
    packed: List[Tuple[str, Dict[str, Any]]] = []
    current: List[AtomicUnit] = []
    current_tokens = 0
    idx = 0
    while idx < len(units):
        unit = units[idx]
        unit_tokens = _token_count(unit.text)
        if current and current_tokens + unit_tokens > config.hard_max_tokens:
            text, page_start, page_end, chunk_type = _finalize_chunk(current)
            packed.append((text, {"page_start": page_start, "page_end": page_end, "chunk_type": chunk_type}))
            current = []
            current_tokens = 0
            continue
        current.append(unit)
        current_tokens += unit_tokens
        next_unit = units[idx + 1] if idx + 1 < len(units) else None
        should_finalize = current_tokens >= config.target_min_tokens
        if should_finalize:
            if next_unit and _dangling_heading(current[-1].text):
                idx += 1
                current.append(next_unit)
                current_tokens += _token_count(next_unit.text)
                next_unit = units[idx + 1] if idx + 1 < len(units) else None
            if current_tokens >= config.target_max_tokens or not next_unit:
                text, page_start, page_end, chunk_type = _finalize_chunk(current)
                packed.append((text, {"page_start": page_start, "page_end": page_end, "chunk_type": chunk_type}))
                current = []
                current_tokens = 0
        idx += 1

    if current:
        text, page_start, page_end, chunk_type = _finalize_chunk(current)
        packed.append((text, {"page_start": page_start, "page_end": page_end, "chunk_type": chunk_type}))
    return packed

def _apply_overlap(chunks: List[str], config: ChunkIntegrityConfig) -> List[str]:
    if config.overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for idx in range(1, len(chunks)):
        prev_tokens = chunks[idx - 1].split()
        curr_tokens = chunks[idx].split()
        if not prev_tokens or not curr_tokens:
            overlapped.append(chunks[idx])
            continue
        overlap_count = min(config.overlap_tokens, len(prev_tokens))
        tail = prev_tokens[-overlap_count:]
        prefix = " ".join(tail).strip()
        merged = f"{prefix} {chunks[idx]}".strip() if prefix else chunks[idx]
        overlapped.append(merged)
    return overlapped

def enforce_chunk_integrity(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    config: Optional[ChunkIntegrityConfig] = None,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    if len(chunks) != len(metadata):
        raise ValueError("chunk and metadata length mismatch")
    config = config or ChunkIntegrityConfig()
    merged_small = 0
    rebuilt_chunks: List[str] = []
    rebuilt_meta: List[Dict[str, Any]] = []

    section_order: List[str] = []
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for text, meta in zip(chunks, metadata):
        section_id = meta.get("section_id") or meta.get("section_path") or meta.get("section_title") or "section"
        if section_id not in grouped:
            section_order.append(section_id)
            grouped[section_id] = []
        grouped[section_id].append((text, meta))

    for section_id in section_order:
        items = grouped[section_id]
        units: List[AtomicUnit] = []
        base_meta = items[0][1] if items else {}
        for text, meta in items:
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            units.extend(build_atomic_units(text, page_start, page_end))
        if not units:
            continue
        packed = _pack_units(units, config)
        packed_texts = [entry[0] for entry in packed if entry[0]]
        packed_meta = [entry[1] for entry in packed if entry[0]]
        packed_texts = _apply_overlap(packed_texts, config)

        section_chunks: List[str] = []
        section_meta: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(packed_texts):
            text = packed_texts[idx]
            meta = dict(base_meta)
            meta.update(packed_meta[idx])
            min_chars = config.min_chars
            role = str(meta.get("chunk_type") or "").lower()
            if role in {"header", "caption", "image_caption"}:
                min_chars = 0
            if len(text) < min_chars and idx + 1 < len(packed_texts):
                packed_texts[idx + 1] = f"{text}\n\n{packed_texts[idx + 1]}".strip()
                merged_small += 1
                idx += 1
                continue
            section_chunks.append(text)
            section_meta.append(meta)
            idx += 1

        if section_chunks:
            last_idx = len(section_chunks) - 1
            last_text = section_chunks[last_idx]
            last_meta = section_meta[last_idx]
            last_role = str(last_meta.get("chunk_type") or "").lower()
            last_min_chars = 0 if last_role in {"header", "caption", "image_caption"} else config.min_chars
            if last_min_chars and len(last_text) < last_min_chars and len(section_chunks) > 1:
                section_chunks[-2] = f"{section_chunks[-2]}\n\n{last_text}".strip()
                section_chunks.pop()
                section_meta.pop()
                merged_small += 1

        rebuilt_chunks.extend(section_chunks)
        rebuilt_meta.extend(section_meta)

    stats = {"merged_small": merged_small}
    if merged_small:
        logger.warning("Merged %s undersized chunks during integrity enforcement", merged_small)
    return rebuilt_chunks, rebuilt_meta, stats

__all__ = [
    "ChunkIntegrityConfig",
    "AtomicUnit",
    "build_atomic_units",
    "clean_text_for_embedding",
    "enforce_chunk_integrity",
    "is_chunk_complete",
    "is_valid_chunk_text",
]
