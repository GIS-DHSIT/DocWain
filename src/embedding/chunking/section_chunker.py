"""Section-aware, sentence-safe chunking for embeddings.

This chunker prefers document sections/headings, keeps bullets/tables intact,
and only cuts at safe boundaries (paragraph, sentence, bullet, or table row).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.api.config import Config
from src.api.pipeline_models import ChunkCandidate, ExtractedDocument, Section, Table
from src.embedding.chunking.sentence_splitter import join_sentences, split_into_sentences

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(
    r"^(?:chapter\b|section\b|appendix\b|\d+(?:\.\d+)+|\d+\.|[ivxlcdm]+\.)\s+.+",
    re.IGNORECASE,
)
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s,:\-]{4,}$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")
_TABLE_ROW_RE = re.compile(r"\|.*\|.*|(?:\S+\s{2,}){2,}\S+")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(r"^\s*(confidential|draft|internal use only)\s*$", re.IGNORECASE)


def _safe_text(item: Any) -> str:
    """Extract text from a list item that might be a string, dict, or object.

    Never calls str() on dicts/objects — that produces metadata garbage.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "full_text", "raw_text", "canonical_text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""
    if hasattr(item, "text"):
        val = getattr(item, "text", "")
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _coalesce_whitespace(text: str) -> str:
    # Preserve bullets and line-based tables by working line-wise.
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Remove excessive blank lines but keep paragraph breaks.
    cleaned_lines: List[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def normalize_text(text: str) -> str:
    """Normalize extracted text without damaging structure."""
    if not text:
        return ""
    # Fix broken hyphenation across line breaks: "exam-\nple" -> "example".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse long newline runs while keeping paragraph boundaries.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _coalesce_whitespace(text)
    text = _strip_boilerplate_lines(text)
    return text


def _strip_boilerplate_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines()]
    if not lines:
        return text
    filtered: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        if _PAGE_NUMBER_RE.match(stripped):
            continue
        if _BOILERPLATE_RE.match(stripped):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def _is_heading_line(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _HEADING_RE.match(clean):
        return True
    if len(clean.split()) <= 10 and _ALL_CAPS_RE.match(clean):
        return True
    return False


def _infer_sections_from_text(text: str, *, fallback_title: str) -> List[Section]:
    """Infer sections using heading-like lines when explicit sections are missing."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    # ── Regex heading detection ───────────────────────────────────────
    sections = []
    current_title = fallback_title
    current_lines: List[str] = []

    def _flush(title: str, lines: Sequence[str]) -> None:
        body = "\n".join(lines).strip()
        if not body:
            return
        section_id = hashlib.sha1(f"{title}|1".encode("utf-8")).hexdigest()[:12]
        sections.append(
            Section(
                section_id=section_id,
                title=title,
                level=1,
                start_page=1,
                end_page=1,
                text=body,
            )
        )

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            current_lines.append("")
            continue
        if _is_heading_line(line) and current_lines:
            _flush(current_title, current_lines)
            current_title = line
            current_lines = []
            continue
        if _is_heading_line(line) and not current_lines:
            current_title = line
            continue
        current_lines.append(raw_line)

    _flush(current_title, current_lines)

    if not sections:
        section_id = hashlib.sha1(f"{fallback_title}|1".encode("utf-8")).hexdigest()[:12]
        sections.append(
            Section(
                section_id=section_id,
                title=fallback_title,
                level=1,
                start_page=1,
                end_page=1,
                text=normalized,
            )
        )

    return sections


@dataclass
class Block:
    text: str
    block_type: str
    page_start: Optional[int]
    page_end: Optional[int]


@dataclass
class Chunk:
    text: str
    section_title: str
    section_path: str
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_index: int
    doc_internal_id: str
    source_filename: str
    sentence_complete: bool


class SectionChunker:
    """Chunk extracted documents along section and sentence boundaries."""

    def __init__(
        self,
        *,
        target_chunk_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
        max_chunk_chars: Optional[int] = None,
        overlap_sentences: int = 2,
    ) -> None:
        target = int(target_chunk_chars or getattr(Config.Retrieval, "CHUNK_SIZE", 900))
        minimum = int(min_chunk_chars or getattr(Config.Retrieval, "MIN_CHUNK_SIZE", 200))
        maximum = int(max_chunk_chars or max(target + target // 2, minimum + 200))
        self.target_chunk_chars = max(200, target)
        self.min_chunk_chars = max(50, minimum)
        self.max_chunk_chars = max(self.target_chunk_chars, maximum)
        self.overlap_sentences = max(0, int(overlap_sentences))

    # -------- public API --------
    def chunk_document(
        self,
        extracted_document: Any,
        *,
        doc_internal_id: str,
        source_filename: str,
    ) -> List[Chunk]:
        sections = self._extract_sections(extracted_document, source_filename=source_filename)
        chunks: List[Chunk] = []
        chunk_index = 0

        for section_path, section, section_blocks in sections:
            if not section_blocks:
                continue
            section_title = (section.title or "Untitled Section").strip() or "Untitled Section"
            section_path = section_path or section_title

            section_chunks = self._chunk_section(
                section_blocks,
                section_title=section_title,
                section_path=section_path,
                doc_internal_id=doc_internal_id,
                source_filename=source_filename,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        if not chunks:
            raise ValueError(f"No chunks produced for {source_filename}")
        return chunks

    # -------- section extraction --------
    def _extract_sections(
        self,
        extracted_document: Any,
        *,
        source_filename: str,
    ) -> List[Tuple[str, Section, List[Block]]]:
        """Return a list of (section_path, section, blocks) tuples."""
        if isinstance(extracted_document, ExtractedDocument):
            return self._extract_from_extracted_document(extracted_document, source_filename=source_filename)

        if isinstance(extracted_document, dict):
            return self._extract_from_dict(extracted_document, source_filename=source_filename)

        if isinstance(extracted_document, str):
            inferred = _infer_sections_from_text(extracted_document, fallback_title="Untitled Section")
            return [(sec.title or "Untitled Section", sec, self._blocks_from_text(sec.text, sec.start_page, sec.end_page)) for sec in inferred]

        raise ValueError(f"Unsupported extracted document type: {type(extracted_document)}")

    def _extract_from_extracted_document(
        self,
        extracted: ExtractedDocument,
        *,
        source_filename: str,
    ) -> List[Tuple[str, Section, List[Block]]]:
        sections = list(extracted.sections or [])
        if not sections and extracted.full_text:
            sections = _infer_sections_from_text(extracted.full_text, fallback_title="Untitled Section")

        section_paths = self._build_section_paths(sections)

        # Prefer chunk_candidates when present because they often retain layout info.
        candidates_by_section: Dict[str, List[ChunkCandidate]] = {}
        for cand in extracted.chunk_candidates or []:
            section_id = cand.section_id or ""
            candidates_by_section.setdefault(section_id, []).append(cand)

        tables_by_page: Dict[int, List[Table]] = {}
        for table in extracted.tables or []:
            page = _safe_int(getattr(table, "page", None), default=0) or 0
            tables_by_page.setdefault(page, []).append(table)

        results: List[Tuple[str, Section, List[Block]]] = []
        for sec in sections:
            sec_id = sec.section_id or ""
            sec_path = section_paths.get(sec_id) or (sec.title or "Untitled Section")
            blocks: List[Block] = []

            section_candidates = candidates_by_section.get(sec_id) or []
            if section_candidates:
                for cand in section_candidates:
                    page = _safe_int(cand.page)
                    block_type = cand.chunk_type or self._classify_block(cand.text)
                    blocks.append(Block(normalize_text(cand.text), block_type, page, page))
            else:
                blocks.extend(self._blocks_from_text(sec.text, sec.start_page, sec.end_page))

            # Add any extracted tables that fall inside the section's page span.
            start_page = _safe_int(sec.start_page, default=1) or 1
            end_page = _safe_int(sec.end_page, default=start_page) or start_page
            for page in range(start_page, end_page + 1):
                for table in tables_by_page.get(page, []):
                    table_text = normalize_text(table.text)
                    if table_text:
                        blocks.append(Block(table_text, "table", page, page))

            results.append((sec_path, sec, self._coalesce_blocks(blocks)))

        return results

    def _extract_from_dict(self, payload: Dict[str, Any], *, source_filename: str) -> List[Tuple[str, Section, List[Block]]]:
        # If the dict already contains structured chunks, keep them and rely on metadata fallbacks.
        if isinstance(payload.get("texts"), list) and payload.get("chunk_metadata"):
            fake_section = Section(
                section_id=hashlib.sha1(f"{source_filename}|structured".encode("utf-8")).hexdigest()[:12],
                title="Structured Content",
                level=1,
                start_page=1,
                end_page=1,
                text="",
            )
            blocks = [Block(normalize_text(_safe_text(t)), "text", None, None) for t in payload.get("texts") or [] if _safe_text(t).strip()]
            return [("Structured Content", fake_section, blocks)]

        sections_raw = payload.get("sections")
        if isinstance(sections_raw, list) and sections_raw:
            sections: List[Section] = []
            for idx, sec in enumerate(sections_raw, start=1):
                if isinstance(sec, Section):
                    sections.append(sec)
                    continue
                if not isinstance(sec, dict):
                    continue
                title = str(sec.get("title") or sec.get("section_title") or f"Section {idx}")
                sec_id = str(sec.get("section_id") or hashlib.sha1(f"{title}|{idx}".encode("utf-8")).hexdigest()[:12])
                sections.append(
                    Section(
                        section_id=sec_id,
                        title=title,
                        level=_safe_int(sec.get("level"), default=1) or 1,
                        start_page=_safe_int(sec.get("start_page"), default=1) or 1,
                        end_page=_safe_int(sec.get("end_page"), default=1) or 1,
                        text=str(sec.get("text") or sec.get("content") or ""),
                    )
                )
            paths = self._build_section_paths(sections)
            return [
                (
                    paths.get(sec.section_id) or sec.title or "Untitled Section",
                    sec,
                    self._blocks_from_text(sec.text, sec.start_page, sec.end_page),
                )
                for sec in sections
            ]

        text_candidates = [
            payload.get("full_text"),
            payload.get("text"),
            payload.get("content"),
        ]
        text_value = next((t for t in text_candidates if isinstance(t, str) and t.strip()), "")
        if not text_value and isinstance(payload.get("pages"), list):
            pages_text = []
            for page in payload.get("pages"):
                if isinstance(page, dict):
                    pages_text.append(str(page.get("text") or page.get("content") or ""))
                else:
                    pages_text.append(str(page))
            text_value = "\n\n".join(pages_text)

        inferred = _infer_sections_from_text(text_value, fallback_title="Untitled Section")
        return [
            (
                sec.title or "Untitled Section",
                sec,
                self._blocks_from_text(sec.text, sec.start_page, sec.end_page),
            )
            for sec in inferred
        ]

    def _build_section_paths(self, sections: Sequence[Section]) -> Dict[str, str]:
        """Build section paths from heading levels when available."""
        paths: Dict[str, str] = {}
        stack: List[Tuple[int, str]] = []

        for sec in sections:
            title = (sec.title or "Untitled Section").strip() or "Untitled Section"
            level = _safe_int(sec.level, default=1) or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            paths[sec.section_id] = " > ".join(item[1] for item in stack)
        return paths

    # -------- block building --------
    def _classify_block(self, text: str) -> str:
        if not text:
            return "text"
        first_line = text.strip().splitlines()[0].strip()
        if _BULLET_RE.match(first_line):
            return "bullet"
        if _TABLE_ROW_RE.search(first_line):
            return "table"
        return "text"

    def _blocks_from_text(self, text: str, start_page: Optional[int], end_page: Optional[int]) -> List[Block]:
        normalized = normalize_text(text or "")
        if not normalized:
            return []

        blocks: List[Block] = []
        lines = normalized.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            stripped = line.strip()

            if not stripped:
                idx += 1
                continue

            if _BULLET_RE.match(stripped):
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
                    # Keep indented continuation lines with the bullet.
                    bullet_lines.append(nxt)
                    idx += 1
                bullet_text = "\n".join(bullet_lines).strip()
                blocks.append(Block(bullet_text, "bullet", start_page, end_page))
                continue

            if _TABLE_ROW_RE.search(stripped):
                table_lines = [line]
                idx += 1
                while idx < len(lines):
                    nxt = lines[idx]
                    nxt_stripped = nxt.strip()
                    if not nxt_stripped:
                        break
                    if _is_heading_line(nxt_stripped) or _BULLET_RE.match(nxt_stripped):
                        break
                    if not _TABLE_ROW_RE.search(nxt_stripped):
                        break
                    table_lines.append(nxt)
                    idx += 1
                table_text = "\n".join(table_lines).strip()
                blocks.append(Block(table_text, "table", start_page, end_page))
                continue

            # Default: paragraph block until the next blank line/heading/bullet/table.
            para_lines = [line]
            idx += 1
            while idx < len(lines):
                nxt = lines[idx]
                nxt_stripped = nxt.strip()
                if not nxt_stripped:
                    break
                if _is_heading_line(nxt_stripped) or _BULLET_RE.match(nxt_stripped) or _TABLE_ROW_RE.search(nxt_stripped):
                    break
                para_lines.append(nxt)
                idx += 1
            para_text = "\n".join(para_lines).strip()
            blocks.append(Block(para_text, "text", start_page, end_page))

        return self._coalesce_blocks(blocks)

    def _coalesce_blocks(self, blocks: Iterable[Block]) -> List[Block]:
        """Drop blanks and normalize types/text."""
        cleaned: List[Block] = []
        for block in blocks:
            text = normalize_text(block.text)
            if not text:
                continue
            block_type = block.block_type or self._classify_block(text)
            cleaned.append(Block(text, block_type, block.page_start, block.page_end))
        return cleaned

    # -------- chunking --------
    def _chunk_section(
        self,
        blocks: Sequence[Block],
        *,
        section_title: str,
        section_path: str,
        doc_internal_id: str,
        source_filename: str,
        start_index: int,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        current_blocks: List[Block] = []
        current_len = 0
        last_chunk_sentences: List[str] = []
        chunk_index = start_index

        def _block_len(block: Block) -> int:
            return len(block.text)

        def _safe_overlap() -> str:
            if not last_chunk_sentences or self.overlap_sentences <= 0:
                return ""
            overlap = last_chunk_sentences[-self.overlap_sentences :]
            return " ".join(overlap).strip()

        def _finalize(force: bool = False) -> None:
            nonlocal current_blocks, current_len, last_chunk_sentences, chunk_index
            if not current_blocks:
                return
            chunk_text, page_start, page_end = self._render_chunk_text(current_blocks)
            if not chunk_text:
                current_blocks = []
                current_len = 0
                return
            if not force and current_len < self.min_chunk_chars and chunks:
                return
            sentence_complete = self._is_sentence_complete(chunk_text, current_blocks[-1].block_type)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    section_title=section_title,
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=chunk_index,
                    doc_internal_id=str(doc_internal_id),
                    source_filename=source_filename,
                    sentence_complete=sentence_complete,
                )
            )
            last_chunk_sentences = split_into_sentences(chunk_text)
            chunk_index += 1
            overlap_text = _safe_overlap()
            current_blocks = []
            current_len = 0
            if overlap_text:
                overlap_block = Block(overlap_text, "text", page_start, page_end)
                current_blocks.append(overlap_block)
                current_len = len(overlap_text)

        for block in blocks:
            # Split oversized blocks at safe boundaries.
            sub_blocks = self._split_oversized_block(block)
            for sub_block in sub_blocks:
                blen = _block_len(sub_block)
                if current_blocks and current_len + blen > self.target_chunk_chars:
                    _finalize(force=False)
                if current_blocks and current_len + blen > self.max_chunk_chars:
                    _finalize(force=True)
                current_blocks.append(sub_block)
                current_len += blen + (2 if sub_block.block_type == "text" else 1)

        _finalize(force=True)
        return chunks

    def _split_oversized_block(self, block: Block) -> List[Block]:
        if len(block.text) <= self.max_chunk_chars:
            return [block]

        if block.block_type == "table":
            return self._split_table_block(block)
        if block.block_type == "bullet":
            return self._split_bullet_block(block)
        return self._split_text_block(block)

    def _split_table_block(self, block: Block) -> List[Block]:
        rows = [ln for ln in block.text.splitlines() if ln.strip()]
        if not rows:
            return []
        chunks: List[Block] = []
        buffer: List[str] = []
        buf_len = 0
        for row in rows:
            row_len = len(row) + 1
            if buffer and buf_len + row_len > self.max_chunk_chars:
                chunks.append(Block("\n".join(buffer), "table", block.page_start, block.page_end))
                buffer = []
                buf_len = 0
            buffer.append(row)
            buf_len += row_len
        if buffer:
            chunks.append(Block("\n".join(buffer), "table", block.page_start, block.page_end))
        return chunks

    def _split_bullet_block(self, block: Block) -> List[Block]:
        # Split bullets on blank lines to avoid breaking a single bullet item.
        items = [item.strip() for item in re.split(r"\n\s*\n", block.text) if item.strip()]
        if len(items) <= 1:
            # Fall back to sentence-based splitting if a single bullet is too long.
            return self._split_text_block(block)
        chunks: List[Block] = []
        buffer: List[str] = []
        buf_len = 0
        for item in items:
            item_len = len(item) + 2
            if buffer and buf_len + item_len > self.max_chunk_chars:
                chunks.append(Block("\n\n".join(buffer), "bullet", block.page_start, block.page_end))
                buffer = []
                buf_len = 0
            buffer.append(item)
            buf_len += item_len
        if buffer:
            chunks.append(Block("\n\n".join(buffer), "bullet", block.page_start, block.page_end))
        return chunks

    def _split_text_block(self, block: Block) -> List[Block]:
        sentences = split_into_sentences(block.text)
        if not sentences:
            return [block]
        sentence_chunks = join_sentences(sentences, self.max_chunk_chars)
        return [Block(chunk, block.block_type, block.page_start, block.page_end) for chunk in sentence_chunks if chunk]

    def _render_chunk_text(self, blocks: Sequence[Block]) -> Tuple[str, Optional[int], Optional[int]]:
        if not blocks:
            return "", None, None
        parts: List[str] = []
        page_start: Optional[int] = None
        page_end: Optional[int] = None
        for block in blocks:
            if block.page_start is not None:
                page_start = block.page_start if page_start is None else min(page_start, block.page_start)
            if block.page_end is not None:
                page_end = block.page_end if page_end is None else max(page_end, block.page_end)
            parts.append(block.text)
        chunk_text = "\n\n".join(part for part in parts if part).strip()
        return chunk_text, page_start, page_end

    def _build_table_chunk_meta(self, table_text: str) -> Optional[Dict[str, Any]]:
        """Extract column headers from table text for chunk metadata."""
        lines = [ln.strip() for ln in table_text.splitlines() if ln.strip()]
        if not lines:
            return None

        # Try pipe-delimited format
        if "|" in lines[0]:
            headers = [h.strip() for h in lines[0].split("|") if h.strip()]
            if headers and len(headers) >= 2:
                return {"table_headers": headers, "table_rows": len(lines) - 1}

        # Try comma-delimited format
        if "," in lines[0]:
            headers = [h.strip() for h in lines[0].split(",") if h.strip()]
            if headers and len(headers) >= 2:
                return {"table_headers": headers, "table_rows": len(lines) - 1}

        return None

    def _is_sentence_complete(self, text: str, block_type: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if block_type in {"bullet", "table"}:
            return True
        return stripped[-1] in {".", "?", "!"}
