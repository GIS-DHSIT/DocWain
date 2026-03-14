from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
import time
from typing import Dict, List, Optional, Tuple

from src.api.dw_document_extractor import DocumentExtractor
from src.api.pipeline_models import ExtractedDocument

from .models import Block, DocumentManifest, DocumentStatus, ExtractedDocumentJSON, Page, Section, Table, Image

logger = get_logger(__name__)

_KEY_VALUE_RE = re.compile(r"^([A-Za-z0-9][^:]{1,64}):\s*(.+)$")
_KEY_VALUE_DASH_RE = re.compile(r"^([A-Za-z0-9][^-]{1,64})\s+-\s+(.+)$")

def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

def _mk_block_id(document_id: str, page_number: Optional[int], text: str, index: int) -> str:
    base = f"{document_id}|{page_number or 0}|{index}|{text[:128]}"
    return f"blk_{_hash_text(base)}"

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\u00a0", " ").split()).strip()

def _detect_block_type(text: str, fallback: str = "paragraph") -> str:
    if not text:
        return "other"
    if text.strip().startswith(("-", "•", "*")):
        return "list_item"
    if _KEY_VALUE_RE.match(text.strip()) or _KEY_VALUE_DASH_RE.match(text.strip()):
        return "key_value"
    return fallback

def _split_blocks_from_text(text: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    blocks: List[Tuple[str, Optional[str], Optional[str]]] = []
    for line in (text or "").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        match = _KEY_VALUE_RE.match(cleaned) or _KEY_VALUE_DASH_RE.match(cleaned)
        if match:
            blocks.append((cleaned, match.group(1).strip(), match.group(2).strip()))
        else:
            blocks.append((cleaned, None, None))
    return blocks

def build_manifest(
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str,
    filename: str,
    file_bytes: Optional[bytes],
    mime_type: Optional[str] = None,
    file_type: Optional[str] = None,
    source_type: Optional[str] = None,
    source_uri: Optional[str] = None,
    source_version: Optional[str] = None,
    page_count: Optional[int] = None,
    language: Optional[str] = None,
    status: DocumentStatus = DocumentStatus.UPLOADED,
) -> DocumentManifest:
    checksum = hashlib.sha256(file_bytes or b"").hexdigest()
    return DocumentManifest(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_id=str(document_id),
        filename=str(filename),
        mime_type=mime_type,
        file_type=file_type,
        size_bytes=len(file_bytes or b""),
        page_count=page_count,
        language=language,
        checksum_sha256=checksum,
        source_type=source_type,
        source_uri=source_uri,
        source_version=source_version,
        ingested_at=time.time(),
        status=status,
    )

def extract_document_json(
    *,
    file_bytes: bytes,
    filename: str,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    mime_type: Optional[str] = None,
    file_type: Optional[str] = None,
    source_type: Optional[str] = None,
    source_uri: Optional[str] = None,
    source_version: Optional[str] = None,
) -> Tuple[ExtractedDocumentJSON, DocumentManifest]:
    extractor = DocumentExtractor()
    extracted_docs = extractor.extract_bytes(file_bytes, filename=filename)
    extracted: Optional[ExtractedDocument] = None
    if isinstance(extracted_docs, dict):
        extracted = extracted_docs.get(filename)
        if extracted is None and extracted_docs:
            extracted = next(iter(extracted_docs.values()))
    if not isinstance(extracted, ExtractedDocument):
        raise ValueError("Extractor did not return ExtractedDocument")

    document_json = build_document_json_from_extracted(extracted, document_id=document_id)
    page_count = max([p.page_number for p in document_json.pages], default=None)

    manifest = build_manifest(
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
        filename=filename,
        file_bytes=file_bytes,
        mime_type=mime_type,
        file_type=file_type,
        source_type=source_type,
        source_uri=source_uri,
        source_version=source_version,
        page_count=page_count,
        language=None,
        status=DocumentStatus.EXTRACTED,
    )
    manifest.extracted_at = time.time()
    return document_json, manifest

def build_document_json_from_extracted(extracted: ExtractedDocument, *, document_id: str) -> ExtractedDocumentJSON:
    pages: Dict[int, Page] = {}
    blocks_by_section: Dict[str, List[str]] = {}
    block_lookup: Dict[str, Block] = {}

    reading_index = 0
    # Include section titles as headings
    for section in extracted.sections or []:
        page_number = section.start_page or section.end_page or 1
        heading_text = _normalize_text(section.title or "")
        if heading_text:
            block_id = _mk_block_id(document_id, page_number, heading_text, reading_index)
            block = Block(
                block_id=block_id,
                type="heading",
                text=heading_text,
                page_number=page_number,
                reading_order=reading_index,
            )
            block_lookup[block_id] = block
            pages.setdefault(page_number, Page(page_number=page_number, blocks=[])).blocks.append(block)
            blocks_by_section.setdefault(section.section_id or heading_text, []).append(block_id)
            reading_index += 1

        for line, key, value in _split_blocks_from_text(section.text or ""):
            page_number = section.start_page or section.end_page or 1
            block_type = _detect_block_type(line)
            block_id = _mk_block_id(document_id, page_number, line, reading_index)
            block = Block(
                block_id=block_id,
                type=block_type,
                text=_normalize_text(line),
                key=key,
                value=value,
                page_number=page_number,
                reading_order=reading_index,
            )
            block_lookup[block_id] = block
            pages.setdefault(page_number, Page(page_number=page_number, blocks=[])).blocks.append(block)
            blocks_by_section.setdefault(section.section_id or section.title, []).append(block_id)
            reading_index += 1

    for candidate in extracted.chunk_candidates or []:
        text = _normalize_text(candidate.text or "")
        if not text:
            continue
        page_number = candidate.page or 1
        block_type = _detect_block_type(text, fallback=candidate.chunk_type or "paragraph")
        block_id = _mk_block_id(document_id, page_number, text, reading_index)
        if block_id in block_lookup:
            continue
        block = Block(
            block_id=block_id,
            type=block_type,
            text=text,
            page_number=page_number,
            reading_order=reading_index,
        )
        block_lookup[block_id] = block
        pages.setdefault(page_number, Page(page_number=page_number, blocks=[])).blocks.append(block)
        if candidate.section_id or candidate.section_title:
            blocks_by_section.setdefault(candidate.section_id or candidate.section_title, []).append(block_id)
        reading_index += 1

    tables: List[Table] = []
    for idx, table in enumerate(extracted.tables or []):
        table_id = f"tbl_{_hash_text(str(idx) + (table.text or ''))}"
        headers: List[str] = []
        rows: List[List[str]] = []
        if table.csv:
            for row in table.csv.splitlines():
                cells = [cell.strip() for cell in row.split(",")]
                if not headers:
                    headers = cells
                else:
                    rows.append(cells)
        elif table.text:
            rows.append([_normalize_text(table.text)])
        tables.append(Table(table_id=table_id, headers=headers, rows=rows, page_number=table.page))

    images: List[Image] = []
    for fig in extracted.figures or []:
        images.append(Image(page_number=fig.page, caption=_normalize_text(fig.caption or "")))

    sections: List[Section] = []
    for section in extracted.sections or []:
        section_id = section.section_id or section.title
        block_ids = blocks_by_section.get(section_id, [])
        if not block_ids:
            # fallback: try by title
            block_ids = blocks_by_section.get(section.title, [])
        sections.append(
            Section(
                section_path=[section.title] if section.title else ["Section"],
                content_refs=block_ids,
                page_range=[section.start_page or 1, section.end_page or section.start_page or 1],
            )
        )

    ordered_pages = [pages[key] for key in sorted(pages.keys())]
    raw_text = extracted.full_text or ""
    return ExtractedDocumentJSON(
        document_id=str(document_id),
        pages=ordered_pages,
        sections=sections,
        tables=tables,
        images=images,
        raw_text=raw_text or None,
    )

def attach_canonical_json(extracted: ExtractedDocument, *, document_id: str) -> ExtractedDocument:
    try:
        canonical = build_document_json_from_extracted(extracted, document_id=document_id)
        extracted.canonical_json = {"document_json": canonical.dict()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to attach canonical JSON: %s", exc)
    return extracted

__all__ = [
    "extract_document_json",
    "build_manifest",
    "build_document_json_from_extracted",
    "attach_canonical_json",
]
