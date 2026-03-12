from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from .metadata_normalizer import normalize_chunk_payload
from .models import Block, DocumentManifest, EntityFactBundle, ExtractedDocumentJSON

logger = get_logger(__name__)

def _hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower())

def _anchors_from_text(text: str, limit: int = 8) -> List[str]:
    tokens = _tokenize(text)
    seen: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.append(token)
        if len(seen) >= limit:
            break
    return seen

def _page_range_from_block(block: Block) -> List[int]:
    page = block.page_number or 1
    return [page, page]

def _section_index(document: ExtractedDocumentJSON) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for section in document.sections:
        for block_id in section.content_refs:
            mapping[block_id] = section.section_path
    return mapping

def build_answerable_chunks(
    *,
    document: ExtractedDocumentJSON,
    manifest: DocumentManifest,
    entities: Optional[EntityFactBundle] = None,
) -> List[Dict[str, Any]]:
    chunk_payloads: List[Dict[str, Any]] = []
    section_map = _section_index(document)
    block_map: Dict[str, Block] = {}
    for page in document.pages:
        for block in page.blocks:
            block_map[block.block_id] = block

    # Section chunks
    for section in document.sections:
        block_texts: List[str] = []
        block_ids = section.content_refs
        for block_id in block_ids:
            block = block_map.get(block_id)
            if block and block.text:
                block_texts.append(block.text)
        text = "\n".join(block_texts).strip()
        if text:
            chunk_payloads.append(
                {
                    "subscription_id": manifest.subscription_id,
                    "profile_id": manifest.profile_id,
                    "document_id": manifest.document_id,
                    "filename": manifest.filename,
                    "source_type": manifest.source_type,
                    "domain": "generic",
                    "chunk_kind": "section_text",
                    "section_path": section.section_path,
                    "page_range": section.page_range,
                    "block_ids": block_ids,
                    "anchors": _anchors_from_text(text),
                    "text": text,
                    "checksum_sha256": manifest.checksum_sha256,
                    "source_version": manifest.source_version,
                    "ingest_timestamp": manifest.ingested_at,
                }
            )

    # Block-derived chunks
    for block in block_map.values():
        text = block.text or ""
        if not text:
            continue
        section_path = section_map.get(block.block_id, [])
        base = {
            "subscription_id": manifest.subscription_id,
            "profile_id": manifest.profile_id,
            "document_id": manifest.document_id,
            "filename": manifest.filename,
            "source_type": manifest.source_type,
            "domain": "generic",
            "section_path": section_path,
            "page_range": _page_range_from_block(block),
            "block_ids": [block.block_id],
            "anchors": _anchors_from_text(text),
            "text": text,
            "checksum_sha256": manifest.checksum_sha256,
            "source_version": manifest.source_version,
            "ingest_timestamp": manifest.ingested_at,
        }
        if block.type == "key_value" and block.key and block.value:
            base["chunk_kind"] = "structured_field"
            base["text"] = f"{block.key}: {block.value}"
            chunk_payloads.append(base)
        elif block.type == "list_item":
            base["chunk_kind"] = "structured_field"
            chunk_payloads.append(base)

    # Table row chunks
    for table in document.tables:
        headers = table.headers or []
        for row in table.rows or []:
            if not row:
                continue
            parts: List[str] = []
            for idx, cell in enumerate(row):
                header = headers[idx] if idx < len(headers) else f"col_{idx + 1}"
                parts.append(f"{header}: {cell}")
            text = "; ".join(parts).strip()
            if not text:
                continue
            chunk_payloads.append(
                {
                    "subscription_id": manifest.subscription_id,
                    "profile_id": manifest.profile_id,
                    "document_id": manifest.document_id,
                    "filename": manifest.filename,
                    "source_type": manifest.source_type,
                    "domain": "generic",
                    "chunk_kind": "table_text",
                    "section_path": [],
                    "page_range": [table.page_number or 1, table.page_number or 1],
                    "block_ids": [],
                    "anchors": _anchors_from_text(text),
                    "text": text,
                    "checksum_sha256": manifest.checksum_sha256,
                    "source_version": manifest.source_version,
                    "ingest_timestamp": manifest.ingested_at,
                }
            )

    # Entity fact chunks
    if entities:
        for fact in entities.facts:
            subject = next((e for e in entities.entities if e.entity_id == fact.subject_id), None)
            text = " ".join([subject.text if subject else "Entity", fact.predicate, fact.object_value or ""]).strip()
            if not text:
                continue
            chunk_payloads.append(
                {
                    "subscription_id": manifest.subscription_id,
                    "profile_id": manifest.profile_id,
                    "document_id": manifest.document_id,
                    "filename": manifest.filename,
                    "source_type": manifest.source_type,
                    "domain": "generic",
                    "chunk_kind": "structured_field",
                    "section_path": fact.evidence.get("section_path") if isinstance(fact.evidence, dict) else [],
                    "page_range": fact.evidence.get("page_range") if isinstance(fact.evidence, dict) else None,
                    "block_ids": fact.evidence.get("block_ids") if isinstance(fact.evidence, dict) else [],
                    "anchors": _anchors_from_text(text),
                    "text": text,
                    "checksum_sha256": manifest.checksum_sha256,
                    "source_version": manifest.source_version,
                    "ingest_timestamp": manifest.ingested_at,
                }
            )

    # Normalize payloads and attach chunk ids
    normalized: List[Dict[str, Any]] = []
    for idx, payload in enumerate(chunk_payloads):
        payload["chunk_id"] = payload.get("chunk_id") or f"chunk_{_hash(manifest.document_id + str(idx) + payload.get('chunk_kind', ''))}"
        normalized.append(normalize_chunk_payload(payload))

    return normalized

__all__ = ["build_answerable_chunks"]
