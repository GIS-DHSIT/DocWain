"""Document context assembler for cross-document intelligence.

Groups chunks by document, extracts per-document metadata, and renders
attributed evidence for LLM prompts and structured comparisons.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import Chunk

# Simple KV pattern: "Label: Value" on a line
_KV_RE = re.compile(r"^([A-Z][A-Za-z /&-]{1,40}):\s+(.+)", re.MULTILINE)


@dataclass
class DocumentContext:
    document_id: str
    document_name: str
    doc_domain: str
    chunks: List[Chunk] = field(default_factory=list)
    fields: Dict[str, str] = field(default_factory=dict)
    section_kinds: List[str] = field(default_factory=list)


def assemble_document_contexts(chunks: List[Chunk]) -> List[DocumentContext]:
    """Group chunks by document_id, extract metadata, build per-doc field maps.

    Returns list of DocumentContext sorted by document_name.
    """
    if not chunks:
        return []

    by_doc: Dict[str, List[Chunk]] = defaultdict(list)
    doc_meta: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        meta = chunk.meta or {}
        doc_id = (
            meta.get("document_id")
            or meta.get("doc_id")
            or "unknown"
        )
        by_doc[doc_id].append(chunk)

        if doc_id not in doc_meta:
            doc_name = (
                getattr(chunk.source, "document_name", None)
                or meta.get("source_name")
                or "Unknown Document"
            )
            doc_domain = meta.get("doc_domain") or "generic"
            doc_meta[doc_id] = {"name": doc_name, "domain": doc_domain}

    contexts: List[DocumentContext] = []
    for doc_id, doc_chunks in by_doc.items():
        info = doc_meta[doc_id]

        # Collect unique section_kinds
        kinds: List[str] = []
        seen_kinds: set = set()
        for c in doc_chunks:
            kind = (c.meta or {}).get("section_kind", "")
            if kind and kind not in seen_kinds:
                kinds.append(kind)
                seen_kinds.add(kind)

        # Extract KV fields from chunk text
        fields: Dict[str, str] = {}
        for c in doc_chunks:
            for match in _KV_RE.finditer(c.text or ""):
                label = match.group(1).strip()
                value = match.group(2).strip()
                if label not in fields:
                    fields[label] = value

        contexts.append(DocumentContext(
            document_id=doc_id,
            document_name=info["name"],
            doc_domain=info["domain"],
            chunks=doc_chunks,
            fields=fields,
            section_kinds=kinds,
        ))

    contexts.sort(key=lambda ctx: ctx.document_name)
    return contexts


def render_attributed_evidence(
    contexts: List[DocumentContext],
    query: str = "",
) -> str:
    """Render evidence grouped by document with clear source attribution.

    Output format for multi-doc:
        === Document: Name.pdf (domain) ===
        [section_kind] chunk text...

    For single doc, omits the header to avoid redundancy.
    """
    if not contexts:
        return ""

    parts: List[str] = []
    show_header = len(contexts) > 1

    for ctx in contexts:
        if show_header:
            parts.append(f"=== Document: {ctx.document_name} ({ctx.doc_domain}) ===")

        for chunk in ctx.chunks:
            kind = (chunk.meta or {}).get("section_kind", "")
            text = (chunk.text or "").strip()
            if not text:
                continue
            if kind:
                parts.append(f"[{kind}] {text}")
            else:
                parts.append(text)

        if show_header:
            parts.append("")  # blank line between docs

    return "\n".join(parts).strip()
