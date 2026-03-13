from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger
from src.utils.payload_utils import (
    get_chunk_hash,
    get_chunk_sentence_complete,
    get_source_name,
)

logger = get_logger(__name__)

@dataclass
class ContextBuildResult:
    context_text: str
    sources: List[Dict[str, Any]]
    selected_chunks: List[Dict[str, Any]]
    token_count: int


class ContextAssembler:
    """Section-aware context assembler with dedup and coherence hints."""

    def __init__(self, max_tokens: int = 4096, max_chunks: int = 12):
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks

    def build(
        self,
        chunks: List[Dict[str, Any]],
        *,
        intent_type: str = "factual",
    ) -> ContextBuildResult:
        logger.debug("build: chunks=%d, intent_type=%s", len(chunks), intent_type)
        if not chunks:
            return ContextBuildResult("", [], [], 0)

        deduped = self._deduplicate(chunks)
        grouped = self._group_by_section(deduped)
        group_order = self._rank_groups(grouped)

        selected_chunks: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        token_budget = self.max_tokens
        source_id = 1

        max_per_section = 3
        if intent_type in {"summarization", "deep_analysis"}:
            max_per_section = 2

        for key in group_order:
            items = grouped[key]
            picked = self._pick_chunks(items, max_per_section)
            for chunk in picked:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                tokens = self._approx_tokens(text)
                if tokens > token_budget:
                    continue
                token_budget -= tokens
                meta = chunk.get("metadata") or {}
                header = self._header(meta)
                context_parts.append(f"[SOURCE-{source_id}] {header}\n{text}\n[/SOURCE-{source_id}]")
                sources.append(
                    {
                        "source_id": source_id,
                        "source_name": self._safe_doc_name(meta),
                        "section": self._safe_section(meta) or None,
                        "page": self._safe_page(meta),
                        "excerpt": text[:400],
                        "score": round(float(chunk.get("score", 0.0)), 4),
                        "chunk_id": meta.get("chunk_id"),
                    }
                )
                selected_chunks.append(chunk)
                source_id += 1
                if source_id > self.max_chunks or token_budget <= 0:
                    token_budget = 0
                    break
            if token_budget <= 0 or source_id > self.max_chunks:
                break

        context_text = "\n\n".join(context_parts)
        result = ContextBuildResult(context_text, sources, selected_chunks, self._approx_tokens(context_text))
        logger.debug("build: returning %d sources, %d tokens", len(result.sources), result.token_count)
        return result

    def _deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.debug("_deduplicate: input chunks=%d", len(chunks))
        seen: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            text = chunk.get("text") or ""
            if not text.strip():
                continue
            meta = chunk.get("metadata") or {}
            chunk_hash = get_chunk_hash(meta) or hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324
            key = str(chunk_hash)
            if key in seen:
                if float(chunk.get("score", 0.0)) > float(seen[key].get("score", 0.0)):
                    seen[key] = chunk
            else:
                seen[key] = chunk
        logger.debug("_deduplicate: output chunks=%d", len(seen))
        return list(seen.values())

    def _group_by_section(self, chunks: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            doc_id = str(meta.get("document_id") or meta.get("doc_id") or get_source_name(meta) or "unknown")
            section = str(meta.get("section_path") or meta.get("section_title") or meta.get("section") or "")
            grouped[(doc_id, section)].append(chunk)
        return grouped

    @staticmethod
    def _rank_groups(grouped: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> List[Tuple[str, str]]:
        scored = []
        for key, items in grouped.items():
            top_score = max(float(item.get("score", 0.0)) for item in items)
            scored.append((key, top_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [key for key, _ in scored]

    def _pick_chunks(self, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        items_sorted = sorted(
            items,
            key=lambda c: (
                not get_chunk_sentence_complete(c.get("metadata") or {}, True),
                -float(c.get("score", 0.0)),
                self._chunk_index(c),
            ),
        )
        picked = []
        for chunk in items_sorted:
            if len(picked) >= limit:
                break
            picked.append(chunk)
            if not get_chunk_sentence_complete(chunk.get("metadata") or {}, True):
                neighbor = self._find_neighbor(items_sorted, chunk)
                if neighbor and neighbor not in picked and len(picked) < limit:
                    picked.append(neighbor)
        return picked

    def _find_neighbor(self, items: List[Dict[str, Any]], chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        idx = self._chunk_index(chunk)
        if idx is None:
            return None
        candidates = []
        for other in items:
            other_idx = self._chunk_index(other)
            if other_idx is None:
                continue
            if other_idx in {idx - 1, idx + 1}:
                candidates.append(other)
        if not candidates:
            return None
        return sorted(candidates, key=lambda c: -float(c.get("score", 0.0)))[0]

    @staticmethod
    def _chunk_index(chunk: Dict[str, Any]) -> Optional[int]:
        meta = chunk.get("metadata") or {}
        idx = meta.get("chunk_index")
        if isinstance(idx, int):
            return idx
        try:
            return int(idx) if idx is not None else None
        except Exception:
            logger.debug("_chunk_index: failed to parse idx=%s", idx, exc_info=True)
            return None

    @staticmethod
    def _safe_doc_name(meta: Dict[str, Any]) -> str:
        value = get_source_name(meta) or meta.get("title")
        if value:
            base = os.path.basename(str(value))
            base = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", base)
            return base.strip() or "Document"
        return "Document"

    @staticmethod
    def _safe_section(meta: Dict[str, Any]) -> str:
        section = meta.get("section_title") or meta.get("section_path") or meta.get("section") or ""
        return str(section).strip()

    @staticmethod
    def _safe_page(meta: Dict[str, Any]) -> Optional[str]:
        page = meta.get("page") or meta.get("page_number")
        if page is None:
            return None
        try:
            return str(int(page))
        except Exception:
            logger.debug("_safe_page: failed to parse page=%s", page, exc_info=True)
            return str(page)

    def _header(self, meta: Dict[str, Any]) -> str:
        parts = [f"Document: {self._safe_doc_name(meta)}"]
        section = self._safe_section(meta)
        if section:
            parts.append(f"Section: {section}")
        page = self._safe_page(meta)
        if page:
            parts.append(f"Page: {page}")
        return ", ".join(parts)

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return len((text or "").split())
