from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.payload_utils import get_source_name
logger = get_logger(__name__)

@dataclass
class ContextBuildResult:
    context_text: str
    sources: List[Dict[str, Any]]
    selected_chunks: List[Dict[str, Any]]
    token_count: int

class ContextAssembler:
    """Builds section-aware context with safe citations and deduplication."""

    def __init__(self, max_tokens: int = 2000, dedup_threshold: float = 0.92, max_chunks: Optional[int] = None):
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold
        self.max_chunks = max_chunks

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]{2,}", (text or "").lower())

    def _cosine_similarity(self, left: str, right: str) -> float:
        left_tokens = Counter(self._tokenize(left))
        right_tokens = Counter(self._tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        dot = sum(left_tokens[t] * right_tokens.get(t, 0) for t in left_tokens)
        left_mag = sum(v * v for v in left_tokens.values()) ** 0.5
        right_mag = sum(v * v for v in right_tokens.values()) ** 0.5
        if left_mag == 0 or right_mag == 0:
            return 0.0
        return dot / (left_mag * right_mag)

    def _deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.get("text") or ""
            if not text.strip():
                continue
            duplicate = False
            for existing in unique:
                sim = self._cosine_similarity(text, existing.get("text") or "")
                if sim >= self.dedup_threshold:
                    if float(chunk.get("score", 0.0)) > float(existing.get("score", 0.0)):
                        unique.remove(existing)
                        unique.append(chunk)
                    duplicate = True
                    break
            if not duplicate:
                unique.append(chunk)
        return unique

    @staticmethod
    def _group_key(meta: Dict[str, Any]) -> Tuple[str, str]:
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or get_source_name(meta) or "unknown")
        section = str(meta.get("section_title") or meta.get("section") or "")
        return doc_id, section

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
        section = meta.get("section_title") or meta.get("section") or ""
        section = str(section).strip()
        return section

    @staticmethod
    def _safe_page(meta: Dict[str, Any]) -> Optional[str]:
        page = meta.get("page") or meta.get("page_number")
        if page is None:
            return None
        try:
            return str(int(page))
        except Exception:
            return str(page)

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return len((text or "").split())

    def build(self, chunks: List[Dict[str, Any]]) -> ContextBuildResult:
        if not chunks:
            return ContextBuildResult("", [], [], 0)

        deduped = self._deduplicate(chunks)
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for chunk in deduped:
            meta = chunk.get("metadata") or {}
            grouped[self._group_key(meta)].append(chunk)

        group_scores: List[Tuple[Tuple[str, str], float]] = []
        for key, items in grouped.items():
            score = max(float(c.get("score", 0.0)) for c in items)
            group_scores.append((key, score))
        group_scores.sort(key=lambda item: item[1], reverse=True)

        selected_chunks: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        token_budget = self.max_tokens
        source_id = 1

        for key, _ in group_scores:
            items = grouped[key]
            items_sorted = sorted(
                items,
                key=lambda c: (c.get("metadata", {}).get("chunk_index") is None, c.get("metadata", {}).get("chunk_index", 0)),
            )
            merged_texts: List[Tuple[str, Dict[str, Any], float]] = []
            current_text = ""
            current_meta: Dict[str, Any] = {}
            current_score = 0.0
            prev_index: Optional[int] = None

            for chunk in items_sorted:
                meta = chunk.get("metadata") or {}
                idx = meta.get("chunk_index")
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                if current_text:
                    if prev_index is not None and idx is not None and idx == prev_index + 1:
                        current_text = f"{current_text}\n{text}"
                        current_score = max(current_score, float(chunk.get("score", 0.0)))
                    else:
                        merged_texts.append((current_text, current_meta, current_score))
                        current_text = text
                        current_meta = meta
                        current_score = float(chunk.get("score", 0.0))
                else:
                    current_text = text
                    current_meta = meta
                    current_score = float(chunk.get("score", 0.0))
                prev_index = idx if isinstance(idx, int) else prev_index

            if current_text:
                merged_texts.append((current_text, current_meta, current_score))

            for text, meta, score in merged_texts:
                tokens = self._approx_tokens(text)
                if tokens > token_budget:
                    continue
                token_budget -= tokens
                safe_doc = self._safe_doc_name(meta)
                safe_section = self._safe_section(meta)
                safe_page = self._safe_page(meta)
                header_parts = [f"Document: {safe_doc}"]
                if safe_section:
                    header_parts.append(f"Section: {safe_section}")
                if safe_page:
                    header_parts.append(f"Page: {safe_page}")
                header = " | ".join(header_parts)

                context_parts.append(f"[SOURCE-{source_id}] {header}\n{text}\n[/SOURCE-{source_id}]")
                sources.append(
                    {
                        "source_id": source_id,
                        "source_name": safe_doc,
                        "section": safe_section or None,
                        "page": safe_page,
                        "excerpt": text[:400],
                        "score": round(float(score), 4),
                        "chunk_id": meta.get("chunk_id"),
                    }
                )
                selected_chunks.append({"text": text, "score": score, "metadata": meta})
                source_id += 1
                if self.max_chunks and source_id > self.max_chunks:
                    token_budget = 0
                if token_budget <= 0:
                    break
            if token_budget <= 0:
                break

        context_text = "\n\n".join(context_parts)
        token_count = self._approx_tokens(context_text)
        logger.info("Context assembled", extra={"sources": len(sources), "tokens": token_count})
        return ContextBuildResult(context_text, sources, selected_chunks, token_count)
