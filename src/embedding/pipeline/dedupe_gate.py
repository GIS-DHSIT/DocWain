from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = get_logger(__name__)

@dataclass(frozen=True)
class DedupeConfig:
    similarity_threshold: float = 0.92
    max_overlap_ratio: float = 0.20

def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned

def _simhash64(text: str) -> int:
    tokens = _normalize_text(text).split()
    if not tokens:
        return 0
    vector = [0] * 64
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        h = int.from_bytes(digest[:8], "big")
        for i in range(64):
            bit = 1 if (h >> i) & 1 else -1
            vector[i] += bit
    result = 0
    for i, score in enumerate(vector):
        if score > 0:
            result |= 1 << i
    return result

def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()

def _overlap_prefix_tokens(prev_tokens: List[str], curr_tokens: List[str]) -> int:
    max_overlap = min(len(prev_tokens), len(curr_tokens))
    for overlap in range(max_overlap, 0, -1):
        if prev_tokens[-overlap:] == curr_tokens[:overlap]:
            return overlap
    return 0

def _trim_overlap(prev_tokens: List[str], curr_tokens: List[str], max_ratio: float) -> List[str]:
    if not curr_tokens:
        return curr_tokens
    max_ratio = max(0.0, min(max_ratio, 1.0))
    while curr_tokens:
        overlap = _overlap_prefix_tokens(prev_tokens, curr_tokens)
        if overlap == 0:
            break
        ratio = overlap / max(1, len(curr_tokens))
        if ratio <= max_ratio:
            break
        allowed = int(max_ratio * len(curr_tokens))
        remove = max(1, overlap - allowed)
        curr_tokens = curr_tokens[remove:]
    return curr_tokens

def apply_dedupe_gate(
    chunks: List[str],
    metadata: List[Dict[str, object]],
    config: DedupeConfig | None = None,
) -> Tuple[List[str], List[Dict[str, object]], Dict[str, int]]:
    if len(chunks) != len(metadata):
        raise ValueError("chunk and metadata length mismatch")
    config = config or DedupeConfig()
    max_distance = int((1 - config.similarity_threshold) * 64)

    deduped_chunks: List[str] = []
    deduped_meta: List[Dict[str, object]] = []
    dropped = 0
    merged = 0

    section_order: List[str] = []
    grouped: Dict[str, List[Tuple[str, Dict[str, object]]]] = {}
    for text, meta in zip(chunks, metadata):
        section_id = str(meta.get("section_id") or meta.get("section_path") or meta.get("section_title") or "section")
        if section_id not in grouped:
            section_order.append(section_id)
            grouped[section_id] = []
        grouped[section_id].append((text, meta))

    for section_id in section_order:
        hashes: List[int] = []
        section_chunks: List[str] = []
        section_meta: List[Dict[str, object]] = []
        for text, meta in grouped[section_id]:
            curr_tokens = text.split()
            if section_chunks:
                prev_tokens = section_chunks[-1].split()
                trimmed_tokens = _trim_overlap(prev_tokens, curr_tokens, config.max_overlap_ratio)
                if trimmed_tokens != curr_tokens:
                    ratio = _overlap_prefix_tokens(prev_tokens, curr_tokens) / max(1, len(curr_tokens))
                    logger.warning("Trimming overlap prefix (ratio=%.2f) for chunk dedupe gate", ratio)
                    text = " ".join(trimmed_tokens).strip()
                    curr_tokens = trimmed_tokens
                    if not text:
                        dropped += 1
                        continue

            sim = _simhash64(text)
            is_dup = False
            for prior_idx, prior in enumerate(hashes):
                if _hamming_distance(sim, prior) <= max_distance:
                    is_dup = True
                    prev_text = section_chunks[prior_idx]
                    if text.startswith(prev_text):
                        if len(text) > len(prev_text):
                            section_chunks[prior_idx] = text
                            merged += 1
                        else:
                            dropped += 1
                        break
                    if prev_text.startswith(text):
                        dropped += 1
                        break
                    section_chunks[prior_idx] = f"{prev_text}\n\n{text}".strip()
                    merged += 1
                    break
            if is_dup:
                continue
            section_chunks.append(text)
            section_meta.append(dict(meta))
            hashes.append(sim)

        if len(section_chunks) > 1:
            cleaned_chunks = [section_chunks[0]]
            cleaned_meta = [section_meta[0]]
            for idx in range(1, len(section_chunks)):
                prev_tokens = cleaned_chunks[-1].split()
                curr_tokens = section_chunks[idx].split()
                trimmed_tokens = _trim_overlap(prev_tokens, curr_tokens, config.max_overlap_ratio)
                if trimmed_tokens != curr_tokens:
                    ratio = _overlap_prefix_tokens(prev_tokens, curr_tokens) / max(1, len(curr_tokens))
                    logger.warning("Trimming overlap prefix (ratio=%.2f) after dedupe merge", ratio)
                    trimmed = " ".join(trimmed_tokens).strip()
                    if not trimmed:
                        dropped += 1
                        continue
                    section_chunks[idx] = trimmed
                cleaned_chunks.append(section_chunks[idx])
                cleaned_meta.append(section_meta[idx])
            section_chunks = cleaned_chunks
            section_meta = cleaned_meta

        deduped_chunks.extend(section_chunks)
        deduped_meta.extend(section_meta)

    stats = {"dropped": dropped, "merged": merged}
    if dropped or merged:
        logger.warning("Dedupe gate dropped=%s merged=%s", dropped, merged)
    return deduped_chunks, deduped_meta, stats

__all__ = ["DedupeConfig", "apply_dedupe_gate"]
