import json
from src.utils.logging_utils import get_logger
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = get_logger(__name__)

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _token_count(text: str) -> int:
    return len(text.split())

@dataclass
class Chunk:
    profile_id: str
    text: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]

class LineFrequencyCleaner:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.line_counts: Dict[str, int] = {}
        self.total = 0

    def observe(self, text: str):
        self.total += 1
        for line in (text or "").splitlines():
            key = line.strip()
            if not key:
                continue
            self.line_counts[key] = self.line_counts.get(key, 0) + 1

    def culprits(self) -> set:
        culprits = set()
        if self.total == 0:
            return culprits
        for line, count in self.line_counts.items():
            if count / self.total >= self.threshold:
                culprits.add(line)
        return culprits

    def clean(self, text: str) -> str:
        culprits = self.culprits()
        kept = []
        for line in (text or "").splitlines():
            if line.strip() in culprits:
                continue
            kept.append(line)
        return "\n".join(kept).strip()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def mmr(anchor: np.ndarray, candidates: List[Tuple[int, np.ndarray]], k: int = 4, lambda_mult: float = 0.5) -> List[int]:
    selected: List[int] = []
    candidate_idxs = [idx for idx, _ in candidates]
    candidate_vecs = {idx: vec for idx, vec in candidates}
    if not candidate_idxs:
        return selected
    for _ in range(min(k, len(candidate_idxs))):
        best_idx = None
        best_score = -math.inf
        for idx in candidate_idxs:
            if idx in selected:
                continue
            relevance = cosine_sim(anchor, candidate_vecs[idx])
            diversity = 0.0
            if selected:
                diversity = max(cosine_sim(candidate_vecs[idx], candidate_vecs[s]) for s in selected)
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
    return selected

def _bundle_adjacent(chunks: List[Chunk], min_size: int = 3, max_size: int = 6) -> List[List[Chunk]]:
    bundles: List[List[Chunk]] = []
    by_doc: Dict[str, List[Chunk]] = {}
    for ch in chunks:
        doc_id = str(ch.metadata.get("doc_id") or ch.metadata.get("document_id") or "doc")
        by_doc.setdefault(doc_id, []).append(ch)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c.metadata.get("chunk_index") or 0)
        i = 0
        while i < len(doc_chunks):
            bundle = doc_chunks[i : i + max_size]
            if len(bundle) >= min_size:
                bundles.append(bundle)
            i += max_size - 1
    return bundles

def _bundle_knn(chunks: List[Chunk], k: int = 6) -> List[List[Chunk]]:
    bundles: List[List[Chunk]] = []
    vectors = [np.array(ch.embedding) if ch.embedding is not None else None for ch in chunks]
    for idx, anchor_vec in enumerate(vectors):
        if anchor_vec is None:
            continue
        sims: List[Tuple[int, np.ndarray]] = []
        for j, vec in enumerate(vectors):
            if j == idx or vec is None:
                continue
            sims.append((j, vec))
        mmr_idx = mmr(anchor_vec, sims, k=min(k, len(sims)))
        bundle = [chunks[idx]] + [chunks[j] for j in mmr_idx]
        if len(bundle) >= 3:
            bundles.append(bundle)
    return bundles

def _call_ollama(prompt: str, model: str, temperature: float = 0.15) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["ollama", "run", model, "--temperature", str(temperature)],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            logger.warning("ollama run failed: %s", proc.stderr.decode("utf-8", errors="ignore"))
            return None
        return proc.stdout.decode("utf-8", errors="ignore").strip()
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ollama invocation failed: %s", exc)
        return None

def _fallback_summary(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    ranked = sorted(sentences, key=lambda s: len(s.split()), reverse=True)
    top = ranked[:3]
    return " ".join(top)

def _fallback_structure(text: str) -> str:
    sentences = _split_sentences(text)
    topic = sentences[0] if sentences else ""
    steps = [s for s in sentences if any(k in s.lower() for k in ["step", "first", "then", "next"])]
    constraints = [s for s in sentences if any(k in s.lower() for k in ["must", "required", "only", "limit"])]
    warnings = [s for s in sentences if any(k in s.lower() for k in ["warn", "caution", "risk", "avoid"])]
    result = {
        "topic": topic,
        "steps": steps,
        "constraints": constraints,
        "warnings": warnings,
        "references": [],
    }
    return json.dumps(result, ensure_ascii=False)

def _bundle_text(bundle: List[Chunk]) -> str:
    parts = []
    for ch in bundle:
        label = ch.metadata.get("section_title") or ch.metadata.get("heading") or ""
        header = f"[{label}]" if label else ""
        parts.append(f"{header}\n{ch.text}")
    return "\n\n".join(parts).strip()

def generate_examples(
    bundles: List[List[Chunk]],
    use_ollama: bool = True,
    ollama_model: str = "llama3.2",
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for bundle in bundles:
        context = _bundle_text(bundle)
        meta = bundle[0].metadata
        tier_used = None
        answer = None
        instruction = None

        # Tier 1: section-aware Q/A
        title = meta.get("section_title") or meta.get("heading")
        if title:
            tier_used = "T1"
            instruction = f"Explain the key steps described in '{title}'. Cite details precisely."
            if use_ollama:
                answer = _call_ollama(
                    f"You are a grounded assistant. Use ONLY the provided context. If missing, say you don't know.\nCONTEXT:\n{context}\n\nQuestion: {instruction}",
                    model=ollama_model,
                    temperature=0.12,
                )

        # Tier 2: faithful summary
        if not answer:
            tier_used = tier_used or "T2"
            instruction = "Summarize the passage precisely. Keep all numbers and constraints."
            if use_ollama:
                answer = _call_ollama(
                    f"You are a grounded assistant. Use ONLY the provided context. If missing, say you don't know.\nCONTEXT:\n{context}\n\nSummarize concisely.",
                    model=ollama_model,
                    temperature=0.1,
                )
            if not answer:
                answer = _fallback_summary(context)

        # Tier 3: structured extraction
        if not answer:
            tier_used = tier_used or "T3"
            instruction = "Extract key fields into JSON: {topic, steps, constraints, warnings, references}."
            answer = _fallback_structure(context)

        # Tier 4: rewrite
        if not answer:
            tier_used = tier_used or "T4"
            instruction = "Rewrite the passage clearly without adding new facts."
            answer = context

        pair = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use ONLY the provided context. If missing, say you don't know.",
                },
                {"role": "user", "content": f"{instruction}\n\nCONTEXT:\n{context}"},
                {"role": "assistant", "content": answer},
            ],
            "meta": {
                "profile_id": bundle[0].profile_id,
                "doc_id": meta.get("doc_id") or meta.get("document_id"),
                "source_file": meta.get("source_file"),
                "pages": meta.get("pages"),
                "chunk_ids": [c.metadata.get("chunk_id") or c.metadata.get("chunk_index") for c in bundle],
                "tier": tier_used,
                "bundle_strategy": meta.get("bundle_strategy", "adjacent"),
            },
        }
        pairs.append(pair)
    return pairs

def build_pairs_for_profile(
    chunks: List[Chunk],
    cleaner: LineFrequencyCleaner,
    min_bundle_size: int = 3,
    max_bundle_size: int = 6,
    use_ollama: bool = True,
    ollama_model: str = "llama3.2",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    drop_counters = {"empty_text": 0, "after_cleanup": 0}
    cleaned_chunks: List[Chunk] = []
    for ch in chunks:
        if not ch.text.strip():
            drop_counters["empty_text"] += 1
            continue
        cleaner.observe(ch.text)
        cleaned_chunks.append(ch)

    culprits = cleaner.culprits()
    final_chunks: List[Chunk] = []
    for ch in cleaned_chunks:
        if culprits:
            cleaned_text = cleaner.clean(ch.text)
        else:
            cleaned_text = ch.text
        if not cleaned_text.strip():
            drop_counters["after_cleanup"] += 1
            continue
        final_chunks.append(Chunk(profile_id=ch.profile_id, text=cleaned_text, embedding=ch.embedding, metadata=ch.metadata))

    bundles_adj = _bundle_adjacent(final_chunks, min_size=min_bundle_size, max_size=max_bundle_size)
    for b in bundles_adj:
        for c in b:
            c.metadata["bundle_strategy"] = "adjacent"
    bundles_knn = _bundle_knn(final_chunks)
    for b in bundles_knn:
        for c in b:
            c.metadata["bundle_strategy"] = "knn_mmr"

    bundles = bundles_adj + bundles_knn
    if not bundles and final_chunks:
        fallback_bundle = final_chunks[:max_bundle_size]
        for c in fallback_bundle:
            c.metadata["bundle_strategy"] = "fallback"
        bundles.append(fallback_bundle)
    pairs = generate_examples(bundles, use_ollama=use_ollama, ollama_model=ollama_model)
    return pairs, drop_counters

__all__ = ["Chunk", "LineFrequencyCleaner", "build_pairs_for_profile"]
