import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.payload_utils import get_document_type, get_source_name

logger = get_logger(__name__)

SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*([^\s]+)"),
    re.compile(r"(?i)\b(aws_access_key_id|aws_secret_access_key)\b\s*[:=]\s*([^\s]+)"),
]

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
    return redacted

def token_count(text: str) -> int:
    return len(text.split())

def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    letters = sum(1 for ch in text if ch.isalpha())
    ascii_letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if letters == 0:
        return "unknown"
    ratio = ascii_letters / letters
    return "en" if ratio >= 0.6 else "non_en"

@dataclass
class ChunkRecord:
    text: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None
    chunk_id: Optional[str] = None

def compute_line_frequencies(chunks: Iterable[ChunkRecord]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for ch in chunks:
        for line in ch.text.splitlines():
            line = line.strip()
            if not line:
                continue
            freq[line] = freq.get(line, 0) + 1
    return freq

def strip_boilerplate(chunks: List[ChunkRecord], threshold: float = 0.6) -> List[ChunkRecord]:
    if not chunks:
        return []
    freq = compute_line_frequencies(chunks)
    total = max(1, len(chunks))
    culprits = {line for line, count in freq.items() if (count / total) > threshold}
    if not culprits:
        return chunks
    cleaned = []
    for ch in chunks:
        kept = [line for line in ch.text.splitlines() if line.strip() not in culprits]
        cleaned.append(ChunkRecord(text="\n".join(kept).strip(), metadata=ch.metadata, vector=ch.vector, chunk_id=ch.chunk_id))
    return cleaned

def dedup_blocks(blocks: List[ChunkRecord], threshold: float = 0.92) -> List[ChunkRecord]:
    seen_hashes = set()
    result = []
    for block in blocks:
        norm = re.sub(r"\s+", " ", block.text.lower()).strip()
        key = hash(norm)
        if key in seen_hashes:
            continue
        # near-duplicate check against recent blocks
        duplicate = False
        for prev in result[-10:]:
            ratio = SequenceMatcher(None, norm, re.sub(r"\s+", " ", prev.text.lower()).strip()).ratio()
            if ratio >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        seen_hashes.add(key)
        result.append(block)
    return result

def merge_adjacent(
    chunks: List[ChunkRecord],
    min_tokens: int,
    merge_window: int,
) -> List[ChunkRecord]:
    if not chunks:
        return []
    groups: Dict[str, List[ChunkRecord]] = {}
    for ch in chunks:
        doc_id = str(ch.metadata.get("document_id") or ch.metadata.get("doc_id") or get_source_name(ch.metadata) or "doc")
        groups.setdefault(doc_id, []).append(ch)
    merged_blocks: List[ChunkRecord] = []
    for _, items in groups.items():
        items.sort(key=lambda c: c.metadata.get("chunk_index") or c.metadata.get("page") or 0)
        i = 0
        while i < len(items):
            window = items[i : i + merge_window]
            buf = []
            ids = []
            vec = window[0].vector if window else None
            meta = dict(window[0].metadata) if window else {}
            for w in window:
                buf.append(w.text)
                if w.chunk_id:
                    ids.append(w.chunk_id)
                if token_count("\n".join(buf)) >= min_tokens:
                    break
            merged_text = "\n\n".join(buf).strip()
            meta = dict(meta)
            meta["merged_chunk_ids"] = ids
            merged_blocks.append(ChunkRecord(text=merged_text, metadata=meta, vector=vec, chunk_id=ids[0] if ids else None))
            i += max(1, len(buf))
    return merged_blocks

def _extract_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _pick_key_sentences(text: str, max_sentences: int = 4) -> List[str]:
    sentences = _extract_sentences(text)
    if not sentences:
        return []
    ranked = sorted(sentences, key=lambda s: len(s.split()), reverse=True)
    return ranked[:max_sentences]

class MultiStrategyPairGenerator:
    def __init__(
        self,
        llm_client=None,
        min_pairs: int = 5,
        max_pairs: int = 40,
        temperature: float = 0.2,
    ):
        self.llm_client = llm_client
        self.min_pairs = min_pairs
        self.max_pairs = max_pairs
        self.temperature = temperature

    def generate(self, blocks: List[ChunkRecord]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        strategies = {
            "summary_qa": 0,
            "span_instruction": 0,
            "extractive_qa": 0,
            "metadata_task": 0,
            "bootstrap": 0,
        }
        pairs: List[Dict[str, Any]] = []
        for block in blocks:
            if len(pairs) >= self.max_pairs:
                break
            new_pairs = self._strategy_summary_qa(block)
            pairs.extend(new_pairs)
            strategies["summary_qa"] += len(new_pairs)
        if len(pairs) < self.min_pairs:
            for block in blocks:
                if len(pairs) >= self.max_pairs:
                    break
                new_pairs = self._strategy_span_instruction(block)
                pairs.extend(new_pairs)
                strategies["span_instruction"] += len(new_pairs)
        if len(pairs) < self.min_pairs:
            for block in blocks:
                if len(pairs) >= self.max_pairs:
                    break
                new_pairs = self._strategy_extractive(block)
                pairs.extend(new_pairs)
                strategies["extractive_qa"] += len(new_pairs)
        if len(pairs) < self.min_pairs:
            for block in blocks:
                if len(pairs) >= self.max_pairs:
                    break
                new_pairs = self._strategy_metadata(block)
                pairs.extend(new_pairs)
                strategies["metadata_task"] += len(new_pairs)
        if len(pairs) < self.min_pairs:
            pairs.extend(self._strategy_bootstrap(pairs))
            strategies["bootstrap"] = len(pairs) - sum(strategies.values())

        deduped = self._dedup_pairs(pairs)
        return deduped[: self.max_pairs], strategies

    def _strategy_summary_qa(self, block: ChunkRecord) -> List[Dict[str, Any]]:
        if not self.llm_client:
            return []
        prompt = (
            "Generate 3-5 high-quality question/answer pairs grounded ONLY in the provided text. "
            "Return strict JSON list with keys instruction and output. Do not include any extra text.\n\n"
            f"TEXT:\n{block.text[:4000]}"
        )
        try:
            raw = self.llm_client.generate(prompt, max_retries=1, temperature=self.temperature)
        except TypeError:
            raw = self.llm_client.generate(prompt, max_retries=1)
        pairs = _parse_pairs(raw)
        return self._attach_source(block, pairs, strategy="summary_qa")

    def _strategy_span_instruction(self, block: ChunkRecord) -> List[Dict[str, Any]]:
        sentences = _pick_key_sentences(block.text, max_sentences=3)
        if not sentences:
            return []
        instruction = "List the key facts or steps described in the passage."
        output = " ".join(sentences)
        pair = {"instruction": instruction, "input": block.text, "output": output}
        return self._attach_source(block, [pair], strategy="span_instruction")

    def _strategy_extractive(self, block: ChunkRecord) -> List[Dict[str, Any]]:
        sentences = _extract_sentences(block.text)
        if not sentences:
            return []
        sentence = max(sentences, key=lambda s: len(s.split()))
        question = f"What does the passage state about: \"{sentence.split()[0]}\"?"
        pair = {"instruction": question, "input": block.text, "output": sentence}
        return self._attach_source(block, [pair], strategy="extractive_qa")

    def _strategy_metadata(self, block: ChunkRecord) -> List[Dict[str, Any]]:
        meta = block.metadata or {}
        pairs = []
        for key in ("section_title", "page"):
            if key in meta and meta.get(key):
                pairs.append(
                    {
                        "instruction": f"Extract {key} from the document metadata.",
                        "input": block.text,
                        "output": str(meta.get(key)),
                    }
                )
        doc_type = get_document_type(meta)
        if doc_type:
            pairs.append(
                {
                    "instruction": "Extract document type from the document metadata.",
                    "input": block.text,
                    "output": str(doc_type),
                }
            )
        source_name = get_source_name(meta)
        if source_name:
            pairs.append(
                {
                    "instruction": "Extract source name from the document metadata.",
                    "input": block.text,
                    "output": str(source_name),
                }
            )
        return self._attach_source(block, pairs, strategy="metadata_task")

    def _strategy_bootstrap(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs:
            return []
        augmented = []
        for pair in pairs[:5]:
            instruction = f"In your own words, {pair['instruction'].lower()}"
            augmented.append(
                {
                    "instruction": instruction,
                    "input": pair.get("input", ""),
                    "output": pair.get("output", ""),
                    "source": pair.get("source", {}),
                }
            )
        return augmented

    def _attach_source(self, block: ChunkRecord, pairs: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        output = []
        for pair in pairs:
            pair = dict(pair)
            pair.setdefault("input", block.text)
            pair.setdefault("output", "")
            pair["instruction"] = pair.get("instruction", "").strip()
            pair["output"] = pair.get("output", "").strip()
            pair["source"] = {
                "document_id": block.metadata.get("document_id") or block.metadata.get("doc_id"),
                "source_name": get_source_name(block.metadata or {}),
                "chunk_ids": block.metadata.get("merged_chunk_ids") or [block.chunk_id],
                "strategy": strategy,
            }
            output.append(pair)
        return output

    def _dedup_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        output = []
        for pair in pairs:
            key = (pair.get("instruction"), pair.get("output"))
            if key in seen:
                continue
            seen.add(key)
            output.append(pair)
        return output

def _parse_pairs(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        if isinstance(data, list):
            out = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                instr = row.get("instruction") or row.get("question")
                ans = row.get("output") or row.get("answer")
                if instr and ans:
                    out.append({"instruction": str(instr), "output": str(ans)})
            return out
    except Exception:
        return []
    return []
