import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.vector_store import build_collection_name

logger = logging.getLogger(__name__)


def _get_llm_client(model_name: Optional[str] = None):
    # Lazy import to avoid circular dependencies
    from src.api.dw_newron import create_llm_client

    return create_llm_client(model_name)


def _sample_chunks(
        profile_id: str,
        subscription_id: str,
        limit: int = 100
) -> List[Dict]:
    """Fetch chunk payloads from Qdrant for a profile/collection."""
    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    collection = build_collection_name(subscription_id)

    def _scroll(filter_key: str):
        pts: List[Dict] = []
        scroll_filter = {"must": [{"key": filter_key, "match": {"value": str(profile_id)}}]}
        offset = None
        while len(pts) < limit:
            res = client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=min(64, limit - len(pts)),
                with_vectors=False,
                with_payload=True,
                offset=offset,
            )
            batch, offset = res
            if not batch:
                break
            pts.extend(batch)
            if offset is None:
                break
        return pts

    points = _scroll("profile_id")
    if not points:
        points = _scroll("profileId")

    chunks = []
    seen = set()
    for pt in points[:limit]:
        payload = pt.payload or {}
        text = payload.get("text") or payload.get("chunk") or ""
        if not text:
            continue
        text = str(text).strip()
        if len(text) < 80:
            continue
        sig = hash(text[:256])
        if sig in seen:
            continue
        seen.add(sig)
        chunks.append(
            {
                "text": text,
                "metadata": payload,
            }
        )
    return chunks


def _parse_qa_pairs(raw: str, expected: int) -> List[Dict[str, str]]:
    """Parse JSON or line-based QA pairs into training records."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        if isinstance(data, list):
            out = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                instr = row.get("instruction") or row.get("question") or row.get("q")
                ans = row.get("output") or row.get("answer") or row.get("a")
                if instr and ans:
                    out.append({"instruction": instr.strip(), "output": ans.strip()})
            if out:
                return out[:expected]
    except Exception:
        pass
    pairs = []
    q, a = None, None
    for line in raw.splitlines():
        lower = line.lower()
        if lower.startswith("q:") or lower.startswith("question:"):
            if q and a:
                pairs.append({"instruction": q.strip(), "output": a.strip()})
                a = None
            q = line.split(":", 1)[1]
        elif lower.startswith("a:") or lower.startswith("answer:"):
            a = line.split(":", 1)[1]
        if len(pairs) >= expected:
            break
    if q and a and len(pairs) < expected:
        pairs.append({"instruction": q.strip(), "output": a.strip()})

    # Fallback: treat whole text as a single answer
    if not pairs:
        cleaned = raw.strip()
        if cleaned:
            pairs.append({"instruction": "Answer questions about the given text", "output": cleaned})
    return pairs[:expected]


def _generate_pairs_for_chunk(chunk_text: str, profile_id: str, model_name: Optional[str], k: int) -> List[Dict[str, str]]:
    llm = _get_llm_client(model_name)
    prompt = f"""You are generating high-quality domain-specific training pairs for profile '{profile_id}'.
Given the document chunk below, produce {k} diverse, factual question/answer pairs that stay within the chunk.
Return strict JSON list of objects: [{{"instruction": "...", "output": "..."}}].

CHUNK:
{chunk_text[:4000]}
"""
    raw = llm.generate(prompt, max_retries=2)
    return _parse_qa_pairs(raw, k)


def build_dataset_from_qdrant(
        profile_id: str,
        subscription_id: str = "default",
        max_points: int = 120,
        questions_per_chunk: int = 2,
        generation_model: Optional[str] = None,
        output_dir: Optional[str] = None
) -> Path:
    """
    Auto-build a JSONL dataset by sampling chunks from Qdrant and generating QA pairs.
    """
    chunks = _sample_chunks(profile_id, subscription_id, limit=max_points)
    if not chunks:
        raise ValueError(f"No chunks found for profile {profile_id} in subscription {subscription_id}")

    records = []
    for chunk in chunks:
        text = chunk["text"]
        pairs = _generate_pairs_for_chunk(text, profile_id, generation_model, questions_per_chunk)
        for pair in pairs:
            records.append(
                {
                    "instruction": pair["instruction"],
                    "output": pair["output"],
                    "input": "",
                    "profile_id": profile_id,
                    "metadata": {
                        "source_file": chunk["metadata"].get("source_file"),
                        "document_id": chunk["metadata"].get("document_id"),
                        "chunk_index": chunk["metadata"].get("chunk_index"),
                    },
                }
            )

    root = Path(output_dir) if output_dir else Path(Config.Path.APP_HOME) / "finetune_artifacts" / profile_id / "auto_datasets"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"qa_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Built auto dataset at %s with %d records", path, len(records))
    return path
