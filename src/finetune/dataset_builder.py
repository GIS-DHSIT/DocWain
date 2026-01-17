import json
import logging
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.vector_store import build_collection_name
from src.finetune import qdrant_discovery

logger = logging.getLogger(__name__)


def _ensure_profile_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create keyword indexes for profile_id/profileId when missing to avoid Qdrant 400 errors."""
    for field in ("profile_id", "profileId"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "already exists" in msg or "index exists" in msg:
                continue
            logger.debug("Ensure payload index %s on %s failed: %s", field, collection_name, exc)


def _get_llm_client(model_name: Optional[str] = None):
    # Lazy import to avoid circular dependencies
    from src.api.dw_newron import create_llm_client

    # Cache per-model to avoid reinitializing Ollama/Gemini clients on every chunk
    global _LLM_CLIENT_CACHE
    try:
        _LLM_CLIENT_CACHE
    except NameError:
        _LLM_CLIENT_CACHE = {}

    key = model_name or "default"
    if key not in _LLM_CLIENT_CACHE:
        _LLM_CLIENT_CACHE[key] = create_llm_client(model_name)
    return _LLM_CLIENT_CACHE[key]


def _sample_chunks(
        profile_id: str,
        subscription_id: Optional[str],
        limit: int = 100,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
) -> List[Dict]:
    """Fetch chunk payloads from Qdrant for a profile/collection."""
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    if not collection_name:
        if not subscription_id:
            raise ValueError("collection_name or subscription_id is required to sample chunks")
        collection_name = build_collection_name(subscription_id)
    target_collection = collection_name
    _ensure_profile_indexes(client, target_collection)

    def _scroll(filter_key: str):
        pts: List[Dict] = []
        scroll_filter = {"must": [{"key": filter_key, "match": {"value": str(profile_id)}}]}
        offset = None
        while len(pts) < limit:
            try:
                res = client.scroll(
                    collection_name=target_collection,
                    scroll_filter=scroll_filter,
                    limit=min(64, limit - len(pts)),
                    with_vectors=False,
                    with_payload=True,
                    offset=offset,
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "index required" in msg and "profileid" in msg:
                    _ensure_profile_indexes(client, target_collection)
                    res = client.scroll(
                        collection_name=target_collection,
                        scroll_filter=scroll_filter,
                        limit=min(64, limit - len(pts)),
                        with_vectors=False,
                        with_payload=True,
                        offset=offset,
                    )
                else:
                    raise
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
    seen: Set[str] = set()
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for pt in points[:limit]:
        payload = pt.payload or {}
        text = payload.get("text") or payload.get("chunk") or ""
        if not text:
            continue
        text = str(text).strip()
        min_len = int(getattr(Config.Retrieval, "MIN_CHUNK_SIZE", 150))
        if len(text) < min_len:
            continue
        identity = hashlib.sha256(
            f"{text}|{payload.get('source_file')}|{payload.get('document_id')}|{payload.get('chunk_index')}".encode("utf-8")
        ).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        payload = dict(payload)
        payload.setdefault("collection_name", target_collection)
        payload.setdefault("profile_id", str(profile_id))
        entry = {"text": text, "metadata": payload}
        group_key = (str(payload.get("source_file") or ""), str(payload.get("document_id") or ""))
        groups.setdefault(group_key, []).append(entry)

    # Sample diversely across documents
    while groups and len(chunks) < limit:
        for key in list(groups.keys()):
            if not groups[key]:
                groups.pop(key, None)
                continue
            chunks.append(groups[key].pop(0))
            if len(chunks) >= limit:
                break
    chunks.sort(
        key=lambda ch: (
            str(ch["metadata"].get("source_file") or ""),
            ch["metadata"].get("page") or ch["metadata"].get("page_number") or 0,
            ch["metadata"].get("chunk_index") or 0,
            str(ch["metadata"].get("chunk_id") or ""),
        )
    )
    return chunks


def _discover_profiles_in_collection(
        client: QdrantClient,
        collection_name: str,
        max_profiles: Optional[int] = None,
        scan_limit: int = 512,
        batch_size: int = 64,
) -> List[str]:
    """
    Scan a Qdrant collection to find unique profile ids stored in payloads.
    Limits the scan to avoid pulling the entire collection.
    """
    profiles: Set[str] = set()
    offset = None
    scanned = 0

    while True:
        if scan_limit and scanned >= scan_limit:
            break
        limit = batch_size if not scan_limit else min(batch_size, scan_limit - scanned)
        if limit <= 0:
            break
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_vectors=False,
            with_payload=True,
            offset=offset,
        )
        if not batch:
            break
        scanned += len(batch)
        for pt in batch:
            payload = pt.payload or {}
            profile = payload.get("profile_id") or payload.get("profileId")
            if profile:
                profiles.add(str(profile))
                if max_profiles and len(profiles) >= max_profiles:
                    return sorted(profiles)
        if offset is None or scanned >= scan_limit:
            break
    return sorted(profiles)


def discover_collections_and_profiles(
        subscription_ids: Optional[List[str]] = None,
        max_profiles_per_collection: Optional[int] = None,
        client: Optional[QdrantClient] = None,
) -> Dict[str, List[str]]:
    """
    Enumerate Qdrant collections and the profile ids they contain.
    If subscription_ids are provided, scanning is limited to that list; otherwise
    every available collection is scanned.
    """
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    try:
        collections = subscription_ids
        if collections is None:
            collections = qdrant_discovery.list_collections(client=client)
    except Exception as exc:
        raise RuntimeError(f"Unable to list Qdrant collections: {exc}") from exc

    discovered: Dict[str, List[str]] = {}
    for sub in collections or []:
        collection_name = build_collection_name(sub)
        try:
            profiles = discover_profiles_for_collection(
                collection_name=collection_name,
                max_profiles=max_profiles_per_collection,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping collection %s due to discovery error: %s", collection_name, exc)
            continue
        if profiles:
            discovered[sub] = profiles
    return discovered


def discover_profiles_for_collection(
        collection_name: str,
        max_profiles: Optional[int] = None,
        scan_limit: int = 512,
        batch_size: int = 64,
        client: Optional[QdrantClient] = None,
) -> List[str]:
    """
    Return a list of profile ids present in a specific Qdrant collection.
    """
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    try:
        return _discover_profiles_in_collection(
            client=client,
            collection_name=collection_name,
            max_profiles=max_profiles,
            scan_limit=scan_limit,
            batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to read collection {collection_name}: {exc}") from exc


def list_profile_ids_in_collection(collection_name: str, client: Optional[QdrantClient] = None) -> List[str]:
    """
    Public helper to resolve all profile ids contained in a collection.
    """
    return discover_profiles_for_collection(collection_name=collection_name, client=client)


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


def _validate_pair(chunk_text: str, pair: Dict[str, str]) -> bool:
    instruction = (pair.get("instruction") or "").strip()
    output = (pair.get("output") or "").strip()
    if not instruction or not output:
        return False
    return output.lower() in chunk_text.lower()


def _generate_pairs_for_chunk(
    chunk_text: str,
    profile_id: str,
    model_name: Optional[str],
    k: int,
    retries: int,
) -> List[Dict[str, str]]:
    llm = _get_llm_client(model_name)
    prompt = f"""You are generating high-quality domain-specific training pairs for profile '{profile_id}'.
Given the document chunk below, produce {k} diverse, factual question/answer pairs that stay within the chunk.
Return strict JSON list of objects: [{{"instruction": "...", "output": "..."}}].

CHUNK:
{chunk_text[:4000]}
"""
    attempts = 0
    while attempts <= retries:
        raw = llm.generate(prompt, max_retries=2)
        pairs = _parse_qa_pairs(raw, k)
        valid = [pair for pair in pairs if _validate_pair(chunk_text, pair)]
        if valid:
            return valid[:k]
        attempts += 1
    return []


def _validate_records(records: List[Dict], profile_id: str, collection_name: Optional[str]) -> None:
    if not records:
        raise ValueError(
            f"No training pairs generated for profile {profile_id} in collection "
            f"{collection_name or 'unknown'}. Ensure chunks contain sufficient text."
        )


def _get_profile_snapshot(
    client: QdrantClient,
    collection: str,
    profile_id: str,
) -> Dict[str, int]:
    _ensure_profile_indexes(client, collection)
    try:
        count = client.count(
            collection_name=collection,
            count_filter={"must": [{"key": "profile_id", "match": {"value": str(profile_id)}}]},
            exact=True,
        )
        return {"count": int(getattr(count, "count", 0))}
    except Exception:
        try:
            count = client.count(
                collection_name=collection,
                count_filter={"must": [{"key": "profileId", "match": {"value": str(profile_id)}}]},
                exact=True,
            )
            return {"count": int(getattr(count, "count", 0))}
        except Exception:
            return {}


def _load_dataset_cache(cache_path: Path) -> Dict[str, Dict]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return {}


def _save_dataset_cache(cache_path: Path, cache: Dict[str, Dict]) -> None:
    try:
        cache_path.write_text(json.dumps(cache, indent=2))
    except Exception as exc:
        logger.warning("Failed to persist dataset cache: %s", exc)


def build_dataset_from_qdrant(
        profile_id: str,
        subscription_id: Optional[str] = "default",
        max_points: int = 120,
        questions_per_chunk: int = 2,
        generation_model: Optional[str] = None,
        output_dir: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
) -> Path:
    """
    Auto-build a JSONL dataset by sampling chunks from Qdrant and generating QA pairs.
    Provide collection_name to bypass subscription-derived naming.
    """
    chunks = _sample_chunks(
        profile_id=profile_id,
        subscription_id=subscription_id,
        limit=max_points,
        client=client,
        collection_name=collection_name,
    )
    if not chunks:
        raise ValueError(f"No chunks found for profile {profile_id} in subscription {subscription_id}")

    records = []
    teacher_model = generation_model or getattr(Config.Finetune, "TEACHER_MODEL", "") or Config.Teams.DEFAULT_MODEL
    retries = int(getattr(Config.Finetune, "QA_RETRY_MAX", 2))
    max_workers = max(1, int(getattr(Config.Finetune, "MAX_CONCURRENCY", 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_pairs_for_chunk,
                chunk["text"],
                profile_id,
                teacher_model,
                questions_per_chunk,
                retries,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            pairs = future.result()
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
    _validate_records(records, profile_id, collection_name or subscription_id)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            payload = rec.get("metadata") or {}
            payload.setdefault("collection_name", collection_name or build_collection_name(subscription_id or "default"))
            payload.setdefault("profile_id", profile_id)
            payload.setdefault("chunk_id", payload.get("chunk_id") or payload.get("id"))
            payload.setdefault("page", payload.get("page") or payload.get("page_number") or 0)
            payload.setdefault("chunk_index", payload.get("chunk_index") or 0)
            rec["metadata"] = payload
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Built auto dataset at %s with %d records", path, len(records))
    return path
