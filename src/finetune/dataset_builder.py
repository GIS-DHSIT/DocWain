import json
from src.utils.logging_utils import get_logger
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict, Optional, Set, Tuple

from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.vector_store import build_collection_name
from src.utils.payload_utils import get_source_name
from src.finetune import qdrant_discovery
from src.finetune.pair_generator import (
    ChunkRecord,
    MultiStrategyPairGenerator,
    normalize_text,
    redact_secrets,
    token_count,
    detect_language,
    strip_boilerplate,
    merge_adjacent,
    dedup_blocks,
)
from src.finetune.schema_probe import load_or_probe
from src.finetune.config_resolver import as_int, as_float

logger = get_logger(__name__)

@dataclass
class DatasetBuildResult:
    dataset_path: Optional[Path]
    diagnostics_path: Path
    status: str
    pair_count: int
    diagnostics: Dict[str, Any]

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

def _payload_get(payload: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = payload
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def _convert_profile_value(value: Any, profile_type: str) -> Any:
    if value is None:
        return None
    if profile_type == "int":
        try:
            return int(value)
        except Exception:
            return value
    if profile_type == "float":
        try:
            return float(value)
        except Exception:
            return value
    return str(value)

def _sample_chunks(
    profile_id: str,
    subscription_id: Optional[str],
    limit: int = 100,
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> List[Dict]:
    """Fetch chunk payloads from Qdrant for a profile/collection using schema probe."""
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    if not collection_name:
        if not subscription_id:
            raise ValueError("collection_name or subscription_id is required to sample chunks")
        collection_name = build_collection_name(subscription_id)
    target_collection = collection_name
    cache_dir = run_dir or (Path(Config.Path.APP_HOME) / "finetune_artifacts" / "_schema_cache")
    schema = load_or_probe(client, target_collection, cache_dir)
    text_path = schema.text_field.path
    profile_path = schema.profile_field.path
    profile_type = schema.profile_type
    _ensure_profile_indexes(client, target_collection)

    def _scroll(path_key: str):
        pts: List[Dict] = []
        scroll_filter = {"must": [{"key": path_key, "match": {"value": _convert_profile_value(profile_id, profile_type)}}]}
        offset = None
        while len(pts) < limit:
            res = client.scroll(
                collection_name=target_collection,
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

    key_path = ".".join(profile_path)
    points = _scroll(key_path)
    if not points and profile_path != ["profile_id"]:
        points = _scroll("profile_id")
    if not points and profile_path != ["profileId"]:
        points = _scroll("profileId")

    chunks = []
    seen: Set[str] = set()
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for pt in points[:limit]:
        payload = pt.payload or {}
        text = _payload_get(payload, text_path) or ""
        if not text:
            continue
        text = str(text).strip()
        min_len = int(getattr(Config.Finetune, "MIN_CHUNK_CHARS", 200))
        if len(text) < min_len:
            continue
        source_name = get_source_name(payload) or ""
        identity = hashlib.sha256(
            f"{text}|{source_name}|{payload.get('document_id')}|{payload.get('chunk_index')}".encode("utf-8")
        ).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        payload = dict(payload)
        payload.setdefault("collection_name", target_collection)
        payload.setdefault("profile_id", str(profile_id))
        entry = {"text": text, "metadata": payload}
        group_key = (str(source_name), str(payload.get("document_id") or ""))
        groups.setdefault(group_key, []).append(entry)

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
            str(get_source_name(ch["metadata"]) or ""),
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
            logger.debug("Skipping collection %s due to discovery error: %s", collection_name, exc)
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

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)

def _build_embedding_bundles(blocks: List[ChunkRecord], k: int = 2, min_sim: float = 0.6) -> List[ChunkRecord]:
    bundles: List[ChunkRecord] = []
    vectors = [b.vector for b in blocks]
    for idx, block in enumerate(blocks):
        if not block.vector:
            continue
        sims: List[Tuple[int, float]] = []
        for j, vec in enumerate(vectors):
            if j == idx or not vec:
                continue
            sim = _cosine_similarity(block.vector, vec)
            if sim >= min_sim:
                sims.append((j, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors = sims[:k]
        if not neighbors:
            continue
        texts = [block.text] + [blocks[j].text for j, _ in neighbors]
        meta = dict(block.metadata)
        meta["bundle_strategy"] = "embedding_neighbors"
        meta["neighbor_ids"] = [blocks[j].chunk_id for j, _ in neighbors]
        bundles.append(ChunkRecord(text="\n\n".join(texts), metadata=meta, vector=block.vector, chunk_id=block.chunk_id))
    return bundles

def _coerce_vector(vec: Any) -> Optional[List[float]]:
    if vec is None:
        return None
    try:
        return [float(x) for x in vec]
    except Exception:
        return None

def _percentile(values: List[int], pct: int) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[int(k)])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)

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
        run_id: Optional[str] = None,
) -> DatasetBuildResult:
    """
    Auto-build a JSONL dataset by sampling chunks from Qdrant and generating QA pairs.
    Provide collection_name to bypass subscription-derived naming.
    """
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    if not collection_name:
        if not subscription_id:
            raise ValueError("collection_name or subscription_id is required")
        collection_name = build_collection_name(subscription_id)

    run_id = run_id or f"run-{int(time.time())}"
    base_root = Path(output_dir) if output_dir else Path(Config.Path.APP_HOME) / "outputs" / "finetune" / run_id
    profile_root = base_root / profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    dataset_path = profile_root / "train.jsonl"
    diagnostics_path = profile_root / "diagnostics.json"

    schema = load_or_probe(client, collection_name, base_root)
    text_path = schema.text_field.path
    profile_path = schema.profile_field.path
    profile_type = schema.profile_type

    profile_key = ".".join(profile_path)
    profile_value = _convert_profile_value(profile_id, profile_type)

    offset = None
    points = []
    while len(points) < max_points:
        batch, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter={"must": [{"key": profile_key, "match": {"value": profile_value}}]},
            limit=min(64, max_points - len(points)),
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        if not batch:
            break
        points.extend(batch)
        if offset is None:
            break
    if not points and profile_key not in {"profile_id", "profileId"}:
        # fallback to common keys
        for fallback_key in ("profile_id", "profileId"):
            offset = None
            points = []
            while len(points) < max_points:
                batch, offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter={"must": [{"key": fallback_key, "match": {"value": profile_value}}]},
                    limit=min(64, max_points - len(points)),
                    with_vectors=True,
                    with_payload=True,
                    offset=offset,
                )
                if not batch:
                    break
                points.extend(batch)
                if offset is None:
                    break
            if points:
                break

    diagnostics: Dict[str, Any] = {
        "profile_id": profile_id,
        "collection_name": collection_name,
        "text_field": text_path,
        "profile_field": profile_path,
        "profile_type": profile_type,
        "filter": {"key": profile_key, "value": profile_value},
        "counts": {},
        "drop_reasons": {},
        "language": {},
        "ocr_confidence": {},
        "strategies": {},
        "sample_payloads": schema.sample_payloads,
        "sample_chunk_ids": [],
    }

    if not points:
        diagnostics["counts"]["chunks_total"] = 0
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
        return DatasetBuildResult(
            dataset_path=None,
            diagnostics_path=diagnostics_path,
            status="skipped_insufficient_pairs",
            pair_count=0,
            diagnostics=diagnostics,
        )

    min_chars = as_int(getattr(Config.Finetune, "MIN_CHUNK_CHARS", 200), "min_chunk_chars", "default")
    min_tokens = as_int(getattr(Config.Finetune, "MIN_MERGED_TOKENS", 120), "min_merged_tokens", "default")
    min_pairs = as_int(getattr(Config.Finetune, "MIN_PAIRS_PER_PROFILE", 5), "min_pairs_per_profile", "default")
    max_pairs = as_int(getattr(Config.Finetune, "MAX_PAIRS_PER_PROFILE", 40), "max_pairs_per_profile", "default")
    merge_window = as_int(getattr(Config.Finetune, "MERGE_WINDOW", 4), "merge_window", "default")
    dedup_threshold = as_float(getattr(Config.Finetune, "DEDUP_THRESHOLD", 0.92), "dedup_threshold", "default")
    gen_model = generation_model or getattr(Config.Finetune, "GENERATION_MODEL", None) or getattr(Config.Finetune, "TEACHER_MODEL", "") or Config.Teams.DEFAULT_MODEL
    gen_temp = as_float(getattr(Config.Finetune, "GENERATION_TEMPERATURE", 0.2), "temperature", "default")

    empty_ids: List[str] = []
    chunks_total = 0
    chunks_with_text = 0
    tokens: List[int] = []
    language_counts: Dict[str, int] = {}
    ocr_scores: List[float] = []
    drop_reasons = {"empty_text": 0, "too_short": 0, "non_english": 0, "duplicate": 0}

    raw_chunks: List[ChunkRecord] = []
    for pt in points:
        chunks_total += 1
        payload = pt.payload or {}
        text = _payload_get(payload, text_path)
        if not text:
            empty_ids.append(str(pt.id))
            drop_reasons["empty_text"] += 1
            continue
        text = redact_secrets(normalize_text(str(text)))
        if len(text) < min_chars:
            drop_reasons["too_short"] += 1
            continue
        lang = detect_language(text)
        language_counts[lang] = language_counts.get(lang, 0) + 1
        if lang == "non_en" and not getattr(Config.Finetune, "ALLOW_NON_ENGLISH", True):
            drop_reasons["non_english"] += 1
            continue
        chunks_with_text += 1
        tokens.append(token_count(text))
        if payload.get("ocr_confidence") is not None:
            try:
                ocr_scores.append(float(payload.get("ocr_confidence")))
            except Exception:
                pass
        raw_chunks.append(
            ChunkRecord(
                text=text,
                metadata=payload,
                vector=_coerce_vector(getattr(pt, "vector", None)),
                chunk_id=str(pt.id),
            )
        )

    cleaned_chunks = strip_boilerplate(raw_chunks, threshold=float(getattr(Config.Finetune, "BOILERPLATE_THRESHOLD", 0.6)))
    merged = merge_adjacent(cleaned_chunks, min_tokens=min_tokens, merge_window=merge_window)
    deduped = dedup_blocks(merged, threshold=dedup_threshold)
    if len(merged) > len(deduped):
        drop_reasons["duplicate"] += len(merged) - len(deduped)
    embedding_bundles = _build_embedding_bundles(deduped)
    blocks = deduped + embedding_bundles

    llm_client = None
    if gen_model:
        try:
            llm_client = _get_llm_client(gen_model)
        except Exception as exc:
            logger.debug("LLM generation unavailable (%s); falling back to heuristic strategies.", exc)

    generator = MultiStrategyPairGenerator(
        llm_client=llm_client,
        min_pairs=min_pairs,
        max_pairs=max_pairs,
        temperature=gen_temp,
    )
    pairs, strategy_counts = generator.generate(blocks)
    diagnostics["strategies"] = strategy_counts
    logger.info(
        "Pair generation profile=%s pairs=%d strategies=%s drops=%s",
        profile_id,
        len(pairs),
        strategy_counts,
        drop_reasons,
    )

    diagnostics["counts"] = {
        "chunks_total": chunks_total,
        "chunks_with_text": chunks_with_text,
        "merged_blocks": len(merged),
        "deduped_blocks": len(deduped),
        "embedding_bundles": len(embedding_bundles),
        "pairs_created": len(pairs),
        "empty_ratio": (drop_reasons["empty_text"] / chunks_total) if chunks_total else 0.0,
    }
    diagnostics["drop_reasons"] = drop_reasons
    diagnostics["language"] = language_counts
    diagnostics["ocr_confidence"] = {
        "avg": sum(ocr_scores) / len(ocr_scores) if ocr_scores else None,
        "min": min(ocr_scores) if ocr_scores else None,
        "max": max(ocr_scores) if ocr_scores else None,
    }
    diagnostics["sample_chunk_ids"] = empty_ids[:5]
    diagnostics["token_stats"] = {
        "avg": sum(tokens) / len(tokens) if tokens else 0.0,
        "p50": _percentile(tokens, 50),
        "p90": _percentile(tokens, 90),
    }

    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    if len(pairs) < min_pairs:
        return DatasetBuildResult(
            dataset_path=None,
            diagnostics_path=diagnostics_path,
            status="skipped_insufficient_pairs",
            pair_count=len(pairs),
            diagnostics=diagnostics,
        )

    with dataset_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "instruction": pair.get("instruction", "").strip(),
                "input": redact_secrets(pair.get("input", "").strip()),
                "output": redact_secrets(pair.get("output", "").strip()),
                "source": pair.get("source", {}),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Built auto dataset at %s with %d records", dataset_path, len(pairs))
    return DatasetBuildResult(
        dataset_path=dataset_path,
        diagnostics_path=diagnostics_path,
        status="success",
        pair_count=len(pairs),
        diagnostics=diagnostics,
    )
