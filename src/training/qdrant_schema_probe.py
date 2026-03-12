import json
from src.utils.logging_utils import get_logger
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

logger = get_logger(__name__)

TEXT_FIELD_CANDIDATES = [
    "canonical_text",
    "text",
    "content",
    "page_content",
    "chunk",
    "chunk_text",
    "document_text",
    "raw_text",
    "summary",
]

PROFILE_FIELD_CANDIDATES = [
    "profile_id",
    "profileId",
    "profile",
    "profileID",
]

def _get_nested(value: Dict[str, Any], path: List[str]) -> Any:
    cur = value
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def _token_count(text: str) -> int:
    return len((text or "").split())

@dataclass
class FieldStat:
    path: List[str]
    non_empty: int = 0
    total: int = 0
    total_chars: int = 0
    total_tokens: int = 0

    def register(self, val: Any):
        self.total += 1
        if val is None:
            return
        if isinstance(val, (dict, list)):
            return
        text = str(val).strip()
        if not text:
            return
        self.non_empty += 1
        self.total_chars += len(text)
        self.total_tokens += _token_count(text)

    @property
    def non_empty_ratio(self) -> float:
        return 0.0 if self.total == 0 else self.non_empty / self.total

    @property
    def avg_chars(self) -> float:
        return 0.0 if self.non_empty == 0 else self.total_chars / self.non_empty

    @property
    def avg_tokens(self) -> float:
        return 0.0 if self.non_empty == 0 else self.total_tokens / self.non_empty

@dataclass
class SchemaProbeResult:
    text_field: FieldStat
    profile_field: FieldStat
    profile_type: str
    observed_keys: List[str]
    sample_payloads: List[Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        return {
            "text_field": {
                "path": self.text_field.path,
                "non_empty_ratio": self.text_field.non_empty_ratio,
                "avg_chars": self.text_field.avg_chars,
                "avg_tokens": self.text_field.avg_tokens,
            },
            "profile_field": {
                "path": self.profile_field.path,
                "type": self.profile_type,
                "non_empty_ratio": self.profile_field.non_empty_ratio,
            },
            "observed_keys": self.observed_keys,
            "sample_payloads": self.sample_payloads,
        }

class SchemaProbe:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        run_dir: Path,
        sample_size: int = 64,
        scroll_page: int = 32,
    ):
        self.client = client
        self.collection = collection
        self.run_dir = run_dir
        self.sample_size = sample_size
        self.scroll_page = scroll_page
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _sample_points(self) -> List[Dict[str, Any]]:
        sampled: List[Dict[str, Any]] = []
        offset = None
        while len(sampled) < self.sample_size:
            limit = min(self.scroll_page, self.sample_size - len(sampled))
            try:
                batch, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=limit,
                    offset=offset,
                    with_vectors=False,
                    with_payload=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Qdrant scroll failed: %s", exc)
                raise
            if not batch:
                break
            for point in batch:
                sampled.append(point.dict())
            if offset is None:
                break
        return sampled

    def _build_candidate_paths(self, payload: Dict[str, Any]) -> List[List[str]]:
        paths: List[List[str]] = []
        for key in TEXT_FIELD_CANDIDATES:
            paths.append([key])
            paths.append(["metadata", key])
            paths.append(["payload", key])
            paths.append(["doc", key])
        # Add leaf keys to improve coverage
        def _walk(prefix: List[str], obj: Any):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_p = prefix + [k]
                    paths.append(new_p)
                    _walk(new_p, v)
            elif isinstance(obj, list):
                if obj and isinstance(obj[0], dict):
                    _walk(prefix + ["0"], obj[0])

        _walk([], payload)
        dedup: List[List[str]] = []
        seen = set()
        for p in paths:
            tup = tuple(p)
            if tup in seen:
                continue
            seen.add(tup)
            dedup.append(p)
        return dedup

    def _score_fields(self, samples: List[Dict[str, Any]]) -> FieldStat:
        candidate_paths = self._build_candidate_paths(samples[0].get("payload", {})) if samples else []
        scores: Dict[Tuple[str, ...], FieldStat] = {}
        for sample in samples:
            payload = sample.get("payload") or {}
            for path in candidate_paths:
                key = tuple(path)
                scores.setdefault(key, FieldStat(path=list(path)))
                scores[key].register(_get_nested(payload, list(path)))
        if not scores:
            raise ValueError("No payload fields found to evaluate")
        sorted_scores = sorted(
            scores.values(),
            key=lambda s: (s.non_empty_ratio, s.avg_chars, s.avg_tokens),
            reverse=True,
        )
        best = sorted_scores[0]
        if best.non_empty == 0:
            raise ValueError(
                f"No training text field detected. Observed keys: {self._collect_keys(samples)}"
            )
        return best

    def _collect_keys(self, samples: List[Dict[str, Any]]) -> List[str]:
        keys = set()
        for s in samples:
            payload = s.get("payload") or {}
            keys.update(payload.keys())
            meta = payload.get("metadata")
            if isinstance(meta, dict):
                keys.update(f"metadata.{k}" for k in meta.keys())
        return sorted(keys)

    def _detect_profile_field(self, samples: List[Dict[str, Any]]) -> Tuple[FieldStat, str]:
        candidate_paths: List[List[str]] = []
        for key in PROFILE_FIELD_CANDIDATES:
            candidate_paths.extend([[key], ["metadata", key], ["payload", key], ["doc", key]])
        scores: Dict[Tuple[str, ...], FieldStat] = {}
        type_counter: Dict[str, int] = {}
        for sample in samples:
            payload = sample.get("payload") or {}
            for path in candidate_paths:
                key = tuple(path)
                scores.setdefault(key, FieldStat(path=list(path)))
                val = _get_nested(payload, list(path))
                scores[key].register(val)
                if val is not None and not isinstance(val, (dict, list)):
                    type_counter[type(val).__name__] = type_counter.get(type(val).__name__, 0) + 1
        if not scores:
            return FieldStat(path=["profile_id"]), "str"
        sorted_scores = sorted(scores.values(), key=lambda s: s.non_empty_ratio, reverse=True)
        best = sorted_scores[0]
        detected_type = "str"
        if type_counter:
            detected_type = max(type_counter.items(), key=lambda kv: kv[1])[0]
        return best, detected_type

    def probe(self) -> SchemaProbeResult:
        samples = self._sample_points()
        if not samples:
            raise ValueError(f"No points found in collection {self.collection}")
        text_field = self._score_fields(samples)
        profile_field, profile_type = self._detect_profile_field(samples)
        result = SchemaProbeResult(
            text_field=text_field,
            profile_field=profile_field,
            profile_type=profile_type,
            observed_keys=self._collect_keys(samples),
            sample_payloads=[s.get("payload") or {} for s in samples[:3]],
        )
        schema_path = self.run_dir / "schema.json"
        schema_path.write_text(json.dumps(result.to_json(), indent=2))
        logger.info("Schema probe complete. Text field: %s", ".".join(text_field.path))
        return result

def run_probe(
    collection: str,
    run_dir: Path,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    sample_size: int = 64,
) -> SchemaProbeResult:
    url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key, timeout=120)
    probe = SchemaProbe(client=client, collection=collection, run_dir=run_dir, sample_size=sample_size)
    return probe.probe()

__all__ = ["SchemaProbe", "SchemaProbeResult", "run_probe", "TEXT_FIELD_CANDIDATES", "FieldStat"]
