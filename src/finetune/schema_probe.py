import json
from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qdrant_client import QdrantClient

logger = get_logger(__name__)

TEXT_CANDIDATES = [
    "text",
    "content",
    "page_content",
    "chunk",
    "chunk_text",
    "document_text",
    "raw_text",
    "summary",
]

PROFILE_CANDIDATES = ["profile_id", "profileId", "profile", "profileID"]

def _get_nested(payload: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = payload
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def _token_count(text: str) -> int:
    return len(text.split())

@dataclass
class FieldStats:
    path: List[str]
    non_empty: int = 0
    total: int = 0
    total_chars: int = 0
    total_tokens: int = 0

    def register(self, val: Any):
        self.total += 1
        if val is None:
            return
        if isinstance(val, (list, dict)):
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
    text_field: FieldStats
    profile_field: FieldStats
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
    def __init__(self, client: QdrantClient, collection: str, sample_size: int = 64):
        self.client = client
        self.collection = collection
        self.sample_size = sample_size

    def _sample_points(self) -> List[Dict[str, Any]]:
        sampled: List[Dict[str, Any]] = []
        offset = None
        while len(sampled) < self.sample_size:
            limit = min(64, self.sample_size - len(sampled))
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                limit=limit,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            if not batch:
                break
            sampled.extend([pt.payload or {} for pt in batch])
            if offset is None:
                break
        return sampled

    def _candidate_paths(self) -> List[List[str]]:
        paths = []
        for key in TEXT_CANDIDATES:
            paths.append([key])
            paths.append(["metadata", key])
            paths.append(["payload", key])
            paths.append(["doc", key])
        return paths

    def _score_text_fields(self, samples: List[Dict[str, Any]]) -> FieldStats:
        scores: Dict[Tuple[str, ...], FieldStats] = {}
        for sample in samples:
            for path in self._candidate_paths():
                key = tuple(path)
                scores.setdefault(key, FieldStats(path=list(path)))
                scores[key].register(_get_nested(sample, list(path)))
        best = sorted(
            scores.values(),
            key=lambda s: (s.non_empty_ratio, s.avg_chars, s.avg_tokens),
            reverse=True,
        )[0]
        if best.non_empty == 0:
            raise ValueError(f"No usable text field detected. Observed keys: {self._collect_keys(samples)}")
        return best

    def _score_profile_field(self, samples: List[Dict[str, Any]]) -> Tuple[FieldStats, str]:
        scores: Dict[Tuple[str, ...], FieldStats] = {}
        types: Dict[str, int] = {}
        for sample in samples:
            for key in PROFILE_CANDIDATES:
                for path in ([key], ["metadata", key], ["payload", key], ["doc", key]):
                    tup = tuple(path)
                    scores.setdefault(tup, FieldStats(path=list(path)))
                    val = _get_nested(sample, list(path))
                    scores[tup].register(val)
                    if val is not None and not isinstance(val, (list, dict)):
                        types[type(val).__name__] = types.get(type(val).__name__, 0) + 1
        best = sorted(scores.values(), key=lambda s: s.non_empty_ratio, reverse=True)[0]
        profile_type = "str"
        if types:
            profile_type = max(types.items(), key=lambda kv: kv[1])[0]
        return best, profile_type

    def _collect_keys(self, samples: List[Dict[str, Any]]) -> List[str]:
        keys = set()
        for sample in samples:
            keys.update(sample.keys())
            meta = sample.get("metadata")
            if isinstance(meta, dict):
                keys.update([f"metadata.{k}" for k in meta.keys()])
        return sorted(keys)

    def probe(self) -> SchemaProbeResult:
        samples = self._sample_points()
        if not samples:
            raise ValueError(f"No points found in collection {self.collection}")
        text_field = self._score_text_fields(samples)
        profile_field, profile_type = self._score_profile_field(samples)
        result = SchemaProbeResult(
            text_field=text_field,
            profile_field=profile_field,
            profile_type=profile_type,
            observed_keys=self._collect_keys(samples),
            sample_payloads=samples[:3],
        )
        return result

def load_or_probe(
    client: QdrantClient,
    collection: str,
    run_dir: Path,
    sample_size: int = 64,
) -> SchemaProbeResult:
    schema_path = run_dir / "schema.json"
    if schema_path.exists():
        try:
            data = json.loads(schema_path.read_text())
            text_field = FieldStats(path=data["text_field"]["path"])
            profile_field = FieldStats(path=data["profile_field"]["path"])
            return SchemaProbeResult(
                text_field=text_field,
                profile_field=profile_field,
                profile_type=data["profile_field"].get("type", "str"),
                observed_keys=data.get("observed_keys", []),
                sample_payloads=data.get("sample_payloads", []),
            )
        except Exception as exc:
            logger.warning("Failed to read cached schema: %s", exc)

    probe = SchemaProbe(client, collection, sample_size=sample_size)
    result = probe.probe()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(result.to_json(), indent=2))
    return result
