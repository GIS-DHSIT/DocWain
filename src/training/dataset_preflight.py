import json
from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = get_logger(__name__)

@dataclass
class DropCounters:
    empty_text: int = 0
    after_cleanup: int = 0
    no_embedding: int = 0
    other: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "empty_text": self.empty_text,
            "after_cleanup": self.after_cleanup,
            "no_embedding": self.no_embedding,
            "other": self.other,
        }

@dataclass
class ProfileStats:
    profile_id: str
    chunks_total: int = 0
    chunks_with_text: int = 0
    tokens: List[int] = field(default_factory=list)
    unique_docs: set = field(default_factory=set)
    unique_sources: set = field(default_factory=set)
    drop_reasons: DropCounters = field(default_factory=DropCounters)
    bundles_created: int = 0
    pairs_created: int = 0
    diagnosis: Dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "chunks_total": self.chunks_total,
            "chunks_with_text": self.chunks_with_text,
            "p50_tokens": self._percentile(50),
            "p90_tokens": self._percentile(90),
            "unique_docs": len(self.unique_docs),
            "unique_sources": len(self.unique_sources),
            "bundles_created": self.bundles_created,
            "pairs_created": self.pairs_created,
            "drop_reasons": self.drop_reasons.to_dict(),
            "diagnosis": self.diagnosis,
        }

    def _percentile(self, pct: int) -> float:
        if not self.tokens:
            return 0.0
        sorted_tokens = sorted(self.tokens)
        k = (len(sorted_tokens) - 1) * (pct / 100)
        f = int(k)
        c = min(f + 1, len(sorted_tokens) - 1)
        if f == c:
            return float(sorted_tokens[int(k)])
        d0 = sorted_tokens[f] * (c - k)
        d1 = sorted_tokens[c] * (k - f)
        return float(d0 + d1)

def save_profile_stats(run_dir: Path, stats: List[ProfileStats]):
    path = run_dir / "preflight.json"
    payload = [s.as_json() for s in stats]
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Preflight stats written to %s", path)

def gate_training(stats: List[ProfileStats], min_pairs: int = 1) -> Dict[str, Dict[str, any]]:
    result: Dict[str, Dict[str, any]] = {}
    for s in stats:
        if s.pairs_created >= min_pairs:
            result[s.profile_id] = {"status": "ready"}
            continue
        diag = s.diagnosis or {}
        diag.setdefault("drop_reasons", s.drop_reasons.to_dict())
        diag.setdefault("counts", {"chunks_total": s.chunks_total, "chunks_with_text": s.chunks_with_text})
        result[s.profile_id] = {
            "status": "skipped_insufficient_pairs",
            "diagnosis": diag,
        }
    return result
