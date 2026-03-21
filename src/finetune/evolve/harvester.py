"""Harvester — collects and merges signals from observer + stored feedback."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


class Harvester:
    def __init__(self, signals_dir: Path):
        self._signals_dir = Path(signals_dir)

    def load_observation_signals(self, iteration: int) -> List[Dict[str, Any]]:
        path = self._signals_dir / f"iter_{iteration}" / "observation_signals.jsonl"
        if not path.exists():
            return []
        signals = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    signals.append(json.loads(line))
        return signals

    def load_interaction_signals(self, feedback_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        if feedback_path is None:
            feedback_path = Path("src/outputs/learning_signals/high_quality.jsonl")
        if not feedback_path.exists():
            return []
        signals = []
        with open(feedback_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                signals.append(self._feedback_to_signal(entry))
        return signals

    def _feedback_to_signal(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        messages = entry.get("messages", [])
        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        metadata = entry.get("metadata", {})
        return {
            "query": query,
            "category": "interaction_quality",
            "subcategory": "feedback",
            "signal_type": "user_feedback",
            "metadata": metadata,
        }

    def merge_and_dedup(self, observation_signals: List[Dict[str, Any]], interaction_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_queries = set()
        merged = []
        for signal in observation_signals + interaction_signals:
            key = signal.get("query", "").strip().lower()
            if key and key not in seen_queries:
                seen_queries.add(key)
                merged.append(signal)
        return merged

    def balance_categories(self, signals: List[Dict[str, Any]], max_per_subcategory: int = 30) -> List[Dict[str, Any]]:
        by_subcat: Dict[str, List[Dict[str, Any]]] = {}
        for s in signals:
            sub = s.get("subcategory", "unknown")
            by_subcat.setdefault(sub, []).append(s)
        balanced = []
        for sub, items in by_subcat.items():
            balanced.extend(items[:max_per_subcategory])
        return balanced

    def summarize(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_cat: Counter = Counter()
        by_subcat: Counter = Counter()
        for s in signals:
            by_cat[s.get("category", "unknown")] += 1
            by_subcat[s.get("subcategory", "unknown")] += 1
        return {"total": len(signals), "by_category": dict(by_cat), "by_subcategory": dict(by_subcat)}

    def save_harvest(self, signals: List[Dict[str, Any]], iteration: int) -> Path:
        iter_dir = self._signals_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        out_path = iter_dir / "harvested_signals.jsonl"
        with open(out_path, "w") as f:
            for s in signals:
                f.write(json.dumps(s) + "\n")
        return out_path
