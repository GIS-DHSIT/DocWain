"""Distiller — cherry-picks best responses from all models for hybrid distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .prompts.teacher_sft import DOCWAIN_PERSONA


class Distiller:
    def __init__(self, distill_every_n=3):
        self._every_n = distill_every_n
        self._system_prompt = DOCWAIN_PERSONA

    def should_distill(self, iteration):
        return iteration > 0 and iteration % self._every_n == 0

    def cherry_pick(self, eval_results, criterion="accuracy"):
        by_query = {}
        for model_name, results in eval_results.items():
            for r in results:
                q = r["query"]
                by_query.setdefault(q, []).append({**r, "model": model_name})
        best = []
        for query, candidates in by_query.items():
            winner = max(candidates, key=lambda c: c["scores"].get(criterion, 0))
            best.append({"query": query, "response": winner["response"], "model": winner["model"]})
        return best

    def build_dataset(self, best_responses):
        dataset = []
        for item in best_responses:
            dataset.append({"messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": item["query"]},
                {"role": "assistant", "content": item["response"]},
            ]})
        return dataset

    def save_dataset(self, dataset, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")
        return path
