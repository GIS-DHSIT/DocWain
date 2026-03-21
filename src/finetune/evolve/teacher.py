import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .prompts.teacher_sft import DOCWAIN_PERSONA

_CONTENT_PATTERN = re.compile(
    r"(?:\$[\d,.]+\s*(?:million|billion|M|B|K))|"
    r"(?:fiscal year \d{4})|"
    r"(?:filed on [A-Z][a-z]+ \d{1,2})|"
    r"(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_CONTENT_THRESHOLD = 3


class Teacher:
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._docwain_system_prompt = DOCWAIN_PERSONA

    def format_sft_pairs(self, signals: List[Dict], ideal_responses: List[str]) -> List[Dict]:
        pairs = []
        for signal, response in zip(signals, ideal_responses):
            if not self._is_pattern_not_content(response):
                continue
            pair = self._format_sft_pair(signal["query"], response)
            pairs.append(pair)
        return pairs

    def format_dpo_pairs(self, signals: List[Dict], improved_responses: List[str]) -> List[Dict]:
        pairs = []
        for signal, improved in zip(signals, improved_responses):
            if not self._is_pattern_not_content(improved):
                continue
            rejected = signal.get("model_response", "")
            if not rejected:
                continue
            pair = self._format_dpo_pair(signal["query"], chosen=improved, rejected=rejected)
            pairs.append(pair)
        return pairs

    def _format_sft_pair(self, query: str, ideal_response: str) -> Dict[str, Any]:
        return {"messages": [
            {"role": "system", "content": self._docwain_system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": ideal_response},
        ]}

    def _format_dpo_pair(self, query: str, chosen: str, rejected: str) -> Dict[str, Any]:
        return {
            "chosen": [
                {"role": "system", "content": self._docwain_system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "system", "content": self._docwain_system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": rejected},
            ],
        }

    def _is_pattern_not_content(self, response: str) -> bool:
        matches = _CONTENT_PATTERN.findall(response)
        return len(matches) < _CONTENT_THRESHOLD

    def _save_output(self, sft_pairs: List[Dict], dpo_pairs: List[Dict], iteration: int) -> None:
        iter_dir = self._output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        with open(iter_dir / "sft_pairs.jsonl", "w") as f:
            for p in sft_pairs:
                f.write(json.dumps(p) + "\n")
        with open(iter_dir / "dpo_pairs.jsonl", "w") as f:
            for p in dpo_pairs:
                f.write(json.dumps(p) + "\n")
        summary = {"sft_count": len(sft_pairs), "dpo_count": len(dpo_pairs), "iteration": iteration}
        with open(iter_dir / "teach_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
