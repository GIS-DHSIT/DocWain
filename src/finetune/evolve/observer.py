"""Observer module: probes DocWain model via Ollama and identifies weaknesses."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.finetune.evolve.config import EvolveConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic keyword sets used for scoring
# ---------------------------------------------------------------------------

_GROUNDING_PHRASES = {
    "according to", "as stated in", "the document", "page", "section",
    "table", "paragraph", "as shown", "based on", "the report",
    "mentioned in", "referenced in", "from the",
}

_REASONING_PHRASES = {
    "because", "therefore", "thus", "since", "as a result",
    "this means", "consequently", "given that", "it follows",
    "the reason", "implies", "suggests that",
}

_UNCERTAINTY_PHRASES = {
    "i'm not sure", "it is unclear", "the document does not specify",
    "there is no mention", "it's ambiguous", "cannot confirm",
    "not enough information", "uncertain", "may not", "might not",
    "insufficient data", "not explicitly stated",
}

_FORMATTING_MARKERS = {
    "- ", "* ", "1.", "2.", "3.", ":", "\n\n", "##", "**",
}


# ---------------------------------------------------------------------------
# ObservationSignal dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObservationSignal:
    """A single weakness signal produced by the Observer."""

    signal_type: str
    query: str
    model_response: str
    category: str
    subcategory: str
    confidence_score: float
    scores: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class Observer:
    """Probes the live DocWain model and scores responses to find weaknesses."""

    def __init__(self, config: EvolveConfig, output_dir: Path) -> None:
        self._config = config
        self._output_dir = Path(output_dir)
        self._endpoint = config.docwain.endpoint
        self._model_name = config.docwain.model_name
        self._weights: Dict[str, float] = dict(config.gate.weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe_model(self, prompts: List[Dict[str, str]]) -> List[ObservationSignal]:
        """Send *prompts* to the model, score responses, return weakness signals."""
        signals: List[ObservationSignal] = []
        for prompt in prompts:
            query = prompt["query"]
            category = prompt["category"]
            subcategory = prompt["subcategory"]
            try:
                response = self._query_docwain(query)
            except Exception:
                logger.warning("Failed to query model for: %s", query, exc_info=True)
                continue

            scores = self._score_response(query, response, category, subcategory)
            if self._is_weak(scores):
                signal = self._build_signal(query, response, category, subcategory, scores)
                signals.append(signal)
        return signals

    # ------------------------------------------------------------------
    # Model interaction
    # ------------------------------------------------------------------

    def _query_docwain(self, query: str) -> str:
        """Send a prompt to the Ollama /api/generate endpoint."""
        url = f"{self._endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": self._model_name,
            "prompt": query,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")

    # ------------------------------------------------------------------
    # Heuristic scoring
    # ------------------------------------------------------------------

    def _score_response(
        self,
        query: str,
        response: str,
        category: str,
        subcategory: str,
    ) -> Dict[str, float]:
        """Return a dict of five criteria scores in [0, 1]."""
        resp_lower = response.lower()
        query_lower = query.lower()

        accuracy = self._score_accuracy(query_lower, resp_lower, subcategory)
        groundedness = self._score_groundedness(resp_lower)
        reasoning = self._score_reasoning(resp_lower)
        formatting = self._score_formatting(response)
        tone = self._score_tone(resp_lower, category, subcategory)

        return {
            "accuracy": round(min(max(accuracy, 0.0), 1.0), 4),
            "groundedness": round(min(max(groundedness, 0.0), 1.0), 4),
            "reasoning": round(min(max(reasoning, 0.0), 1.0), 4),
            "formatting": round(min(max(formatting, 0.0), 1.0), 4),
            "tone": round(min(max(tone, 0.0), 1.0), 4),
        }

    # --- individual criterion scorers ---

    @staticmethod
    def _score_accuracy(query_lower: str, resp_lower: str, subcategory: str) -> float:
        """Keyword overlap + subcategory bonus heuristic."""
        # Simple keyword overlap between query and response
        query_tokens = set(re.findall(r"\w+", query_lower))
        resp_tokens = set(re.findall(r"\w+", resp_lower))
        if not query_tokens:
            return 0.5
        overlap = len(query_tokens & resp_tokens) / len(query_tokens)

        # Bonus for subcategory-relevant keywords in response
        subcat_bonus = 0.0
        subcat_keywords = {
            "table_extraction": {"table", "row", "column", "cell", "header"},
            "layout_parsing": {"layout", "header", "footer", "column", "sidebar"},
            "cross_reference": {"reference", "appendix", "citation", "footnote", "see"},
            "section_hierarchy": {"section", "subsection", "heading", "chapter", "level"},
            "multi_page_reasoning": {"page", "pages", "across", "combined", "overall"},
            "uncertainty_handling": {"unclear", "uncertain", "not specified", "ambiguous"},
            "adaptive_tone": {"summary", "brief", "technical", "simple", "executive"},
        }
        relevant = subcat_keywords.get(subcategory, set())
        if relevant:
            hits = len(relevant & resp_tokens)
            subcat_bonus = min(hits * 0.1, 0.3)

        return min(overlap + subcat_bonus, 1.0)

    @staticmethod
    def _score_groundedness(resp_lower: str) -> float:
        """How well the response grounds itself in document references."""
        hits = sum(1 for phrase in _GROUNDING_PHRASES if phrase in resp_lower)
        # Scale: 0 hits -> 0.2 (baseline), 4+ hits -> 1.0
        return min(0.2 + hits * 0.2, 1.0)

    @staticmethod
    def _score_reasoning(resp_lower: str) -> float:
        """Presence of reasoning/explanatory language."""
        hits = sum(1 for phrase in _REASONING_PHRASES if phrase in resp_lower)
        return min(0.2 + hits * 0.2, 1.0)

    @staticmethod
    def _score_formatting(response: str) -> float:
        """Checks structural formatting markers (lists, headings, etc)."""
        if not response.strip():
            return 0.1
        hits = sum(1 for marker in _FORMATTING_MARKERS if marker in response)
        length_bonus = min(len(response) / 500, 0.3)
        return min(0.3 + hits * 0.1 + length_bonus, 1.0)

    @staticmethod
    def _score_tone(resp_lower: str, category: str, subcategory: str) -> float:
        """Evaluate tone appropriateness."""
        base = 0.5
        # Uncertainty handling should express caveats
        if subcategory == "uncertainty_handling":
            caveat_hits = sum(1 for p in _UNCERTAINTY_PHRASES if p in resp_lower)
            return min(0.2 + caveat_hits * 0.2, 1.0)
        # Adaptive tone: reward clear explanatory language
        if subcategory == "adaptive_tone":
            clarity_words = {"means", "simply", "in other words", "essentially", "summary"}
            hits = sum(1 for w in clarity_words if w in resp_lower)
            return min(0.3 + hits * 0.15, 1.0)
        # Generic tone: polite, professional
        polite = {"please", "thank", "note that", "important", "consider"}
        hits = sum(1 for w in polite if w in resp_lower)
        return min(base + hits * 0.1, 1.0)

    # ------------------------------------------------------------------
    # Weakness classification
    # ------------------------------------------------------------------

    def _is_weak(self, scores: Dict[str, float]) -> bool:
        """Return True when the weighted composite score is below 0.6."""
        composite = sum(scores[k] * self._weights[k] for k in self._weights)
        return composite < 0.6

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        query: str,
        response: str,
        category: str,
        subcategory: str,
        scores: Dict[str, float],
    ) -> ObservationSignal:
        """Create an ObservationSignal from scored probe results."""
        composite = sum(scores[k] * self._weights[k] for k in self._weights)
        return ObservationSignal(
            signal_type=f"{subcategory}_weakness",
            query=query,
            model_response=response,
            category=category,
            subcategory=subcategory,
            confidence_score=round(composite, 4),
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_signals(self, signals: List[ObservationSignal], iteration: int) -> Path:
        """Write signals to signals/{iter_N}/observation_signals.jsonl."""
        iter_dir = self._output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        out_path = iter_dir / "observation_signals.jsonl"
        with open(out_path, "w") as f:
            for sig in signals:
                f.write(json.dumps(asdict(sig)) + "\n")
        logger.info("Saved %d signals to %s", len(signals), out_path)
        return out_path
