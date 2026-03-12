from __future__ import annotations

import concurrent.futures
from src.utils.logging_utils import get_logger
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.utils.redis_cache import RedisJsonCache, hash_query, stamp_cache_payload

logger = get_logger(__name__)

@dataclass
class CompanionClassification:
    intent_expected: str
    sentiment: str
    style_directives: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intent_expected": self.intent_expected,
            "sentiment": self.sentiment,
            "style_directives": self.style_directives,
        }

class CompanionClassifier:
    """Fast, cached intent/sentiment/style classifier for companion responses."""

    _FRUSTRATED = [
        "bad answer",
        "wrong",
        "not right",
        "useless",
        "worst",
        "nonsense",
        "incorrect",
        "not correct",
        "not accurate",
        "doesn't make sense",
        "does not make sense",
    ]
    _THANKFUL = [
        "thanks",
        "thank you",
        "great",
        "awesome",
        "good job",
        "perfect",
        "appreciate",
    ]
    _POSITIVE = [
        "nice",
        "good",
        "cool",
        "amazing",
        "well done",
    ]
    _NEGATIVE = [
        "hate",
        "angry",
        "disappointed",
        "frustrated",
    ]
    _HUMOR_REQUEST = [
        "joke",
        "funny",
        "lighthearted",
        "casual",
        "witty",
        "pun",
        "make it fun",
    ]

    def __init__(self, redis_client: Optional[Any] = None, ttl_seconds: Optional[int] = None):
        companion_config = getattr(Config, "Companion", CompanionConfig)
        ttl_default = ttl_seconds if ttl_seconds is not None else companion_config.CLASSIFIER_TTL_SECONDS
        ttl_default = int(ttl_default or 600)
        self.cache = RedisJsonCache(redis_client, default_ttl=ttl_default)
        self.ttl_seconds = ttl_default

    def classify(
        self,
        user_query: str,
        last_turns_summary: str,
        intent_from_query_intelligence: str,
        *,
        session_id: Optional[str] = None,
    ) -> CompanionClassification:
        normalized_query = self._normalize(user_query)
        cache_key = self._cache_key(session_id, normalized_query)
        cached = self.cache.get_json(cache_key, feature="companion_classifier")
        if cached:
            return CompanionClassification(
                intent_expected=cached.get("intent_expected") or intent_from_query_intelligence or "factual",
                sentiment=cached.get("sentiment") or "neutral",
                style_directives=cached.get("style_directives") or {},
            )

        intent_expected = self._normalize_intent(intent_from_query_intelligence or "")
        sentiment, sentiment_signals, low_confidence = self._detect_sentiment(normalized_query)

        if low_confidence:
            sentiment = self._maybe_llm_sentiment(normalized_query) or sentiment

        style_directives = self._build_style_directives(
            normalized_query,
            intent_expected=intent_expected,
            sentiment=sentiment,
            last_turns_summary=last_turns_summary,
        )

        result = CompanionClassification(
            intent_expected=intent_expected,
            sentiment=sentiment,
            style_directives=style_directives,
        )
        payload = result.as_dict()
        payload["sentiment_signals"] = sentiment_signals
        self.cache.set_json(cache_key, stamp_cache_payload(payload), feature="companion_classifier", ttl=self.ttl_seconds)
        return result

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _cache_key(session_id: Optional[str], normalized_query: str) -> str:
        session = session_id or "default"
        return f"cls:{session}:{hash_query(normalized_query)}"

    def _detect_sentiment(self, normalized_query: str) -> Tuple[str, List[str], bool]:
        signals: List[str] = []

        def _match(keywords: List[str]) -> List[str]:
            return [kw for kw in keywords if kw in normalized_query]

        frustrated = _match(self._FRUSTRATED)
        if frustrated:
            return "frustrated", frustrated, False
        thankful = _match(self._THANKFUL)
        if thankful:
            return "thankful", thankful, False
        positive = _match(self._POSITIVE)
        if positive:
            return "positive", positive, False
        negative = _match(self._NEGATIVE)
        if negative:
            return "negative", negative, False
        return "neutral", signals, True

    def _maybe_llm_sentiment(self, normalized_query: str) -> Optional[str]:
        companion_config = getattr(Config, "Companion", CompanionConfig)
        if not getattr(companion_config, "CLASSIFIER_USE_LLM", False):
            return None
        model_name = getattr(companion_config, "CLASSIFIER_MODEL", "")
        if not model_name:
            return None
        timeout = float(getattr(companion_config, "CLASSIFIER_TIMEOUT_SEC", 0.4))

        prompt = (
            "Classify the user sentiment into one of: neutral, positive, negative, frustrated, thankful. "
            "Return only the label.\n\n"
            f"User message: {normalized_query}"
        )

        def _run() -> Optional[str]:
            try:
                from src.llm.gateway import get_llm_gateway
                text = get_llm_gateway().generate(prompt)
                text = (text or "").strip().lower()
                for label in ["neutral", "positive", "negative", "frustrated", "thankful"]:
                    if label in text:
                        return label
            except Exception as exc:  # noqa: BLE001
                logger.debug("Companion classifier LLM fallback failed: %s", exc)
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=timeout)
            except Exception:
                return None

    def _build_style_directives(
        self,
        normalized_query: str,
        *,
        intent_expected: str,
        sentiment: str,
        last_turns_summary: str,
    ) -> Dict[str, Any]:
        humor_requested = any(token in normalized_query for token in self._HUMOR_REQUEST)
        follow_up = bool(last_turns_summary)
        sensitive_domain = self._is_sensitive_domain(normalized_query)

        tone = "friendly"
        if sentiment in {"frustrated", "negative"}:
            tone = "supportive"
        if sensitive_domain:
            tone = "professional"

        humor_level = 1 if humor_requested else 0
        if sensitive_domain or sentiment in {"frustrated", "negative"}:
            humor_level = 0

        structure_required = intent_expected in {
            "comparison",
            "summary",
            "how-to",
            "troubleshooting",
            "extraction",
            "analysis",
        } or any(token in normalized_query for token in ["table", "steps", "step", "bullet", "list", "section"])

        return {
            "tone": tone,
            "humor_level": humor_level,
            "acknowledgement_required": True,
            "structure_required": structure_required,
            "follow_up": follow_up,
        }

    @staticmethod
    def _normalize_intent(intent: str) -> str:
        normalized = (intent or "").strip().lower()
        if normalized in {"procedural", "instruction/how-to", "how-to", "howto"}:
            return "how-to"
        if normalized in {"summary", "summarization"}:
            return "summary"
        if normalized in {"reasoning", "analysis", "deep_analysis"}:
            return "analysis"
        if normalized in {"extraction", "field_extraction", "numeric_lookup"}:
            return "extraction"
        if normalized == "comparison":
            return "comparison"
        if normalized == "troubleshooting":
            return "troubleshooting"
        return "factual"

    @staticmethod
    def _is_sensitive_domain(normalized_query: str) -> bool:
        tokens = [
            "medical",
            "health",
            "diagnosis",
            "patient",
            "hipaa",
            "legal",
            "contract",
            "compliance",
            "law",
            "court",
            "finance",
            "financial",
            "investment",
            "account",
            "bank",
            "tax",
            "insurance",
            "security",
            "safety",
        ]
        return any(tok in normalized_query for tok in tokens)

class CompanionConfig:
    """Placeholder to avoid attribute errors when Config.Companion is absent."""

    CLASSIFIER_TTL_SECONDS = int(os.getenv("COMPANION_CLASSIFIER_TTL_SECONDS", "600"))
    CLASSIFIER_USE_LLM = os.getenv("COMPANION_CLASSIFIER_USE_LLM", "false").lower() in {"1", "true", "yes", "on"}
    CLASSIFIER_MODEL = os.getenv("COMPANION_CLASSIFIER_MODEL", "")
    CLASSIFIER_TIMEOUT_SEC = float(os.getenv("COMPANION_CLASSIFIER_TIMEOUT_SEC", "0.4"))
