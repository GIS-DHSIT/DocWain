from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.api.config import Config
from src.api.genai_client import generate_text
from src.nlp.intent_rules import match_intent_rules
from src.nlp.sentiment_rules import match_sentiment_rules
from src.policy.response_policy import INFO_MODE, ResponseModeClassifier, build_docwain_intro
from src.prompting.persona import enforce_docwain_identity, get_docwain_persona, sanitize_response

logger = get_logger(__name__)

@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    method: str
    matched_rule: Optional[str] = None

@dataclass(frozen=True)
class SentimentResult:
    sentiment: str
    confidence: float
    feedback_type: str
    should_recover: bool
    matched_rule: Optional[str] = None

@dataclass(frozen=True)
class RouteDecision:
    intent: IntentResult
    sentiment: Optional[SentimentResult]
    direct_response: bool
    response_text: Optional[str]
    use_retrieval: bool
    policy: str

_INTENT_FALLBACK_PROMPT = (
    "Classify the user message into one intent: GREETING, THANKS_OR_PRAISE, "
    "NEGATIVE_FEEDBACK, SMALL_TALK, META, DOCUMENT_TASK, CLARIFICATION, UNKNOWN. "
    "Return strict JSON: {{\"intent\": string, \"confidence\": number between 0 and 1}}. "
    "Message: {text}"
)

_SENTIMENT_FALLBACK_PROMPT = (
    "Analyze sentiment for the user message. "
    "Return strict JSON: {{\"sentiment\": \"positive|negative|neutral|mixed\", "
    "\"confidence\": number between 0 and 1, "
    "\"feedback_type\": \"praise|thanks|complaint|frustration|none\"}}. "
    "Message: {text}"
)

_DOC_TASK_CUES = re.compile(
    r"\b(summarize|summarise|analyze|analyse|extract|find|identify|compare|rank|list|generate|" \
    r"review|explain|highlight|locate|compile|report)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = re.compile(r"\b(what|why|how|which|when|where|who)\b", re.IGNORECASE)

def _intent_from_rules(text: str) -> Optional[IntentResult]:
    match = match_intent_rules(text)
    if not match:
        return None
    return IntentResult(intent=match.intent, confidence=match.confidence, method="rules", matched_rule=match.rule_name)

def _sentiment_from_rules(text: str) -> Optional[SentimentResult]:
    match = match_sentiment_rules(text)
    if not match:
        return None
    should_recover = match.sentiment in {"negative", "mixed"}
    return SentimentResult(
        sentiment=match.sentiment,
        confidence=match.confidence,
        feedback_type=match.feedback_type,
        should_recover=should_recover,
        matched_rule=match.rule_name,
    )

def _has_document_task_cues(text: str) -> bool:
    if _QUESTION_WORDS.search(text):
        return True
    return bool(_DOC_TASK_CUES.search(text))

def _classify_with_ollama(model_name: str, prompt: str, llm_client=None) -> Optional[dict]:
    if not model_name and llm_client is None:
        return None
    try:
        full_prompt = f"Return only JSON.\n\n{prompt}"
        if llm_client is not None:
            content = llm_client.generate(full_prompt)
        else:
            from src.llm.gateway import get_llm_gateway
            content = get_llm_gateway().generate(full_prompt)
        return json.loads((content or "").strip())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama classification failed: %s", exc)
        return None

def _classify_with_genai(prompt: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY") or getattr(Config.Model, "GEMINI_API_KEY", "")
    model = os.getenv("DOCWAIN_SMALLTALK_MODEL") or getattr(Config.Model, "GEMINI_MODEL_NAME", "")
    if not api_key or not model:
        return None
    try:
        text, _ = generate_text(api_key=api_key, model=model, prompt=prompt)
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GenAI classification failed: %s", exc)
        return None

def detect_intent(text: str) -> IntentResult:
    text = (text or "").strip()
    if not text:
        return IntentResult(intent="UNKNOWN", confidence=0.0, method="empty")

    rule_match = _intent_from_rules(text)
    if rule_match:
        return rule_match

    if _has_document_task_cues(text):
        return IntentResult(intent="DOCUMENT_TASK", confidence=0.7, method="heuristic")

    model_name = os.getenv("DOCWAIN_SMALLTALK_MODEL", "").strip()
    prompt = _INTENT_FALLBACK_PROMPT.format(text=text)
    payload = _classify_with_ollama(model_name, prompt) or _classify_with_genai(prompt)
    if payload:
        intent = str(payload.get("intent", "UNKNOWN")).upper()
        confidence = float(payload.get("confidence", 0.0))
        return IntentResult(intent=intent, confidence=confidence, method="llm")

    return IntentResult(intent="DOCUMENT_TASK", confidence=0.55, method="fallback")

def analyze_sentiment(text: str) -> SentimentResult:
    text = (text or "").strip()
    if not text:
        return SentimentResult(
            sentiment="neutral",
            confidence=0.0,
            feedback_type="none",
            should_recover=False,
            matched_rule=None,
        )

    rule_match = _sentiment_from_rules(text)
    if rule_match:
        return rule_match

    model_name = os.getenv("DOCWAIN_SMALLTALK_MODEL", "").strip()
    prompt = _SENTIMENT_FALLBACK_PROMPT.format(text=text)
    payload = _classify_with_ollama(model_name, prompt) or _classify_with_genai(prompt)
    if payload:
        sentiment = str(payload.get("sentiment", "neutral")).lower()
        feedback_type = str(payload.get("feedback_type", "none")).lower()
        confidence = float(payload.get("confidence", 0.0))
        should_recover = sentiment in {"negative", "mixed"}
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            feedback_type=feedback_type,
            should_recover=should_recover,
            matched_rule=None,
        )

    return SentimentResult(
        sentiment="neutral",
        confidence=0.5,
        feedback_type="none",
        should_recover=False,
        matched_rule=None,
    )

def _build_direct_response(intent: IntentResult, sentiment: Optional[SentimentResult], response_mode: str, user_text: str = "") -> str:
    # Try dynamic engine first.
    if user_text:
        try:
            from src.intelligence.conversational_nlp import generate_conversational_response
            resp = generate_conversational_response(user_text)
            if resp and resp.text:
                return resp.text
        except Exception:
            pass

    if intent.intent == "GREETING":
        response = build_docwain_intro()
    elif intent.intent == "THANKS_OR_PRAISE":
        response = (
            "Glad that helped. I can summarize, extract key data, or compare documents—what would you like next?"
        )
    elif intent.intent == "NEGATIVE_FEEDBACK":
        response = (
            "Sorry about that. I can retry with deeper retrieval or focus on a specific document/section. Which should I prioritize?"
        )
    elif intent.intent == "META":
        response = build_docwain_intro()
    elif intent.intent == "SMALL_TALK":
        response = (
            "Hi! If you have a document task, I'm ready to help."
        )
    elif intent.intent == "CLARIFICATION":
        response = (
            "Happy to clarify. Tell me which part you want me to explain again, or paste the snippet to review."
        )
    else:
        response = "How can I help with your documents?"

    if sentiment and sentiment.sentiment == "mixed" and intent.intent != "META":
        response = (
            "I hear you. Want me to retry with deeper retrieval or focus on a specific document/section?"
        )

    if response_mode == INFO_MODE and intent.intent != "META":
        return build_docwain_intro()

    return response

def route_message(user_text: str, state: Optional[dict] = None) -> RouteDecision:
    state = state or {}
    threshold = float(os.getenv("DOCWAIN_INTENT_THRESHOLD", "0.65"))
    persona_enabled = os.getenv("DOCWAIN_PERSONA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    sentiment_enabled = os.getenv("DOCWAIN_SENTIMENT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    intent = detect_intent(user_text)
    sentiment = analyze_sentiment(user_text) if sentiment_enabled else None
    response_mode = ResponseModeClassifier.classify(user_text)

    direct_intents = {"GREETING", "THANKS_OR_PRAISE", "NEGATIVE_FEEDBACK", "SMALL_TALK", "CLARIFICATION", "META"}
    direct_response = intent.intent in direct_intents and intent.confidence >= threshold
    if response_mode == INFO_MODE:
        direct_response = True
        intent = IntentResult(intent="META", confidence=1.0, method="policy", matched_rule="response_mode")

    response_text = None
    policy = "task"
    if direct_response:
        response_text = _build_direct_response(intent, sentiment, response_mode, user_text=user_text)
        if persona_enabled:
            persona_text = get_docwain_persona(
                profile_id=state.get("profile_id"),
                subscription_id=state.get("subscription_id"),
                redis_client=state.get("redis_client"),
            )
            response_text = enforce_docwain_identity(response_text, user_text, persona_text, response_mode=response_mode)
        response_text = sanitize_response(response_text)
        policy = "smalltalk"

    return RouteDecision(
        intent=intent,
        sentiment=sentiment,
        direct_response=direct_response,
        response_text=response_text,
        use_retrieval=not direct_response,
        policy=policy,
    )
