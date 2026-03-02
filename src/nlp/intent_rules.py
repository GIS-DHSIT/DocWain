from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IntentRuleMatch:
    intent: str
    confidence: float
    rule_name: str


_GREETINGS = re.compile(
    r"\b(hi|hello|hey|hiya|yo|greetings|good\s+morning|good\s+afternoon|good\s+evening|howdy)\b",
    re.IGNORECASE,
)
_THANKS = re.compile(
    r"\b(thanks|thank\s+you|thx|ty|cheers|appreciate\s+it|much\s+appreciated)\b",
    re.IGNORECASE,
)
_PRAISE = re.compile(
    r"\b(good\s+work|great\s+job|nice|awesome|well\s+done|excellent|brilliant|fantastic)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(bad|not\s+good|wrong|incorrect|useless|terrible|does\s*n['’]?t\s+work|broken|poor|awful)\b",
    re.IGNORECASE,
)
_CLARIFICATION = re.compile(
    r"\b(what\s+did\s+you\s+mean|repeat\s+that|say\s+that\s+again|explain\s+again|clarify|can\s+you\s+repeat)\b",
    re.IGNORECASE,
)
_SMALL_TALK = re.compile(
    r"\b(how\s+are\s+you|what's\s+up|whats\s+up|how's\s+it\s+going|how\s+do\s+you\s+do)\b",
    re.IGNORECASE,
)
_META = re.compile(
    r"\bwho\s+are\s+you\b|"
    r"\bwhat\s+are\s+you\b|"
    r"\bwhat\s+is\s+docwain\b|"
    r"\bwhat\s+can\s+you\s+do\b|"
    r"\bwhat\s+else\s+can\s+you\s+do\b|"
    r"\bwhat\s+all\s+can\s+(?:you|docwain)\s+do\b|"
    r"\bwhat\s+else\s+can\s+you\s+help\s+with\b|"
    r"\bhow\s+can\s+(?:you|docwain)\s+help(?:\s+me)?\b|"
    r"\bshow\s+(?:me\s+)?what\s+(?:you|docwain)\s+can\s+do\b|"
    r"\bhow\s+do\s+you\s+work\b|"
    r"\bthis\s+is\s+not\s+the\s+right\s+answer\b|"
    r"\bnot\s+good\b|"
    r"\bthis\s+is\s+bad\b|"
    r"\bthank\s+you\b|"
    r"\b(thanks|thx|ty)\b|"
    r"\bwonderful\b",
    re.IGNORECASE,
)


def match_intent_rules(text: str) -> Optional[IntentRuleMatch]:
    if not text:
        return None
    if _GREETINGS.search(text):
        return IntentRuleMatch(intent="GREETING", confidence=0.95, rule_name="greeting")
    if _THANKS.search(text) or _PRAISE.search(text):
        return IntentRuleMatch(intent="THANKS_OR_PRAISE", confidence=0.9, rule_name="thanks_or_praise")
    if _NEGATIVE.search(text):
        return IntentRuleMatch(intent="NEGATIVE_FEEDBACK", confidence=0.9, rule_name="negative_feedback")
    if _CLARIFICATION.search(text):
        return IntentRuleMatch(intent="CLARIFICATION", confidence=0.85, rule_name="clarification")
    if _SMALL_TALK.search(text):
        return IntentRuleMatch(intent="SMALL_TALK", confidence=0.75, rule_name="small_talk")
    if _META.search(text):
        return IntentRuleMatch(intent="META", confidence=0.95, rule_name="meta_question")
    return None
