from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SentimentRuleMatch:
    sentiment: str
    feedback_type: str
    confidence: float
    rule_name: str


_THANKS = re.compile(
    r"\b(thanks|thank\s+you|thx|ty|cheers|appreciate\s+it|much\s+appreciated)\b",
    re.IGNORECASE,
)
_PRAISE = re.compile(
    r"\b(great|nice|awesome|good\s+job|well\s+done|love\s+it|perfect|excellent)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(bad|not\s+good|wrong|incorrect|useless|terrible|does\s*n['’]?t\s+work|awful|frustrated|annoyed)\b",
    re.IGNORECASE,
)
_FRUSTRATION = re.compile(r"[!?]{2,}")
_POSITIVE_EMOJI = re.compile(r"[😊😄😁👍✨❤️]", re.UNICODE)
_NEGATIVE_EMOJI = re.compile(r"[😞😠😡👎💀]", re.UNICODE)


def match_sentiment_rules(text: str) -> Optional[SentimentRuleMatch]:
    if not text:
        return None
    positive = bool(_THANKS.search(text) or _PRAISE.search(text) or _POSITIVE_EMOJI.search(text))
    negative = bool(_NEGATIVE.search(text) or _NEGATIVE_EMOJI.search(text))
    frustrated = bool(_FRUSTRATION.search(text))

    if positive and negative:
        return SentimentRuleMatch(
            sentiment="mixed",
            feedback_type="frustration" if frustrated else "complaint",
            confidence=0.75,
            rule_name="mixed",
        )
    if negative:
        return SentimentRuleMatch(
            sentiment="negative",
            feedback_type="frustration" if frustrated else "complaint",
            confidence=0.85,
            rule_name="negative",
        )
    if positive:
        return SentimentRuleMatch(
            sentiment="positive",
            feedback_type="thanks" if _THANKS.search(text) else "praise",
            confidence=0.85,
            rule_name="positive",
        )
    return None
