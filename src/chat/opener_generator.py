from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Optional

BANNED_OPENERS = [
    "i reviewed the documents",
    "i reviewed the document",
    "i reviewed the documents and pulled",
    "here are the most relevant passages",
    "here's the most relevant passages",
    "here are the relevant passages",
    "here's the most relevant info",
    "here is the most relevant info",
    "based on the passages",
    "based on the passages above",
    "based on the passages provided",
    "based on the retrieved context",
    "based on the documents",
    "from the documents",
]


def contains_banned_opener(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(lowered.startswith(phrase) for phrase in BANNED_OPENERS)


def _seed_from_query(query: str) -> int:
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pick_variant(variants: List[str], seed: int) -> str:
    if not variants:
        return ""
    rng = random.Random(seed)
    return rng.choice(variants)


def generate_opener(
    *,
    intent: str,
    sentiment: str,
    follow_up: bool,
    style_directives: Optional[Dict[str, object]] = None,
    query: str = "",
) -> str:
    style_directives = style_directives or {}
    humor_level = int(style_directives.get("humor_level") or 0)
    tone = str(style_directives.get("tone") or "friendly")

    seed = _seed_from_query(query or intent or sentiment)
    intent_key = (intent or "factual").strip().lower()
    sentiment_key = (sentiment or "neutral").strip().lower()

    if sentiment_key == "frustrated":
        variants = [
            "Sorry about that - I'll fix this properly.",
            "Thanks for the pushback - I'll correct it now.",
            "I hear you - I'll tighten this up with better evidence.",
        ]
    elif sentiment_key == "thankful":
        variants = [
            "Happy to help.",
            "Glad that helped.",
            "Anytime - happy to help.",
        ]
    elif sentiment_key == "negative":
        variants = [
            "I hear you - let's get this right.",
            "Got it - I'll be extra careful with the evidence.",
        ]
    else:
        variants = _intent_variants(intent_key, humor_level=humor_level, tone=tone)

    opener = _pick_variant(variants, seed)
    if follow_up:
        # Avoid explicit continuation phrases in openers.
        pass

    if not opener or contains_banned_opener(opener):
        opener = "Got it - here's what I found."

    return opener


def _intent_variants(intent_key: str, *, humor_level: int, tone: str) -> List[str]:
    base: Dict[str, List[str]] = {
        "factual": [
            "Here's what I found.",
            "Here's the answer I can support.",
            "Here's what I can confirm.",
        ],
        "summary": [
            "Here's a concise summary.",
            "Here's a clean summary.",
        ],
        "comparison": [
            "Here's a side-by-side comparison.",
            "Here's a direct comparison.",
        ],
        "how-to": [
            "Here's a step-by-step answer.",
            "Here's how to approach it.",
        ],
        "troubleshooting": [
            "Here's a structured troubleshooting pass.",
            "Let's walk through fixes step by step.",
        ],
        "extraction": [
            "Here are the key details.",
            "Here are the extracted items.",
        ],
        "analysis": [
            "Here's a focused analysis.",
            "Here's the analysis I can support.",
        ],
    }

    variants = list(base.get(intent_key, base["factual"]))

    if humor_level > 0 and tone in {"friendly", "supportive"}:
        variants.extend(
            [
                "Alright - I'll keep this crisp and helpful.",
                "Let's make this painless - here's the answer.",
            ]
        )

    return variants
