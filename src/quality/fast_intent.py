from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class IntentResult:
    expected: str
    delivered: str
    mismatch: bool
    explicit: bool


_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference|differences)\b", re.IGNORECASE)
_PROCEDURAL_RE = re.compile(r"\b(steps|step-by-step|how to|how do i|procedure|process|install|configure|setup|set up)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"\b(summary|summarize|overview|brief|high[- ]level)\b", re.IGNORECASE)


def detect_intent(query: str, query_intent: Optional[str] = None) -> tuple[str, bool]:
    if query_intent:
        intent = str(query_intent).lower()
        if intent in {"comparison", "procedural", "summary", "qa", "analysis", "reasoning", "extraction"}:
            return intent, intent in {"comparison", "procedural", "summary"}
    if _COMPARE_RE.search(query):
        return "comparison", True
    if _PROCEDURAL_RE.search(query):
        return "procedural", True
    if _SUMMARY_RE.search(query):
        return "summary", True
    return "qa", False


def _has_table(answer: str) -> bool:
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if "|" in line and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if "---" in next_line and "|" in next_line:
                return True
    return False


def _has_steps(answer: str) -> bool:
    for line in (answer or "").splitlines():
        if re.match(r"^\s*\d+[\).]\s+", line):
            return True
    return False


def _has_headings(answer: str) -> bool:
    for line in (answer or "").splitlines():
        if re.match(r"^\s*#{1,6}\s+\S+", line):
            return True
    return False


def infer_delivery(answer: str) -> str:
    if _has_table(answer):
        return "comparison"
    if _has_steps(answer):
        return "procedural"
    if _has_headings(answer):
        return "summary"
    return "qa"


def evaluate_intent_alignment(query: str, answer: str, query_intent: Optional[str] = None) -> IntentResult:
    expected, explicit = detect_intent(query, query_intent=query_intent)
    delivered = infer_delivery(answer)
    mismatch = explicit and expected != delivered
    return IntentResult(expected=expected, delivered=delivered, mismatch=mismatch, explicit=explicit)
