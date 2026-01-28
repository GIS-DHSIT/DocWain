from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


_ROBOT_OPENERS = [
    "as an ai",
    "as a language model",
    "i'm an ai",
    "i am an ai",
    "i cannot",
    "i can't",
    "i will now",
    "i am unable",
]


@dataclass
class StructureResult:
    has_table: bool
    has_steps: bool
    has_headings: bool
    long_paragraphs: int
    robot_opener: bool
    robot_phrase: str | None


def detect_robot_opener(answer: str) -> Tuple[bool, str | None]:
    start = (answer or "").strip().lower()
    for phrase in _ROBOT_OPENERS:
        if start.startswith(phrase):
            return True, phrase
    return False, None


def has_table(answer: str) -> bool:
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if "|" in line and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if "---" in next_line and "|" in next_line:
                return True
    return False


def has_steps(answer: str) -> bool:
    for line in (answer or "").splitlines():
        if re.match(r"^\s*\d+[\).]\s+", line):
            return True
    return False


def has_headings(answer: str) -> bool:
    for line in (answer or "").splitlines():
        if re.match(r"^\s*#{1,6}\s+\S+", line):
            return True
    return False


def count_long_paragraphs(answer: str, *, max_lines: int = 5) -> int:
    paragraphs = []
    current: List[str] = []
    for line in (answer or "").splitlines():
        if not line.strip():
            if current:
                paragraphs.append(current)
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(current)
    return sum(1 for para in paragraphs if len(para) > max_lines)


def evaluate_structure(answer: str) -> StructureResult:
    robot, phrase = detect_robot_opener(answer)
    return StructureResult(
        has_table=has_table(answer),
        has_steps=has_steps(answer),
        has_headings=has_headings(answer),
        long_paragraphs=count_long_paragraphs(answer),
        robot_opener=robot,
        robot_phrase=phrase,
    )
