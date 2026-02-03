from __future__ import annotations

import re
from typing import Optional


_DISALLOWED_BULLETS = [
    "\u2022",  # bullet
    "\u2023",  # triangular bullet
    "\u2043",  # hyphen bullet
    "\u2219",  # bullet operator
    "\u00b7",  # middle dot
    "\u25cf",  # black circle
    "\u25cb",  # white circle
    "\u25e6",  # white bullet
    "\u25aa",  # small square
    "\u25ab",  # white small square
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2212",  # minus sign
    "\u2192",  # right arrow
    "\u00bb",  # right-pointing double angle
    "\u203a",  # right-pointing single angle
]

_BULLET_RE = re.compile(r"^(\s*)(?:" + "|".join(re.escape(b) for b in _DISALLOWED_BULLETS) + r")\s+")
_NUMBER_PAREN_RE = re.compile(r"^(\s*)(\d+)\)\s+")
_LETTER_PAREN_RE = re.compile(r"^(\s*)[A-Za-z]\)\s+")
_LETTER_DOT_RE = re.compile(r"^(\s*)[A-Za-z]\.\s+")
_NOT_STATED_RE = re.compile(r"\bnot stated in the retrieved sections\b", re.IGNORECASE)


def _limit_intro_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sentences])


def enforce_response_formatting(
    *,
    text: str,
    wants_table: bool = False,
    table_markdown: Optional[str] = None,
    intro: Optional[str] = None,
) -> str:
    if wants_table and table_markdown:
        intro_text = _limit_intro_sentences(intro or "Here is the table based on the retrieved sections.")
        base = "\n\n".join([intro_text, table_markdown]).strip()
        return base

    if not text:
        return text

    text = _NOT_STATED_RE.sub("the retrieved sections do not include that detail", text)

    in_code_block = False
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            lines.append(line)
            continue
        if in_code_block:
            lines.append(line)
            continue
        updated = _NUMBER_PAREN_RE.sub(r"\1\2. ", line)
        updated = _LETTER_PAREN_RE.sub(r"\1- ", updated)
        updated = _LETTER_DOT_RE.sub(r"\1- ", updated)
        updated = _BULLET_RE.sub(r"\1- ", updated)
        lines.append(updated)

    return "\n".join(lines)


__all__ = ["enforce_response_formatting"]
