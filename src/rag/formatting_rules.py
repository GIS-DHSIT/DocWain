from __future__ import annotations

import re
from typing import Optional


_BULLET_RE = re.compile(r"^[\s>*•–—·]+", re.MULTILINE)


def _sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"[.!?]+", text.replace("\n", " "))
    return len([p for p in parts if p.strip()])


def _ensure_leading_sentence(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return text
    if stripped.startswith(("-", "*")) or re.match(r"^\d+\.", stripped):
        intro = "Here is the answer based on the retrieved sections."
        return intro + "\n\n" + text
    return text


def _normalize_bullets(text: str) -> str:
    return _BULLET_RE.sub("- ", text)


def _append_takeaways(text: str) -> str:
    takeaways = [
        "If you want a deeper dive, tell me which document or section to prioritize.",
        "If you need a comparison or table, let me know the fields you care about.",
    ]
    return text.rstrip() + "\n\nTakeaways:\n" + "\n".join(f"- {line}" for line in takeaways)


def apply(text: str, *, wants_table: bool = False, table_markdown: Optional[str] = None) -> str:
    updated = text or ""
    updated = _normalize_bullets(updated)
    updated = _ensure_leading_sentence(updated)
    if wants_table and table_markdown and table_markdown not in updated:
        updated = updated.rstrip() + "\n\n" + table_markdown
    if _sentence_count(updated) < 5:
        updated = _append_takeaways(updated)
    return updated.strip()


__all__ = ["apply"]
