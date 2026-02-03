from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Sequence


def _sanitize_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "; ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _coerce_row(columns: Sequence[str], record: Any) -> List[str]:
    if isinstance(record, Mapping):
        return [_sanitize_cell(record.get(col, "")) for col in columns]
    if isinstance(record, (list, tuple)):
        values = list(record)
        if len(values) < len(columns):
            values = values + [""] * (len(columns) - len(values))
        return [_sanitize_cell(v) for v in values[: len(columns)]]
    return [_sanitize_cell(record)] + [""] * (len(columns) - 1)


def render_markdown_table(columns: Sequence[str], records: Iterable[Any]) -> str:
    cols = [_sanitize_cell(col) for col in columns]
    if not cols:
        return ""
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for record in records:
        row = _coerce_row(cols, record)
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join([header, separator] + rows)


__all__ = ["render_markdown_table"]
