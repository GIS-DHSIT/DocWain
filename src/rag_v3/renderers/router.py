from __future__ import annotations

from typing import Any

from ..enterprise import render_enterprise


def render_answer(*, domain: str, intent: str, schema: Any, strict: bool = False) -> str:
    return render_enterprise(schema, intent, domain=domain, strict=strict)


def render(*, domain: str, intent: str, schema: Any, strict: bool = False) -> str:
    return render_answer(domain=domain, intent=intent, schema=schema, strict=strict)
