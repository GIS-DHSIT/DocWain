from __future__ import annotations

from ..types import GenericSchema, MISSING_REASON


def render_generic(schema: GenericSchema, intent: str, strict: bool = False) -> str:
    facts_items = (schema.facts.items if schema.facts else None) or []
    if not facts_items:
        if strict:
            return ""
        if schema.facts and schema.facts.missing_reason:
            return schema.facts.missing_reason
        return ""
    facts = [fact.value for fact in facts_items if fact.value]
    if not facts:
        if strict:
            return ""
        return ""
    cleaned = [" ".join(fact.split()) for fact in facts]
    if len(cleaned) == 1:
        return cleaned[0].strip()
    lines = ["Key details:"]
    for fact in cleaned:
        lines.append(f"- {fact}")
    return "\n".join(lines).strip()
