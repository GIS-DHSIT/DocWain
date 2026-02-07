from __future__ import annotations

from ..types import LegalSchema, MISSING_REASON


def render_legal(schema: LegalSchema, intent: str, strict: bool = False) -> str:
    clauses_items = (schema.clauses.items if schema.clauses else None) or []
    if not clauses_items:
        if strict:
            return ""
        if schema.clauses and schema.clauses.missing_reason:
            return schema.clauses.missing_reason
        return MISSING_REASON
    clauses = []
    for clause in clauses_items:
        text = clause.text.strip()
        if clause.title:
            clauses.append(f"{clause.title}: {text}")
        elif text:
            clauses.append(text)
    if not clauses:
        if strict:
            return ""
        return MISSING_REASON
    cleaned = [" ".join(c.split()) for c in clauses]
    if len(cleaned) == 1:
        return cleaned[0]
    lines = ["Relevant clauses:"]
    for clause in cleaned:
        lines.append(f"- {clause}")
    return "\n".join(lines)
