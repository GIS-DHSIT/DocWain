from __future__ import annotations

from ..types import MISSING_REASON, MultiEntitySchema


def render_multi(schema: MultiEntitySchema, intent: str, strict: bool = False) -> str:
    entities = schema.entities or []
    if not entities:
        if strict:
            return ""
        return schema.missing_reason or MISSING_REASON

    has_docs = any(item.document_name or item.document_id for item in entities)
    has_named = any(item.label and item.label.lower() not in {"document", "candidate"} for item in entities)

    if has_named and not has_docs:
        lead = "There are multiple candidates in the documents."
    elif has_named and has_docs:
        lead = "There are multiple candidates across documents."
    else:
        lead = "There are multiple documents that match this request."

    parts = []
    for item in entities:
        label = item.label or item.document_name or item.document_id or "Document"
        doc_ref = item.document_name or item.document_id
        if doc_ref and doc_ref not in label:
            parts.append(f"{label} ({doc_ref})")
        else:
            parts.append(label)

    if len(parts) == 1:
        return f"{lead} {parts[0]}."
    lines = [lead, "Matched entities:"]
    lines.extend(f"- {item}" for item in parts)
    return "\n".join(lines)
