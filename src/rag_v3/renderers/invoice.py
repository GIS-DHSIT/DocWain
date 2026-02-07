from __future__ import annotations

from typing import List

from ..types import InvoiceSchema, MISSING_REASON


NO_ITEMS_MESSAGE = "Invoice pages retrieved don’t show itemized products/services—only totals/parties."


def render_invoice(schema: InvoiceSchema, intent: str, strict: bool = False) -> str:
    if intent == "products_list":
        items = (schema.items.items if schema.items else None) or []
        if items:
            rendered = [_format_item(item.description) for item in items if item.description]
            if rendered:
                lines = ["Items listed on the invoice:"]
                lines.extend(f"- {item}" for item in rendered)
                return "\n".join(lines)
        if schema.items and schema.items.missing_reason:
            return schema.items.missing_reason
        return NO_ITEMS_MESSAGE

    if intent == "totals":
        totals_items = (schema.totals.items if schema.totals else None) or []
        if totals_items:
            totals = [_format_field(item.label, item.value) for item in totals_items if item.value]
            if totals:
                lines = ["Totals shown:"]
                lines.extend(f"- {item}" for item in totals)
                return "\n".join(lines)
        if schema.totals and schema.totals.missing_reason:
            return schema.totals.missing_reason
        return MISSING_REASON

    if strict:
        return _render_totals_parties(schema, prefer_totals=True) or ""

    summary = _render_totals_parties(schema, prefer_totals=True)
    if summary:
        return summary
    items = (schema.items.items if schema.items else None) or []
    if items:
        rendered = [_format_item(item.description) for item in items if item.description]
        if rendered:
            lines = ["Items listed on the invoice:"]
            lines.extend(f"- {item}" for item in rendered)
            return "\n".join(lines)
    if schema.items and schema.items.missing_reason:
        return schema.items.missing_reason
    return MISSING_REASON


def _render_totals_parties(schema: InvoiceSchema, prefer_totals: bool = True) -> str:
    parts: List[str] = []
    totals_items = (schema.totals.items if schema.totals else None) or []
    parties_items = (schema.parties.items if schema.parties else None) or []
    terms_items = (schema.terms.items if schema.terms else None) or []
    if prefer_totals and totals_items:
        totals = [_format_field(item.label, item.value) for item in totals_items if item.value]
        if totals:
            parts.append("Totals shown:")
            parts.extend(f"- {item}" for item in totals)
    if parties_items:
        parties = [_format_field(item.label, item.value) for item in parties_items if item.value]
        if parties:
            parts.append("Parties listed:")
            parts.extend(f"- {item}" for item in parties)
    if terms_items:
        terms = [_format_field(item.label, item.value) for item in terms_items if item.value]
        if terms:
            parts.append("Terms noted:")
            parts.extend(f"- {item}" for item in terms)
    return "\n".join(parts).strip()


def _format_item(text: str) -> str:
    return " ".join(text.split())


def _format_field(label: str | None, value: str) -> str:
    if label:
        return f"{label}: {value}"
    return value
