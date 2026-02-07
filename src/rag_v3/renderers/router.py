from __future__ import annotations

from typing import Any

from .generic import render_generic
from .hr import render_hr
from .invoice import render_invoice
from .legal import render_legal
from .multi import render_multi
from ..types import GenericSchema, HRSchema, InvoiceSchema, LegalSchema, MultiEntitySchema


def render_answer(*, domain: str, intent: str, schema: Any, strict: bool = False) -> str:
    if isinstance(schema, MultiEntitySchema):
        return render_multi(schema, intent, strict=strict)
    if domain == "invoice" and isinstance(schema, InvoiceSchema):
        return render_invoice(schema, intent, strict=strict)
    if domain == "hr" and isinstance(schema, HRSchema):
        return render_hr(schema, intent, strict=strict)
    if domain == "legal" and isinstance(schema, LegalSchema):
        return render_legal(schema, intent, strict=strict)
    if isinstance(schema, GenericSchema):
        return render_generic(schema, intent, strict=strict)
    return ""


def render(*, domain: str, intent: str, schema: Any, strict: bool = False) -> str:
    return render_answer(domain=domain, intent=intent, schema=schema, strict=strict)
