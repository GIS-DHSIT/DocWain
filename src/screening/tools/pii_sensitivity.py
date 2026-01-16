from __future__ import annotations

import re
from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class PIISensitivityTool(ScreeningTool):
    name = "pii_sensitivity"
    category = "Security & Confidentiality"
    default_weight = 0.12
    tool_version = "1.0"

    def _spans(self, pattern: re.Pattern[str], text: str, label: str) -> List[dict]:
        spans = []
        for match in pattern.finditer(text):
            spans.append({"label": label, "text": match.group(0), "start": match.start(), "end": match.end()})
        return spans

    def run(self, ctx: ScreeningContext):
        email_spans = self._spans(EMAIL_RE, ctx.text, "email")
        phone_spans = self._spans(PHONE_RE, ctx.text, "phone")
        ssn_spans = self._spans(SSN_RE, ctx.text, "id")

        all_spans = email_spans + phone_spans + ssn_spans
        count = len(all_spans)

        reasons = []
        actions = ["tag"]

        if count > 0:
            reasons.append(f"Detected {count} potential PII item(s).")
            actions.append("redact")
        else:
            reasons.append("No obvious PII detected.")

        score = clamp(count / 5)
        raw_features = {"pii_count": count, "emails": len(email_spans), "phones": len(phone_spans), "ids": len(ssn_spans)}

        return self.result(
            ctx, score, reasons, raw_features=raw_features, actions=actions, evidence_spans=all_spans[:50]
        )
