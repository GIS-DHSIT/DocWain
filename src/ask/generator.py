from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .models import EvidenceQuality, Plan


class ResponseGenerator:
    def generate(
        self,
        *,
        plan: Plan,
        bundle: Dict[str, Any],
        quality: EvidenceQuality,
    ) -> Dict[str, Any]:
        lines: List[str] = []
        ack = _acknowledgement(plan)
        lines.append(ack)

        body_lines, gaps = _render_body(plan, bundle)
        lines.extend(body_lines)

        gaps = _augment_gaps(plan, gaps)
        if gaps:
            lines.append("What's missing: " + "; ".join(gaps))

        if quality.quality == "LOW":
            lines.append("Note: Complete information is not available in the current profile documents.")

        response = "\n".join([line for line in lines if line])
        response = _normalize_output_text(response)
        return {"response": response, "gaps": gaps}


def _acknowledgement(plan: Plan) -> str:
    intent = plan.intent
    short_query = _short_query(plan.query)
    if intent == "compare":
        return f"Here is a clear comparison for {short_query} based on this profile's documents."
    if intent == "summarize":
        return f"Here is a concise summary for {short_query} based on this profile's documents."
    if intent == "rank":
        return f"Here is a ranked view for {short_query} grounded in the available evidence."
    if intent == "draft":
        return f"Here is a draft for {short_query} grounded in the profile documents."
    if intent == "extract":
        return f"Here is what I found for {short_query} in the current profile documents."
    if intent == "greet":
        return "Hello! I'm DocWain, your document intelligence assistant."
    return f"Here is what I found for {short_query} in the current profile documents."


def _render_body(plan: Plan, bundle: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    documents = bundle.get("documents") or []
    gaps: List[str] = []
    lines: List[str] = []
    for doc in documents:
        file_name = doc.get("file_name") or "document"
        objects = doc.get("objects") or []
        for obj in objects:
            fields = obj.get("fields") or {}
            evidence_map = obj.get("evidence_map") or {}
            doc_gaps = obj.get("gaps") or []
            gaps.extend(doc_gaps)
            summary = _field_summary(file_name, fields)
            citation = _citation_from_evidence(file_name, evidence_map)
            if summary:
                lines.append(f"- {summary} {citation}")
            else:
                lines.append(f"- {file_name} {citation}")
    return lines, _dedupe(gaps)

def _field_summary(file_name: str, fields: Dict[str, Any]) -> str:
    parts: List[str] = []
    names = fields.get("names") or []
    skills = fields.get("skills") or []
    orgs = fields.get("organizations") or []
    dates = fields.get("dates") or []
    subject = names[0] if names else file_name
    if names:
        parts.append(f"{subject}")
    if skills:
        parts.append("skills include " + ", ".join(skills))
    if orgs:
        parts.append("organizations mentioned: " + ", ".join(orgs))
    if dates:
        parts.append("dates referenced: " + ", ".join(dates))
    if not parts:
        return ""
    if parts and parts[0] == subject:
        return f"{subject} - {', '.join(parts[1:])}" if len(parts) > 1 else subject
    return ", ".join(parts)


def _citation_from_evidence(file_name: str, evidence_map: Dict[str, Any]) -> str:
    for ev_list in evidence_map.values():
        if not ev_list:
            continue
        ev = ev_list[0]
        page = ev.get("page")
        page_str = f"p.{page}" if page is not None else "p.?"
        return f"({file_name}, {page_str})"
    return f"({file_name}, p.?)"


def _dedupe(items: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _augment_gaps(plan: Plan, gaps: List[str]) -> List[str]:
    augmented = list(gaps or [])
    hints = [h for h in (plan.entity_hints or []) if h]
    if any(hint.startswith("certification") for hint in hints):
        if not any("certification" in gap for gap in augmented):
            augmented.append("certifications not explicitly listed")
    return _dedupe(augmented)


def _short_query(query: str) -> str:
    _ = query
    return "your request"


def _normalize_output_text(text: str) -> str:
    if not text:
        return ""
    # Insert spaces in long artifact-like tokens (e.g., ProficientinPython).
    def _fix_token(token: str) -> str:
        if len(token) < 18 or " " in token:
            return token
        fixed = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
        fixed = re.sub(r"([A-Za-z])(\d)", r"\1 \2", fixed)
        fixed = re.sub(r"(\d)([A-Za-z])", r"\1 \2", fixed)
        return fixed

    words = text.split(" ")
    return " ".join(_fix_token(word) for word in words)


__all__ = ["ResponseGenerator"]
