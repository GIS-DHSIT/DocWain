from __future__ import annotations

import re
from typing import Dict, List, Set

from ..models import ScreeningContext
from .base import ScreeningTool, clamp


class TemplateConformanceTool(ScreeningTool):
    name = "template_conformance"
    category = "Compliance & Template Conformance"
    default_weight = 0.10
    tool_version = "1.0"

    def _extract_sections(self, text: str) -> List[str]:
        sections: List[str] = []
        seen: Set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) < 3:
                continue
            normalized = re.sub(r"[^a-z0-9 ]", "", line.lower()).strip()
            if not normalized:
                continue
            if line.isupper() or line.endswith(":") or re.match(r"^[0-9]+[\.\)]", line):
                if normalized not in seen:
                    sections.append(normalized)
                    seen.add(normalized)
            elif len(normalized.split()) <= 5:
                # treat short, title-cased lines as potential headings
                if line[:1].isupper():
                    if normalized not in seen:
                        sections.append(normalized)
                        seen.add(normalized)
        return sections

    def _check_order(self, sections: List[str], required_order: List[str]) -> bool:
        """Return True if order is respected."""
        index_map: Dict[str, int] = {}
        for idx, sec in enumerate(sections):
            index_map[sec] = min(index_map.get(sec, idx), idx) if sec in index_map else idx
        last_index = -1
        for required in required_order:
            required_norm = re.sub(r"[^a-z0-9 ]", "", required.lower())
            candidate_indices = [index_map[s] for s in index_map if required_norm in s]
            if not candidate_indices:
                continue
            candidate = min(candidate_indices)
            if candidate < last_index:
                return False
            last_index = candidate
        return True

    def run(self, ctx: ScreeningContext):
        doc_type = (ctx.doc_type or ctx.metadata.get("doc_type") or "").upper()
        template = ctx.config.doc_type_templates.get(doc_type) if ctx.config else None
        sections_found = self._extract_sections(ctx.text)

        if not template:
            reasons = ["No template configured for this document type; skipping template enforcement."]
            return self.result(
                ctx,
                0.0,
                reasons,
                raw_features={"sections_found": list(sections_found), "template_used": None},
                actions=["tag"],
            )

        mandatory = [s.lower() for s in template.get("mandatory_sections", [])]
        heading_order = [s.lower() for s in template.get("heading_order", [])]
        missing = [sec for sec in mandatory if not any(sec in found for found in sections_found)]

        reasons = []
        score = 0.0

        if missing:
            reasons.append(f"Missing mandatory sections: {', '.join(missing)}.")
            score += min(1.0, len(missing) / max(len(mandatory), 1))

        order_ok = self._check_order(list(sections_found), heading_order) if heading_order else True
        if not order_ok:
            reasons.append("Headings appear out of the expected order.")
            score += 0.25

        if not reasons:
            reasons.append("All mandatory sections found and ordered.")

        raw_features = {
            "sections_found": list(sections_found),
            "missing_sections": missing,
            "template_used": doc_type,
            "order_ok": order_ok,
        }

        actions = ["tag"]
        if missing:
            actions.append("warn")

        return self.result(ctx, clamp(score), reasons, raw_features=raw_features, actions=actions)
