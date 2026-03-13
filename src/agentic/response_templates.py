from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ResponseTemplate:
    name: str
    format_style: str
    heading: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    guidance: str = ""

    def render_guidance(self) -> str:
        parts: List[str] = []
        if self.heading:
            parts.append(f"Use heading: {self.heading}.")
        if self.columns:
            parts.append("Table columns (strict): " + " | ".join(self.columns) + ".")
        if self.guidance:
            parts.append(self.guidance)
        return " ".join(parts).strip()

    def render_outline(self) -> str:
        heading = self.heading or "Answer"
        if self.format_style == "comparison_table" and self.columns:
            header = " | ".join(self.columns)
            separator = " | ".join(["---"] * len(self.columns))
            return (
                f"## {heading}\n"
                f"| {header} |\n"
                f"| {separator} |\n"
                f"| ... | ... | ... |\n\n"
                "## Key Takeaways\n"
                "- ...\n"
                "- ...\n"
            )
        if self.format_style == "extraction_table" and self.columns:
            header = " | ".join(self.columns)
            separator = " | ".join(["---"] * len(self.columns))
            return (
                f"## {heading}\n"
                f"| {header} |\n"
                f"| {separator} |\n"
                f"| ... | ... | ... |\n\n"
                "## Notes\n"
                "- ...\n"
            )
        if self.format_style == "numbered_steps":
            return (
                f"## {heading}\n"
                "1. ...\n"
                "2. ...\n"
                "3. ...\n\n"
                "## Takeaway\n"
                "..."
            )
        if self.format_style == "troubleshooting":
            return (
                "## Symptoms\n"
                "- ...\n\n"
                "## Likely Causes\n"
                "- ...\n"
                "- ...\n\n"
                "## Fix Steps\n"
                "1. ...\n"
                "2. ...\n"
                "3. ...\n"
            )
        if self.format_style == "factual":
            return (
                "## Answer\n"
                "... \n\n"
                "## Evidence\n"
                "- ...\n"
                "- ...\n\n"
                "## Where This Came From\n"
                "- ...\n"
            )
        if self.format_style == "summary_sections":
            return (
                "## Summary\n"
                "... \n\n"
                "## Key Points\n"
                "- ...\n"
                "- ...\n\n"
                "## Takeaway\n"
                "..."
            )
        if self.format_style == "sections":
            return (
                "## Summary\n"
                "... \n\n"
                "## Evidence\n"
                "- ...\n"
                "- ...\n\n"
                "## Analysis\n"
                "- ...\n"
                "- ...\n\n"
                "## Takeaway\n"
                "..."
            )
        if self.format_style == "bullets":
            return (
                f"## {heading}\n"
                "- ...\n"
                "- ...\n\n"
                "## Takeaway\n"
                "..."
            )
        return (
            f"## {heading}\n"
            "... \n\n"
            "## Key Points\n"
            "- ...\n"
            "- ...\n\n"
            "## Takeaway\n"
            "..."
        )


class ResponseTemplateSelector:
    """Selects response templates based on intent and explicit instructions."""

    @staticmethod
    def select(intent: str, instructions: Dict[str, str | bool | None]) -> ResponseTemplate:
        logger.debug("select called with intent=%s", intent)
        normalized_intent = ResponseTemplateSelector._normalize_intent(intent)
        style = (instructions.get("style") or "").lower()
        format_hint = (instructions.get("format") or "").lower()
        wants_table = bool(instructions.get("use_table")) or "table" in format_hint
        wants_steps = bool(instructions.get("step_by_step")) or "step" in format_hint
        wants_bullets = bool(instructions.get("use_bullets")) or "bullet" in format_hint or "list" in format_hint
        wants_sections = "section" in format_hint
        wants_analysis = bool(instructions.get("analysis")) or normalized_intent == "analysis"
        wants_detailed = str(instructions.get("brevity") or "").lower() == "detailed"

        if normalized_intent == "comparison":
            return ResponseTemplate(
                name="comparison_table",
                format_style="comparison_table",
                heading="Comparison",
                columns=["Document", "Key Evidence", "Differences / Implications"],
                guidance="Follow the table with a short Key Takeaways section (2-4 bullets) grounded in citations.",
            )

        if normalized_intent == "how-to" or wants_steps:
            return ResponseTemplate(
                name="step_by_step",
                format_style="numbered_steps",
                heading="Steps",
                guidance="Use numbered steps, include prerequisites when applicable, and cite evidence.",
            )

        if normalized_intent == "troubleshooting":
            return ResponseTemplate(
                name="troubleshooting",
                format_style="troubleshooting",
                heading="Troubleshooting",
                guidance="Use sections: Symptoms, Likely Causes, Fix Steps. Keep fixes grounded in citations.",
            )

        if normalized_intent == "extraction" or wants_table:
            return ResponseTemplate(
                name="extraction_table",
                format_style="extraction_table",
                heading="Extracted Items",
                columns=["Item", "Value", "Notes"],
                guidance="Provide the table first, then a short Notes section explaining any caveats with citations.",
            )

        if normalized_intent == "summary":
            return ResponseTemplate(
                name="summary_sections",
                format_style="summary_sections",
                heading="Summary",
                guidance="Use headings and bullets. Keep the summary concise and grounded in citations.",
            )

        if wants_analysis or wants_detailed or normalized_intent == "analysis":
            return ResponseTemplate(
                name="deep_analysis",
                format_style="sections",
                heading="Summary",
                guidance="Use headings: Summary, Evidence, Analysis, and Takeaway. Include bullets under Evidence and Analysis.",
            )

        if wants_bullets:
            return ResponseTemplate(
                name="bullet_summary",
                format_style="bullets",
                heading="Key Points",
                guidance="Start with a one-sentence summary, then 3-6 bullet points.",
            )

        if wants_sections:
            return ResponseTemplate(
                name="sections",
                format_style="sections",
                heading="Summary",
                guidance="Use headings and bullets to keep the structure clear.",
            )

        if style == "narrative":
            return ResponseTemplate(
                name="narrative",
                format_style="narrative",
                heading="Answer",
                guidance="Write 4-6 concise sentences with citations, no bullets.",
            )

        logger.debug("select returning template=factual (default)")
        return ResponseTemplate(
            name="factual",
            format_style="factual",
            heading="Answer",
            guidance="Start with a short Answer section, then Evidence bullets, then Where This Came From (if helpful).",
        )

    @staticmethod
    def _normalize_intent(intent: str) -> str:
        normalized = (intent or "").strip().lower()
        if normalized in {"procedural", "instruction/how-to", "how-to", "howto"}:
            return "how-to"
        if normalized in {"summary", "summarization"}:
            return "summary"
        if normalized in {"reasoning", "analysis", "deep_analysis"}:
            return "analysis"
        if normalized in {"extraction", "field_extraction", "numeric_lookup"}:
            return "extraction"
        if normalized == "comparison":
            return "comparison"
        if normalized == "troubleshooting":
            return "troubleshooting"
        return "factual"
