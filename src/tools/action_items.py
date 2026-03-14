"""
Action item and deadline extraction tool for DocWain.

Extracts tasks, deadlines, and obligations from documents using deterministic
patterns (modal verbs, deadline phrases, assignment patterns) plus optional
LLM enhancement for contextual priority and implicit deadlines.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.base import register_tool, standard_response

logger = get_logger(__name__)

# ── Modal verb patterns ──────────────────────────────────────────────────

_MODAL_RE = re.compile(
    r"\b(?:shall|must|will|should|needs?\s+to|required?\s+to|obligated?\s+to|"
    r"is\s+responsible\s+for|agrees?\s+to|undertakes?\s+to)\b",
    re.IGNORECASE,
)

# ── Deadline patterns ────────────────────────────────────────────────────

_DEADLINE_RE = re.compile(
    r"\b(?:by|before|no\s+later\s+than|within|due\s+(?:date|by|on)?|"
    r"deadline|expires?\s+on|effective\s+(?:from|until)|on\s+or\s+before)\b"
    r"\s*[:\-]?\s*"
    r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{2,4}|"
    r"\d{1,4}\s+(?:days?|weeks?|months?|years?|business\s+days?|calendar\s+days?)|"
    r"immediately|forthwith|promptly|upon\s+\w+)",
    re.IGNORECASE,
)

_DATE_STANDALONE_RE = re.compile(
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s*\d{2,4})\b",
    re.IGNORECASE,
)

# ── Assignment patterns ──────────────────────────────────────────────────

_ASSIGNEE_RE = re.compile(
    r"(?:(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:shall|must|will|should|agrees?\s+to)|"
    r"assigned\s+to\s+(\w+(?:\s+\w+)?)|"
    r"responsibility\s+of\s+(\w+(?:\s+\w+)?))",
    re.IGNORECASE,
)

# ── Priority / urgency words ─────────────────────────────────────────────

_HIGH_PRIORITY_RE = re.compile(
    r"\b(?:immediately|urgent|critical|mandatory|forthwith|without\s+delay|"
    r"as\s+soon\s+as\s+practicable|time\s+is\s+of\s+the\s+essence|"
    r"imperative|essential|crucial)\b",
    re.IGNORECASE,
)

_LOW_PRIORITY_RE = re.compile(
    r"\b(?:may|optional|recommended|suggested|consider|where\s+practicable|"
    r"if\s+possible|at\s+(?:its|their)\s+discretion)\b",
    re.IGNORECASE,
)

# ── Domain-specific category hints ───────────────────────────────────────

_DOMAIN_CATEGORIES: Dict[str, List[tuple[re.Pattern, str]]] = {
    "legal": [
        (re.compile(r"\b(?:indemnif|liability|damages|breach|termination)\b", re.IGNORECASE), "obligation"),
        (re.compile(r"\b(?:deliver|provide|supply|furnish|submit)\b", re.IGNORECASE), "deliverable"),
        (re.compile(r"\b(?:pay|payment|fee|cost|reimburse|invoice)\b", re.IGNORECASE), "payment"),
        (re.compile(r"\b(?:notice|notify|inform|disclose|report)\b", re.IGNORECASE), "notification"),
        (re.compile(r"\b(?:confidential|proprietary|trade\s+secret|nda)\b", re.IGNORECASE), "confidentiality"),
    ],
    "medical": [
        (re.compile(r"\b(?:follow[\s-]?up|appointment|visit|check[\s-]?up|return)\b", re.IGNORECASE), "follow_up"),
        (re.compile(r"\b(?:medication|prescri|dose|tablet|capsule|inject)\b", re.IGNORECASE), "medication"),
        (re.compile(r"\b(?:test|lab|blood|imaging|x[\s-]?ray|mri|ct\s+scan)\b", re.IGNORECASE), "diagnostic"),
        (re.compile(r"\b(?:refer|specialist|consult)\b", re.IGNORECASE), "referral"),
    ],
    "hr": [
        (re.compile(r"\b(?:onboard|orientation|training|induction)\b", re.IGNORECASE), "onboarding"),
        (re.compile(r"\b(?:review|evaluation|assessment|appraisal)\b", re.IGNORECASE), "review"),
        (re.compile(r"\b(?:document|form|paperwork|sign|submit)\b", re.IGNORECASE), "documentation"),
    ],
    "invoice": [
        (re.compile(r"\b(?:pay|payment|remit|transfer|settle)\b", re.IGNORECASE), "payment"),
        (re.compile(r"\b(?:deliver|ship|dispatch|fulfil)\b", re.IGNORECASE), "delivery"),
        (re.compile(r"\b(?:return|refund|credit|dispute)\b", re.IGNORECASE), "dispute"),
    ],
}

@dataclass
class ActionItem:
    """An extracted action item from a document."""

    description: str
    deadline: Optional[str] = None
    priority: str = "medium"  # high, medium, low
    assignee: Optional[str] = None
    source_document: Optional[str] = None
    category: str = "general"
    source_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "description": self.description,
            "priority": self.priority,
            "category": self.category,
        }
        if self.deadline:
            result["deadline"] = self.deadline
        if self.assignee:
            result["assignee"] = self.assignee
        if self.source_document:
            result["source_document"] = self.source_document
        return result

def _extract_deadline(text: str) -> Optional[str]:
    """Extract deadline phrase from text."""
    m = _DEADLINE_RE.search(text)
    if m and m.group(1):
        return m.group(1).strip()
    m2 = _DATE_STANDALONE_RE.search(text)
    if m2:
        return m2.group(1).strip()
    return None

def _extract_assignee(text: str) -> Optional[str]:
    """Extract assignee from text."""
    m = _ASSIGNEE_RE.search(text)
    if m:
        assignee = m.group(1) or m.group(2) or m.group(3)
        if assignee:
            assignee = assignee.strip()
            noise = {"the", "a", "an", "each", "every", "any", "all", "said", "such"}
            if assignee.lower() not in noise and len(assignee) > 1:
                return assignee
    return None

def _determine_priority(text: str) -> str:
    """Determine priority from urgency indicators."""
    if _HIGH_PRIORITY_RE.search(text):
        return "high"
    if _LOW_PRIORITY_RE.search(text):
        return "low"
    return "medium"

def _determine_category(text: str, domain: Optional[str] = None) -> str:
    """Determine action item category based on content and domain."""
    if domain and domain in _DOMAIN_CATEGORIES:
        for pattern, cat in _DOMAIN_CATEGORIES[domain]:
            if pattern.search(text):
                return cat

    for domain_cats in _DOMAIN_CATEGORIES.values():
        for pattern, cat in domain_cats:
            if pattern.search(text):
                return cat

    return "general"

def _clean_description(text: str) -> str:
    """Clean extracted text into a concise action description."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 200:
        text = text[:197] + "..."
    return text

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for action item extraction."""
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [s.strip() for s in parts if s and len(s.strip()) > 10]

def extract_action_items(
    text: str,
    domain: Optional[str] = None,
    source_document: Optional[str] = None,
) -> List[ActionItem]:
    """
    Extract action items from document text using deterministic patterns.

    Looks for modal verbs (shall, must, will, should, needs to, required to),
    deadline phrases, and assignment patterns. Categorizes by domain.
    """
    if not text or len(text.strip()) < 10:
        return []

    sentences = _split_into_sentences(text)
    items: List[ActionItem] = []
    seen_descriptions: set = set()

    for sentence in sentences:
        if not _MODAL_RE.search(sentence):
            continue

        description = _clean_description(sentence)
        desc_lower = description.lower()

        if desc_lower in seen_descriptions:
            continue
        seen_descriptions.add(desc_lower)

        deadline = _extract_deadline(sentence)
        assignee = _extract_assignee(sentence)
        priority = _determine_priority(sentence)
        category = _determine_category(sentence, domain)

        items.append(
            ActionItem(
                description=description,
                deadline=deadline,
                priority=priority,
                assignee=assignee,
                source_document=source_document,
                category=category,
                source_text=sentence,
            )
        )

    items.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.priority, 1))

    return items

def render_action_items(items: List[ActionItem]) -> str:
    """Render action items into a readable format."""
    if not items:
        return "No action items found in the document(s)."

    lines = [f"**{len(items)} Action Item(s) Found**\n"]

    priority_groups = {"high": [], "medium": [], "low": []}
    for item in items:
        priority_groups.get(item.priority, priority_groups["medium"]).append(item)

    priority_labels = {
        "high": "High Priority",
        "medium": "Medium Priority",
        "low": "Low Priority",
    }

    for priority in ("high", "medium", "low"):
        group = priority_groups[priority]
        if not group:
            continue

        lines.append(f"\n**{priority_labels[priority]}**")
        for i, item in enumerate(group, 1):
            parts = [f"{i}. {item.description}"]
            if item.deadline:
                parts.append(f"   - Deadline: {item.deadline}")
            if item.assignee:
                parts.append(f"   - Assignee: {item.assignee}")
            if item.category != "general":
                parts.append(f"   - Category: {item.category}")
            if item.source_document:
                parts.append(f"   - Source: {item.source_document}")
            lines.append("\n".join(parts))

    return "\n".join(lines)

@register_tool("action_items")
async def action_items_handler(
    payload: Dict[str, Any], correlation_id: str | None = None
) -> Dict[str, Any]:
    """Handle action item extraction requests."""
    input_data = payload.get("input") or payload
    text = input_data.get("text", "")
    query = input_data.get("query", "")
    domain = input_data.get("domain")
    source_document = input_data.get("source_document")

    if not text:
        return standard_response(
            "action_items",
            status="error",
            context_found=False,
            grounded=False,
            error={"message": "No text provided for action item extraction"},
            correlation_id=correlation_id,
        )

    items = extract_action_items(text, domain=domain, source_document=source_document)
    rendered = render_action_items(items)

    return standard_response(
        "action_items",
        result={
            "action_items": [item.to_dict() for item in items],
            "count": len(items),
            "rendered": rendered,
        },
        grounded=True,
        context_found=bool(items),
        correlation_id=correlation_id,
    )
