"""
Response Formatter for DocWain Intelligence Layer.

This module provides intelligent response formatting that:
- Acknowledges user intent before answering
- Formats responses based on query type and domain
- Provides structured, clear outcomes
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = get_logger(__name__)

class QueryIntent(Enum):
    """Classification of user query intent."""

    RETRIEVE = "retrieve"           # User wants to get specific information
    SUMMARIZE = "summarize"         # User wants a summary/overview
    COMPARE = "compare"             # User wants to compare items
    EXTRACT = "extract"             # User wants specific data extracted
    EXPLAIN = "explain"             # User wants explanation/clarification
    LIST = "list"                   # User wants a list of items
    VALIDATE = "validate"           # User wants to verify information
    SEARCH = "search"               # User is searching for something
    UNKNOWN = "unknown"

# Intent detection patterns
INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
    QueryIntent.RETRIEVE: [
        r"\bwhat\s+is\b", r"\bwhat's\b", r"\btell\s+me\b", r"\bshow\s+me\b",
        r"\bget\s+(?:me\s+)?(?:the)?\b", r"\bfind\s+(?:the)?\b", r"\bgive\s+me\b",
        r"\bwhat\s+are\b", r"\bwho\s+is\b", r"\bwhere\s+is\b",
    ],
    QueryIntent.SUMMARIZE: [
        r"\bsummar", r"\boverview\b", r"\bbrief\b", r"\bhighlight",
        r"\bkey\s+points\b", r"\bmain\s+points\b", r"\bin\s+short\b",
    ],
    QueryIntent.COMPARE: [
        r"\bcompare\b", r"\bdifference\s+between\b", r"\bvs\.?\b",
        r"\bversus\b", r"\bbetter\s+than\b", r"\bsimilar\s+to\b",
    ],
    QueryIntent.EXTRACT: [
        r"\bextract\b", r"\bpull\s+out\b", r"\bget\s+all\b",
        r"\blist\s+all\b", r"\bfind\s+all\b", r"\bcollect\b",
    ],
    QueryIntent.EXPLAIN: [
        r"\bexplain\b", r"\bhow\s+does\b", r"\bwhy\s+(?:is|does|do)\b",
        r"\bwhat\s+does\s+.+\s+mean\b", r"\bclarif", r"\bdescribe\b",
    ],
    QueryIntent.LIST: [
        r"\blist\b", r"\benumerate\b", r"\bwhat\s+are\s+(?:the|all)\b",
        r"\bgive\s+me\s+(?:a\s+)?list\b", r"\bshow\s+all\b",
    ],
    QueryIntent.VALIDATE: [
        r"\bis\s+(?:it|this|that)\s+(?:true|correct|right)\b",
        r"\bverif", r"\bconfirm\b", r"\bcheck\s+if\b", r"\bdoes\s+.+\s+have\b",
    ],
    QueryIntent.SEARCH: [
        r"\bsearch\b", r"\blook\s+(?:for|up)\b", r"\bfind\b",
        r"\bwhere\s+can\s+i\s+find\b", r"\blocate\b",
    ],
}

# Acknowledgement templates by intent
ACKNOWLEDGEMENT_TEMPLATES: Dict[QueryIntent, List[str]] = {
    QueryIntent.RETRIEVE: [
        "I understand you're looking for {topic}. Here's what I found:",
        "You want to know about {topic}. Based on the documents:",
        "Let me get that information about {topic} for you:",
    ],
    QueryIntent.SUMMARIZE: [
        "I'll provide you with a summary of {topic}:",
        "Here's an overview of {topic} from the documents:",
        "Let me summarize {topic} for you:",
    ],
    QueryIntent.COMPARE: [
        "I understand you want to compare {topic}. Here's the analysis:",
        "Let me compare {topic} based on the available information:",
    ],
    QueryIntent.EXTRACT: [
        "I'll extract {topic} from the documents:",
        "Here's the extracted information about {topic}:",
    ],
    QueryIntent.EXPLAIN: [
        "Let me explain {topic} based on the documents:",
        "I'll clarify {topic} for you:",
        "Here's an explanation of {topic}:",
    ],
    QueryIntent.LIST: [
        "Here's a list of {topic} from the documents:",
        "I'll list out {topic} for you:",
    ],
    QueryIntent.VALIDATE: [
        "Let me verify {topic} for you:",
        "I'll check {topic} against the documents:",
    ],
    QueryIntent.SEARCH: [
        "I searched for {topic} and here's what I found:",
        "Based on my search for {topic}:",
    ],
    QueryIntent.UNKNOWN: [
        "I understand your question about {topic}. Here's what I found:",
        "Regarding {topic}, here's the information from the documents:",
    ],
}

@dataclass
class FormattedResponse:
    """A formatted response with acknowledgement and content."""

    acknowledgement: str
    content: str
    intent: QueryIntent
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)

    def to_string(self, include_sources: bool = True) -> str:
        """Convert to full response string."""
        parts = [self.acknowledgement, "", self.content]

        if include_sources and self.sources:
            parts.append("")
            parts.append("**Sources:**")
            for source in self.sources[:5]:  # Limit to 5 sources
                parts.append(f"- {source}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "acknowledgement": self.acknowledgement,
            "content": self.content,
            "intent": self.intent.value,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
        }

class ResponseFormatter:
    """
    Formats responses with intelligent acknowledgement.

    Analyzes user queries to determine intent, then formats
    responses with appropriate acknowledgement and structure.
    """

    def __init__(
        self,
        default_acknowledgement: str = "Based on the available documents:",
        include_confidence_note: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the formatter.

        Args:
            default_acknowledgement: Default acknowledgement for unknown intents.
            include_confidence_note: Whether to add confidence notes for low confidence.
            confidence_threshold: Threshold below which to add confidence notes.
        """
        self.default_acknowledgement = default_acknowledgement
        self.include_confidence_note = include_confidence_note
        self.confidence_threshold = confidence_threshold

        # Compile patterns for efficiency
        self._compiled_patterns: Dict[QueryIntent, List[re.Pattern]] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def detect_intent(self, query: str) -> tuple[QueryIntent, float]:
        """
        Detect the intent of a user query.

        Args:
            query: The user's query string.

        Returns:
            Tuple of (intent, confidence_score).
        """
        query_lower = query.lower()
        intent_scores: Dict[QueryIntent, int] = {}

        for intent, patterns in self._compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(query_lower))
            if matches > 0:
                intent_scores[intent] = matches

        if not intent_scores:
            return QueryIntent.UNKNOWN, 0.3

        # Get intent with most pattern matches
        best_intent = max(intent_scores, key=intent_scores.get)
        max_matches = intent_scores[best_intent]

        # Calculate confidence based on match count
        confidence = min(0.5 + (max_matches * 0.15), 0.95)

        return best_intent, confidence

    def extract_topic(self, query: str) -> str:
        """
        Extract the main topic from a query.

        Args:
            query: The user's query string.

        Returns:
            The extracted topic or a generic placeholder.
        """
        # Remove common question words and extract topic
        cleaned = query.strip()

        # Remove question marks and common prefixes
        cleaned = re.sub(r'\?+$', '', cleaned)
        cleaned = re.sub(
            r'^(?:what|who|where|when|why|how|can|could|would|is|are|do|does|'
            r'please|tell me|show me|give me|find|get|list)\s+',
            '', cleaned, flags=re.IGNORECASE
        )

        # Remove follow-up question words (e.g., "are" after "what")
        cleaned = re.sub(r'^(?:is|are|was|were)\s+', '', cleaned, flags=re.IGNORECASE)

        # Remove articles
        cleaned = re.sub(r'^(?:the|a|an)\s+', '', cleaned, flags=re.IGNORECASE)

        # Truncate if too long
        if len(cleaned) > 100:
            cleaned = cleaned[:100].rsplit(' ', 1)[0] + "..."

        return cleaned if cleaned else "your question"

    def generate_acknowledgement(
        self,
        query: str,
        intent: Optional[QueryIntent] = None,
        topic: Optional[str] = None,
    ) -> str:
        """
        Generate an acknowledgement for the query.

        Args:
            query: The user's query.
            intent: Optional pre-detected intent.
            topic: Optional pre-extracted topic.

        Returns:
            An acknowledgement string.
        """
        if intent is None:
            intent, _ = self.detect_intent(query)

        if topic is None:
            topic = self.extract_topic(query)

        templates = ACKNOWLEDGEMENT_TEMPLATES.get(
            intent,
            ACKNOWLEDGEMENT_TEMPLATES[QueryIntent.UNKNOWN]
        )

        # Rotate templates deterministically by query hash so different
        # queries get different phrasing while the same query stays stable
        idx = hash(query) % len(templates) if query else 0
        template = templates[idx]

        return template.format(topic=topic)

    def format_response(
        self,
        query: str,
        content: str,
        sources: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FormattedResponse:
        """
        Format a complete response with acknowledgement.

        Args:
            query: The user's original query.
            content: The answer content to format.
            sources: Optional list of source documents.
            confidence: Optional confidence score for the answer.
            metadata: Optional additional metadata.

        Returns:
            A FormattedResponse object.
        """
        intent, intent_confidence = self.detect_intent(query)
        topic = self.extract_topic(query)

        acknowledgement = self.generate_acknowledgement(query, intent, topic)

        # Add confidence note if needed
        final_content = content
        answer_confidence = confidence if confidence is not None else intent_confidence

        if (self.include_confidence_note and
            answer_confidence < self.confidence_threshold):
            confidence_note = (
                "\n\n*Note: I have moderate confidence in this answer. "
                "Please verify this information if it's critical.*"
            )
            final_content = content + confidence_note

        return FormattedResponse(
            acknowledgement=acknowledgement,
            content=final_content,
            intent=intent,
            confidence=answer_confidence,
            sources=sources or [],
            metadata=metadata or {},
        )

    def format_error_response(
        self,
        query: str,
        error_message: str,
        suggestion: Optional[str] = None,
    ) -> FormattedResponse:
        """
        Format an error response with helpful guidance.

        Args:
            query: The user's original query.
            error_message: The error description.
            suggestion: Optional suggestion for the user.

        Returns:
            A FormattedResponse for the error.
        """
        topic = self.extract_topic(query)

        acknowledgement = f"I understand you're asking about {topic}."

        content_parts = [
            f"Unfortunately, {error_message.lower()}"
        ]

        if suggestion:
            content_parts.append(f"\n\n**Suggestion:** {suggestion}")

        return FormattedResponse(
            acknowledgement=acknowledgement,
            content="\n".join(content_parts),
            intent=QueryIntent.UNKNOWN,
            confidence=0.0,
            metadata={"is_error": True},
        )

    def format_no_results_response(
        self,
        query: str,
        searched_documents: Optional[List[str]] = None,
    ) -> FormattedResponse:
        """
        Format a response when no results are found.

        Args:
            query: The user's original query.
            searched_documents: Optional list of documents that were searched.

        Returns:
            A FormattedResponse for no results.
        """
        topic = self.extract_topic(query)

        acknowledgement = f"I understand you're looking for information about {topic}."

        content_parts = [
            "I searched through the available documents but couldn't find "
            "information directly related to your question."
        ]

        if searched_documents:
            content_parts.append(
                f"\n\nI searched in: {', '.join(searched_documents[:3])}"
                + (" and more..." if len(searched_documents) > 3 else "")
            )

        content_parts.append(
            "\n\n**Suggestions:**\n"
            "- Try rephrasing your question\n"
            "- Check if the document contains this information\n"
            "- Ask about a specific document by name"
        )

        return FormattedResponse(
            acknowledgement=acknowledgement,
            content="\n".join(content_parts),
            intent=QueryIntent.SEARCH,
            confidence=0.0,
            metadata={"no_results": True},
        )

def format_acknowledged_response(
    query: str,
    content: str,
    sources: Optional[List[str]] = None,
    confidence: Optional[float] = None,
) -> str:
    """
    Convenience function to format a response with acknowledgement.

    Args:
        query: The user's query.
        content: The answer content.
        sources: Optional source documents.
        confidence: Optional confidence score.

    Returns:
        The full formatted response as a string.
    """
    formatter = ResponseFormatter()
    response = formatter.format_response(query, content, sources, confidence)
    return response.to_string()

__all__ = [
    "QueryIntent",
    "FormattedResponse",
    "ResponseFormatter",
    "format_acknowledged_response",
    "INTENT_PATTERNS",
    "ACKNOWLEDGEMENT_TEMPLATES",
]
