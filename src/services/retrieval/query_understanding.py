from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class QueryUnderstandingResult:
    original_query: str
    normalized_query: str
    intent: str
    sub_queries: List[str] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    explicit_hints: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def query_variants(self) -> List[str]:
        variants: List[str] = []
        for candidate in [self.normalized_query] + self.expanded_queries + self.sub_queries:
            cleaned = re.sub(r"\s+", " ", candidate or "").strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)
        return variants

class QueryUnderstanding:
    """Normalize and enrich user queries for higher-accuracy retrieval."""

    _SUMMARY_HINTS = ("summary", "summarize", "overview", "brief", "high-level")
    _COMPARE_HINTS = ("compare", "difference", "vs", "versus", "contrast")
    _PROCEDURAL_HINTS = ("how", "steps", "procedure", "process", "configure", "setup", "install")

    def analyze(self, query: str, profile_context: Optional[Dict[str, Any]] = None) -> QueryUnderstandingResult:
        normalized = self.normalize_query(query)
        intent = self.detect_intent(normalized)
        metadata_filters, explicit_hints = self.extract_metadata(normalized, profile_context or {})
        sub_queries = self.generate_subqueries(normalized, intent)
        expanded_queries = self.expand_queries(normalized, intent, profile_context or {})

        return QueryUnderstandingResult(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            sub_queries=sub_queries,
            expanded_queries=expanded_queries,
            metadata_filters=metadata_filters,
            explicit_hints=explicit_hints,
        )

    @staticmethod
    def normalize_query(query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query or "").strip()
        return cleaned

    def detect_intent(self, query: str) -> str:
        lowered = (query or "").lower()
        if any(term in lowered for term in self._SUMMARY_HINTS):
            return "summarization"
        if any(term in lowered for term in self._COMPARE_HINTS):
            return "comparison"
        if any(term in lowered for term in self._PROCEDURAL_HINTS):
            return "procedural"
        return "fact_lookup"

    def generate_subqueries(self, query: str, intent: str) -> List[str]:
        sub_queries: List[str] = []
        lowered = query.lower()

        if intent == "comparison" and (" vs " in lowered or " versus " in lowered or "compare" in lowered):
            parts = re.split(r"\bvs\b|\bversus\b|\bcompare\b", query, flags=re.IGNORECASE)
            parts = [p.strip(" :,-") for p in parts if p.strip()]
            if len(parts) >= 2:
                left, right = parts[0], parts[1]
                sub_queries.append(left)
                sub_queries.append(right)
                sub_queries.append(f"Compare {left} and {right}")
        elif intent == "summarization":
            focus = self._strip_question_words(query)
            if focus and focus != query:
                sub_queries.append(f"summary of {focus}")
                sub_queries.append(f"key points about {focus}")
        elif intent == "procedural":
            focus = self._strip_question_words(query)
            if focus and focus != query:
                sub_queries.append(f"steps to {focus}")
                sub_queries.append(f"procedure for {focus}")
        else:
            simplified = self._strip_question_words(query)
            if simplified and simplified != query:
                sub_queries.append(simplified)

        # Limit to 1-3 subqueries, preserving order
        seen = []
        for candidate in sub_queries:
            cleaned = re.sub(r"\s+", " ", candidate or "").strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen[:3]

    def expand_queries(self, query: str, intent: str, profile_context: Dict[str, Any]) -> List[str]:
        expansions: List[str] = []
        terms = self._profile_terms(profile_context)
        if terms:
            expansions.append(f"{query} {' '.join(terms[:5])}")

        if intent == "comparison":
            expansions.append(query.replace(" vs ", " versus "))
        elif intent == "procedural" and not query.lower().startswith("how"):
            expansions.append(f"how to {query}")
        elif intent == "summarization" and "summary" not in query.lower():
            expansions.append(f"summary {query}")

        seen: List[str] = []
        for candidate in expansions:
            cleaned = re.sub(r"\s+", " ", candidate or "").strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen[:3]

    @staticmethod
    def _strip_question_words(query: str) -> str:
        lowered = query.lower()
        lowered = re.sub(r"^(what|which|who|where|when|how)\b", "", lowered).strip()
        lowered = re.sub(r"\bplease\b", "", lowered).strip()
        lowered = re.sub(r"\?", "", lowered).strip()
        return lowered

    def extract_metadata(self, query: str, profile_context: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, List[str]]]:
        filters: Dict[str, Any] = {}
        explicit: Dict[str, List[str]] = {}

        # Section titles
        section_matches = re.findall(r"\bsection\s*[:\"']?\s*([A-Za-z0-9 _\-]{3,60})", query, flags=re.IGNORECASE)
        section_titles = [m.strip() for m in section_matches if m.strip()]
        if section_titles:
            filters["section_titles"] = section_titles
            explicit["section_titles"] = section_titles

        # Page numbers
        page_numbers: List[int] = []
        for match in re.finditer(r"\bpage[s]?\s*(\d+)(?:\s*(?:-|to|–)\s*(\d+))?", query, re.IGNORECASE):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            if end < start:
                start, end = end, start
            span = min(end - start, 10)
            page_numbers.extend(range(start, start + span + 1))
        if page_numbers:
            filters["page_numbers"] = sorted(set(page_numbers))
            explicit["page_numbers"] = [str(p) for p in page_numbers]

        # File names / document hints
        file_matches = re.findall(r"\b[\w\-\.]+\.(?:pdf|docx?|pptx?|xlsx?|csv|txt)\b", query, flags=re.IGNORECASE)
        if file_matches:
            filters["source_files"] = file_matches
            explicit["source_files"] = file_matches

        doc_type = self._infer_doc_type(query)
        if doc_type:
            filters["doc_types"] = [doc_type]

        title_matches = re.findall(r"\btitle\s*[:\"']?\s*([A-Za-z0-9 _\-]{3,80})", query, flags=re.IGNORECASE)
        if title_matches:
            filters["titles"] = [m.strip() for m in title_matches if m.strip()]

        product_matches = re.findall(r"\bproduct\s*[:\"']?\s*([A-Za-z0-9 _\-]{2,60})", query, flags=re.IGNORECASE)
        if product_matches:
            filters["product_names"] = [m.strip() for m in product_matches if m.strip()]
            explicit["product_names"] = filters["product_names"]

        person_matches: List[str] = []
        person_matches += re.findall(
            r"\b(?:person|candidate|employee|author|speaker|contact)\s*[:\"']?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
            query,
        )
        person_matches += re.findall(
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})'s\s+(?:resume|experience|profile|bio|background)",
            query,
        )
        if person_matches:
            cleaned = [m.strip() for m in person_matches if m.strip()]
            if cleaned:
                filters["person_names"] = cleaned
                explicit["person_names"] = cleaned

        place_matches: List[str] = []
        place_matches += re.findall(
            r"\b(?:location|place|city|site|office|campus|facility)\s*[:\"']?\s*([A-Za-z][A-Za-z0-9 ,\-]{2,60})",
            query,
        )
        if place_matches:
            cleaned = [m.strip() for m in place_matches if m.strip()]
            if cleaned:
                filters["place_names"] = cleaned
                explicit["place_names"] = cleaned

        event_matches: List[str] = []
        event_matches += re.findall(
            r"\b(?:event|conference|summit|webinar|meeting|incident|outage|launch)\s*[:\"']?\s*([A-Za-z0-9][A-Za-z0-9 _\-]{2,80})",
            query,
        )
        event_matches += re.findall(
            r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4})\s+"
            r"(conference|summit|webinar|meeting|incident|outage|launch)\b",
            query,
        )
        if event_matches:
            cleaned = []
            for match in event_matches:
                if isinstance(match, tuple):
                    match = match[0]
                cleaned.append(str(match).strip())
            cleaned = [m for m in cleaned if m]
            if cleaned:
                filters["event_names"] = cleaned
                explicit["event_names"] = cleaned

        if profile_context.get("hints"):
            filters.setdefault("document_hints", profile_context.get("hints"))

        return filters, explicit

    @staticmethod
    def _infer_doc_type(query: str) -> Optional[str]:
        q = (query or "").lower()
        if any(word in q for word in ["slide", "slides", "deck", "presentation", "ppt"]):
            return "presentation"
        if any(word in q for word in ["spreadsheet", "excel", "xlsx", "csv", "sheet"]):
            return "spreadsheet"
        if any(word in q for word in ["invoice", "receipt", "bill"]):
            return "invoice"
        if any(word in q for word in ["contract", "agreement", "msa"]):
            return "contract"
        if any(word in q for word in ["resume", "cv"]):
            return "resume"
        if "report" in q:
            return "report"
        return None

    @staticmethod
    def _profile_terms(profile_context: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for key in ("keywords", "hints"):
            for term in profile_context.get(key) or []:
                cleaned = str(term).strip()
                if cleaned and cleaned not in terms:
                    terms.append(cleaned)
        return terms
