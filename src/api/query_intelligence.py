import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.kg.entity_extractor import EntityExtractor

logger = get_logger(__name__)

@dataclass
class QueryAnalysis:
    normalized_query: str
    intent: str
    sub_queries: List[str] = field(default_factory=list)
    expanded_query: str = ""
    expansion_terms: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    used_llm: bool = False
    sentiment: str = "neutral"
    sentiment_signals: List[str] = field(default_factory=list)
    instructions: Dict[str, Any] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def query_variants(self) -> List[str]:
        variants = []
        for candidate in [self.expanded_query] + self.sub_queries:
            if candidate and candidate not in variants:
                variants.append(candidate)
        return variants

class QueryIntelligence:
    """Analyze, expand, and decompose queries to improve retrieval recall and precision."""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self._entity_extractor = EntityExtractor()

    def analyze(self, query: str, profile_context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        normalized = re.sub(r"\s+", " ", query or "").strip()
        intent = self._detect_intent(normalized)
        metadata_filters = self._extract_metadata_filters(normalized, profile_context or {})
        sentiment, sentiment_signals = self._detect_sentiment(normalized)
        instructions = self._parse_instructions(normalized)
        entities = self.extract_entities(normalized)

        sub_queries = self._decompose_query(normalized)
        used_llm = False
        if self._use_llm() and self.llm_client and self._should_llm_decompose(normalized, sub_queries):
            llm_subs = self._llm_decompose(normalized)
            if llm_subs:
                sub_queries = self._merge_subqueries(sub_queries, llm_subs)
                used_llm = True

        expansion_terms = self._expand_with_vocab(normalized, profile_context or {})
        expanded_query = normalized
        if expansion_terms:
            expanded_query = f"{normalized} {' '.join(expansion_terms)}"

        return QueryAnalysis(
            normalized_query=normalized,
            intent=intent,
            sub_queries=sub_queries,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            metadata_filters=metadata_filters,
            used_llm=used_llm,
            sentiment=sentiment,
            sentiment_signals=sentiment_signals,
            instructions=instructions,
            entities=[entity.__dict__ for entity in entities],
        )

    def extract_entities(self, query: str) -> List[Any]:
        """Expose deterministic entity extraction for downstream KG probe."""
        try:
            return self._entity_extractor.extract(query or "")
        except Exception:
            return []

    @staticmethod
    def _use_llm() -> bool:
        return bool(getattr(Config.Retrieval, "QUERY_INTELLIGENCE_USE_LLM", True))

    @staticmethod
    def _should_llm_decompose(query: str, sub_queries: List[str]) -> bool:
        if sub_queries:
            return False
        if len(query.split()) >= 10:
            return True
        return any(token in query.lower() for token in [" and ", " vs ", " versus ", " compare ", " difference "])

    @staticmethod
    def _merge_subqueries(existing: List[str], extra: List[str]) -> List[str]:
        merged = []
        for candidate in existing + extra:
            candidate = re.sub(r"\s+", " ", candidate or "").strip()
            if candidate and candidate not in merged:
                merged.append(candidate)
        return merged

    def _llm_decompose(self, query: str) -> List[str]:
        prompt = f"""You are a query decomposition assistant. Break the user query into 2-4 focused sub-questions
that could be answered from documents. Output strict JSON only.

User query: {query}

Return format:
{{"sub_questions": ["...","..."]}}"""
        try:
            response = self.llm_client.generate(prompt, max_retries=2, backoff=0.4)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM query decomposition failed: %s", exc)
            return []
        try:
            payload = self._extract_json(response)
            sub_questions = payload.get("sub_questions") or []
            return [str(q).strip() for q in sub_questions if str(q).strip()]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse LLM decomposition JSON: %s", exc)
            return []

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            return json.loads(raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group(0))

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["summary", "summarize", "overview", "brief"]):
            return "summary"
        if any(word in q for word in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        if any(word in q for word in ["list", "extract", "identify", "show", "find"]):
            return "extraction"
        if any(word in q for word in ["why", "cause", "impact", "implication", "analyze", "evaluate", "reason"]):
            return "reasoning"
        if any(word in q for word in ["how", "steps", "procedure", "configure", "setup", "process"]):
            return "procedural"
        if any(word in q for word in ["error", "issue", "fail", "troubleshoot", "bug"]):
            return "troubleshooting"
        return "factual"

    @staticmethod
    def _detect_sentiment(query: str) -> tuple[str, List[str]]:
        q = query.lower()
        signals: List[str] = []
        if any(word in q for word in ["frustrated", "annoyed", "upset", "angry", "doesn't work", "not working"]):
            signals.append("frustration")
        if any(word in q for word in ["confused", "unclear", "not sure", "don't understand"]):
            signals.append("confusion")
        if any(word in q for word in ["thanks", "thank you", "appreciate"]):
            signals.append("gratitude")
        if any(word in q for word in ["urgent", "asap", "immediately", "right away"]):
            signals.append("urgency")
        if not signals:
            return "neutral", []
        # prioritize negative/urgent before gratitude
        for ordered in ["frustration", "confusion", "urgency", "gratitude"]:
            if ordered in signals:
                return ordered, signals
        return signals[0], signals

    @staticmethod
    def _parse_instructions(query: str) -> Dict[str, Any]:
        q = query.lower()
        instructions: Dict[str, Any] = {}

        if any(word in q for word in ["table", "tabular", "matrix"]):
            instructions["use_table"] = True
            instructions["format"] = "table"
        if any(word in q for word in ["step by step", "step-by-step", "steps"]):
            instructions["step_by_step"] = True
            instructions["format"] = "steps"
        if any(word in q for word in ["bullet", "bullets", "list"]):
            instructions["use_bullets"] = True
            instructions.setdefault("format", "bullets")
        if any(word in q for word in ["with sections", "in sections", "use sections", "sectioned"]):
            instructions["format"] = "sections"
        if any(word in q for word in ["brief", "short", "concise"]):
            instructions["brevity"] = "brief"
        if any(word in q for word in ["detailed", "in depth", "deep dive"]):
            instructions["brevity"] = "detailed"
            instructions["analysis"] = True
        if any(word in q for word in ["analyze", "analysis", "deep analysis", "in-depth analysis"]):
            instructions["analysis"] = True
        if any(word in q for word in ["formal", "executive tone"]):
            instructions["tone"] = "formal"
        if any(word in q for word in ["casual", "informal"]):
            instructions["tone"] = "informal"
        if any(word in q for word in ["empathetic", "with empathy", "be kind", "supportive"]):
            instructions["empathy"] = True
        if any(word in q for word in ["narrative", "story"]):
            instructions["style"] = "narrative"

        return instructions

    @staticmethod
    def _decompose_query(query: str) -> List[str]:
        lowered = query.lower()
        sub_queries: List[str] = []

        if " vs " in lowered or " versus " in lowered:
            parts = re.split(r"\bvs\b|\bversus\b", query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                left = parts[0].strip(" :,-")
                right = parts[1].strip(" :,-")
                if left:
                    sub_queries.append(left)
                if right:
                    sub_queries.append(right)
                if left and right:
                    sub_queries.append(f"Compare {left} and {right}")
            return sub_queries

        if " and " in lowered and "between" not in lowered:
            parts = [p.strip(" ,") for p in re.split(r"\band\b", query, flags=re.IGNORECASE) if p.strip()]
            if len(parts) >= 2:
                sub_queries.extend(parts[:3])

        return sub_queries

    @staticmethod
    def _expand_with_vocab(query: str, profile_context: Dict[str, Any]) -> List[str]:
        terms = []
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        keywords = profile_context.get("keywords") or []
        hints = profile_context.get("hints") or []

        for word in keywords:
            cleaned = str(word).strip()
            if not cleaned:
                continue
            token = cleaned.lower()
            if token not in query_tokens and len(token) > 3:
                terms.append(cleaned)
            if len(terms) >= 8:
                break

        if not terms and hints:
            for hint in hints[:3]:
                cleaned = str(hint).strip()
                if cleaned and cleaned.lower() not in query_tokens:
                    terms.append(cleaned)

        return terms

    @staticmethod
    def _extract_metadata_filters(query: str, profile_context: Dict[str, Any]) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}

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

        section_match = re.search(r"\bsection\s*[:\"']?\s*([A-Za-z0-9 \-_]{3,40})", query, re.IGNORECASE)
        if section_match:
            section_title = section_match.group(1).strip()
            if section_title:
                filters["section_titles"] = [section_title]

        doc_type = QueryIntelligence._infer_doc_type(query)
        if doc_type:
            filters["doc_types"] = [doc_type]

        if QueryIntelligence._needs_high_confidence(query):
            filters["min_confidence"] = float(getattr(Config.Retrieval, "MIN_OCR_CONFIDENCE", 60.0))

        if profile_context.get("hints"):
            filters.setdefault("document_hints", profile_context.get("hints"))

        return filters

    @staticmethod
    def _infer_doc_type(query: str) -> Optional[str]:
        q = query.lower()
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
    def _needs_high_confidence(query: str) -> bool:
        q = query.lower()
        return any(word in q for word in ["quote", "verbatim", "exact", "ocr", "scanned"])
