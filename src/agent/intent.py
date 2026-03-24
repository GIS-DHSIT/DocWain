"""Intent Analyzer — the UNDERSTAND step of the Core Agent pipeline.

Replaces six competing intent classifiers with one LLM-native system.
Fast-paths conversational queries (greetings, farewells, meta questions)
without an LLM call; everything else gets analyzed via build_understand_prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.generation.prompts import build_understand_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid enum values
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = frozenset(
    {"extract", "compare", "summarize", "overview", "investigate", "lookup", "aggregate", "list", "conversational"}
)
_VALID_OUTPUT_FORMATS = frozenset({"table", "bullets", "sections", "numbered", "prose"})
_VALID_COMPLEXITIES = frozenset({"simple", "complex"})

# ---------------------------------------------------------------------------
# Conversational detection patterns
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings|yo)\b",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^\s*(?:bye|goodbye|see\s+you|thanks|thank\s+you|cheers)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"^\s*(?:who\s+are\s+you|what\s+can\s+you\s+do|help)\s*[?!.]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# QueryUnderstanding dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryUnderstanding:
    """Result of the UNDERSTAND step — structured intent analysis."""

    task_type: str
    complexity: str
    resolved_query: str
    output_format: str
    relevant_documents: List[Dict[str, Any]]
    cross_profile: bool
    sub_tasks: Optional[List[str]] = None
    entities: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    @property
    def is_conversational(self) -> bool:
        return self.task_type == "conversational"

    @property
    def is_complex(self) -> bool:
        return self.complexity == "complex" and bool(self.sub_tasks)


# ---------------------------------------------------------------------------
# IntentAnalyzer
# ---------------------------------------------------------------------------


class IntentAnalyzer:
    """LLM-native intent analysis with conversational fast-path."""

    def __init__(self, llm_gateway: Any, redis_client: Any = None) -> None:
        self._llm = llm_gateway
        self._redis = redis_client

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        doc_intelligence: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        kg_hints: Optional[Dict[str, Any]] = None,
    ) -> QueryUnderstanding:
        """Analyze user intent and return a structured QueryUnderstanding.

        Fast-paths greetings/farewells/meta questions without an LLM call.
        For real queries, builds a prompt via ``build_understand_prompt`` and
        parses the LLM's JSON response.  After parsing, enriches
        ``relevant_documents`` using answerable_topics overlap and optional
        KG entity hints.
        """
        # Fast-path: conversational queries need no LLM call
        if self._is_conversational(query):
            logger.debug("Fast-path conversational: %s", query[:60])
            return QueryUnderstanding(
                task_type="conversational",
                complexity="simple",
                resolved_query=query,
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
            )

        # Fast-path: simple queries with clear intent patterns (skip LLM call)
        heuristic = self._heuristic_classify(query, conversation_history)
        if heuristic is not None:
            logger.debug("Fast-path heuristic: %s -> %s", query[:60], heuristic.task_type)
            self._enrich_relevant_documents(heuristic, query, doc_intelligence, kg_hints)
            return heuristic

        # Build cache key: intent:{profile_id}:{sha1(query)}
        _query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()
        _cache_key = f"intent:{profile_id}:{_query_hash}"
        _INTENT_TTL = 300  # 5 minutes

        # Check Redis cache before calling LLM
        if self._redis is not None:
            try:
                cached_raw = self._redis.get(_cache_key)
                if cached_raw:
                    if isinstance(cached_raw, (bytes, bytearray)):
                        cached_raw = cached_raw.decode("utf-8")
                    cached_data = json.loads(cached_raw)
                    result = QueryUnderstanding(
                        task_type=cached_data["task_type"],
                        complexity=cached_data["complexity"],
                        resolved_query=cached_data["resolved_query"],
                        output_format=cached_data["output_format"],
                        relevant_documents=cached_data.get("relevant_documents", []),
                        cross_profile=cached_data.get("cross_profile", False),
                        sub_tasks=cached_data.get("sub_tasks"),
                        entities=cached_data.get("entities", []),
                        needs_clarification=cached_data.get("needs_clarification", False),
                        clarification_question=cached_data.get("clarification_question"),
                    )
                    logger.debug("[IntentAnalyzer] Cache hit for key %s", _cache_key)
                    self._enrich_relevant_documents(result, query, doc_intelligence, kg_hints)
                    return result
            except Exception:
                logger.debug("[IntentAnalyzer] Redis cache read failed — proceeding without cache")

        # Build prompt and call LLM
        prompt = build_understand_prompt(query, doc_intelligence, conversation_history)
        try:
            raw = self._llm.generate(
                prompt,
                system="You are a document intelligence query analyzer. Respond with valid JSON only. Do NOT use <think> tags. Output JSON immediately.",
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception:
            logger.exception("LLM call failed for intent analysis")
            result = self._safe_defaults(query)
            self._enrich_relevant_documents(result, query, doc_intelligence, kg_hints)
            return result

        result = self._parse_response(raw, query)

        # Cache successful LLM parse result (not fallback defaults)
        if self._redis is not None and result.task_type != "summarize":
            try:
                cache_payload = json.dumps({
                    "task_type": result.task_type,
                    "complexity": result.complexity,
                    "resolved_query": result.resolved_query,
                    "output_format": result.output_format,
                    "relevant_documents": result.relevant_documents,
                    "cross_profile": result.cross_profile,
                    "sub_tasks": result.sub_tasks,
                    "entities": result.entities,
                    "needs_clarification": result.needs_clarification,
                    "clarification_question": result.clarification_question,
                })
                self._redis.setex(_cache_key, _INTENT_TTL, cache_payload)
                logger.debug("[IntentAnalyzer] Cached intent result for key %s (TTL=%ds)", _cache_key, _INTENT_TTL)
            except Exception:
                logger.debug("[IntentAnalyzer] Redis cache write failed — continuing without cache")

        self._enrich_relevant_documents(result, query, doc_intelligence, kg_hints)
        return result

    # -- enrichment ---------------------------------------------------------

    @staticmethod
    def _match_topics(
        query: str,
        doc_intelligence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score documents by word overlap between query and answerable_topics.

        Returns list of ``{"document_id": str, "topic_score": int}`` sorted
        by score descending, only including documents with score > 0.
        """
        query_words = set(query.lower().split())
        results: List[Dict[str, Any]] = []

        for doc in doc_intelligence:
            doc_id = doc.get("document_id")
            if not doc_id:
                continue
            intel = doc.get("intelligence") or {}
            topics: List[str] = intel.get("answerable_topics") or []
            if not topics:
                continue

            best_score = 0
            for topic in topics:
                topic_words = set(topic.lower().split())
                overlap = len(query_words & topic_words)
                if overlap > best_score:
                    best_score = overlap

            if best_score > 0:
                results.append({"document_id": doc_id, "topic_score": best_score})

        results.sort(key=lambda x: x["topic_score"], reverse=True)
        return results

    @staticmethod
    def _enrich_relevant_documents(
        result: QueryUnderstanding,
        query: str,
        doc_intelligence: List[Dict[str, Any]],
        kg_hints: Optional[Dict[str, Any]],
    ) -> None:
        """Enrich relevant_documents with topic matches and KG hints."""
        existing_ids = {d.get("document_id") for d in result.relevant_documents}

        # Add topic-matched documents
        for match in IntentAnalyzer._match_topics(query, doc_intelligence):
            if match["document_id"] not in existing_ids:
                result.relevant_documents.append(match)
                existing_ids.add(match["document_id"])

        # Add KG-hinted documents
        if kg_hints:
            for doc_id in kg_hints.get("target_doc_ids", []):
                if doc_id not in existing_ids:
                    result.relevant_documents.append({"document_id": doc_id, "source": "kg"})
                    existing_ids.add(doc_id)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _heuristic_classify(
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Optional[QueryUnderstanding]:
        """Fast heuristic classification for unambiguous query patterns.

        Returns None if the query is ambiguous and needs LLM analysis.
        Only fires for simple, single-turn queries without pronoun references.
        """
        q = query.strip().lower()

        # Skip heuristic if query likely references conversation context
        if conversation_history and any(w in q for w in ["it", "that", "this", "those", "them", "the same"]):
            return None

        # Skip very short or very long queries (ambiguous or complex)
        if len(q) < 10 or len(q) > 300:
            return None

        # Pattern matching for clear intent signals
        task_type: Optional[str] = None
        output_format = "prose"

        if any(q.startswith(w) for w in ["list ", "list all ", "enumerate ", "name all "]):
            task_type = "list"
            output_format = "bullets"
        elif any(q.startswith(w) for w in ["compare ", "contrast "]) or " vs " in q or " versus " in q:
            task_type = "compare"
            output_format = "table"
        elif any(w in q for w in [
            "tell me about the documents", "tell me about all", "what do we have",
            "give me an overview", "overview of all", "describe the documents",
            "what documents", "about the documents", "about the collection",
        ]):
            task_type = "overview"
            output_format = "sections"
        elif any(q.startswith(w) for w in ["summarize ", "summary of ", "give me a summary"]):
            task_type = "summarize"
            output_format = "sections"
        elif any(q.startswith(w) for w in ["extract "]):
            task_type = "extract"
            output_format = "prose"
        elif any(w in q for w in ["step by step", "steps to", "procedure for"]):
            task_type = "extract"
            output_format = "numbered"
        elif any(w in q for w in ["how many", "total ", "count ", "sum of"]):
            task_type = "aggregate"
            output_format = "prose"

        # Let all other queries go through LLM analysis for better classification
        if task_type is None:
            return None

        return QueryUnderstanding(
            task_type=task_type,
            complexity="simple",
            resolved_query=query,
            output_format=output_format,
            relevant_documents=[],
            cross_profile=False,
        )

    @staticmethod
    def _is_conversational(query: str) -> bool:
        """Return True if *query* is a greeting, farewell, or meta question."""
        return bool(
            _GREETING_RE.search(query)
            or _FAREWELL_RE.search(query)
            or _META_RE.search(query)
        )

    @staticmethod
    def _parse_response(raw: str, original_query: str) -> QueryUnderstanding:
        """Parse LLM JSON response into QueryUnderstanding.

        Strips markdown fences, validates enum fields, and falls back to
        safe defaults on any parse failure.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # If response is empty but thinking block exists, try to extract JSON from thinking
        if not text and raw:
            think_match = re.search(r"\{[^{}]*\"task_type\"[^{}]*\}", raw, re.DOTALL)
            if think_match:
                text = think_match.group()
                logger.debug("Extracted intent JSON from thinking block")

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse intent JSON (len=%d): %.200s", len(raw or ""), raw or "(empty)")
            return IntentAnalyzer._safe_defaults(original_query)

        # Validate and coerce enum fields
        task_type = data.get("task_type", "lookup")
        if task_type not in _VALID_TASK_TYPES:
            logger.warning("Invalid task_type '%s', falling back to 'lookup'", task_type)
            task_type = "lookup"

        output_format = data.get("output_format", "prose")
        if output_format not in _VALID_OUTPUT_FORMATS:
            output_format = "prose"

        complexity = data.get("complexity", "simple")
        if complexity not in _VALID_COMPLEXITIES:
            complexity = "simple"

        return QueryUnderstanding(
            task_type=task_type,
            complexity=complexity,
            resolved_query=data.get("resolved_query", original_query),
            output_format=output_format,
            relevant_documents=data.get("relevant_documents", []),
            cross_profile=bool(data.get("cross_profile", False)),
            sub_tasks=data.get("sub_tasks"),
            entities=data.get("entities", []),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question"),
        )

    @staticmethod
    def _safe_defaults(query: str) -> QueryUnderstanding:
        """Return a safe-default QueryUnderstanding when parsing fails.

        Defaults to 'summarize' with 'sections' format to produce structured,
        detailed responses rather than the minimal 1-3 sentence 'lookup' style.
        """
        return QueryUnderstanding(
            task_type="summarize",
            complexity="simple",
            resolved_query=query,
            output_format="sections",
            relevant_documents=[],
            cross_profile=False,
        )
