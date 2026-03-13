"""Core Agent Orchestrator — UNDERSTAND -> RETRIEVE -> REASON -> COMPOSE.

Ties the entire DocWain intelligence pipeline together.  For complex
queries it spawns dynamic sub-agents in parallel via ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from src.agent.intent import IntentAnalyzer, QueryUnderstanding
from src.agent.subagent import DynamicSubAgent
from src.generation.composer import compose_response
from src.generation.reasoner import Reasoner, ReasonerResult
from src.retrieval.context_builder import build_context
from src.retrieval.reranker import rerank_chunks
from src.retrieval.retriever import UnifiedRetriever
from src.agent.domain_dispatch import DomainDispatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords for query expansion filtering
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "only",
    "own", "same", "than", "too", "very", "just", "about", "up", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "many", "much", "tell", "show", "give", "get",
    "document", "documents", "file", "files", "please",
}

# ---------------------------------------------------------------------------
# Query expansion synonyms by task type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dynamic evidence count by task type
# ---------------------------------------------------------------------------

_EVIDENCE_TOP_K: Dict[str, int] = {
    "lookup": 4,
    "extract": 6,
    "list": 8,
    "summarize": 8,
    "compare": 8,
    "investigate": 8,
    "aggregate": 6,
}

_TASK_SYNONYMS: Dict[str, List[str]] = {
    "extract": ["extract", "find", "identify", "locate", "what is", "what are"],
    "compare": ["compare", "contrast", "difference", "versus", "vs", "similarity"],
    "summarize": ["summarize", "summary", "overview", "key points", "highlights"],
    "investigate": ["investigate", "analyze", "examine", "assess", "evaluate", "risk"],
    "lookup": ["what", "who", "when", "where", "how much"],
    "list": ["list", "enumerate", "name", "all", "each"],
    "aggregate": ["total", "count", "sum", "average", "how many"],
}

# ---------------------------------------------------------------------------
# Conversational response fragments
# ---------------------------------------------------------------------------

_CONVERSATIONAL_RESPONSES: Dict[str, str] = {
    "greeting": "Hello! I'm DocWain, your document intelligence assistant. How can I help you today?",
    "farewell": "Goodbye! Feel free to come back anytime you need help with your documents.",
    "thanks": "You're welcome! Let me know if there's anything else I can help you find in your documents.",
    "meta": "I'm DocWain, an AI document intelligence assistant. I can search, extract, compare, and summarize information from your uploaded documents. Just ask me a question!",
}

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings|yo)\b",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^\s*(?:bye|goodbye|see\s+you)\b",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\s*(?:thanks|thank\s+you|cheers)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"^\s*(?:who\s+are\s+you|what\s+can\s+you\s+do|help)\s*[?!.]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# CoreAgent
# ---------------------------------------------------------------------------


class CoreAgent:
    """Orchestrates the full UNDERSTAND -> RETRIEVE -> REASON -> COMPOSE pipeline."""

    MAX_SUBAGENTS = 5
    SUBAGENT_TIMEOUT = 30.0

    def __init__(
        self,
        llm_gateway: Any,
        qdrant_client: Any,
        embedder: Any,
        mongodb: Any,
        kg_query_service: Any = None,
        cross_encoder: Any = None,
    ) -> None:
        self._llm = llm_gateway
        self._mongodb = mongodb
        self._intent_analyzer = IntentAnalyzer(llm_gateway=llm_gateway)
        self._retriever = UnifiedRetriever(qdrant_client=qdrant_client, embedder=embedder)
        self._reasoner = Reasoner(llm_gateway=llm_gateway)
        self.kg_query_service = kg_query_service
        self._cross_encoder = cross_encoder
        self._domain_dispatcher = DomainDispatcher(llm_gateway=llm_gateway)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        user_id: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]],
        *,
        agent_name: Optional[str] = None,
        document_id: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline and return an AnswerPayload dict."""
        if not subscription_id or not str(subscription_id).strip():
            raise ValueError("subscription_id is required")
        if not profile_id or not str(profile_id).strip():
            raise ValueError("profile_id is required")

        timing: Dict[str, float] = {}

        # --- UNDERSTAND ---
        t0 = time.monotonic()
        doc_intelligence = self._load_doc_intelligence(subscription_id)
        doc_intelligence_dict = {
            d.get("document_id", ""): d.get("intelligence", d)
            for d in doc_intelligence
        }

        # KG probe — extract entities from query and find related docs/chunks
        kg_hints: Dict[str, Any] = {}
        if self.kg_query_service:
            try:
                query_entities = self.kg_query_service.extract_entities(query)
                if query_entities:
                    kg_result = self.kg_query_service.query(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        domain_hint=None,
                        entities=query_entities,
                    )
                    kg_hints = {
                        "target_doc_ids": kg_result.doc_ids,
                        "target_chunk_ids": kg_result.chunk_ids,
                        "entities": query_entities,
                    }
            except Exception as exc:
                logger.debug("KG probe failed (non-fatal): %s", exc)

        # Trim doc_intelligence for intent analysis (only needs summaries/topics, not full entities)
        trimmed_intel = []
        for d in doc_intelligence[:10]:  # cap at 10 docs
            trimmed = {
                "document_id": d.get("document_id", ""),
                "profile_id": d.get("profile_id", ""),
                "profile_name": d.get("profile_name", ""),
            }
            intel = d.get("intelligence") or {}
            trimmed["summary"] = (intel.get("summary") or "")[:200]
            trimmed["answerable_topics"] = (intel.get("answerable_topics") or [])[:5]
            trimmed["document_type"] = intel.get("document_type", "")
            trimmed_intel.append(trimmed)

        understanding = self._intent_analyzer.analyze(
            query, subscription_id, profile_id, trimmed_intel, conversation_history,
            kg_hints=kg_hints,
        )
        timing["understand_ms"] = round((time.monotonic() - t0) * 1000, 1)

        if understanding.is_conversational:
            return self._handle_conversational(query)

        # --- DOMAIN DISPATCH (only when explicitly requested) ---
        if agent_name:
            domain_result = self._domain_dispatcher.try_handle(
                query=understanding.resolved_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                evidence=[],
                doc_context={},
                agent_name=agent_name,
                document_id=document_id,
            )
            if domain_result is not None:
                domain_result.setdefault("metadata", {})["timing"] = timing
                return domain_result

        # --- RETRIEVE ---
        t0 = time.monotonic()
        profile_ids = self._resolve_profile_scope(understanding, profile_id)

        document_ids: Optional[List[str]] = None
        if document_id:
            document_ids = [document_id]
        elif understanding.relevant_documents:
            doc_ids = [
                d.get("document_id", "") for d in understanding.relevant_documents
                if d.get("document_id")
            ]
            if doc_ids:
                document_ids = doc_ids

        # Merge KG-hinted doc IDs into retrieval scope
        kg_doc_ids = kg_hints.get("target_doc_ids", [])
        if kg_doc_ids:
            if document_ids is None:
                document_ids = list(kg_doc_ids)
            else:
                existing = set(document_ids)
                for did in kg_doc_ids:
                    if did not in existing:
                        document_ids.append(did)

        # Enhance query for better retrieval coverage
        enhanced_query = self._enhance_query(
            understanding.resolved_query,
            understanding.task_type,
            doc_intelligence,
            understanding.entities,
        )

        retrieval_result = self._retriever.retrieve(
            enhanced_query,
            subscription_id,
            profile_ids,
            document_ids=document_ids,
        )

        # Dynamic evidence count by task type
        evidence_top_k = _EVIDENCE_TOP_K.get(understanding.task_type, 6)

        reranked = rerank_chunks(
            understanding.resolved_query,  # rerank against original query, not expanded
            retrieval_result.chunks,
            top_k=evidence_top_k,
            cross_encoder=self._cross_encoder,
        )
        evidence, doc_context = build_context(reranked, doc_intelligence_dict)

        # Inject KG entity relationships into doc_context for richer reasoning
        if kg_hints.get("target_doc_ids") and self.kg_query_service:
            try:
                kg_entities = kg_hints.get("entities", [])
                if kg_entities:
                    kg_context = [
                        f"{e.get('value', '')} ({e.get('type', '')})"
                        for e in (kg_entities if isinstance(kg_entities, list) else [])
                        if isinstance(e, dict) and e.get("value")
                    ]
                    if kg_context:
                        existing = doc_context.get("entities") or []
                        for kc in kg_context[:5]:
                            if kc not in existing:
                                existing.append(kc)
                        doc_context["entities"] = existing[:25]
            except Exception:
                logger.debug("KG context enrichment failed (non-fatal)")

        timing["retrieve_ms"] = round((time.monotonic() - t0) * 1000, 1)

        # --- POST-RETRIEVAL DOMAIN DISPATCH ---
        if agent_name:
            post_domain_result = self._domain_dispatcher.try_handle(
                query=understanding.resolved_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                evidence=evidence,
                doc_context=doc_context,
                agent_name=agent_name,
                document_id=document_id,
            )
            if post_domain_result is not None:
                post_domain_result.setdefault("metadata", {})["timing"] = timing
                return post_domain_result

        # --- REASON ---
        t0 = time.monotonic()
        # Thinking mode disabled: on T4 GPU, Qwen3 thinking tokens add
        # 10-20s latency per call with minimal quality gain.
        use_thinking = False

        # Single-GPU: skip sub-agent decomposition — parallel LLM calls
        # serialize on Ollama, causing timeouts. Use the reasoner directly.
        reason_result = self._reasoner.reason(
            query=understanding.resolved_query,
            task_type=understanding.task_type,
            output_format=understanding.output_format,
            evidence=evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
            use_thinking=use_thinking,
        )
        timing["reason_ms"] = round((time.monotonic() - t0) * 1000, 1)

        # --- COMPOSE ---
        metadata = {
            "usage": reason_result.usage,
            "timing": timing,
            "profiles_searched": retrieval_result.profiles_searched,
        }
        return compose_response(
            text=reason_result.text,
            evidence=evidence,
            grounded=reason_result.grounded,
            task_type=understanding.task_type,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Conversational handler
    # ------------------------------------------------------------------

    def _handle_conversational(self, query: str) -> Dict[str, Any]:
        """Return a short friendly response for conversational queries."""
        if _GREETING_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["greeting"]
        elif _FAREWELL_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["farewell"]
        elif _THANKS_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["thanks"]
        elif _META_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["meta"]
        else:
            text = _CONVERSATIONAL_RESPONSES["greeting"]

        return compose_response(
            text=text,
            evidence=[],
            grounded=False,
            task_type="conversational",
        )

    # ------------------------------------------------------------------
    # Complex query handler — spawns sub-agents
    # ------------------------------------------------------------------

    def _handle_complex(
        self,
        understanding: QueryUnderstanding,
        evidence: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> ReasonerResult:
        """Split evidence across sub-tasks and run sub-agents in parallel."""
        sub_tasks = (understanding.sub_tasks or [])[:self.MAX_SUBAGENTS]
        if not sub_tasks:
            return self._reasoner.reason(
                query=understanding.resolved_query,
                task_type=understanding.task_type,
                output_format=understanding.output_format,
                evidence=evidence,
                doc_context=doc_context,
                conversation_history=conversation_history,
                use_thinking=False,
            )

        # Partition evidence round-robin
        partitions: List[List[Dict[str, Any]]] = [[] for _ in sub_tasks]
        for idx, item in enumerate(evidence):
            partitions[idx % len(sub_tasks)].append(item)

        agents = [
            DynamicSubAgent(
                llm_gateway=self._llm,
                role=task,
                evidence=partition,
                doc_context=doc_context,
            )
            for task, partition in zip(sub_tasks, partitions)
        ]

        results = []
        max_workers = min(len(agents), 3)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(agent.execute): agent for agent in agents}
            for future in as_completed(futures, timeout=self.SUBAGENT_TIMEOUT):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning("Sub-agent future failed: %s", exc)

        synthesis_evidence = []
        for idx, r in enumerate(results, start=1):
            if r.success and r.text:
                synthesis_evidence.append({
                    "source_index": idx,
                    "source_name": f"sub-agent: {r.task[:50]}",
                    "section": r.task,
                    "page": 0,
                    "text": r.text,
                    "score": 1.0,
                    "document_id": "",
                    "profile_id": "",
                    "chunk_id": f"subagent-{idx}",
                })

        return self._reasoner.reason(
            query=understanding.resolved_query,
            task_type=understanding.task_type,
            output_format=understanding.output_format,
            evidence=synthesis_evidence if synthesis_evidence else evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
            use_thinking=False,
        )

    # ------------------------------------------------------------------
    # Profile scope resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_profile_scope(
        understanding: QueryUnderstanding,
        requesting_profile_id: str,
    ) -> List[str]:
        """Determine which profile IDs to search."""
        profile_ids = {requesting_profile_id}
        if understanding.cross_profile and understanding.relevant_documents:
            for doc in understanding.relevant_documents:
                pid = doc.get("profile_id")
                if pid:
                    profile_ids.add(pid)
        return list(profile_ids)

    # ------------------------------------------------------------------
    # Query enhancement for better retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_query(
        query: str,
        task_type: str,
        doc_intelligence: List[Dict[str, Any]],
        entities: List[str],
    ) -> str:
        """Expand the query with task synonyms, entities, and topic keywords.

        Adds relevant terms to improve dense retrieval recall without changing
        the semantic meaning. The original query always leads.
        """
        expansion_terms: Set[str] = set()
        query_lower = query.lower()

        # 1. Add task-type synonyms that aren't already in the query
        synonyms = _TASK_SYNONYMS.get(task_type, [])
        for syn in synonyms:
            if syn not in query_lower:
                expansion_terms.add(syn)

        # 2. Add entities from intent analysis (already extracted by LLM)
        for entity in entities[:5]:
            if entity.lower() not in query_lower:
                expansion_terms.add(entity)

        # 3. Mine doc_intelligence for matching topic keywords
        query_words = set(query_lower.split()) - _STOPWORDS
        for doc in doc_intelligence:
            intel = doc.get("intelligence") or {}
            topics: List[str] = intel.get("answerable_topics") or []
            for topic in topics:
                topic_words = set(topic.lower().split()) - _STOPWORDS
                overlap = query_words & topic_words
                if len(overlap) >= 2:
                    # This topic is relevant — add non-overlapping keywords
                    new_words = topic_words - query_words - _STOPWORDS
                    for w in list(new_words)[:3]:
                        if len(w) > 2:
                            expansion_terms.add(w)

            # 4. Add matching entity names from doc intelligence
            doc_entities = intel.get("entities") or []
            for ent in doc_entities[:10]:
                ent_name = ent.get("name", str(ent)) if isinstance(ent, dict) else str(ent)
                ent_lower = ent_name.lower()
                ent_words = set(ent_lower.split()) - _STOPWORDS
                if ent_words & query_words and ent_lower not in query_lower:
                    expansion_terms.add(ent_name)

        # Cap expansion to avoid noise
        expansion_list = list(expansion_terms)[:8]
        if not expansion_list:
            return query

        enhanced = f"{query} {' '.join(expansion_list)}"
        logger.debug("Query enhanced: '%s' -> '%s'", query[:60], enhanced[:120])
        return enhanced

    # ------------------------------------------------------------------
    # Document intelligence loader
    # ------------------------------------------------------------------

    def _load_doc_intelligence(self, subscription_id: str) -> List[Dict[str, Any]]:
        """Load document intelligence metadata from MongoDB."""
        try:
            cursor = self._mongodb.find(
                {"subscription_id": subscription_id, "intelligence": {"$exists": True}},
                {
                    "document_id": 1,
                    "profile_id": 1,
                    "profile_name": 1,
                    "intelligence.summary": 1,
                    "intelligence.entities": 1,
                    "intelligence.answerable_topics": 1,
                    "intelligence.key_facts": 1,
                    "intelligence.document_type": 1,
                },
            )
            return list(cursor)
        except Exception:
            logger.exception("Failed to load doc intelligence for subscription=%s", subscription_id)
            return []
