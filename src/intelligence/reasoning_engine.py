"""
ReasoningEngine — Unified intelligence orchestrator for DocWain.

Orchestrates the full pipeline:
    Conversational Check → UNDERSTAND → RETRIEVE → GENERATE → VERIFY → FORMAT

Uses a single Qwen3-14B-AWQ model via vLLM with native thinking mode
for complex queries.  All domain knowledge is learned from documents,
not hardcoded.
"""
from __future__ import annotations

import contextvars
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures (kept for backward compat with tests)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """A single piece of evidence from retrieval."""
    text: str
    score: float
    source_name: str
    page: Optional[str] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Unified intelligence engine — orchestrates UNDERSTAND → RETRIEVE →
    GENERATE → VERIFY → FORMAT.

    One public method: ``answer(query, ...) → response dict``.
    Response dict shape is unchanged from the old engine.
    """

    def __init__(
        self,
        *,
        llm_client: Any,
        thinking_client: Optional[Any] = None,
        qdrant_client: Any,
        embedder: Any,
        collection_name: str,
        subscription_id: str,
        profile_id: str,
        max_iterations: int = 3,
    ):
        self._llm = llm_client
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._collection = collection_name
        self._subscription_id = subscription_id
        self._profile_id = profile_id
        self._max_iterations = max_iterations

        # Lazy-init intelligence components
        self._understanding = None
        self._generator = None
        self._verifier = None
        self._verification_gate = None
        self._format_enforcer = None
        self._conversational = None
        self._lightweight_intent = None

    # ------------------------------------------------------------------
    # Lazy component initialization
    # ------------------------------------------------------------------

    def _get_understanding(self):
        if self._understanding is None:
            from src.intelligence.understand import QueryUnderstanding
            self._understanding = QueryUnderstanding(self._llm)
        return self._understanding

    def _get_generator(self):
        if self._generator is None:
            from src.intelligence.generator import IntelligentGenerator
            self._generator = IntelligentGenerator(self._llm)
        return self._generator

    def _get_verifier(self):
        if self._verifier is None:
            from src.intelligence.verifier import Verifier
            self._verifier = Verifier(self._llm)
        return self._verifier

    def _get_verification_gate(self):
        if self._verification_gate is None:
            from src.intelligence.verifier import VerificationGate
            self._verification_gate = VerificationGate()
        return self._verification_gate

    def _get_format_enforcer(self):
        if self._format_enforcer is None:
            from src.intelligence.format_enforcer import FormatEnforcer
            self._format_enforcer = FormatEnforcer()
        return self._format_enforcer

    def _get_conversational(self):
        if self._conversational is None:
            try:
                from src.intelligence.conversational import ConversationalDetector
                self._conversational = ConversationalDetector(self._embedder)
            except Exception:
                self._conversational = False  # sentinel: init failed
        return self._conversational if self._conversational is not False else None

    def _get_lightweight_intent(self):
        if self._lightweight_intent is None:
            try:
                from src.intelligence.lightweight_intent import LightweightIntentDetector
                self._lightweight_intent = LightweightIntentDetector(self._embedder)
            except Exception:
                self._lightweight_intent = False
        return self._lightweight_intent if self._lightweight_intent is not False else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        *,
        profile_context: str = "",
        conversation_history: str = "",
        persona_prompt: str = "",
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Main entry point.  Adaptively routes through the intelligence pipeline.

        Returns a response dict compatible with DocWain's answer format:
        ``{"response", "sources", "grounded", "context_found", "metadata"}``.
        """
        t0 = time.perf_counter()

        # ----------------------------------------------------------
        # Step 0: Conversational intent check (~5ms)
        # ----------------------------------------------------------
        conv_detector = self._get_conversational()
        if conv_detector:
            try:
                conv_result = conv_detector.detect(query)
                if conv_result is not None:
                    intent_type, response_text = conv_result
                    elapsed = (time.perf_counter() - t0) * 1000
                    logger.info("Conversational intent detected: %s (%.0fms)", intent_type, elapsed)
                    return {
                        "response": response_text,
                        "sources": [],
                        "grounded": True,
                        "context_found": False,
                        "metadata": {
                            "engine": "reasoning_engine",
                            "fast_path": True,
                            "intent": intent_type,
                            "complexity": "simple",
                            "conversational": True,
                            "timing_ms": {"total": round(elapsed, 1)},
                        },
                    }
            except Exception as exc:
                logger.debug("Conversational detection failed: %s", exc)

        # ----------------------------------------------------------
        # Step 1: Decide trivially simple vs needs UNDERSTAND
        # ----------------------------------------------------------
        from src.intelligence.understand import QueryUnderstanding

        history_list = self._parse_conversation_history(conversation_history)
        is_trivial = QueryUnderstanding.is_trivially_simple(query, history_list)

        if is_trivial:
            return self._trivial_path(
                query,
                profile_context=profile_context,
                conversation_history=conversation_history,
                t0=t0,
            )

        # ----------------------------------------------------------
        # Step 2: Full intelligent path
        # ----------------------------------------------------------
        return self._intelligent_path(
            query,
            profile_context=profile_context,
            conversation_history=conversation_history,
            history_list=history_list,
            task_type=task_type,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Trivial path — 1 LLM call (GENERATE only)
    # ------------------------------------------------------------------

    def _trivial_path(
        self,
        query: str,
        *,
        profile_context: str = "",
        conversation_history: str = "",
        t0: float,
    ) -> Dict[str, Any]:
        """Fast path for trivially simple queries — direct retrieve + generate."""
        # Quick intent detection
        intent_detector = self._get_lightweight_intent()
        intent = "extract"
        if intent_detector:
            try:
                intent, _ = intent_detector.detect(query)
            except Exception:
                pass

        # RETRIEVE
        evidence = self._search_simple(query)
        elapsed_retrieval = time.perf_counter() - t0

        if not evidence:
            return self._no_evidence_response(query, elapsed_retrieval)

        # GENERATE — single call, no thinking mode
        t1 = time.perf_counter()
        answer_text, sources = self._generate_simple(query, evidence, intent, conversation_history)
        elapsed_generate = time.perf_counter() - t1

        # FORMAT
        enforcer = self._get_format_enforcer()
        answer_text = enforcer.enforce(answer_text, "prose")

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ReasoningEngine trivial_path: %.0fms (retrieval=%.0fms, generate=%.0fms) evidence=%d",
            total_ms, elapsed_retrieval * 1000, elapsed_generate * 1000, len(evidence),
        )

        return {
            "response": answer_text,
            "sources": sources,
            "grounded": True,
            "context_found": True,
            "metadata": {
                "engine": "reasoning_engine",
                "fast_path": True,
                "intent": intent,
                "complexity": "simple",
                "iterations": 1,
                "evidence_count": len(evidence),
                "confidence": sum(e.score for e in evidence[:5]) / min(len(evidence), 5),
                "gaps": [],
                "verification": {"ok": True, "skipped": True, "unsupported": []},
                "timing_ms": {
                    "total": round(total_ms, 1),
                    "retrieval": round(elapsed_retrieval * 1000, 1),
                    "generate": round(elapsed_generate * 1000, 1),
                    "verify": 0.0,
                },
            },
        }

    # ------------------------------------------------------------------
    # Intelligent path — UNDERSTAND → RETRIEVE → GENERATE → VERIFY → FORMAT
    # ------------------------------------------------------------------

    def _intelligent_path(
        self,
        query: str,
        *,
        profile_context: str = "",
        conversation_history: str = "",
        history_list: List[dict],
        task_type: str = "",
        t0: float,
    ) -> Dict[str, Any]:
        """Full intelligent pipeline with UNDERSTAND → GENERATE → VERIFY."""

        # --- UNDERSTAND (1 LLM call) ---
        t_understand = time.perf_counter()
        understanding = self._understand(query, history_list, profile_context)
        elapsed_understand = time.perf_counter() - t_understand

        logger.info(
            "UNDERSTAND: intent=%s format=%s complexity=%s thinking=%s subs=%d (%.0fms)",
            understanding.primary_intent, understanding.output_format,
            understanding.complexity, understanding.thinking_required,
            len(understanding.sub_intents), elapsed_understand * 1000,
        )

        # Check if clarification needed
        if understanding.needs_clarification and understanding.clarification_question:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "response": understanding.clarification_question,
                "sources": [],
                "grounded": True,
                "context_found": False,
                "metadata": {
                    "engine": "reasoning_engine",
                    "fast_path": False,
                    "intent": understanding.primary_intent,
                    "complexity": understanding.complexity,
                    "needs_clarification": True,
                    "timing_ms": {"total": round(elapsed, 1)},
                },
            }

        # --- RETRIEVE (no LLM) ---
        t_retrieve = time.perf_counter()
        if understanding.sub_intents and len(understanding.sub_intents) > 1:
            # Multi-part: parallel retrieval per sub-intent
            evidence = self._search_multi(understanding)
        else:
            evidence = self._search_for_query(understanding.resolved_query or query)
        elapsed_retrieval = time.perf_counter() - t_retrieve

        if not evidence:
            return self._no_evidence_response(query, time.perf_counter() - t0)

        # --- GENERATE (1 LLM call, with thinking mode if needed) ---
        t_generate = time.perf_counter()
        generator = self._get_generator()
        evidence_dicts = [self._evidence_to_dict(e) for e in evidence[:16]]

        gen_result = generator.generate(
            query=understanding.resolved_query or query,
            evidence=evidence_dicts,
            understanding=understanding,
            conversation_history=conversation_history,
        )
        answer_text = gen_result.text
        elapsed_generate = time.perf_counter() - t_generate

        # Build sources list
        sources = self._build_sources(evidence[:16])

        # --- VERIFY (conditional, 0-1 LLM call) ---
        t_verify = time.perf_counter()
        gate = self._get_verification_gate()
        verification_result = {"ok": True, "skipped": True, "unsupported": []}

        if gate.needs_verification(evidence_dicts, answer_text, understanding):
            verifier = self._get_verifier()
            v_result = verifier.verify(answer_text, evidence_dicts)
            verification_result = {
                "ok": v_result.supported,
                "skipped": False,
                "unsupported": v_result.unsupported_claims,
            }

            if not v_result.supported and v_result.unsupported_claims:
                # Try to fix
                fixed = verifier.handle_failure(answer_text, v_result)
                if fixed:
                    answer_text = fixed
                else:
                    # Re-generate with strict mode
                    logger.info("Verification failed, re-generating with strict mode")
                    gen_result = generator.generate(
                        query=understanding.resolved_query or query,
                        evidence=evidence_dicts,
                        understanding=understanding,
                        conversation_history=conversation_history,
                    )
                    answer_text = gen_result.text
        elapsed_verify = time.perf_counter() - t_verify

        # --- FORMAT ENFORCEMENT (no LLM) ---
        enforcer = self._get_format_enforcer()
        answer_text = enforcer.enforce(answer_text, understanding.output_format)

        total_ms = (time.perf_counter() - t0) * 1000
        avg_score = sum(e.score for e in evidence[:5]) / min(len(evidence), 5) if evidence else 0.0

        logger.info(
            "ReasoningEngine intelligent_path: %.0fms (understand=%.0fms, retrieval=%.0fms, "
            "generate=%.0fms, verify=%.0fms) evidence=%d grounded=%s",
            total_ms, elapsed_understand * 1000, elapsed_retrieval * 1000,
            elapsed_generate * 1000, elapsed_verify * 1000,
            len(evidence), verification_result["ok"],
        )

        return {
            "response": answer_text,
            "sources": sources,
            "grounded": verification_result["ok"],
            "context_found": True,
            "metadata": {
                "engine": "reasoning_engine",
                "fast_path": False,
                "intent": understanding.primary_intent,
                "complexity": understanding.complexity,
                "iterations": 1,
                "evidence_count": len(evidence),
                "confidence": avg_score,
                "gaps": [],
                "verification": verification_result,
                "output_format": understanding.output_format,
                "thinking_used": understanding.thinking_required,
                "timing_ms": {
                    "total": round(total_ms, 1),
                    "understand": round(elapsed_understand * 1000, 1),
                    "retrieval": round(elapsed_retrieval * 1000, 1),
                    "generate": round(elapsed_generate * 1000, 1),
                    "verify": round(elapsed_verify * 1000, 1),
                },
            },
        }

    # ------------------------------------------------------------------
    # UNDERSTAND — query analysis via LLM
    # ------------------------------------------------------------------

    def _understand(self, query: str, history_list: list, profile_context: str):
        """Run QueryUnderstanding to analyze the query."""
        understanding = self._get_understanding()

        # Build domain context from profile
        domain_context = self._build_domain_context(profile_context)

        return understanding.understand(query, history_list, domain_context)

    def _build_domain_context(self, profile_context: str) -> dict:
        """Extract domain info from profile context string and any document profiles in Qdrant."""
        # Parse the profile_context string for domain info
        domain_labels = set()
        terminology = set()
        field_types = set()

        if profile_context:
            for line in profile_context.split("\n"):
                line = line.strip()
                if line.startswith("- ") and "(domain:" in line:
                    # Extract domain from lines like "- resume.pdf (domain: hr)"
                    domain_match = re.search(r"\(domain:\s*([^)]+)\)", line)
                    if domain_match:
                        domain_labels.add(domain_match.group(1).strip())

        # Try to get document profiles from a quick Qdrant sample
        try:
            from src.api.vector_store import build_qdrant_filter
            qdrant_filter = build_qdrant_filter(
                subscription_id=self._subscription_id,
                profile_id=self._profile_id,
            )
            # Get a small sample of chunks with their domain profiles
            sample_vector = self._embedder.encode(["document content"], normalize_embeddings=True)[0]
            response = self._qdrant.query_points(
                collection_name=self._collection,
                query=sample_vector.tolist(),
                using="content_vector",
                query_filter=qdrant_filter,
                limit=5,
                with_payload=True,
            )
            for point in response.points:
                payload = point.payload or {}
                profile = payload.get("domain_profile", {})
                if profile:
                    if profile.get("domain"):
                        domain_labels.add(profile["domain"])
                    terminology.update(profile.get("key_terminology", []))
                    field_types.update(profile.get("field_types", []))
        except Exception:
            pass  # Domain context is optional, not critical

        return {
            "domain_labels": list(domain_labels) if domain_labels else ["general"],
            "key_terminology": list(terminology)[:20],
            "field_types": list(field_types)[:15],
            "structure_patterns": [],
        }

    # ------------------------------------------------------------------
    # SEARCH — evidence retrieval from Qdrant
    # ------------------------------------------------------------------

    def _search_simple(self, query: str) -> List[EvidenceItem]:
        """Simple single-query search."""
        return self._search_for_query(query, top_k=20)

    def _search_for_query(self, query: str, top_k: int = 30) -> List[EvidenceItem]:
        """Execute a semantic search against Qdrant."""
        from src.api.vector_store import build_qdrant_filter

        try:
            query_vector = self._embedder.encode([query], normalize_embeddings=True)[0]
        except Exception as exc:
            logger.debug("Embedding failed: %s", exc)
            return []

        try:
            qdrant_filter = build_qdrant_filter(
                subscription_id=self._subscription_id,
                profile_id=self._profile_id,
            )
        except Exception:
            qdrant_filter = None

        try:
            response = self._qdrant.query_points(
                collection_name=self._collection,
                query=query_vector.tolist(),
                using="content_vector",
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.debug("Qdrant search failed: %s", exc)
            return []

        return self._points_to_evidence(response.points)

    def _search_multi(self, understanding) -> List[EvidenceItem]:
        """Parallel search for multi-part queries."""
        queries = [understanding.resolved_query or ""]
        for sub in understanding.sub_intents:
            if sub.target:
                queries.append(f"{sub.target} {sub.scope}" if sub.scope else sub.target)

        all_evidence: List[EvidenceItem] = []
        seen_chunks = set()

        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = {
                executor.submit(ctx.run, self._search_for_query, q, 15): q
                for q in queries
            }
            for future in as_completed(futures):
                try:
                    items = future.result()
                    for item in items:
                        if item.chunk_id not in seen_chunks:
                            seen_chunks.add(item.chunk_id)
                            all_evidence.append(item)
                except Exception as exc:
                    logger.debug("Multi-search failed for query: %s", exc)

        all_evidence.sort(key=lambda e: -e.score)
        return all_evidence[:30]

    def _points_to_evidence(self, points) -> List[EvidenceItem]:
        """Convert Qdrant points to EvidenceItem list."""
        items = []
        for point in points:
            payload = point.payload or {}
            text = (
                payload.get("canonical_text")
                or payload.get("embedding_text")
                or payload.get("content")
                or payload.get("text")
                or ""
            )
            if not text.strip():
                continue
            score = float(getattr(point, "score", 0.0))
            if score < 0.20:
                continue

            source_name = (
                payload.get("source_name")
                or payload.get("document_name")
                or payload.get("file_name")
                or "Document"
            )
            items.append(EvidenceItem(
                text=text.strip(),
                score=score,
                source_name=source_name,
                page=str(payload.get("page") or payload.get("page_number") or ""),
                section=payload.get("section_title") or payload.get("section") or "",
                chunk_id=str(payload.get("chunk_id") or payload.get("id") or getattr(point, "id", "")),
                document_id=str(payload.get("document_id") or ""),
                metadata=payload,
            ))

        items.sort(key=lambda e: -e.score)
        return items

    # ------------------------------------------------------------------
    # GENERATE — simple path (no UNDERSTAND)
    # ------------------------------------------------------------------

    def _generate_simple(
        self,
        query: str,
        evidence: List[EvidenceItem],
        intent: str,
        conversation_history: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Direct generation for trivially simple queries."""
        sources = []
        context_parts = []
        for i, item in enumerate(evidence[:10], 1):
            doc_name = os.path.basename(item.source_name).rsplit(".", 1)[0] if item.source_name else "Document"
            header = f"Document: {doc_name}"
            if item.section:
                header += f", Section: {item.section}"
            if item.page:
                header += f", Page: {item.page}"

            context_parts.append(f"[SOURCE-{i}] {header}\n{item.text}\n")
            sources.append({
                "source_id": i,
                "source_name": doc_name,
                "section": item.section or None,
                "page": item.page or None,
                "excerpt": item.text[:400],
                "score": round(item.score, 4),
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
            })

        context_text = "\n".join(context_parts)

        prompt = f"""You are DocWain, a document intelligence assistant.

DOCUMENT CONTEXT:
{context_text}

RULES:
1. Answer ONLY from the document context above.
2. Cite [SOURCE-N] inline after every factual claim.
3. If the information is not in the context, say so.
4. Be concise and direct.

QUESTION: {query}"""

        try:
            if hasattr(self._llm, "generate_with_metadata"):
                answer_text, _ = self._llm.generate_with_metadata(
                    prompt, options={"temperature": 0.2, "max_tokens": 1024}
                )
            else:
                answer_text = self._llm.generate(prompt)
        except Exception as exc:
            logger.error("Simple generation failed: %s", exc)
            answer_text = self._evidence_summary_fallback(evidence, query)

        return answer_text.strip(), sources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_to_dict(item: EvidenceItem) -> dict:
        return {
            "text": item.text,
            "source_name": item.source_name,
            "page": item.page,
            "section": item.section,
            "score": item.score,
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
        }

    @staticmethod
    def _build_sources(evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        sources = []
        for i, item in enumerate(evidence, 1):
            doc_name = os.path.basename(item.source_name).rsplit(".", 1)[0] if item.source_name else "Document"
            sources.append({
                "source_id": i,
                "source_name": doc_name,
                "section": item.section or None,
                "page": item.page or None,
                "excerpt": item.text[:400],
                "score": round(item.score, 4),
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
            })
        return sources

    @staticmethod
    def _parse_conversation_history(history: str) -> List[dict]:
        """Parse conversation history string into list of dicts."""
        if not history:
            return []
        turns = []
        for line in history.strip().split("\n"):
            line = line.strip()
            if line.startswith("Previous queries:"):
                queries = line.replace("Previous queries:", "").strip().split(" | ")
                for q in queries:
                    if q.strip():
                        turns.append({"role": "user", "content": q.strip()})
            elif line:
                turns.append({"role": "context", "content": line})
        return turns[-6:]  # last 3 turns

    def _evidence_summary_fallback(self, evidence: List[EvidenceItem], query: str) -> str:
        """Build a simple evidence summary when generation fails."""
        lines = [f"Based on the available documents for your query about '{query[:100]}':"]
        for i, item in enumerate(evidence[:5], 1):
            excerpt = item.text[:200].strip()
            lines.append(f"\n[SOURCE-{i}] From {item.source_name}: \"{excerpt}...\"")
        lines.append("\n\nPlease refer to the source documents for complete details.")
        return "\n".join(lines)

    def _no_evidence_response(self, query: str, elapsed: float) -> Dict[str, Any]:
        return {
            "response": (
                "I searched the available documents but couldn't find information "
                "relevant to your question. Please verify that the relevant documents "
                "have been uploaded to this profile."
            ),
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {
                "engine": "reasoning_engine",
                "intent": "unknown",
                "evidence_count": 0,
                "verification": {"ok": False, "skipped": True, "unsupported": []},
                "timing_ms": {"total": round(elapsed * 1000, 1)},
            },
        }
