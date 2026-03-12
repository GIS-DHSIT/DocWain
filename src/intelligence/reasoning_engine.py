"""
ReasoningEngine — Unified intelligence algorithm for DocWain.

Replaces scattered heuristic modules with a single LLM-driven loop:
    THINK → SEARCH → REASON → (loop if gaps) → GENERATE → VERIFY

Uses the thinking model for fast reasoning steps (~100-200ms each)
and the main model for final generation (~800ms).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchAction:
    """A single targeted search to execute."""
    query: str
    strategy: str = "semantic"          # semantic | keyword | entity_scoped
    target_sections: List[str] = field(default_factory=list)
    target_doc_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SearchPlan:
    """LLM-generated plan for what evidence to retrieve."""
    intent: str                          # factual | comparison | summary | extraction | reasoning | procedural
    complexity: str                      # simple | moderate | complex
    actions: List[SearchAction] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    reasoning: str = ""


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


@dataclass
class EvidenceAssessment:
    """LLM evaluation of retrieved evidence."""
    sufficient: bool
    confidence: float                    # 0.0 to 1.0
    gaps: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class Verification:
    """LLM verification of the generated answer."""
    ok: bool
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Prompts — kept minimal and structured for fast JSON responses
# ---------------------------------------------------------------------------

_THINK_PROMPT = """\
You are analyzing a user query to plan evidence retrieval from a document store.

USER QUERY: {query}

PROFILE CONTEXT (available documents):
{profile_context}

{prior_evidence_block}

TASK: Produce a JSON search plan. Think about:
1. What is the user actually asking? (intent)
2. How complex is this? (simple=1 search, moderate=2, complex=3)
3. What specific searches should we run to find the answer?
4. What entities/names/terms are critical?

Respond with ONLY valid JSON:
{{
  "intent": "factual|comparison|summary|extraction|reasoning|procedural",
  "complexity": "simple|moderate|complex",
  "actions": [
    {{"query": "search text", "strategy": "semantic|keyword|entity_scoped", "target_sections": [], "target_doc_ids": [], "reason": "why this search"}}
  ],
  "key_entities": ["entity1", "entity2"],
  "reasoning": "brief explanation of plan"
}}"""

_REASON_PROMPT = """\
You are evaluating retrieved evidence to determine if it's sufficient to answer a query.

USER QUERY: {query}
SEARCH PLAN INTENT: {intent}

EVIDENCE RETRIEVED ({count} chunks):
{evidence_text}

TASK: Assess this evidence. Consider:
1. Does it contain enough information to fully answer the query?
2. Are there gaps — things the user asked about that aren't covered?
3. Are there contradictions between evidence pieces?
4. What are the key findings?

Respond with ONLY valid JSON:
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "gaps": ["gap1", "gap2"],
  "contradictions": ["contradiction1"],
  "key_findings": ["finding1", "finding2"],
  "reasoning": "brief assessment"
}}"""

_VERIFY_PROMPT = """\
You are verifying an answer against evidence for factual accuracy.

ANSWER:
{answer}

EVIDENCE:
{evidence_text}

TASK: Check every factual claim in the answer against the evidence.
Flag any claim that is NOT directly supported by the evidence.

Respond with ONLY valid JSON:
{{
  "ok": true/false,
  "unsupported_claims": ["claim that has no evidence support"],
  "reasoning": "brief verification summary"
}}"""


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Unified intelligence engine.

    One public method: answer(query, ...) → response dict.
    Uses thinking model for fast reasoning, main model for generation.
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
        self._thinker = thinking_client or llm_client
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._collection = collection_name
        self._subscription_id = subscription_id
        self._profile_id = profile_id
        self._max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _is_simple_query(query: str, task_type: str = "") -> bool:
        """Decide fast vs full path using query characteristics and task type."""
        if task_type in {"compare", "summarize", "rank", "generate", "reasoning"}:
            return False
        if task_type in {"extract", "list", "qa"}:
            return len(query.split()) <= 20
        word_count = len(query.split())
        if word_count <= 12:
            return True
        return False

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
        Main entry point. Classifies query and dispatches to fast or full path.

        Fast path (simple queries): SEARCH → GENERATE → optional VERIFY (2 calls)
        Full path (complex queries): THINK → SEARCH → REASON → GENERATE → VERIFY (4-5 calls)

        Returns a response dict compatible with DocWain's answer format.
        """
        t0 = time.perf_counter()
        fast_path = self._is_simple_query(query, task_type)
        if fast_path:
            return self._fast_path(
                query,
                profile_context=profile_context,
                conversation_history=conversation_history,
                persona_prompt=persona_prompt,
                t0=t0,
            )
        return self._full_path(
            query,
            profile_context=profile_context,
            conversation_history=conversation_history,
            persona_prompt=persona_prompt,
            task_type=task_type,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Fast path — simple queries: SEARCH → GENERATE → optional VERIFY
    # ------------------------------------------------------------------

    def _fast_path(
        self,
        query: str,
        *,
        profile_context: str = "",
        conversation_history: str = "",
        persona_prompt: str = "",
        t0: float,
    ) -> Dict[str, Any]:
        """Fast path for simple queries — 2 LLM calls (SEARCH → GENERATE), optional VERIFY."""
        # Build a simple plan: single semantic search with the raw query
        plan = SearchPlan(
            intent="factual",
            complexity="simple",
            actions=[SearchAction(query=query, strategy="semantic")],
            key_entities=[],
            reasoning="fast_path: direct semantic search",
        )

        # SEARCH
        all_evidence = self._search(plan)
        elapsed_retrieval = time.perf_counter() - t0

        if not all_evidence:
            resp = self._no_evidence_response(query, elapsed_retrieval)
            resp["metadata"]["fast_path"] = True
            return resp

        # GENERATE
        t1 = time.perf_counter()
        answer_text, sources = self._generate(
            query, all_evidence, None, plan,
            conversation_history=conversation_history,
            persona_prompt=persona_prompt,
        )
        elapsed_generate = time.perf_counter() - t1

        # Adaptive VERIFY — skip when confidence signals are strong
        citation_count = len(sources)
        top_evidence_score = all_evidence[0].score if all_evidence else 0.0
        skip_verify = citation_count >= 3 and top_evidence_score >= 0.8

        elapsed_verify = 0.0
        if skip_verify:
            verification = Verification(ok=True, reasoning="skipped: high confidence fast path")
        else:
            t2 = time.perf_counter()
            verification = self._verify(answer_text, all_evidence)
            elapsed_verify = time.perf_counter() - t2
            if not verification.ok and verification.unsupported_claims:
                answer_text = self._strip_unsupported(answer_text, verification)

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ReasoningEngine fast_path completed: %.0fms (retrieval=%.0fms, generate=%.0fms, verify=%.0fms) "
            "evidence=%d grounded=%s skip_verify=%s",
            total_ms, elapsed_retrieval * 1000, elapsed_generate * 1000,
            elapsed_verify * 1000, len(all_evidence), verification.ok, skip_verify,
        )

        return {
            "response": answer_text,
            "sources": sources,
            "grounded": verification.ok,
            "context_found": True,
            "metadata": {
                "engine": "reasoning_engine",
                "fast_path": True,
                "intent": "factual",
                "complexity": "simple",
                "iterations": 1,
                "evidence_count": len(all_evidence),
                "confidence": top_evidence_score,
                "gaps": [],
                "verification": {
                    "ok": verification.ok,
                    "skipped": skip_verify,
                    "unsupported": verification.unsupported_claims,
                },
                "timing_ms": {
                    "total": round(total_ms, 1),
                    "retrieval": round(elapsed_retrieval * 1000, 1),
                    "generate": round(elapsed_generate * 1000, 1),
                    "verify": round(elapsed_verify * 1000, 1),
                },
            },
        }

    # ------------------------------------------------------------------
    # Full path — complex queries: THINK → SEARCH → REASON → GENERATE → VERIFY
    # ------------------------------------------------------------------

    def _full_path(
        self,
        query: str,
        *,
        profile_context: str = "",
        conversation_history: str = "",
        persona_prompt: str = "",
        task_type: str = "",
        t0: float,
    ) -> Dict[str, Any]:
        """Full path for complex queries — THINK → SEARCH → REASON → GENERATE → VERIFY."""
        all_evidence: List[EvidenceItem] = []

        # --- Iterative retrieval loop ---
        plan: Optional[SearchPlan] = None
        assessment: Optional[EvidenceAssessment] = None

        for iteration in range(self._max_iterations):
            # THINK
            prior_block = ""
            if iteration > 0 and assessment:
                prior_block = self._format_prior_evidence_block(all_evidence, assessment)

            plan = self._think(query, profile_context, prior_block)
            logger.debug(
                "ReasoningEngine iteration=%d intent=%s complexity=%s actions=%d",
                iteration, plan.intent, plan.complexity, len(plan.actions),
            )

            # SEARCH — parallel execution of all search actions
            new_evidence = self._search(plan)
            all_evidence = self._merge_evidence(all_evidence, new_evidence)

            if not all_evidence:
                logger.debug("ReasoningEngine: no evidence found after iteration %d", iteration)
                if iteration == 0:
                    continue  # retry with broader search
                break

            # REASON
            assessment = self._reason(query, plan, all_evidence)
            logger.debug(
                "ReasoningEngine: sufficient=%s confidence=%.2f gaps=%d",
                assessment.sufficient, assessment.confidence, len(assessment.gaps),
            )

            # DECIDE — exit if sufficient or simple query
            if assessment.sufficient or plan.complexity == "simple":
                break
            if not assessment.gaps:
                break  # no specific gaps to address

        elapsed_retrieval = time.perf_counter() - t0

        if not all_evidence:
            resp = self._no_evidence_response(query, elapsed_retrieval)
            resp["metadata"]["fast_path"] = False
            return resp

        # GENERATE — main model, one call
        t1 = time.perf_counter()
        answer_text, sources = self._generate(
            query, all_evidence, assessment, plan,
            conversation_history=conversation_history,
            persona_prompt=persona_prompt,
        )
        elapsed_generate = time.perf_counter() - t1

        # Adaptive VERIFY — skip when confidence is high and citations are strong
        citation_count = len(sources)
        skip_verify = (
            assessment is not None
            and citation_count >= 3
            and assessment.confidence >= 0.8
        )

        elapsed_verify = 0.0
        if skip_verify:
            verification = Verification(ok=True, reasoning="skipped: high confidence full path")
        else:
            t2 = time.perf_counter()
            verification = self._verify(answer_text, all_evidence)
            elapsed_verify = time.perf_counter() - t2
            if not verification.ok and verification.unsupported_claims:
                answer_text = self._strip_unsupported(answer_text, verification)

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ReasoningEngine full_path completed: %.0fms (retrieval=%.0fms, generate=%.0fms, verify=%.0fms) "
            "evidence=%d grounded=%s skip_verify=%s",
            total_ms, elapsed_retrieval * 1000, elapsed_generate * 1000,
            elapsed_verify * 1000, len(all_evidence), verification.ok, skip_verify,
        )

        return {
            "response": answer_text,
            "sources": sources,
            "grounded": verification.ok,
            "context_found": True,
            "metadata": {
                "engine": "reasoning_engine",
                "fast_path": False,
                "intent": plan.intent if plan else "unknown",
                "complexity": plan.complexity if plan else "unknown",
                "iterations": min(iteration + 1, self._max_iterations),
                "evidence_count": len(all_evidence),
                "confidence": assessment.confidence if assessment else 0.0,
                "gaps": assessment.gaps if assessment else [],
                "verification": {
                    "ok": verification.ok,
                    "skipped": skip_verify,
                    "unsupported": verification.unsupported_claims,
                },
                "timing_ms": {
                    "total": round(total_ms, 1),
                    "retrieval": round(elapsed_retrieval * 1000, 1),
                    "generate": round(elapsed_generate * 1000, 1),
                    "verify": round(elapsed_verify * 1000, 1),
                },
            },
        }

    # ------------------------------------------------------------------
    # THINK — LLM analyzes query, produces search plan
    # ------------------------------------------------------------------

    def _think(
        self,
        query: str,
        profile_context: str,
        prior_evidence_block: str,
    ) -> SearchPlan:
        prompt = _THINK_PROMPT.format(
            query=query,
            profile_context=profile_context or "No profile context available.",
            prior_evidence_block=prior_evidence_block,
        )
        raw = self._call_thinker(prompt, max_tokens=512)
        parsed = self._parse_json(raw)
        if not parsed:
            # Fallback: single semantic search with the original query
            return SearchPlan(
                intent="factual",
                complexity="simple",
                actions=[SearchAction(query=query, strategy="semantic")],
                key_entities=[],
                reasoning="fallback: direct semantic search",
            )
        actions = []
        for a in parsed.get("actions", []):
            actions.append(SearchAction(
                query=a.get("query", query),
                strategy=a.get("strategy", "semantic"),
                target_sections=a.get("target_sections", []),
                target_doc_ids=a.get("target_doc_ids", []),
                reason=a.get("reason", ""),
            ))
        if not actions:
            actions = [SearchAction(query=query, strategy="semantic")]

        return SearchPlan(
            intent=parsed.get("intent", "factual"),
            complexity=parsed.get("complexity", "simple"),
            actions=actions,
            key_entities=parsed.get("key_entities", []),
            reasoning=parsed.get("reasoning", ""),
        )

    # ------------------------------------------------------------------
    # SEARCH — parallel execution against Qdrant
    # ------------------------------------------------------------------

    def _search(self, plan: SearchPlan) -> List[EvidenceItem]:
        if not plan.actions:
            return []

        if len(plan.actions) == 1:
            return self._execute_search(plan.actions[0])

        # Parallel search for multiple actions
        results: List[EvidenceItem] = []
        max_workers = min(len(plan.actions), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._execute_search, action): action
                for action in plan.actions
            }
            for future in as_completed(futures):
                try:
                    items = future.result()
                    results.extend(items)
                except Exception as exc:
                    action = futures[future]
                    logger.debug("Search action failed: %s — %s", action.query[:60], exc)
        return results

    def _execute_search(self, action: SearchAction) -> List[EvidenceItem]:
        """Execute a single search action against Qdrant."""
        from src.api.vector_store import build_qdrant_filter

        try:
            query_vector = self._embedder.encode([action.query], normalize_embeddings=True)[0]
        except Exception as exc:
            logger.debug("Embedding failed for search action: %s", exc)
            return []

        # Build filters
        try:
            qdrant_filter = build_qdrant_filter(
                subscription_id=self._subscription_id,
                profile_id=self._profile_id,
                document_id=action.target_doc_ids if action.target_doc_ids else None,
            )
        except Exception:
            qdrant_filter = None

        top_k = 20 if action.strategy == "semantic" else 30

        try:
            results = self._qdrant.search(
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.debug("Qdrant search failed: %s", exc)
            return []

        items = []
        for point in results:
            payload = point.payload or {}
            text = (
                payload.get("content")
                or payload.get("text")
                or payload.get("chunk_text")
                or ""
            )
            if not text.strip():
                continue
            score = float(getattr(point, "score", 0.0))
            if score < 0.20:  # absolute floor — below this is noise
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

        # Sort by score descending
        items.sort(key=lambda e: -e.score)
        return items

    # ------------------------------------------------------------------
    # REASON — LLM evaluates evidence sufficiency
    # ------------------------------------------------------------------

    def _reason(
        self,
        query: str,
        plan: SearchPlan,
        evidence: List[EvidenceItem],
    ) -> EvidenceAssessment:
        evidence_text = self._format_evidence_for_prompt(evidence[:20])  # cap for prompt size
        prompt = _REASON_PROMPT.format(
            query=query,
            intent=plan.intent,
            count=len(evidence),
            evidence_text=evidence_text,
        )
        raw = self._call_thinker(prompt, max_tokens=512)
        parsed = self._parse_json(raw)
        if not parsed:
            # Fallback: assume sufficient if we have any evidence
            return EvidenceAssessment(
                sufficient=len(evidence) >= 3,
                confidence=min(0.8, max(e.score for e in evidence)) if evidence else 0.0,
                key_findings=[],
                reasoning="fallback assessment",
            )
        return EvidenceAssessment(
            sufficient=parsed.get("sufficient", True),
            confidence=float(parsed.get("confidence", 0.5)),
            gaps=parsed.get("gaps", []),
            contradictions=parsed.get("contradictions", []),
            key_findings=parsed.get("key_findings", []),
            reasoning=parsed.get("reasoning", ""),
        )

    # ------------------------------------------------------------------
    # GENERATE — main model produces grounded answer
    # ------------------------------------------------------------------

    def _generate(
        self,
        query: str,
        evidence: List[EvidenceItem],
        assessment: Optional[EvidenceAssessment],
        plan: Optional[SearchPlan],
        *,
        conversation_history: str = "",
        persona_prompt: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Build numbered sources
        sources = []
        context_parts = []
        for i, item in enumerate(evidence[:16], 1):  # max 16 sources
            import os
            doc_name = os.path.basename(item.source_name).rsplit(".", 1)[0] if item.source_name else "Document"
            header_parts = [f"Document: {doc_name}"]
            if item.section:
                header_parts.append(f"Section: {item.section}")
            if item.page:
                header_parts.append(f"Page: {item.page}")
            header = ", ".join(header_parts)

            context_parts.append(f"[SOURCE-{i}] {header}\n{item.text}\n[/SOURCE-{i}]")
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

        context_text = "\n\n".join(context_parts)

        # Build generation prompt
        findings_block = ""
        if assessment and assessment.key_findings:
            findings_block = "\nKEY FINDINGS FROM EVIDENCE ANALYSIS:\n" + "\n".join(
                f"- {f}" for f in assessment.key_findings
            )
        contradictions_block = ""
        if assessment and assessment.contradictions:
            contradictions_block = "\nCONTRADICTIONS DETECTED:\n" + "\n".join(
                f"- {c}" for c in assessment.contradictions
            )
        history_block = ""
        if conversation_history:
            history_block = f"\nCONVERSATION HISTORY:\n{conversation_history}\n"

        intent_guidance = self._intent_guidance(plan.intent if plan else "factual")

        prompt = f"""{persona_prompt or 'You are DocWain-Agent, a document intelligence model.'}

DOCUMENT CONTEXT:
{context_text}
{findings_block}{contradictions_block}{history_block}
TASK: Answer the user's question using ONLY the document context above.
Intent: {plan.intent if plan else 'factual'} query.
{intent_guidance}

GROUNDING RULES (MANDATORY):
1. Use ONLY the document context above. If information is missing, say so.
2. EVERY factual claim must cite with [SOURCE-N] immediately after the claim.
3. Prefer exact quotes of figures, names, dates with citations.
4. If sources disagree, note the discrepancy with citations to both.
5. Do not invent, generalize, or pad beyond the provided evidence.

USER QUESTION: {query}

Provide the answer now with inline citations."""

        try:
            options = {
                "temperature": 0.2,
                "top_p": 0.85,
                "num_predict": 2048,
                "num_ctx": 8192,
            }
            if hasattr(self._llm, "generate_with_metadata"):
                answer_text, _ = self._llm.generate_with_metadata(prompt, options=options)
            else:
                answer_text = self._llm.generate(prompt)
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            answer_text = self._evidence_summary_fallback(evidence, query)

        return answer_text.strip(), sources

    # ------------------------------------------------------------------
    # VERIFY — thinking model checks for hallucination
    # ------------------------------------------------------------------

    def _verify(self, answer: str, evidence: List[EvidenceItem]) -> Verification:
        if not answer.strip():
            return Verification(ok=False, unsupported_claims=["empty answer"])

        evidence_text = self._format_evidence_for_prompt(evidence[:12])
        prompt = _VERIFY_PROMPT.format(
            answer=answer[:3000],  # cap answer size
            evidence_text=evidence_text,
        )
        raw = self._call_thinker(prompt, max_tokens=256)
        parsed = self._parse_json(raw)
        if not parsed:
            return Verification(ok=True, reasoning="verification parse failed, assuming ok")
        return Verification(
            ok=parsed.get("ok", True),
            unsupported_claims=parsed.get("unsupported_claims", []),
            reasoning=parsed.get("reasoning", ""),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_thinker(self, prompt: str, max_tokens: int = 512) -> str:
        """Call the thinking model with tight token budget."""
        try:
            options = {
                "temperature": 0.05,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            }
            if hasattr(self._thinker, "generate_with_metadata"):
                text, _ = self._thinker.generate_with_metadata(prompt, options=options)
            else:
                text = self._thinker.generate(prompt)
            # Strip thinking tokens if present
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as exc:
            logger.debug("Thinker call failed: %s", exc)
            return ""

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        if not text:
            return None
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _format_evidence_for_prompt(self, evidence: List[EvidenceItem]) -> str:
        parts = []
        for i, item in enumerate(evidence, 1):
            doc = item.source_name or "Document"
            section = f" | Section: {item.section}" if item.section else ""
            page = f" | Page: {item.page}" if item.page else ""
            parts.append(f"[{i}] ({doc}{section}{page}, score={item.score:.2f})\n{item.text[:600]}")
        return "\n\n".join(parts)

    def _format_prior_evidence_block(
        self,
        evidence: List[EvidenceItem],
        assessment: EvidenceAssessment,
    ) -> str:
        lines = ["PRIOR RETRIEVAL RESULTS:"]
        lines.append(f"Found {len(evidence)} evidence chunks (confidence={assessment.confidence:.2f})")
        if assessment.gaps:
            lines.append("GAPS STILL MISSING:")
            for gap in assessment.gaps:
                lines.append(f"  - {gap}")
        if assessment.key_findings:
            lines.append("FINDINGS SO FAR:")
            for finding in assessment.key_findings[:5]:
                lines.append(f"  - {finding}")
        lines.append("\nRefine the search plan to fill these gaps. Use different search terms or strategies.")
        return "\n".join(lines)

    def _merge_evidence(
        self,
        existing: List[EvidenceItem],
        new: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        """Merge evidence, dedup by chunk_id, keep highest score."""
        seen: Dict[str, EvidenceItem] = {}
        for item in existing + new:
            key = item.chunk_id or hash(item.text[:100])
            if key not in seen or item.score > seen[key].score:
                seen[key] = item
        merged = sorted(seen.values(), key=lambda e: -e.score)
        return merged[:30]  # cap total evidence

    def _strip_unsupported(self, answer: str, verification: Verification) -> str:
        """Remove sentences containing unsupported claims."""
        if not verification.unsupported_claims:
            return answer
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        claim_terms = set()
        for claim in verification.unsupported_claims:
            claim_terms.update(w.lower() for w in claim.split() if len(w) > 3)
        filtered = []
        for sentence in sentences:
            sentence_words = set(w.lower() for w in sentence.split() if len(w) > 3)
            overlap = len(claim_terms & sentence_words)
            if overlap < 3:  # keep sentences with low overlap with unsupported claims
                filtered.append(sentence)
        return " ".join(filtered) if filtered else answer

    def _evidence_summary_fallback(self, evidence: List[EvidenceItem], query: str) -> str:
        """Build a simple evidence summary when generation fails."""
        lines = [f"Based on the available documents for your query about '{query[:100]}':"]
        for i, item in enumerate(evidence[:5], 1):
            excerpt = item.text[:200].strip()
            lines.append(f"\n[SOURCE-{i}] From {item.source_name}: \"{excerpt}...\"")
        lines.append("\n\nPlease refer to the source documents for complete details.")
        return "\n".join(lines)

    def _intent_guidance(self, intent: str) -> str:
        """Provide intent-specific generation guidance."""
        guidance = {
            "comparison": "Compare entities systematically. Use a table or side-by-side structure. Highlight commonalities AND differences.",
            "summary": "Provide a concise synthesis. Start with a 1-2 sentence overview, then key details.",
            "extraction": "Extract and list the requested fields clearly. Use structured format.",
            "reasoning": "Analyze the evidence logically. Present your reasoning chain step by step.",
            "procedural": "List steps in order. Be specific about each step.",
            "factual": "Answer directly and concisely. Cite the relevant source immediately.",
        }
        return guidance.get(intent, guidance["factual"])

    def _no_evidence_response(self, query: str, elapsed: float) -> Dict[str, Any]:
        return {
            "response": (
                "I searched the available documents but couldn't find information "
                f"relevant to your question. Please verify that the relevant documents "
                f"have been uploaded to this profile."
            ),
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {
                "engine": "reasoning_engine",
                "intent": "unknown",
                "evidence_count": 0,
                "timing_ms": {"total": round(elapsed * 1000, 1)},
            },
        }
