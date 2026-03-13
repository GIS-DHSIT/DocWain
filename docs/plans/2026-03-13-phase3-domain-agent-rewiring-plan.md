# Phase 3: Domain Agent Rewiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route all domain agent tasks through CoreAgent as one unified path, eliminating the legacy agent_name dispatch block in main.py.

**Architecture:** After UNDERSTAND, CoreAgent checks if the query is a domain-specific task (via MLP pre-filter + NLU detection). If yes, it delegates to the specialized DomainAgent, wraps the result through COMPOSE, and returns. If no, it continues the standard RETRIEVE → REASON → COMPOSE pipeline. This makes CoreAgent the single entry point for everything.

**Tech Stack:** Python 3.11, pytest, existing `src/agentic/domain_agents.py` (DomainAgent, get_domain_agent, detect_agent_task), `src/intent/intent_classifier.py` (IntentDomainClassifier), `src/agent/core_agent.py` (CoreAgent)

---

### Task 1: Create DomainDispatcher

**Files:**
- Create: `src/agent/domain_dispatch.py`
- Test: `tests/test_intelligence_v2/test_domain_dispatch.py`

**Step 1: Write the failing test**

```python
"""Tests for DomainDispatcher — detects domain tasks and delegates to agents."""
from unittest.mock import MagicMock, patch
import pytest

from src.agent.domain_dispatch import DomainDispatcher


class TestDomainDispatcher:
    """DomainDispatcher detection and delegation tests."""

    def test_no_detection_returns_none(self):
        """Non-domain queries should return None (fall through to RAG)."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())
        result = dispatcher.try_handle(
            query="What is the revenue for Q3?",
            subscription_id="sub_1",
            profile_id="prof_1",
            evidence=[],
            doc_context={},
        )
        assert result is None

    def test_explicit_agent_name_dispatches(self):
        """Explicit agent_name should dispatch to that domain agent."""
        mock_llm = MagicMock()
        dispatcher = DomainDispatcher(llm_gateway=mock_llm)

        mock_agent = MagicMock()
        mock_agent.domain = "hr"
        mock_agent.get_capabilities.return_value = ["interview_questions"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Here are interview questions...",
            sources=[],
            structured_data={},
            task_type="interview_questions",
        )

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}):
            result = dispatcher.try_handle(
                query="Generate interview questions for this candidate",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[{"text": "Resume content..."}],
                doc_context={},
                agent_name="hr",
            )
        assert result is not None
        assert result["response"] == "Here are interview questions..."
        assert result["metadata"]["agent_handled"] is True

    def test_auto_detect_dispatches(self):
        """Auto-detected domain task should dispatch without explicit agent_name."""
        mock_llm = MagicMock()
        dispatcher = DomainDispatcher(llm_gateway=mock_llm)

        mock_agent = MagicMock()
        mock_agent.domain = "legal"
        mock_agent.get_capabilities.return_value = ["compliance_check"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Compliance analysis...",
            sources=[],
            structured_data={},
            task_type="compliance_check",
        )

        with patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "legal", "task_type": "compliance_check"}), \
             patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent):
            result = dispatcher.try_handle(
                query="Check GDPR compliance of this contract",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[{"text": "Contract text..."}],
                doc_context={},
            )
        assert result is not None
        assert result["metadata"]["agent"] == "legal"

    def test_agent_failure_returns_none(self):
        """If agent execution fails, return None so RAG pipeline handles it."""
        mock_llm = MagicMock()
        dispatcher = DomainDispatcher(llm_gateway=mock_llm)

        mock_agent = MagicMock()
        mock_agent.domain = "hr"
        mock_agent.get_capabilities.return_value = ["interview_questions"]
        mock_agent.execute.return_value = MagicMock(
            success=False,
            output="",
            sources=[],
            structured_data={},
            task_type="interview_questions",
            error="LLM timeout",
        )

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}):
            result = dispatcher.try_handle(
                query="Generate interview questions",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[],
                doc_context={},
                agent_name="hr",
            )
        assert result is None

    def test_evidence_passed_as_context(self):
        """Retrieved evidence should be passed to the domain agent as text context."""
        mock_llm = MagicMock()
        dispatcher = DomainDispatcher(llm_gateway=mock_llm)

        mock_agent = MagicMock()
        mock_agent.domain = "invoice"
        mock_agent.get_capabilities.return_value = ["payment_anomaly_detection"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Anomaly found...",
            sources=[],
            structured_data={},
            task_type="payment_anomaly_detection",
        )

        evidence = [
            {"text": "Invoice line 1: $500", "source_name": "invoice.pdf"},
            {"text": "Invoice line 2: $1000", "source_name": "invoice.pdf"},
        ]

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "invoice", "task_type": "payment_anomaly_detection"}):
            dispatcher.try_handle(
                query="Check for payment anomalies",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=evidence,
                doc_context={},
                agent_name="invoice",
            )

        call_args = mock_agent.execute.call_args
        ctx = call_args[0][1]  # second positional arg is context
        assert "Invoice line 1: $500" in ctx["text"]
        assert "Invoice line 2: $1000" in ctx["text"]

    def test_media_propagated(self):
        """Media from agent structured_data should be propagated to response."""
        mock_llm = MagicMock()
        dispatcher = DomainDispatcher(llm_gateway=mock_llm)

        mock_agent = MagicMock()
        mock_agent.domain = "analytics_viz"
        mock_agent.get_capabilities.return_value = ["generate_chart"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Chart generated",
            sources=[],
            structured_data={"media": [{"type": "image", "url": "/tmp/chart.png"}]},
            task_type="generate_chart",
        )

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "analytics_viz", "task_type": "generate_chart"}):
            result = dispatcher.try_handle(
                query="Generate a bar chart of revenue",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[],
                doc_context={},
                agent_name="analytics_viz",
            )
        assert result is not None
        assert result["media"] == [{"type": "image", "url": "/tmp/chart.png"}]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_intelligence_v2/test_domain_dispatch.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.agent.domain_dispatch'"

**Step 3: Write minimal implementation**

```python
"""Domain Dispatcher — detects domain-specific tasks and delegates to
specialized agents within the CoreAgent pipeline.

Sits between UNDERSTAND and RETRIEVE in the pipeline:
    UNDERSTAND → [domain dispatch?]
        YES → DomainAgent.execute(task, context) → format response
        NO  → continue to RETRIEVE → REASON → COMPOSE
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agentic.domain_agents import detect_agent_task, get_domain_agent

logger = logging.getLogger(__name__)


class DomainDispatcher:
    """Detects domain tasks and delegates to specialized agents.

    Returns a formatted response dict on success, or None to signal
    the caller should continue with the standard RAG pipeline.
    """

    def __init__(self, llm_gateway: Any) -> None:
        self._llm = llm_gateway

    def try_handle(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        evidence: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
        *,
        agent_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt domain agent dispatch.

        Returns a response dict (same shape as compose_response) if a domain
        agent handled the query successfully, or None if the query should
        fall through to the standard RAG pipeline.
        """
        # Determine domain and task
        detection = self._detect(query, agent_name)
        if detection is None:
            return None

        domain = detection["domain"]
        task_type = detection["task_type"]

        # Get the agent
        agent = get_domain_agent(domain, llm_client=self._llm)
        if agent is None:
            logger.debug("No agent found for domain=%s", domain)
            return None

        # Build context from retrieved evidence
        context = self._build_agent_context(query, evidence, doc_context)

        # Execute
        try:
            result = agent.execute(task_type, context)
        except Exception:
            logger.exception("Domain agent %s failed on task %s", domain, task_type)
            return None

        if not result.success or not result.output:
            logger.debug("Domain agent %s returned failure for task %s", domain, task_type)
            return None

        # Format response
        response: Dict[str, Any] = {
            "response": result.output,
            "sources": result.sources or [],
            "grounded": True,
            "context_found": bool(evidence),
            "metadata": {
                "agent": domain,
                "agent_task": task_type,
                "agent_handled": True,
                "engine": "docwain_core_agent",
            },
        }

        # Propagate media (charts/images)
        if (result.structured_data
                and isinstance(result.structured_data.get("media"), list)):
            response["media"] = result.structured_data["media"]

        return response

    @staticmethod
    def _detect(query: str, agent_name: Optional[str]) -> Optional[Dict[str, str]]:
        """Detect domain and task type.

        If agent_name is provided explicitly, use it directly.
        Otherwise, use detect_agent_task for auto-detection.
        """
        if agent_name:
            # Explicit agent: detect task type within that domain
            detection = detect_agent_task(query, domain=agent_name)
            if detection:
                return detection
            # Fallback: use the agent_name as domain with first capability
            agent = get_domain_agent(agent_name)
            if agent:
                caps = agent.get_capabilities()
                return {"domain": agent_name, "task_type": caps[0] if caps else "default"}
            return None

        # Auto-detection
        return detect_agent_task(query)

    @staticmethod
    def _build_agent_context(
        query: str,
        evidence: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build context dict for the domain agent from retrieved evidence."""
        context: Dict[str, Any] = {"query": query}

        if evidence:
            text_parts = [e.get("text", "") for e in evidence if e.get("text")]
            if text_parts:
                context["text"] = "\n\n".join(text_parts)

        return context
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_intelligence_v2/test_domain_dispatch.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add src/agent/domain_dispatch.py tests/test_intelligence_v2/test_domain_dispatch.py
git commit -m "feat(agent): domain dispatcher for routing tasks to specialized agents"
```

---

### Task 2: Integrate DomainDispatcher into CoreAgent

**Files:**
- Modify: `src/agent/core_agent.py:59-210` (CoreAgent class)
- Test: `tests/test_intelligence_v2/test_core_domain_integration.py`

**Step 1: Write the failing test**

```python
"""Tests for CoreAgent domain dispatch integration."""
from unittest.mock import MagicMock, patch
import pytest


def _make_core_agent(**overrides):
    """Create a CoreAgent with mocked dependencies."""
    from src.agent.core_agent import CoreAgent
    defaults = {
        "llm_gateway": MagicMock(),
        "qdrant_client": MagicMock(),
        "embedder": MagicMock(),
        "mongodb": MagicMock(),
    }
    defaults.update(overrides)
    # Mock MongoDB cursor
    defaults["mongodb"].find.return_value = []
    return CoreAgent(**defaults)


class TestCoreDomainIntegration:
    """CoreAgent routes domain tasks through DomainDispatcher."""

    def test_domain_task_skips_retrieve_reason(self):
        """When DomainDispatcher handles query, RETRIEVE and REASON are skipped."""
        agent = _make_core_agent()

        domain_response = {
            "response": "Interview questions generated",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"agent": "hr", "agent_task": "interview_questions", "agent_handled": True},
        }

        with patch("src.agent.core_agent.IntentAnalyzer") as MockAnalyzer, \
             patch("src.agent.core_agent.DomainDispatcher") as MockDispatcher:
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="extract",
                complexity="simple",
                resolved_query="Generate interview questions for this candidate",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )
            # DomainDispatcher returns a response → skip RETRIEVE/REASON
            mock_dispatcher = MockDispatcher.return_value
            mock_dispatcher.try_handle.return_value = domain_response

            result = agent.handle(
                query="Generate interview questions for this candidate",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="hr",
            )

        assert result["response"] == "Interview questions generated"
        assert result["metadata"]["agent_handled"] is True

    def test_non_domain_task_continues_rag(self):
        """When DomainDispatcher returns None, standard RAG pipeline runs."""
        agent = _make_core_agent()

        with patch("src.agent.core_agent.IntentAnalyzer") as MockAnalyzer, \
             patch("src.agent.core_agent.DomainDispatcher") as MockDispatcher, \
             patch.object(agent, "_retriever") as mock_retriever, \
             patch.object(agent, "_reasoner") as mock_reasoner:
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="lookup",
                complexity="simple",
                resolved_query="What is the revenue?",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )
            # DomainDispatcher returns None → continue RAG
            mock_dispatcher = MockDispatcher.return_value
            mock_dispatcher.try_handle.return_value = None

            mock_retriever.retrieve.return_value = MagicMock(
                chunks=[], profiles_searched=["prof_1"],
            )
            mock_reasoner.reason.return_value = MagicMock(
                text="Revenue is $1M", grounded=True, usage={},
            )

            result = agent.handle(
                query="What is the revenue?",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
            )

        # RETRIEVE and REASON should have been called
        mock_retriever.retrieve.assert_called_once()
        mock_reasoner.reason.assert_called_once()

    def test_agent_name_passed_to_dispatcher(self):
        """agent_name from request should be forwarded to DomainDispatcher."""
        agent = _make_core_agent()

        with patch("src.agent.core_agent.IntentAnalyzer") as MockAnalyzer, \
             patch("src.agent.core_agent.DomainDispatcher") as MockDispatcher:
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="extract",
                complexity="simple",
                resolved_query="Check compliance",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )
            mock_dispatcher = MockDispatcher.return_value
            mock_dispatcher.try_handle.return_value = {
                "response": "Compliant", "sources": [], "grounded": True,
                "context_found": True, "metadata": {"agent": "legal", "agent_handled": True},
            }

            agent.handle(
                query="Check compliance",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="legal",
            )

        call_kwargs = mock_dispatcher.try_handle.call_args[1]
        assert call_kwargs.get("agent_name") == "legal" or \
               mock_dispatcher.try_handle.call_args[1].get("agent_name") == "legal"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_intelligence_v2/test_core_domain_integration.py -v`
Expected: FAIL (DomainDispatcher not imported in core_agent.py)

**Step 3: Modify CoreAgent to integrate DomainDispatcher**

In `src/agent/core_agent.py`, make these changes:

1. Add import at the top (after existing imports):
```python
from src.agent.domain_dispatch import DomainDispatcher
```

2. In `__init__`, add the dispatcher:
```python
self._domain_dispatcher = DomainDispatcher(llm_gateway=llm_gateway)
```

3. In `handle()`, after the UNDERSTAND block and before RETRIEVE, add domain dispatch:
```python
        # --- DOMAIN DISPATCH (between UNDERSTAND and RETRIEVE) ---
        domain_result = self._domain_dispatcher.try_handle(
            query=understanding.resolved_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            evidence=[],  # No evidence yet — pre-retrieval dispatch
            doc_context={},
            agent_name=agent_name,
            document_id=document_id,
        )
        if domain_result is not None:
            domain_result.setdefault("metadata", {})["timing"] = timing
            return domain_result
```

**Full diff for `src/agent/core_agent.py`:**

Add to imports (after line 20):
```python
from src.agent.domain_dispatch import DomainDispatcher
```

Add to `__init__` (after line 78):
```python
        self._domain_dispatcher = DomainDispatcher(llm_gateway=llm_gateway)
```

Add after the UNDERSTAND block timing line (after line 136, before `if understanding.is_conversational:`):
```python
        # --- DOMAIN DISPATCH ---
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_intelligence_v2/test_core_domain_integration.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/agent/core_agent.py tests/test_intelligence_v2/test_core_domain_integration.py
git commit -m "feat(agent): integrate domain dispatch into CoreAgent pipeline"
```

---

### Task 3: Post-retrieval domain dispatch (evidence-aware)

**Files:**
- Modify: `src/agent/core_agent.py:140-210` (handle method, after RETRIEVE)
- Modify: `src/agent/domain_dispatch.py` (no changes needed — already accepts evidence)
- Test: `tests/test_intelligence_v2/test_post_retrieval_dispatch.py`

**Step 1: Write the failing test**

```python
"""Tests for post-retrieval domain dispatch — agent gets evidence context."""
from unittest.mock import MagicMock, patch
import pytest


def _make_core_agent():
    from src.agent.core_agent import CoreAgent
    mongodb = MagicMock()
    mongodb.find.return_value = []
    return CoreAgent(
        llm_gateway=MagicMock(),
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        mongodb=mongodb,
    )


class TestPostRetrievalDispatch:
    """Domain dispatch after RETRIEVE provides evidence to agents."""

    def test_post_retrieval_dispatch_with_evidence(self):
        """If pre-retrieval dispatch returns None but agent_name is set,
        try again after RETRIEVE with evidence."""
        agent = _make_core_agent()

        evidence = [
            {"text": "Resume: 5 years Python", "source_name": "resume.pdf", "document_id": "d1",
             "score": 0.9, "page": 1, "section": "Experience", "source_index": 1,
             "profile_id": "prof_1", "chunk_id": "c1"},
        ]

        domain_response = {
            "response": "Skill gap analysis complete",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"agent": "hr", "agent_task": "skill_gap_analysis", "agent_handled": True},
        }

        with patch("src.agent.core_agent.IntentAnalyzer") as MockAnalyzer, \
             patch("src.agent.core_agent.DomainDispatcher") as MockDispatcher, \
             patch.object(agent, "_retriever") as mock_retriever, \
             patch("src.agent.core_agent.rerank_chunks", return_value=evidence), \
             patch("src.agent.core_agent.build_context", return_value=(evidence, {})):

            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="extract",
                complexity="simple",
                resolved_query="Analyze skill gaps",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )

            mock_dispatcher = MockDispatcher.return_value
            # First call (pre-retrieval): returns None
            # Second call (post-retrieval with evidence): returns response
            mock_dispatcher.try_handle.side_effect = [None, domain_response]

            mock_retriever.retrieve.return_value = MagicMock(
                chunks=evidence, profiles_searched=["prof_1"],
            )

            result = agent.handle(
                query="Analyze skill gaps",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="hr",
            )

        assert result["response"] == "Skill gap analysis complete"
        assert mock_dispatcher.try_handle.call_count == 2
        # Second call should have evidence
        second_call = mock_dispatcher.try_handle.call_args_list[1]
        assert len(second_call[1].get("evidence", second_call[0][3] if len(second_call[0]) > 3 else [])) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_intelligence_v2/test_post_retrieval_dispatch.py -v`
Expected: FAIL (CoreAgent doesn't try domain dispatch after RETRIEVE yet)

**Step 3: Modify CoreAgent to add post-retrieval domain dispatch**

In `src/agent/core_agent.py`, after the RETRIEVE block builds evidence (after the `build_context` call, before REASON), add:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_intelligence_v2/test_post_retrieval_dispatch.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add src/agent/core_agent.py tests/test_intelligence_v2/test_post_retrieval_dispatch.py
git commit -m "feat(agent): post-retrieval domain dispatch with evidence context"
```

---

### Task 4: Remove legacy agent_name dispatch from main.py

**Files:**
- Modify: `src/main.py:955-1023` (remove agent_name dispatch block)
- Test: `tests/test_intelligence_v2/test_main_simplified.py`

**Step 1: Write the failing test**

```python
"""Tests that main.py no longer has legacy agent dispatch — all goes through execute_request."""
from unittest.mock import MagicMock, patch
import pytest


def test_ask_with_agent_name_goes_through_execute_request():
    """Requests with agent_name should go through execute_request, not legacy dispatch."""
    from src.execution.common import ExecutionResult
    from src.mode.execution_mode import ExecutionMode

    mock_result = ExecutionResult(
        answer={
            "response": "Domain agent handled via CoreAgent",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"agent": "hr", "agent_handled": True},
        },
        mode=ExecutionMode.AGENT,
        debug={},
        stream=None,
    )

    with patch("src.main.execute_request", return_value=mock_result) as mock_exec, \
         patch("src.main._prepare_execution") as mock_prep, \
         patch("src.main._persist_chat_turn", return_value="sess_1"), \
         patch("src.main.normalize_answer", side_effect=lambda x: x):

        mock_prep.return_value = ("sess_1", {}, ExecutionMode.AGENT, MagicMock())

        # Simulate the ask endpoint internals — agent_name should NOT
        # trigger the old get_domain_agent path
        from src.main import ask_endpoint_handles_agent_name_via_core
        # This is a marker test — the real assertion is that main.py
        # no longer imports get_domain_agent in the ask handler


def test_no_legacy_agent_dispatch_in_ask():
    """Verify main.py ask handler does NOT contain the legacy agent dispatch block."""
    import inspect
    from src.main import handle_ask
    source = inspect.getsource(handle_ask)
    # The old block had: get_domain_agent, detect_agent_task, _retrieve_rag_context
    assert "get_domain_agent" not in source, "Legacy get_domain_agent still in ask handler"
    assert "detect_agent_task" not in source, "Legacy detect_agent_task still in ask handler"
```

Note: The exact test shape depends on whether `handle_ask` is the function name. We'll adapt during implementation.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_intelligence_v2/test_main_simplified.py -v`
Expected: FAIL (legacy agent dispatch block still present)

**Step 3: Remove legacy agent_name dispatch block from main.py**

In `src/main.py`, remove lines 955-1023 (the entire `# ── Agent dispatch: explicit agent_name routes directly to domain agent ──` block). The code should go straight from the conversation-only handling to `_prepare_execution` + `execute_request`.

Before (lines ~955-1026):
```python
    # ── Agent dispatch: explicit agent_name routes directly to domain agent ──
    _agent_name = getattr(request, "agent_name", None)
    if _agent_name:
        try:
            from src.agentic.domain_agents import get_domain_agent, detect_agent_task
            ...
        except Exception as _agent_exc:
            logger.debug(...)

    session_id, session_state, mode, ctx = _prepare_execution(request, agent_mode)
```

After:
```python
    session_id, session_state, mode, ctx = _prepare_execution(request, agent_mode)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_intelligence_v2/test_main_simplified.py -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `pytest tests/test_intelligence_v2/ -v`
Expected: All tests pass (61 existing + new tests)

**Step 6: Commit**

```bash
git add src/main.py tests/test_intelligence_v2/test_main_simplified.py
git commit -m "refactor(main): remove legacy agent dispatch, all requests route through CoreAgent"
```

---

### Task 5: End-to-end test — domain query through unified pipeline

**Files:**
- Test: `tests/test_intelligence_v2/test_e2e_domain_routing.py`

**Step 1: Write the end-to-end test**

```python
"""End-to-end test: domain agent query flows through CoreAgent pipeline."""
from unittest.mock import MagicMock, patch
import pytest


def test_e2e_domain_query_through_core_agent():
    """A domain agent query (agent_name='hr') should:
    1. Go through execute_request → CoreAgent.handle
    2. CoreAgent runs UNDERSTAND
    3. DomainDispatcher detects domain task
    4. DomainAgent executes and returns response
    5. Response is formatted through standard answer structure
    """
    from src.agent.core_agent import CoreAgent
    from src.agentic.domain_agents import AgentTaskResult

    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"task_type": "extract", "complexity": "simple", "resolved_query": "Generate interview questions", "output_format": "prose", "relevant_documents": [], "cross_profile": false}'

    mock_mongodb = MagicMock()
    mock_mongodb.find.return_value = []

    mock_agent_result = AgentTaskResult(
        task_type="interview_questions",
        success=True,
        output="Here are 5 interview questions:\n1. Tell me about your Python experience...",
        sources=[],
        structured_data={},
    )

    mock_agent = MagicMock()
    mock_agent.domain = "hr"
    mock_agent.get_capabilities.return_value = ["interview_questions"]
    mock_agent.execute.return_value = mock_agent_result

    with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
         patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}):

        core = CoreAgent(
            llm_gateway=mock_llm,
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            mongodb=mock_mongodb,
        )

        result = core.handle(
            query="Generate interview questions for this candidate",
            subscription_id="sub_1",
            profile_id="prof_1",
            user_id="user_1",
            session_id="sess_1",
            conversation_history=None,
            agent_name="hr",
        )

    assert "interview questions" in result["response"].lower() or "interview" in result["response"].lower()
    assert result["metadata"]["agent_handled"] is True
    assert result["metadata"]["agent"] == "hr"


def test_e2e_auto_detect_domain_no_agent_name():
    """Without agent_name, auto-detection should still route to domain agent."""
    from src.agent.core_agent import CoreAgent
    from src.agentic.domain_agents import AgentTaskResult

    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"task_type": "extract", "complexity": "simple", "resolved_query": "Check GDPR compliance", "output_format": "prose", "relevant_documents": [], "cross_profile": false}'

    mock_mongodb = MagicMock()
    mock_mongodb.find.return_value = []

    mock_agent_result = AgentTaskResult(
        task_type="compliance_check",
        success=True,
        output="GDPR compliance analysis complete.",
        sources=[],
        structured_data={},
    )

    mock_agent = MagicMock()
    mock_agent.domain = "legal"
    mock_agent.get_capabilities.return_value = ["compliance_check"]
    mock_agent.execute.return_value = mock_agent_result

    with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
         patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "legal", "task_type": "compliance_check"}):

        core = CoreAgent(
            llm_gateway=mock_llm,
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            mongodb=mock_mongodb,
        )

        result = core.handle(
            query="Check GDPR compliance of this contract",
            subscription_id="sub_1",
            profile_id="prof_1",
            user_id="user_1",
            session_id="sess_1",
            conversation_history=None,
        )

    assert result["metadata"]["agent_handled"] is True
    assert result["metadata"]["agent"] == "legal"


def test_e2e_non_domain_falls_through_to_rag():
    """A non-domain query should bypass domain dispatch and use RAG."""
    from src.agent.core_agent import CoreAgent

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        # UNDERSTAND call
        '{"task_type": "lookup", "complexity": "simple", "resolved_query": "What is Q3 revenue?", "output_format": "prose", "relevant_documents": [], "cross_profile": false}',
        # REASON call
        "The Q3 revenue was $2.5M based on the financial report.",
    ]

    mock_mongodb = MagicMock()
    mock_mongodb.find.return_value = []

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = MagicMock(
        chunks=[{"text": "Q3 revenue: $2.5M", "score": 0.95}],
        profiles_searched=["prof_1"],
    )

    with patch("src.agent.domain_dispatch.detect_agent_task", return_value=None), \
         patch("src.agent.core_agent.rerank_chunks", return_value=[{"text": "Q3 revenue: $2.5M", "score": 0.95, "source_name": "report.pdf", "document_id": "d1", "page": 3, "section": "Financials", "source_index": 1, "profile_id": "p1", "chunk_id": "c1"}]), \
         patch("src.agent.core_agent.build_context", return_value=([{"text": "Q3 revenue: $2.5M", "score": 0.95, "source_name": "report.pdf", "document_id": "d1"}], {})):

        core = CoreAgent(
            llm_gateway=mock_llm,
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            mongodb=mock_mongodb,
        )
        core._retriever = mock_retriever

        result = core.handle(
            query="What is Q3 revenue?",
            subscription_id="sub_1",
            profile_id="prof_1",
            user_id="user_1",
            session_id="sess_1",
            conversation_history=None,
        )

    assert result is not None
    assert result.get("metadata", {}).get("agent_handled") is not True
    mock_retriever.retrieve.assert_called_once()
```

**Step 2: Run test**

Run: `pytest tests/test_intelligence_v2/test_e2e_domain_routing.py -v`
Expected: 3 passed

**Step 3: Commit**

```bash
git add tests/test_intelligence_v2/test_e2e_domain_routing.py
git commit -m "test(e2e): domain agent routing through unified CoreAgent pipeline"
```

---

### Task 6: Run full test suite and verify

**Step 1: Run all Phase 2 + Phase 3 tests**

Run: `pytest tests/test_intelligence_v2/ -v`
Expected: All tests pass (61 existing + ~13 new = ~74 total)

**Step 2: Run the broader test suite**

Run: `pytest tests/ -v --timeout=60 2>&1 | tail -30`
Expected: No regressions

**Step 3: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: phase 3 cleanup and final verification"
```
