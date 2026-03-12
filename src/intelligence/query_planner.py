"""
Multi-step query planner for DocWain.

Decomposes complex queries into execution steps for higher quality answers.
Detects patterns like "compare AND rank", "find X then write Y", conditional queries.
Each step runs through a simplified pipeline call.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = get_logger(__name__)

# ── Multi-step detection patterns ─────────────────────────────────────────

_MULTI_STEP_PATTERNS = [
    # "compare X and then rank them"
    re.compile(
        r"\b(?:compare|analyze).*\b(?:then|and\s+then|after\s+that|also|and)\b.*\b(?:rank|sort|order|list)\b",
        re.IGNORECASE,
    ),
    # "find X then write/generate Y"
    re.compile(
        r"\b(?:find|search|get|extract|identify)\b.*\b(?:then|and\s+then|and)\b.*\b(?:write|generate|create|draft|compose)\b",
        re.IGNORECASE,
    ),
    # "if X, what Y" / conditional
    re.compile(
        r"\b(?:if|assuming|given\s+that|in\s+case)\b.*[,;]\s*\b(?:what|which|who|how|list|find)\b",
        re.IGNORECASE,
    ),
    # "first X, then Y"
    re.compile(
        r"\b(?:first|step\s+1)\b.*\b(?:then|next|second|step\s+2)\b",
        re.IGNORECASE,
    ),
    # "X and also Y" with distinct verbs
    re.compile(
        r"\b(?:summarize|extract|list|compare|rank)\b.*\b(?:and\s+also|additionally|plus|as\s+well\s+as)\b.*\b(?:summarize|extract|list|compare|rank)\b",
        re.IGNORECASE,
    ),
]

# ── Query decomposition ──────────────────────────────────────────────────

_CONJUNCTION_SPLIT = re.compile(
    r"\b(?:then|and\s+then|after\s+that|next|additionally|also|plus)\b",
    re.IGNORECASE,
)

_CONDITIONAL_SPLIT = re.compile(
    r"\b(?:if|assuming|given\s+that|in\s+case)\b(.+?)[,;]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)

@dataclass
class QueryStep:
    """A single step in a multi-step query plan."""

    step_number: int
    query: str
    intent: str  # "retrieve", "compare", "generate", "filter", "rank"
    depends_on: List[int] = field(default_factory=list)  # step numbers
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step_number,
            "query": self.query,
            "intent": self.intent,
        }
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.result is not None:
            d["result"] = self.result[:200]
        return d

@dataclass
class QueryPlan:
    """A plan for executing a multi-step query."""

    original_query: str
    steps: List[QueryStep]
    synthesis_strategy: str  # "merge", "chain", "conditional"
    is_multi_step: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "steps": [s.to_dict() for s in self.steps],
            "synthesis_strategy": self.synthesis_strategy,
            "is_multi_step": self.is_multi_step,
        }

def _classify_step_intent(query: str) -> str:
    """Classify the intent of a single query step."""
    lower = query.lower().strip()

    intent_patterns = [
        (r"\b(?:compare|versus|vs|difference|contrast)\b", "compare"),
        (r"\b(?:rank|sort|order|top|best|worst|most|least)\b", "rank"),
        (r"\b(?:write|generate|create|draft|compose|build)\b", "generate"),
        (r"\b(?:filter|select|only|exclude|where|with)\b", "filter"),
        (r"\b(?:summarize|summary|overview|key\s+points)\b", "summarize"),
        (r"\b(?:find|search|get|extract|identify|list|what|who|which)\b", "retrieve"),
    ]

    for pattern, intent in intent_patterns:
        if re.search(pattern, lower):
            return intent
    return "retrieve"

def is_multi_step_query(query: str) -> bool:
    """Check if a query requires multi-step decomposition."""
    if not query or len(query) < 20:
        return False
    for pattern in _MULTI_STEP_PATTERNS:
        if pattern.search(query):
            return True
    return False

def _decompose_conjunction(query: str) -> List[str]:
    """Split a query at conjunction boundaries into sub-queries."""
    parts = _CONJUNCTION_SPLIT.split(query)
    cleaned = [p.strip().strip(",;.") for p in parts if p and len(p.strip()) > 10]
    return cleaned if len(cleaned) > 1 else [query]

def _decompose_conditional(query: str) -> Optional[tuple[str, str]]:
    """Decompose a conditional query into condition + question."""
    m = _CONDITIONAL_SPLIT.search(query)
    if m:
        condition = m.group(1).strip()
        question = m.group(2).strip()
        if len(condition) > 5 and len(question) > 5:
            return (condition, question)
    return None

def decompose_query(query: str, max_steps: int = 3) -> QueryPlan:
    """
    Decompose a complex query into a multi-step execution plan.

    Returns a QueryPlan with ordered steps and synthesis strategy.
    """
    if not is_multi_step_query(query):
        return QueryPlan(
            original_query=query,
            steps=[QueryStep(step_number=1, query=query, intent=_classify_step_intent(query))],
            synthesis_strategy="merge",
            is_multi_step=False,
        )

    # Try conditional decomposition first
    conditional = _decompose_conditional(query)
    if conditional:
        condition, question = conditional
        steps = [
            QueryStep(
                step_number=1,
                query=condition,
                intent="retrieve",
            ),
            QueryStep(
                step_number=2,
                query=question,
                intent=_classify_step_intent(question),
                depends_on=[1],
            ),
        ]
        return QueryPlan(
            original_query=query,
            steps=steps[:max_steps],
            synthesis_strategy="conditional",
            is_multi_step=True,
        )

    # Conjunction decomposition
    sub_queries = _decompose_conjunction(query)
    if len(sub_queries) > 1:
        steps = []
        for i, sq in enumerate(sub_queries[:max_steps]):
            step = QueryStep(
                step_number=i + 1,
                query=sq,
                intent=_classify_step_intent(sq),
                depends_on=[i] if i > 0 else [],
            )
            steps.append(step)
        return QueryPlan(
            original_query=query,
            steps=steps,
            synthesis_strategy="chain",
            is_multi_step=True,
        )

    # Fallback: single step
    return QueryPlan(
        original_query=query,
        steps=[QueryStep(step_number=1, query=query, intent=_classify_step_intent(query))],
        synthesis_strategy="merge",
        is_multi_step=False,
    )

def _llm_decompose(
    query: str, llm_client: Any, timeout: float = 5.0, max_steps: int = 3
) -> Optional[List[str]]:
    """Use LLM to decompose a complex query (optional enhancement)."""
    if llm_client is None:
        return None

    prompt = (
        f"Break this complex question into {max_steps} or fewer simple sub-questions. "
        f"Return ONLY the sub-questions, one per line.\n\n"
        f"Question: {query}\n\n"
        f"Sub-questions:"
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            result = future.result(timeout=timeout)

        if not result:
            return None

        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in result.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        return lines[:max_steps] if len(lines) > 1 else None

    except (FuturesTimeout, Exception) as exc:
        logger.debug("LLM decomposition failed: %s", exc)
        return None

def _call_llm(llm_client: Any, prompt: str) -> str:
    """Call LLM client."""
    if hasattr(llm_client, "generate"):
        result = llm_client.generate(prompt)
        if isinstance(result, tuple):
            return result[0] if result[0] else ""
        return result or ""
    return ""

def execute_plan(
    plan: QueryPlan,
    pipeline_fn: Callable[..., Dict[str, Any]],
    pipeline_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a multi-step query plan using the pipeline function.

    Each step runs through the pipeline. Results are synthesized according
    to the plan's strategy.
    """
    if not plan.is_multi_step or len(plan.steps) <= 1:
        return pipeline_fn(**{**pipeline_kwargs, "query": plan.original_query})

    step_results: List[Dict[str, Any]] = []

    for step in plan.steps:
        try:
            kwargs = {**pipeline_kwargs, "query": step.query, "enable_decomposition": False}
            result = pipeline_fn(**kwargs)
            step.result = result.get("response", "")
            step_results.append(result)
        except Exception as exc:
            logger.warning("Step %d failed: %s", step.step_number, exc)
            step.result = ""
            step_results.append({"response": "", "sources": [], "context_found": False})

    return _synthesize_results(plan, step_results)

def _synthesize_results(
    plan: QueryPlan, step_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge step results into a single response based on synthesis strategy."""
    if not step_results:
        return {"response": "", "sources": [], "context_found": False, "grounded": False}

    all_sources: List[Dict[str, Any]] = []
    responses: List[str] = []
    context_found = False
    grounded = False

    for i, result in enumerate(step_results):
        response = result.get("response", "")
        if response:
            step_label = f"**Step {i + 1}** ({plan.steps[i].intent}): "
            if plan.synthesis_strategy == "chain":
                responses.append(f"{step_label}\n{response}")
            else:
                responses.append(response)

        sources = result.get("sources", [])
        all_sources.extend(sources)

        if result.get("context_found"):
            context_found = True
        if result.get("grounded"):
            grounded = True

    combined_response = "\n\n".join(responses) if responses else ""

    # Deduplicate sources
    seen_sources: set = set()
    unique_sources: List[Dict[str, Any]] = []
    for src in all_sources:
        key = str(src.get("document_id", "")) + str(src.get("chunk_id", ""))
        if key not in seen_sources:
            seen_sources.add(key)
            unique_sources.append(src)

    return {
        "response": combined_response,
        "sources": unique_sources,
        "context_found": context_found,
        "grounded": grounded,
        "metadata": {
            "multi_step": True,
            "plan": plan.to_dict(),
            "steps_executed": len(step_results),
        },
    }
