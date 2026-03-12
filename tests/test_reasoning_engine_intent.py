"""Tests for enriched THINK prompt with domain awareness and intent decomposition."""
import inspect
from dataclasses import fields

from src.intelligence.reasoning_engine import (
    ReasoningEngine,
    SearchPlan,
    _THINK_PROMPT,
)


class TestThinkPromptPlaceholders:
    """Verify _THINK_PROMPT contains the required placeholders and instructions."""

    def test_prompt_has_conversation_context_placeholder(self):
        assert "{conversation_context}" in _THINK_PROMPT

    def test_prompt_has_query_placeholder(self):
        assert "{query}" in _THINK_PROMPT

    def test_prompt_has_profile_context_placeholder(self):
        assert "{profile_context}" in _THINK_PROMPT

    def test_prompt_has_prior_evidence_block_placeholder(self):
        assert "{prior_evidence_block}" in _THINK_PROMPT

    def test_prompt_contains_intent_instruction(self):
        assert "INTENT" in _THINK_PROMPT
        assert "What does the user want to KNOW or DO" in _THINK_PROMPT

    def test_prompt_contains_scope_instruction(self):
        assert "SCOPE" in _THINK_PROMPT

    def test_prompt_contains_implicit_instruction(self):
        assert "IMPLICIT" in _THINK_PROMPT
        assert "implied but not stated" in _THINK_PROMPT

    def test_prompt_contains_depth_instruction(self):
        assert "DEPTH" in _THINK_PROMPT

    def test_prompt_json_includes_user_intent_field(self):
        assert '"user_intent"' in _THINK_PROMPT

    def test_prompt_json_includes_implicit_context_field(self):
        assert '"implicit_context"' in _THINK_PROMPT


class TestSearchPlanFields:
    """Verify SearchPlan dataclass has the new fields."""

    def test_search_plan_has_user_intent_field(self):
        field_names = {f.name for f in fields(SearchPlan)}
        assert "user_intent" in field_names

    def test_search_plan_has_implicit_context_field(self):
        field_names = {f.name for f in fields(SearchPlan)}
        assert "implicit_context" in field_names

    def test_search_plan_user_intent_defaults_empty(self):
        plan = SearchPlan(intent="factual", complexity="simple")
        assert plan.user_intent == ""

    def test_search_plan_implicit_context_defaults_empty(self):
        plan = SearchPlan(intent="factual", complexity="simple")
        assert plan.implicit_context == ""

    def test_search_plan_accepts_user_intent(self):
        plan = SearchPlan(
            intent="factual",
            complexity="simple",
            user_intent="find revenue figures",
        )
        assert plan.user_intent == "find revenue figures"

    def test_search_plan_accepts_implicit_context(self):
        plan = SearchPlan(
            intent="factual",
            complexity="simple",
            implicit_context="user likely wants latest quarter",
        )
        assert plan.implicit_context == "user likely wants latest quarter"


class TestThinkMethodSignature:
    """Verify _think() accepts and uses conversation_context parameter."""

    def test_think_accepts_conversation_context_param(self):
        sig = inspect.signature(ReasoningEngine._think)
        assert "conversation_context" in sig.parameters

    def test_think_conversation_context_has_default(self):
        sig = inspect.signature(ReasoningEngine._think)
        param = sig.parameters["conversation_context"]
        assert param.default == ""
