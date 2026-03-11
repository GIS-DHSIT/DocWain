"""Tests for TaskSpec dataclass and its routing methods."""

import json
import pytest

from src.intelligence.task_spec import (
    TaskSpec,
    VALID_INTENTS,
    VALID_DOMAINS,
    VALID_OUTPUT_FORMATS,
    VALID_SCOPES,
    VALID_COMPLEXITIES,
)


class TestTaskSpecRouting:
    """Verify routing methods return correct values for each complexity."""

    def test_chunk_limit_simple(self):
        ts = TaskSpec(complexity="simple")
        assert ts.get_chunk_limit() == 4

    def test_chunk_limit_medium(self):
        ts = TaskSpec(complexity="medium")
        assert ts.get_chunk_limit() == 10

    def test_chunk_limit_complex(self):
        ts = TaskSpec(complexity="complex")
        assert ts.get_chunk_limit() == 16

    def test_context_chars_simple(self):
        ts = TaskSpec(complexity="simple")
        assert ts.get_context_chars() == 3072

    def test_context_chars_complex(self):
        ts = TaskSpec(complexity="complex")
        assert ts.get_context_chars() == 12288

    def test_num_predict_simple(self):
        ts = TaskSpec(complexity="simple")
        assert ts.get_num_predict() == 1536  # Includes ~1K thinking overhead

    def test_num_predict_complex(self):
        ts = TaskSpec(complexity="complex")
        assert ts.get_num_predict() == 2048

    def test_agent_mode_complex(self):
        ts = TaskSpec(complexity="complex")
        assert ts.should_use_agent_mode() is True

    def test_agent_mode_medium(self):
        ts = TaskSpec(complexity="medium")
        assert ts.should_use_agent_mode() is False


class TestTaskSpecAutoTools:
    """Verify tool auto-selection from intent+domain routing matrix."""

    def test_hr_compare_tools(self):
        ts = TaskSpec(intent="compare", domain="hr")
        tools = ts.get_auto_tools()
        assert "resumes" in tools
        assert "insights" in tools

    def test_medical_extract_tools(self):
        ts = TaskSpec(intent="extract", domain="medical")
        tools = ts.get_auto_tools()
        assert "medical" in tools

    def test_generate_content_tools(self):
        ts = TaskSpec(intent="generate", domain="content")
        tools = ts.get_auto_tools()
        assert "email_drafting" in tools or "content_generate" in tools

    def test_factual_general_empty_tools(self):
        ts = TaskSpec(intent="factual", domain="general")
        tools = ts.get_auto_tools()
        assert isinstance(tools, list)

    def test_tools_deduplicated(self):
        ts = TaskSpec(intent="summarize", domain="hr")
        tools = ts.get_auto_tools()
        assert len(tools) == len(set(tools))


class TestTaskSpecNaturalText:
    """Verify human-readable task description generation."""

    def test_basic_natural_text(self):
        ts = TaskSpec(intent="compare", domain="hr", scope="cross_document")
        text = ts.to_natural_text()
        assert "compare" in text.lower()
        assert "hr" in text.lower()
        assert "cross_document" in text.lower()

    def test_entities_in_natural_text(self):
        ts = TaskSpec(entities=["Python", "Java"])
        text = ts.to_natural_text()
        assert "Python" in text
        assert "Java" in text

    def test_constraints_in_natural_text(self):
        ts = TaskSpec(constraints={"min_experience": 5})
        text = ts.to_natural_text()
        assert "min_experience" in text
        assert "5" in text

    def test_output_format_in_natural_text(self):
        ts = TaskSpec(output_format="table")
        text = ts.to_natural_text()
        assert "table" in text.lower()

    def test_paragraph_format_omitted(self):
        ts = TaskSpec(output_format="paragraph")
        text = ts.to_natural_text()
        assert "paragraph" not in text.lower()


class TestTaskSpecSerialization:
    """Verify JSON round-trip serialization."""

    def test_to_dict(self):
        ts = TaskSpec(intent="compare", domain="hr", entities=["Python"])
        d = ts.to_dict()
        assert d["intent"] == "compare"
        assert d["domain"] == "hr"
        assert d["entities"] == ["Python"]

    def test_to_json(self):
        ts = TaskSpec(intent="factual")
        raw = ts.to_json()
        parsed = json.loads(raw)
        assert parsed["intent"] == "factual"

    def test_from_dict(self):
        d = {"intent": "rank", "domain": "medical", "entities": ["patient"]}
        ts = TaskSpec.from_dict(d)
        assert ts.intent == "rank"
        assert ts.domain == "medical"
        assert ts.entities == ["patient"]

    def test_from_json(self):
        raw = '{"intent": "summarize", "domain": "legal", "confidence": 0.9}'
        ts = TaskSpec.from_json(raw)
        assert ts.intent == "summarize"
        assert ts.domain == "legal"
        assert ts.confidence == 0.9

    def test_roundtrip(self):
        original = TaskSpec(
            intent="compare", domain="hr", output_format="table",
            entities=["Python", "Java"], constraints={"min_exp": 3},
            scope="cross_document", complexity="complex", confidence=0.85,
        )
        raw = original.to_json()
        restored = TaskSpec.from_json(raw)
        assert restored.intent == original.intent
        assert restored.domain == original.domain
        assert restored.entities == original.entities
        assert restored.constraints == original.constraints

    def test_invalid_intent_coerced(self):
        ts = TaskSpec.from_dict({"intent": "nonsense"})
        assert ts.intent == "factual"

    def test_invalid_domain_coerced(self):
        ts = TaskSpec.from_dict({"domain": "xyz"})
        assert ts.domain == "general"

    def test_confidence_clamped(self):
        ts = TaskSpec.from_dict({"confidence": 1.5})
        assert ts.confidence == 1.0
        ts2 = TaskSpec.from_dict({"confidence": -0.5})
        assert ts2.confidence == 0.0

    def test_entities_coerced_from_string(self):
        ts = TaskSpec.from_dict({"entities": "Python"})
        assert ts.entities == ["Python"]

    def test_missing_fields_have_defaults(self):
        ts = TaskSpec.from_dict({})
        assert ts.intent == "factual"
        assert ts.domain == "general"
        assert ts.complexity == "medium"


class TestTaskSpecValidEnums:
    """Verify all valid enum values are accepted."""

    @pytest.mark.parametrize("intent", list(VALID_INTENTS))
    def test_valid_intents(self, intent):
        ts = TaskSpec.from_dict({"intent": intent})
        assert ts.intent == intent

    @pytest.mark.parametrize("domain", list(VALID_DOMAINS))
    def test_valid_domains(self, domain):
        ts = TaskSpec.from_dict({"domain": domain})
        assert ts.domain == domain

    @pytest.mark.parametrize("fmt", list(VALID_OUTPUT_FORMATS))
    def test_valid_output_formats(self, fmt):
        ts = TaskSpec.from_dict({"output_format": fmt})
        assert ts.output_format == fmt

    @pytest.mark.parametrize("scope", list(VALID_SCOPES))
    def test_valid_scopes(self, scope):
        ts = TaskSpec.from_dict({"scope": scope})
        assert ts.scope == scope

    @pytest.mark.parametrize("comp", list(VALID_COMPLEXITIES))
    def test_valid_complexities(self, comp):
        ts = TaskSpec.from_dict({"complexity": comp})
        assert ts.complexity == comp
