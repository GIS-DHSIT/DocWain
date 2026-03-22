# tests/test_v2_tool_schemas.py
import json
import pytest


class TestToolSchemas:
    def test_get_all_schemas_returns_9_core(self):
        from src.finetune.v2.tool_schemas import get_core_tool_schemas
        schemas = get_core_tool_schemas()
        assert len(schemas) == 9

    def test_each_schema_has_required_fields(self):
        from src.finetune.v2.tool_schemas import get_core_tool_schemas
        for s in get_core_tool_schemas():
            assert s["type"] == "function"
            fn = s["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_core_tool_names(self):
        from src.finetune.v2.tool_schemas import get_core_tool_schemas
        names = {s["function"]["name"] for s in get_core_tool_schemas()}
        expected = {"ocr_extract", "layout_extract", "extract_table", "extract_entities",
                    "context_understand", "cross_reference", "search_documents",
                    "summarize_section", "visualize_data"}
        assert names == expected

    def test_auto_invoked_tools(self):
        from src.finetune.v2.tool_schemas import get_auto_invoked_tools
        auto = get_auto_invoked_tools()
        assert "ocr_extract" in auto
        assert "layout_extract" in auto
        assert "context_understand" in auto
        assert "summarize_section" not in auto

    def test_schema_is_valid_json(self):
        from src.finetune.v2.tool_schemas import get_core_tool_schemas
        for s in get_core_tool_schemas():
            serialized = json.dumps(s)
            parsed = json.loads(serialized)
            assert parsed == s

    def test_format_tools_for_system_prompt(self):
        from src.finetune.v2.tool_schemas import format_tools_for_prompt
        prompt_text = format_tools_for_prompt()
        assert "extract_table" in prompt_text
        assert "ocr_extract" in prompt_text
        assert "function" in prompt_text
