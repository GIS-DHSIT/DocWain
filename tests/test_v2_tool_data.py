import json, pytest
from pathlib import Path


class TestToolDataGenerator:
    def test_generate_single_tool_scenario(self):
        from src.finetune.v2.tool_data_generator import generate_single_tool_examples
        examples = generate_single_tool_examples()
        assert len(examples) >= 100
        for ex in examples[:5]:
            assert "messages" in ex
            content = str(ex["messages"])
            assert "<tool_call>" in content

    def test_generate_parallel_tool_scenario(self):
        from src.finetune.v2.tool_data_generator import generate_parallel_tool_examples
        examples = generate_parallel_tool_examples()
        assert len(examples) >= 50
        for ex in examples[:5]:
            content = str(ex["messages"])
            assert content.count("<tool_call>") >= 2

    def test_generate_no_tool_scenario(self):
        from src.finetune.v2.tool_data_generator import generate_no_tool_examples
        examples = generate_no_tool_examples()
        assert len(examples) >= 50
        for ex in examples[:5]:
            content = str(ex["messages"])
            assert "<tool_call>" not in content

    def test_generate_auto_invocation_scenario(self):
        from src.finetune.v2.tool_data_generator import generate_auto_invocation_examples
        examples = generate_auto_invocation_examples()
        assert len(examples) >= 50
        for ex in examples[:5]:
            content = str(ex["messages"])
            assert "<tool_response>" in content

    def test_build_full_dataset(self, tmp_path):
        from src.finetune.v2.tool_data_generator import build_tool_calling_dataset
        path = build_tool_calling_dataset(tmp_path / "tools.jsonl")
        assert path.exists()
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) >= 250
