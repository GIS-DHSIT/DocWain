# tests/test_v2_dataset.py
import json
import pytest
from pathlib import Path


class TestDatasetRegistry:
    def test_list_available_datasets(self):
        from src.finetune.v2.dataset_download import list_datasets
        ds = list_datasets()
        assert "docvqa" in ds
        assert "chartvqa" in ds
        assert "pubtabnet" in ds
        assert "doclaynet" in ds

    def test_dataset_info_has_required_fields(self):
        from src.finetune.v2.dataset_download import get_dataset_info
        info = get_dataset_info("docvqa")
        assert "hf_id" in info
        assert "split" in info
        assert "phase" in info
        assert "sample_size" in info


class TestDatasetPreprocess:
    def test_format_vision_sft_pair(self):
        from src.finetune.v2.dataset_preprocess import format_vision_sft
        pair = format_vision_sft(
            image_path="/tmp/test.png",
            question="What is in the table?",
            answer="The table shows quarterly revenue.",
            tools_json="[]",
        )
        assert "messages" in pair
        assert len(pair["messages"]) == 3
        assert pair["messages"][0]["role"] == "system"
        assert "<image>" in pair["messages"][1]["content"]

    def test_format_tool_call_pair(self):
        from src.finetune.v2.dataset_preprocess import format_tool_call_sft
        pair = format_tool_call_sft(
            query="Extract the table on page 3",
            tool_calls=[{"name": "extract_table", "arguments": {"page": 3}}],
            tool_results=[{"rows": [["a", "b"]], "cols": ["c1", "c2"]}],
            final_answer="The table contains...",
            tools_json="[]",
        )
        assert "messages" in pair
        content = str(pair["messages"])
        assert "<tool_call>" in content

    def test_format_parallel_tool_calls(self):
        from src.finetune.v2.dataset_preprocess import format_tool_call_sft
        pair = format_tool_call_sft(
            query="Compare page 3 table with appendix",
            tool_calls=[
                {"name": "extract_table", "arguments": {"page": 3}},
                {"name": "cross_reference", "arguments": {"claim": "spending", "scope": "appendix"}},
            ],
            tool_results=[{"rows": []}, {"related": []}],
            final_answer="Comparison shows...",
            tools_json="[]",
        )
        content = str(pair["messages"])
        assert content.count("<tool_call>") == 2
