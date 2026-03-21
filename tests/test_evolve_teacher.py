import json
import pytest
from pathlib import Path


class TestTeacherPrompts:
    def test_sft_prompt_has_required_sections(self):
        from src.finetune.evolve.prompts.teacher_sft import build_sft_prompt
        prompt = build_sft_prompt(
            query="What are the Q3 figures?",
            category="document_understanding",
            subcategory="table_extraction",
        )
        assert "DocWain" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_dpo_prompt_includes_model_response(self):
        from src.finetune.evolve.prompts.teacher_dpo import build_dpo_prompt
        prompt = build_dpo_prompt(
            query="What are the Q3 figures?",
            model_response="The figures are unavailable.",
            scores={"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.3, "formatting": 0.5, "tone": 0.5},
        )
        assert "figures are unavailable" in prompt
        assert "accuracy" in prompt.lower()
        assert isinstance(prompt, str)


class TestTeacher:
    def test_generate_sft_pair(self):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        t._docwain_system_prompt = "You are DocWain."
        pair = t._format_sft_pair(
            query="What are Q3 figures?",
            ideal_response="Based on the financial table, Q3 revenue was $1.2M.",
        )
        assert "messages" in pair
        msgs = pair["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "DocWain" in msgs[0]["content"]

    def test_generate_dpo_pair(self):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        t._docwain_system_prompt = "You are DocWain."
        pair = t._format_dpo_pair(
            query="What are Q3 figures?",
            chosen="Based on the financial table, Q3 revenue was $1.2M.",
            rejected="The figures are unavailable.",
        )
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"][-1]["role"] == "assistant"
        assert pair["rejected"][-1]["role"] == "assistant"

    def test_filter_content_references(self):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        assert t._is_pattern_not_content("When reading a table, first identify the headers and row structure.")
        assert not t._is_pattern_not_content("The company reported revenue of $45.2 million for fiscal year 2025 in their annual report filed on March 15.")

    def test_save_teaching_output(self, tmp_path):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        t._output_dir = tmp_path
        sft_pairs = [{"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        dpo_pairs = [{"chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
                      "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}]}]
        t._save_output(sft_pairs, dpo_pairs, iteration=1)
        iter_dir = tmp_path / "iter_1"
        assert (iter_dir / "sft_pairs.jsonl").exists()
        assert (iter_dir / "dpo_pairs.jsonl").exists()
        with open(iter_dir / "sft_pairs.jsonl") as f:
            data = json.loads(f.readline())
        assert "messages" in data
