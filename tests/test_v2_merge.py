import pytest


class TestMergePromote:
    def test_v2_modelfile_content(self):
        from src.finetune.v2.merge_promote import generate_v2_modelfile
        content = generate_v2_modelfile("/path/to/model.gguf")
        assert "DocWain" in content
        assert "vision" in content.lower()
        assert "tool" in content.lower() or "tool_call" in content
        assert "temperature" in content.lower()

    def test_promotion_plan(self):
        from src.finetune.v2.merge_promote import plan_promotion
        plan = plan_promotion()
        actions = [p["action"] for p in plan]
        assert "backup_v1" in actions
        assert "create_v2" in actions
        assert "update_latest" in actions

    def test_regression_test_criteria(self):
        from src.finetune.v2.merge_promote import get_regression_criteria
        criteria = get_regression_criteria()
        assert "persona_match" in criteria
        assert "rag_accuracy" in criteria
        assert all(v > 0 for v in criteria.values())
