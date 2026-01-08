from src.api import dw_newron


def test_enterprise_rag_system_has_answer_question():
    assert hasattr(dw_newron.EnterpriseRAGSystem, "answer_question")
    assert callable(getattr(dw_newron.EnterpriseRAGSystem, "answer_question"))
