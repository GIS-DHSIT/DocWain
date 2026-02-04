def test_dw_newron_does_not_reference_undefined_prompt_variable():
    # Regression guard: we previously referenced `prompt` after refactoring generation,
    # which raised NameError at runtime inside EnterpriseRAGSystem.answer_question.
    with open("src/api/dw_newron.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "prompt_text=prompt" not in content

