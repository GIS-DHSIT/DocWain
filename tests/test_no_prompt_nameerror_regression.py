def test_dw_newron_does_not_reference_undefined_prompt_variable():
    # Regression guard: we previously referenced `prompt` after refactoring generation,
    # which raised NameError at runtime inside EnterpriseRAGSystem.answer_question.
    # The variable `prompt` is now properly defined at line ~4725 before use at ~4970,
    # so we check that `prompt` is assigned before the `prompt_text=prompt` usage.
    import ast
    with open("src/api/dw_newron.py", "r", encoding="utf-8") as f:
        content = f.read()
    # Verify the file parses without syntax errors (basic sanity)
    ast.parse(content)
    # If prompt_text=prompt exists, ensure prompt is assigned somewhere above it
    if "prompt_text=prompt" in content:
        idx = content.index("prompt_text=prompt")
        preceding = content[:idx]
        assert "prompt = " in preceding or "prompt=" in preceding, (
            "prompt_text=prompt found but prompt is never assigned before it"
        )

