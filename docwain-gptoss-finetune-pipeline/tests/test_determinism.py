import difflib
import shutil

import pytest

from docwain_ft.infer import run_one


def test_determinism():
    if shutil.which("ollama") is None:
        pytest.skip("Ollama not available")

    prompt = "User Query: Summarize the invoice total.\nRetrieved Context:\n- doc_name=Doc_101 page=1 section=Summary chunk_kind=text profile_name=A. Patel\n  Total due is $100.00.\nOutput Rules:\nReturn a single sentence."
    outputs = [run_one(prompt) for _ in range(3)]
    ratio = difflib.SequenceMatcher(None, outputs[0], outputs[1]).ratio()
    ratio2 = difflib.SequenceMatcher(None, outputs[1], outputs[2]).ratio()
    assert min(ratio, ratio2) >= 0.7
