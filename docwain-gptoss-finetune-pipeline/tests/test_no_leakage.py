import re

from tests.conftest import load_eval_outputs


UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
KEYWORDS = ["subscription_id", "profile_id", "chunk_id", "embedding"]


def test_no_leakage():
    rows = load_eval_outputs()
    for row in rows:
        answer = row["answer"]
        assert UUID_PATTERN.search(answer) is None
        for key in KEYWORDS:
            assert key not in answer
