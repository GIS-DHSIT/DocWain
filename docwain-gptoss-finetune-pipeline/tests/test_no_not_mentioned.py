from tests.conftest import load_eval_outputs


def test_no_not_mentioned():
    rows = load_eval_outputs()
    for row in rows:
        assert "Not Mentioned" not in row["answer"]
