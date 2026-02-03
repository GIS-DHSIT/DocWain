import re

from tests.conftest import load_eval_outputs


DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONEY_PATTERN = re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b")


def _extract_claims(text: str) -> set[str]:
    claims = set()
    for pattern in [DATE_PATTERN, MONEY_PATTERN, NAME_PATTERN]:
        claims.update(pattern.findall(text))
    return claims


def test_policy_grounding():
    rows = load_eval_outputs()
    for row in rows:
        answer = row["answer"]
        context = row["context"]
        for claim in _extract_claims(answer):
            if "Inference" in answer:
                continue
            assert claim in context, f"Claim '{claim}' not found in context"
