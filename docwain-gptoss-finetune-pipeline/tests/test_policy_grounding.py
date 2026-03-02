import re
from decimal import Decimal, InvalidOperation

from docwain_ft.eval_harness import load_eval_outputs


DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONEY_PATTERN = re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b")


def _extract_claims(text: str) -> set[str]:
    claims = set()
    for pattern in [DATE_PATTERN, MONEY_PATTERN, NAME_PATTERN]:
        claims.update(pattern.findall(text))
    return claims


def _parse_money(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace("$", "").replace(",", ""))
    except InvalidOperation:
        return None


def _money_is_derived(claim: str, context: str) -> bool:
    claim_value = _parse_money(claim)
    if claim_value is None:
        return False
    context_values = [_parse_money(match) for match in MONEY_PATTERN.findall(context)]
    context_values = [value for value in context_values if value is not None]
    if len(context_values) < 2:
        return False
    for i, left in enumerate(context_values):
        for right in context_values[i + 1 :]:
            if (left - right).copy_abs() == claim_value:
                return True
    return False


def test_policy_grounding():
    rows = load_eval_outputs()
    for row in rows:
        answer = row["answer"]
        context = row["context"]
        for claim in _extract_claims(answer):
            if "Inference" in answer:
                continue
            if claim in context:
                continue
            if NAME_PATTERN.fullmatch(claim) and f"{claim}:" in answer:
                continue
            if MONEY_PATTERN.fullmatch(claim) and _money_is_derived(claim, context):
                continue
            assert claim in context, f"Claim '{claim}' not found in context"
