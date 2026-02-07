from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .types import GenericSchema, HRSchema, InvoiceSchema, LegalSchema, LLMBudget, MultiEntitySchema

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    status: str
    reason: str = ""


FALLBACK_STATUS = JudgeResult(status="fail", reason="fallback")


def judge_answer(
    *,
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    llm_client: Optional[object],
    budget: LLMBudget,
    sources_present: bool = True,
    correlation_id: Optional[str] = None,
) -> JudgeResult:
    if not answer:
        return JudgeResult(status="fail", reason="empty_answer")
    if not sources_present:
        return JudgeResult(status="fail", reason="no_sources")

    heuristic = _heuristic_judge(answer, schema, intent)
    if heuristic.status != "uncertain":
        return heuristic

    if llm_client and budget.consume():
        llm_result = _llm_judge(answer, schema, intent, llm_client, correlation_id)
        if llm_result:
            return llm_result

    return JudgeResult(status="fail", reason="uncertain_no_llm")


def _heuristic_judge(
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
) -> JudgeResult:
    if _has_forbidden_tokens(answer):
        return JudgeResult(status="fail", reason="forbidden_tokens")

    if isinstance(schema, MultiEntitySchema):
        lowered = answer.lower()
        if "multiple" not in lowered or not any(tok in lowered for tok in ("candidates", "documents")):
            return JudgeResult(status="fail", reason="multi_entity_not_explicit")

    if isinstance(schema, InvoiceSchema) and intent == "products_list":
        items = schema.items.items if schema.items else None
        if not items:
            if "itemized products/services" in answer:
                return JudgeResult(status="pass")
            if _extract_candidate_items(answer):
                return JudgeResult(status="fail", reason="items_without_evidence")
            return JudgeResult(status="pass")

        allowed_items = {_normalize(item.description) for item in items if item.description}
        for candidate in _extract_candidate_items(answer):
            if _normalize(candidate) not in allowed_items:
                return JudgeResult(status="fail", reason="hallucinated_item")

    has_spans = any(True for _ in _iter_spans(schema))
    if not has_spans:
        return JudgeResult(status="fail", reason="no_evidence_spans")

    answer_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", answer))
    if answer_numbers:
        evidence_numbers = _collect_evidence_numbers(schema)
        if not evidence_numbers:
            return JudgeResult(status="uncertain", reason="numbers_without_evidence")
        if answer_numbers - evidence_numbers:
            return JudgeResult(status="fail", reason="hallucinated_number")

    return JudgeResult(status="pass")


def _collect_evidence_numbers(schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema) -> set:
    numbers = set()
    for span in _iter_spans(schema):
        numbers.update(re.findall(r"\d+(?:[.,]\d+)?", span))
    return numbers


def _iter_spans(schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema) -> Iterable[str]:
    if isinstance(schema, InvoiceSchema):
        items = schema.items.items if schema.items else None
        for item in items or []:
            for span in item.evidence_spans:
                yield span.snippet
        for group in (schema.totals, schema.parties, schema.terms):
            group_items = group.items if group else None
            for item in group_items or []:
                for span in item.evidence_spans:
                    yield span.snippet
    elif isinstance(schema, HRSchema):
        candidates = schema.candidates.items if schema.candidates else None
        for cand in candidates or []:
            for span in cand.evidence_spans:
                yield span.snippet
    elif isinstance(schema, LegalSchema):
        clauses = schema.clauses.items if schema.clauses else None
        for clause in clauses or []:
            for span in clause.evidence_spans:
                yield span.snippet
    elif isinstance(schema, GenericSchema):
        facts = schema.facts.items if schema.facts else None
        for fact in facts or []:
            for span in fact.evidence_spans:
                yield span.snippet
    elif isinstance(schema, MultiEntitySchema):
        for entity in schema.entities or []:
            for span in entity.evidence_spans:
                yield span.snippet


def _extract_candidate_items(answer: str) -> list[str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if line.startswith("-"):
            candidates.append(line.lstrip("- ").strip())
    if not candidates and "Items listed" in answer:
        after = answer.split(":", 1)[-1]
        for part in after.split(";"):
            part = part.strip(" .")
            if part:
                candidates.append(part)
    return candidates


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _llm_judge(
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    llm_client: object,
    correlation_id: Optional[str],
) -> Optional[JudgeResult]:
    prompt = _build_prompt(answer, schema, intent)
    try:
        raw = llm_client.generate(prompt, max_retries=1, backoff=0.4)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 LLM judge failed: %s",
            exc,
            extra={"stage": "judge", "correlation_id": correlation_id},
        )
        return None

    payload = _extract_json(raw)
    verdict = (payload.get("verdict") or "").lower() if isinstance(payload, dict) else ""
    if verdict in {"pass", "fail"}:
        return JudgeResult(status=verdict, reason=str(payload.get("reason") or ""))
    return None


def _build_prompt(
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
) -> str:
    evidence = []
    for span in _iter_spans(schema):
        snippet = " ".join(span.split())[:200]
        if snippet:
            evidence.append(f"- {snippet}")
    return (
        "Check whether the answer is fully supported by the evidence snippets. "
        "Return strict JSON only with fields: verdict (pass/fail) and reason.\n\n"
        f"INTENT: {intent}\n"
        f"ANSWER: {answer}\n"
        "EVIDENCE:\n"
        + "\n".join(evidence)
        + "\n\nJSON:"
    )


def _extract_json(raw: object) -> dict:
    if not raw:
        return {}
    text = str(raw).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    if "{" in text and "}" in text:
        snippet = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def _has_forbidden_tokens(answer: str) -> bool:
    lowered = answer.lower()
    for token in (
        "understanding & scope",
        "evidence & gaps",
        "invoice summary",
        "files searched",
        "files used",
        "documents searched",
        "sources used",
    ):
        if token in lowered:
            return True
    if "answer:" in lowered or re.search(r"^\s*answer\s*[:\-]", lowered):
        return True
    return False


def judge(
    *,
    answer: str,
    schema: InvoiceSchema | HRSchema | LegalSchema | GenericSchema | MultiEntitySchema,
    intent: str,
    llm_client: Optional[object],
    budget: LLMBudget,
    sources_present: bool = True,
    correlation_id: Optional[str] = None,
) -> JudgeResult:
    return judge_answer(
        answer=answer,
        schema=schema,
        intent=intent,
        llm_client=llm_client,
        budget=budget,
        sources_present=sources_present,
        correlation_id=correlation_id,
    )
