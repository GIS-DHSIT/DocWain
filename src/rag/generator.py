from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

from src.rag.answer_stability import (
    build_finalizer_prompt,
    compress_context_for_retry,
    derive_manifest,
    is_bad_refusal,
    is_effectively_empty,
    select_schema,
    validate_schema_output,
)


def generate_rag_answer(
    *,
    llm_client: Any,
    query: str,
    context: str,
    chunks: Sequence[Any],
    prompt_builder: Callable[[str, str, str, str], str],
    approx_token_count: Callable[[str], int],
    num_ctx: int,
    num_predict: int,
    safety_margin: int = 128,
) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic generation wrapper:
    - If evidence exists, always attempt an answer (partial if needed).
    - Handles Ollama thinking-only output via a finalizer call (no CoT leakage).
    - Retries on truncation (done_reason='length'), empty output, bad refusals, and schema violations.
    """
    has_context = bool(chunks) and bool((context or "").strip())
    evidence_strong = (len(chunks) >= 2) or (len(context or "") >= 800)

    schema = select_schema(query)
    schema_selector = (
        "S1: Structured Answer (sections + bullets)\n"
        "S2: Field Extraction (key:value blocks)\n"
        "S3: Comparison/Ranking Table (Markdown table)\n"
        "S4: Timeline (chronological bullets)\n"
        f"Chosen schema: {schema.schema_id} ({schema.title})\n"
        f"{schema.instructions}"
    )

    manifest_text, dominant_lang, dominant_category, context_titles = derive_manifest(chunks)

    def _finalizer_builder(max_tokens_hint: str) -> Dict[str, Any]:
        return {
            "build_prompt": lambda thinking: build_finalizer_prompt(
                draft_thinking=thinking,
                query=query,
                context_titles=context_titles,
                manifest_text=manifest_text,
                schema=schema,
                max_tokens_hint=max_tokens_hint,
            )
        }

    def _call_llm(prompt_text: str, *, max_tokens_hint: str) -> Tuple[str, Dict[str, Any]]:
        if hasattr(llm_client, "generate_with_meta"):
            payload = llm_client.generate_with_meta(
                prompt_text,
                max_retries=1,
                backoff=0.0,
                options={"num_ctx": num_ctx, "num_predict": num_predict},
                finalizer=_finalizer_builder(max_tokens_hint) if has_context else None,
            )
            return str(payload.get("response") or "").strip(), payload
        return str(llm_client.generate(prompt_text) or "").strip(), {}

    def _is_retryable(answer_text: str, payload: Dict[str, Any], *, schema_check: bool) -> Tuple[bool, str]:
        done_reason = str(payload.get("done_reason") or "").lower()
        if done_reason == "length":
            return True, "done_reason_length"
        try:
            prompt_eval_count = int(payload.get("prompt_eval_count")) if payload.get("prompt_eval_count") is not None else None
        except Exception:
            prompt_eval_count = None
        if prompt_eval_count is not None:
            available = int(num_ctx) - int(prompt_eval_count) - int(safety_margin)
            if available < 256:
                return True, f"low_token_budget:{available}"
        if is_effectively_empty(answer_text):
            return True, "empty_or_too_short"
        if has_context and is_bad_refusal(answer_text):
            return True, "bad_refusal_with_evidence"
        if schema_check:
            ok, reason = validate_schema_output(answer_text or "", schema)
            if not ok:
                return True, f"schema_invalid:{reason}"
        return False, "ok"

    compressed_context = ""

    prompt0 = prompt_builder(context or "", manifest_text, schema_selector, "Aim for <= 700 tokens; be structured and human-like.")
    approx_prompt_tokens = approx_token_count(prompt0)
    if evidence_strong and approx_prompt_tokens > max(0, num_ctx - safety_margin - 256):
        compressed_context = compress_context_for_retry(chunks, top_k=6, max_excerpt_chars=1200)
        prompt0 = prompt_builder(compressed_context, manifest_text, schema_selector, "Aim for <= 700 tokens; be structured and human-like.")

    attempts: List[Dict[str, Any]] = []
    best_answer = ""

    answer0, payload0 = _call_llm(prompt0, max_tokens_hint="Aim for <= 700 tokens; be structured and human-like.")
    attempts.append(
        {
            "attempt": 0,
            "done_reason": payload0.get("done_reason"),
            "prompt_eval_count": payload0.get("prompt_eval_count"),
            "eval_count": payload0.get("eval_count"),
            "finalizer_used": bool(payload0.get("finalizer_used")),
            "used_compressed_context": bool(compressed_context),
            "response_len": len(answer0 or ""),
        }
    )
    best_answer = answer0
    retryable, _ = _is_retryable(answer0, payload0, schema_check=schema.schema_id == "S3")

    if retryable:
        if not compressed_context:
            compressed_context = compress_context_for_retry(chunks, top_k=6, max_excerpt_chars=1200)
        prompt1 = prompt_builder(
            compressed_context,
            manifest_text,
            schema_selector,
            "Keep the answer concise (<= 600 tokens). Prefer structured bullets/tables.",
        )
        answer1, payload1 = _call_llm(prompt1, max_tokens_hint="Keep the answer concise (<= 600 tokens). Prefer structured bullets/tables.")
        attempts.append(
            {
                "attempt": 1,
                "done_reason": payload1.get("done_reason"),
                "prompt_eval_count": payload1.get("prompt_eval_count"),
                "eval_count": payload1.get("eval_count"),
                "finalizer_used": bool(payload1.get("finalizer_used")),
                "used_compressed_context": True,
                "response_len": len(answer1 or ""),
            }
        )
        if not is_effectively_empty(answer1):
            best_answer = answer1
        retryable2, _ = _is_retryable(answer1, payload1, schema_check=schema.schema_id == "S3")

        if retryable2:
            schema_selector_strict = schema_selector + "\n\nHARD OUTPUT LIMIT: Do not exceed 350-500 tokens."
            prompt2 = prompt_builder(
                compressed_context,
                manifest_text,
                schema_selector_strict,
                "HARD LIMIT: Do not exceed 350-500 tokens.",
            )
            answer2, payload2 = _call_llm(prompt2, max_tokens_hint="HARD LIMIT: Do not exceed 350-500 tokens.")
            attempts.append(
                {
                    "attempt": 2,
                    "done_reason": payload2.get("done_reason"),
                    "prompt_eval_count": payload2.get("prompt_eval_count"),
                    "eval_count": payload2.get("eval_count"),
                    "finalizer_used": bool(payload2.get("finalizer_used")),
                    "used_compressed_context": True,
                    "response_len": len(answer2 or ""),
                }
            )
            if not is_effectively_empty(answer2):
                best_answer = answer2

    answer = best_answer or ""

    if has_context and is_bad_refusal(answer):
        answer = ""

    if has_context and is_effectively_empty(answer):
        sources_only = "\n".join(f"- {t}" for t in (context_titles or [])[:8]) or "- (none)"
        answer = (
            "I found relevant excerpts in your documents, but some requested details are missing from the retrieved content.\n\n"
            "What I checked (sources):\n"
            f"{sources_only}\n\n"
            "Partial answer:\n"
            "- Available details: Missing/Not found in retrieved excerpts.\n"
            "- Inferred details: None (insufficient explicit evidence in retrieved excerpts).\n"
            "- Missing/Not found in retrieved excerpts: Please specify the exact field(s) or candidate name to narrow retrieval."
        ).strip()

    diagnostics = {
        "dominant_language": dominant_lang,
        "dominant_category": dominant_category,
        "schema": schema.schema_id,
        "attempts": attempts,
    }
    return answer, diagnostics
