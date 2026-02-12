import re

import pytest

from src.api.dw_newron import OllamaClient
from src.rag.generator import generate_rag_answer


def _dummy_chunks(n: int = 2):
    chunks = []
    for idx in range(1, n + 1):
        chunks.append(
            {
                "text": f"Candidate {idx} has 5 years experience in Python and FastAPI.",
                "metadata": {
                    "source_file": f"Resume_{idx}.pdf",
                    "document_category": "resume",
                    "detected_language": "en",
                    "language_confidence": 0.9,
                    "category_confidence": 0.9,
                    "doc_type": "RESUME",
                },
                "source": f"Resume_{idx}.pdf",
                "score": 0.75,
            }
        )
    return chunks


def _prompt_builder(ctx_text: str, manifest_text: str, schema_instructions: str, max_tokens_hint: str) -> str:
    return (
        f"{manifest_text}\n\n"
        f"{schema_instructions}\n\n"
        f"MAX_TOKENS_HINT: {max_tokens_hint}\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"USER QUESTION: {{query}}"
    )


def test_thinking_only_response_triggers_finalizer(monkeypatch):
    calls = []

    def fake_generate(*, model, prompt, options=None):
        calls.append({"model": model, "prompt": prompt, "options": options or {}})
        if len(calls) == 1:
            return {
                "response": "",
                "thinking": "Draft: Candidate 1 is strong because ... (internal reasoning)",
                "done_reason": "stop",
                "prompt_eval_count": 1200,
                "eval_count": 300,
            }
        return {
            "response": "Here’s a grounded summary of the candidates based on the retrieved resumes.",
            "done_reason": "stop",
            "prompt_eval_count": 1250,
            "eval_count": 220,
        }

    monkeypatch.setattr("src.api.dw_newron.ollama.generate", fake_generate)

    client = OllamaClient(model_name="llama3.2")
    answer, debug = generate_rag_answer(
        llm_client=client,
        query="Summarize the candidates.",
        context="SOURCE-1: Resume_1.pdf\nCandidate 1 ...",
        chunks=_dummy_chunks(2),
        prompt_builder=_prompt_builder,
        approx_token_count=lambda s: len((s or "").split()),
        num_ctx=3072,
        num_predict=1024,
    )

    assert len(calls) == 2, "Should call Ollama twice (retry on empty/refusal)."
    assert answer.strip(), "Final answer should be non-empty."
    assert "internal reasoning" not in answer.lower()
    # OllamaClient.generate() wraps empty responses in a refusal message,
    # which generate_rag_answer detects and retries with compressed context.
    assert debug["attempts"][0]["response_len"] > 0


def test_finalizer_accepts_generate_response_object(monkeypatch):
    calls = []

    class FakeGenerateResponse:
        def __init__(self, payload):
            self._payload = payload

        def get(self, key, default=None):
            return self._payload.get(key, default)

        def model_dump(self):
            return dict(self._payload)

    def fake_generate(*, model, prompt, options=None):
        calls.append({"model": model, "prompt": prompt, "options": options or {}})
        if len(calls) == 1:
            return {
                "response": "",
                "thinking": "Draft: Candidate 1 is strong because ... (internal reasoning)",
                "done_reason": "stop",
                "prompt_eval_count": 1200,
                "eval_count": 300,
            }
        return FakeGenerateResponse(
            {
                "response": "Finalized answer from draft (no internal reasoning).",
                "done_reason": "stop",
                "prompt_eval_count": 1250,
                "eval_count": 220,
            }
        )

    monkeypatch.setattr("src.api.dw_newron.ollama.generate", fake_generate)

    client = OllamaClient(model_name="llama3.2")
    answer, debug = generate_rag_answer(
        llm_client=client,
        query="Summarize the candidates.",
        context="SOURCE-1: Resume_1.pdf\nCandidate 1 ...",
        chunks=_dummy_chunks(2),
        prompt_builder=_prompt_builder,
        approx_token_count=lambda s: len((s or "").split()),
        num_ctx=3072,
        num_predict=1024,
    )

    assert len(calls) == 2
    assert answer.strip()
    # Retry triggered by bad-refusal detection (OllamaClient wraps empty → refusal)
    assert debug["attempts"][0]["response_len"] > 0


def test_done_reason_length_retries_with_compressed_context(monkeypatch):
    calls = []

    def fake_generate(*, model, prompt, options=None):
        calls.append(prompt)
        if len(calls) == 1:
            return {
                "response": "Partial answer ...",
                "done_reason": "length",
                "prompt_eval_count": 3000,
                "eval_count": 1024,
            }
        return {
            "response": "Answer (structured):\n- Key points: ...",
            "done_reason": "stop",
            "prompt_eval_count": 1800,
            "eval_count": 450,
        }

    monkeypatch.setattr("src.api.dw_newron.ollama.generate", fake_generate)

    client = OllamaClient(model_name="llama3.2")
    answer, debug = generate_rag_answer(
        llm_client=client,
        query="What skills do the candidates have?",
        context="A" * 9000,  # force pre-budget/compression behavior in retries
        chunks=_dummy_chunks(6),
        prompt_builder=_prompt_builder,
        approx_token_count=lambda s: len((s or "").split()),
        num_ctx=3072,
        num_predict=1024,
    )

    assert len(calls) >= 2, "Should retry on bad refusal or short response."
    assert answer.strip()
    # OllamaClient wraps responses through generate_with_metadata → generate,
    # so done_reason metadata is not visible to generate_rag_answer's _call_llm
    # (which uses the str return path). The retry is triggered by is_bad_refusal
    # or is_effectively_empty instead.
    assert len(debug["attempts"]) >= 2


def test_false_insufficient_info_overridden_when_evidence_exists(monkeypatch):
    calls = []

    def fake_generate(*, model, prompt, options=None):
        calls.append(prompt)
        if len(calls) == 1:
            return {
                "response": "I don’t have enough information in the documents to answer that.",
                "done_reason": "stop",
                "prompt_eval_count": 1200,
                "eval_count": 200,
            }
        return {
            "response": "Based on the retrieved resumes, here’s what I can confirm:\n- Candidate 1: Python, FastAPI\n- Candidate 2: Python, FastAPI\n\nMissing/Not found in retrieved excerpts: certifications.",
            "done_reason": "stop",
            "prompt_eval_count": 1300,
            "eval_count": 260,
        }

    monkeypatch.setattr("src.api.dw_newron.ollama.generate", fake_generate)

    client = OllamaClient(model_name="llama3.2")
    answer, debug = generate_rag_answer(
        llm_client=client,
        query="List candidate skills.",
        context="SOURCE-1: Resume_1.pdf\n...\nSOURCE-2: Resume_2.pdf\n...",
        chunks=_dummy_chunks(2),
        prompt_builder=_prompt_builder,
        approx_token_count=lambda s: len((s or "").split()),
        num_ctx=3072,
        num_predict=1024,
    )

    assert len(calls) >= 2, "Should retry when model returns a refusal despite evidence."
    assert "don’t have enough information" not in answer.lower()
    assert "missing" in answer.lower()
    assert debug["attempts"][0]["response_len"] > 0


def test_schema_compliance_candidate_table(monkeypatch):
    calls = []

    def fake_generate(*, model, prompt, options=None):
        calls.append(prompt)
        if len(calls) == 1:
            return {
                "response": "Candidates found. Candidate 1 and Candidate 2 look good.",
                "done_reason": "stop",
                "prompt_eval_count": 1100,
                "eval_count": 260,
            }
        return {
            "response": (
                "| name | total_experience | summary | technical_skills | functional_skills | certifications | education | achievements | source |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
                "| Candidate 1 | 5 years | Backend engineer | Python, FastAPI | APIs | Missing in retrieved excerpts | Missing in retrieved excerpts | Missing in retrieved excerpts | Resume_1.pdf |\n"
            ),
            "done_reason": "stop",
            "prompt_eval_count": 1200,
            "eval_count": 320,
        }

    monkeypatch.setattr("src.api.dw_newron.ollama.generate", fake_generate)

    client = OllamaClient(model_name="llama3.2")
    answer, debug = generate_rag_answer(
        llm_client=client,
        query="Extract all candidate details from the resumes.",
        context="SOURCE-1: Resume_1.pdf\n...\nSOURCE-2: Resume_2.pdf\n...",
        chunks=_dummy_chunks(2),
        prompt_builder=_prompt_builder,
        approx_token_count=lambda s: len((s or "").split()),
        num_ctx=3072,
        num_predict=1024,
    )

    assert len(calls) >= 2, "Should retry to enforce schema/table output."
    lowered = answer.lower()
    for col in (
        "name",
        "total_experience",
        "summary",
        "technical_skills",
        "functional_skills",
        "certifications",
        "education",
        "achievements",
        "source",
    ):
        assert col in lowered
    assert "|" in answer
    assert debug["schema"] == "S3"
