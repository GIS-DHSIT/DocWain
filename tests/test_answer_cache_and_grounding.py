import re

from src.rag.context_reasoning import KeyFact, NumericClaim, WorkingContext
from src.rag.grounding_guard import verify_grounding
from src.rag.query_cache import build_cache_key, compute_retrieval_fingerprint, is_query_answer_consistent, normalize_query
from src.rag.response_formatter import format_structured_response


class DummyChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata
        self.id = metadata.get("id", "")
        self.source = metadata.get("source")


def test_cache_key_changes_on_query_change():
    key_one = build_cache_key(
        subscription_id="sub",
        session_id="sess",
        user_id="user",
        model_name="model",
        prompt_version="v1",
        normalized_query=normalize_query("What is the total?"),
        retrieval_fingerprint="abc",
    )
    key_two = build_cache_key(
        subscription_id="sub",
        session_id="sess",
        user_id="user",
        model_name="model",
        prompt_version="v1",
        normalized_query=normalize_query("List products and services"),
        retrieval_fingerprint="abc",
    )
    assert key_one != key_two


def test_cache_key_changes_on_context_change():
    chunks_one = [
        DummyChunk("Total: $100.00", {"document_id": "doc1", "source_file": "a.pdf", "chunk_id": "c1"}),
    ]
    chunks_two = [
        DummyChunk("Total: $200.00", {"document_id": "doc1", "source_file": "a.pdf", "chunk_id": "c2"}),
    ]
    fp_one = compute_retrieval_fingerprint(chunks_one)
    fp_two = compute_retrieval_fingerprint(chunks_two)
    assert fp_one != fp_two


def test_cache_rejects_mismatched_intent():
    normalized = normalize_query("List products and services")
    response = "The subtotal is $100.00."
    assert not is_query_answer_consistent(normalized, response, "PRODUCTS_SERVICES")


def test_grounding_rejects_unseen_currency():
    chunks = [DummyChunk("Total: £500.00", {"document_id": "doc1", "source_file": "a.pdf"})]
    result = verify_grounding("The total is £638.60.", chunks, intent="TOTALS")
    assert not result.passed


def test_response_minimum_structure():
    context = WorkingContext(
        resolved_scope=["Invoice_A.pdf"],
        key_facts=[
            KeyFact(label="Vendor", value="Acme Corp", doc_name="Invoice_A.pdf", section="Header", chunk_id="c1"),
            KeyFact(label="Service", value="Consulting", doc_name="Invoice_A.pdf", section="Lines", chunk_id="c2"),
        ],
        tables=[],
        numeric_claims=[
            NumericClaim(label="Total", value="$100.00", unit="$", doc_name="Invoice_A.pdf", chunk_id="c3"),
        ],
        contradictions=[],
        missing_fields=[],
    )
    formatted = format_structured_response(
        query="Summarize the invoice",
        intent="SUMMARIZE",
        context=context,
        doc_names=["Invoice_A.pdf"],
    )
    sentences = [s for s in re.split(r"[.!?]+", formatted.text) if s.strip()]
    assert len(sentences) >= 5
    assert "Evidence:" in formatted.text
    assert "Takeaways:" in formatted.text
    assert formatted.text.count("\n- ") >= 2
