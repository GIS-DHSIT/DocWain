import re

from src.policy.response_policy import (
    INFO_MODE,
    TASK_MODE,
    ResponseModeClassifier,
    apply_evidence_gate,
    build_docwain_intro,
    build_evidence_ledger,
    BANNED_HEDGE_PHRASES,
)


def _ledger(text: str, source_name: str = "Contract.pdf"):
    sources = [
        {"source_id": 1, "source_name": source_name, "excerpt": text},
    ]
    chunks = [
        {"text": text, "metadata": {"source_name": source_name, "chunk_id": "c1"}},
    ]
    ledger = build_evidence_ledger(chunks, sources)
    return ledger, sources


def test_task_mode_strips_docwain_intro():
    query = "Summarize the contract terms."
    assert ResponseModeClassifier.classify(query) == TASK_MODE

    ledger, _ = _ledger("The contract starts on 2024-01-01.")
    answer = (
        "I’m DocWain — a document-based AI assistant. I do not store your privacy info. [SOURCE-1]\n"
        "- The contract starts on 2024-01-01. [SOURCE-1]"
    )

    cleaned, _ = apply_evidence_gate(answer, ledger, response_mode=TASK_MODE)
    forbidden = [
        "i’m docwain",
        "document-based ai assistant",
        "do not store",
        "docs section",
        "privacy",
    ]
    lowered = cleaned.lower()
    for phrase in forbidden:
        assert phrase not in lowered


def test_info_mode_returns_short_intro():
    query = "What is DocWain?"
    assert ResponseModeClassifier.classify(query) == INFO_MODE
    intro = build_docwain_intro()
    assert "docwain" in intro.lower()
    assert len(intro) <= 500  # Structured response format


def test_info_mode_detects_capability_variants():
    queries = [
        "What else can you do?",
        "What all can DocWain do?",
        "How can you help me?",
        "What else can you help with?",
        "How do I compare resumes?",
    ]
    for query in queries:
        assert ResponseModeClassifier.classify(query) == INFO_MODE


def test_hallucinated_doc_name_removed():
    ledger, _ = _ledger("The SLA is 99.9%.", source_name="RealDoc.pdf")
    answer = (
        "- FakeDoc.pdf states the SLA is 99.9%. [SOURCE-1]\n"
        "- RealDoc.pdf states the SLA is 99.9%. [SOURCE-1]"
    )
    cleaned, _ = apply_evidence_gate(answer, ledger, response_mode=TASK_MODE)
    assert "fakedoc.pdf" not in cleaned.lower()
    assert "realdoc.pdf" in cleaned.lower()


def test_banned_hedge_phrases_removed():
    ledger, _ = _ledger("The project began on 2024-03-01.")
    answer = (
        "Project details not mentioned in the documents. [SOURCE-1]\n"
        "- The project began on 2024-03-01. [SOURCE-1]"
    )
    cleaned, _ = apply_evidence_gate(answer, ledger, response_mode=TASK_MODE)
    lowered = cleaned.lower()
    for phrase in BANNED_HEDGE_PHRASES:
        assert phrase not in lowered


def test_conflicting_date_range_removed_but_facts_kept():
    ledger, _ = _ledger("Start date is 2025-01-10. End date is 2024-12-31.")
    answer = (
        "- Timeline: from 2025-01-10 to 2024-12-31. [SOURCE-1]\n"
        "- Start date is 2025-01-10. [SOURCE-1]"
    )
    cleaned, _ = apply_evidence_gate(answer, ledger, response_mode=TASK_MODE)
    assert re.search(r"from 2025-01-10 to 2024-12-31", cleaned) is None
    assert "start date is 2025-01-10" in cleaned.lower()
