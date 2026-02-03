import time

from src.rag.cache_guard import CacheGuard
from src.rag.entity_detector import EntityDetectionResult


def _entities():
    return EntityDetectionResult(people=["Alice"], products=[], documents=["DocA"], raw_matches=[])


def test_cache_key_changes_on_corpus_fingerprint():
    guard = CacheGuard(ttl_seconds=600)
    context_a = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        session_id="sess",
        query_text="What is the total?",
        intent_type="lookup_fact",
        scope="single_doc",
        target_docs=["doc-1"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        doc_set=["doc-1"],
    )
    context_b = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        session_id="sess",
        query_text="What is the total?",
        intent_type="lookup_fact",
        scope="single_doc",
        target_docs=["doc-1"],
        entities=_entities(),
        corpus_fingerprint="fp-2",
        model_id="llama3.2",
        doc_set=["doc-1"],
    )
    assert guard.build_cache_key(context_a) != guard.build_cache_key(context_b)


def test_cache_rejects_stale_payload():
    guard = CacheGuard(ttl_seconds=600)
    context = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        session_id="sess",
        query_text="Summarize",
        intent_type="summarize",
        scope="multi_doc",
        target_docs=["doc-1", "doc-2"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        doc_set=["doc-1", "doc-2"],
        wants_all_docs=True,
    )
    cache_key = guard.build_cache_key(context)
    payload = {
        "cache_key": cache_key,
        "created_at": time.time() - 700,
        "metadata": context.to_metadata(),
        "response": {"response": "ok"},
    }
    decision = guard.evaluate_cached_payload(context=context, cache_key=cache_key, cached_payload=payload)
    assert not decision.hit


def test_cache_bypasses_multi_doc_without_doc_set():
    guard = CacheGuard(ttl_seconds=600)
    context = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        session_id="sess",
        query_text="Compare all documents",
        intent_type="compare",
        scope="multi_doc",
        target_docs=["doc-1", "doc-2"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        doc_set=[],
        wants_all_docs=True,
    )
    cache_key = guard.build_cache_key(context)
    payload = {
        "cache_key": cache_key,
        "created_at": time.time(),
        "metadata": context.to_metadata(),
        "response": {"response": "ok"},
    }
    decision = guard.evaluate_cached_payload(context=context, cache_key=cache_key, cached_payload=payload)
    assert not decision.hit
