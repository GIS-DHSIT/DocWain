import time

from src.rag.cache_guard_v2 import CacheGuardV2
from src.rag.entity_detector import EntityDetectionResult


def _entities():
    return EntityDetectionResult(people=["Alice"], products=[], documents=["DocA"], raw_matches=[])


def test_cache_miss_on_different_query():
    guard = CacheGuardV2(ttl_seconds=600)
    context_a = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        query_text="What is the total?",
        intent_type="lookup_fact",
        scope_type="single_doc",
        target_docs=["doc-1"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        retrieval_signature="sig-1",
        is_vague=False,
    )
    context_b = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        query_text="Summarize the invoice",
        intent_type="summarize",
        scope_type="single_doc",
        target_docs=["doc-1"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        retrieval_signature="sig-1",
        is_vague=False,
    )
    key_a = guard.build_cache_key(context_a)
    key_b = guard.build_cache_key(context_b)
    payload = {
        "cache_key": key_a,
        "created_at": time.time(),
        "metadata": context_a.to_metadata(),
        "response": {"response": "ok"},
    }
    decision = guard.evaluate_cached_payload(context=context_b, cache_key=key_b, cached_payload=payload)
    assert not decision.hit


def test_cache_miss_on_corpus_fingerprint_change():
    guard = CacheGuardV2(ttl_seconds=600)
    context_a = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        query_text="What is the total?",
        intent_type="lookup_fact",
        scope_type="single_doc",
        target_docs=["doc-1"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        retrieval_signature="sig-1",
        is_vague=False,
    )
    context_b = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        query_text="What is the total?",
        intent_type="lookup_fact",
        scope_type="single_doc",
        target_docs=["doc-1"],
        entities=_entities(),
        corpus_fingerprint="fp-2",
        model_id="llama3.2",
        retrieval_signature="sig-1",
        is_vague=False,
    )
    key_a = guard.build_cache_key(context_a)
    key_b = guard.build_cache_key(context_b)
    payload = {
        "cache_key": key_a,
        "created_at": time.time(),
        "metadata": context_a.to_metadata(),
        "response": {"response": "ok"},
    }
    decision = guard.evaluate_cached_payload(context=context_b, cache_key=key_b, cached_payload=payload)
    assert not decision.hit


def test_cache_bypass_on_vague_query():
    guard = CacheGuardV2(ttl_seconds=600)
    context = guard.build_context(
        subscription_id="sub",
        profile_id="profile",
        query_text="Summarize",
        intent_type="summarize",
        scope_type="multi_doc",
        target_docs=["doc-1", "doc-2"],
        entities=_entities(),
        corpus_fingerprint="fp-1",
        model_id="llama3.2",
        retrieval_signature="sig-1",
        is_vague=True,
    )
    key = guard.build_cache_key(context)
    payload = {
        "cache_key": key,
        "created_at": time.time(),
        "metadata": context.to_metadata(),
        "response": {"response": "ok"},
    }
    decision = guard.evaluate_cached_payload(context=context, cache_key=key, cached_payload=payload)
    assert not decision.hit
