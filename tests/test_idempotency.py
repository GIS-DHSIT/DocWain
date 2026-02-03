from src.utils import idempotency
from src.utils.idempotency import acquire_lock, release_lock


def test_idempotency_lock_memory_backend():
    idempotency.get_redis_client = lambda: None
    first = acquire_lock(stage="embedding", document_id="doc-1", subscription_id="sub-1", ttl_seconds=5)
    second = acquire_lock(stage="embedding", document_id="doc-1", subscription_id="sub-1", ttl_seconds=5)

    assert first.acquired is True
    assert second.acquired is False

    release_lock(first)

    third = acquire_lock(stage="embedding", document_id="doc-1", subscription_id="sub-1", ttl_seconds=5)
    assert third.acquired is True
    release_lock(third)
