from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.rag.doc_inventory import DocInventoryItem
from src.rag.qdrant_profile_digest import build_profile_digest


@dataclass
class FakePoint:
    payload: Dict[str, object]


@dataclass
class ScrollResult:
    points: List[FakePoint]


class FakeRedis:
    def __init__(self):
        self.store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value


class FakeQdrant:
    def __init__(self, payloads_by_doc: Dict[str, List[Dict[str, object]]]):
        self.payloads_by_doc = payloads_by_doc
        self.calls = 0

    def scroll(self, *, collection_name: str, scroll_filter: Dict[str, object], limit: int, with_payload: bool, with_vectors: bool):
        self.calls += 1
        doc_id = None
        source_file = None
        for cond in scroll_filter.get("must", []):
            if cond.get("key") == "document_id":
                doc_id = cond.get("match", {}).get("value")
            if cond.get("key") == "source_file":
                source_file = cond.get("match", {}).get("value")
        key = str(doc_id or source_file or "")
        payloads = self.payloads_by_doc.get(key, [])[:limit]
        points = [FakePoint(payload=p) for p in payloads]
        return ScrollResult(points=points)


def test_profile_digest_build_and_invalidate():
    payloads = {
        "doc-1": [{"text": "Alice Summary", "document_id": "doc-1", "page": 1}],
        "doc-2": [{"text": "Bob Report", "document_id": "doc-2", "page": 2}],
        "doc-3": [{"text": "Cara Invoice", "document_id": "doc-3", "page": 3}],
    }
    qdrant = FakeQdrant(payloads_by_doc=payloads)
    redis = FakeRedis()

    docs = [
        DocInventoryItem(doc_id="doc-1", source_file="a.pdf", document_name="a.pdf", doc_type="report"),
        DocInventoryItem(doc_id="doc-2", source_file="b.pdf", document_name="b.pdf", doc_type="report"),
    ]

    digest1 = build_profile_digest(
        qdrant_client=qdrant,
        collection_name="col",
        profile_id="profile",
        subscription_id="sub",
        redis_client=redis,
        doc_inventory=docs,
    )
    assert digest1["doc_count"] == 2
    first_calls = qdrant.calls

    digest2 = build_profile_digest(
        qdrant_client=qdrant,
        collection_name="col",
        profile_id="profile",
        subscription_id="sub",
        redis_client=redis,
        doc_inventory=docs,
    )
    assert digest2["doc_count"] == 2
    assert qdrant.calls == first_calls

    docs_updated = docs + [
        DocInventoryItem(doc_id="doc-3", source_file="c.pdf", document_name="c.pdf", doc_type="invoice"),
    ]
    digest3 = build_profile_digest(
        qdrant_client=qdrant,
        collection_name="col",
        profile_id="profile",
        subscription_id="sub",
        redis_client=redis,
        doc_inventory=docs_updated,
    )
    assert digest3["doc_count"] == 3
    assert qdrant.calls > first_calls
