from __future__ import annotations

from dataclasses import dataclass

from src.ask.models import EvidenceChunk, Plan
from src.ask.repair_loop import RepairLoop
from src.cache.redis_keys import RedisKeys
from src.cache.redis_store import RedisStore
from tests.rag_v2_helpers import FakeRedis


@dataclass
class FakeRetriever:
    calls: int = 0

    def retrieve(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        document_ids = kwargs.get("document_ids") or []
        if "doc-python" in document_ids:
            return [
                EvidenceChunk(
                    text="Alice Johnson lists Python among skills.",
                    score=0.9,
                    metadata={},
                    file_name="Alice Resume.pdf",
                    document_id="doc-python",
                    section_id="sec-1",
                    page=1,
                    chunk_kind="section_text",
                    snippet="Alice Johnson lists Python among skills.",
                    snippet_sha="sha1",
                )
            ]
        return []


def test_repair_loop_improves_evidence():
    redis = FakeRedis()
    store = RedisStore(redis)
    keys = RedisKeys(subscription_id="sub-1", profile_id="profile-A")
    store.set_entity_index(
        keys,
        {
            "entities": {
                "python": [
                    {
                        "file_name": "Alice Resume.pdf",
                        "document_id": "doc-python",
                        "section_id": "sec-1",
                        "page": 1,
                        "snippet_sha": "sha1",
                    }
                ]
            },
            "updated_at": 0,
        },
    )

    plan = Plan(
        intent="extract",
        scope={"subscription_id": "sub-1", "profile_id": "profile-A", "document_id": None},
        query_rewrites=["who is skilled in python?", "python skills"],
        entity_hints=["python"],
        expected_answer_shape="bullets",
        query="who is skilled in python?",
    )

    initial = [
        EvidenceChunk(
            text="General skills overview without the target skill mentioned.",
            score=0.5,
            metadata={},
            file_name="Other.pdf",
            document_id="doc-other",
            section_id="sec-2",
            page=1,
            chunk_kind="section_text",
            snippet="General skills overview without the target skill mentioned.",
            snippet_sha="sha0",
        )
    ]

    retriever = FakeRetriever()
    loop = RepairLoop(redis_client=redis)
    repaired, quality, meta = loop.run(
        plan=plan,
        evidence=initial,
        retriever=retriever,
        subscription_id="sub-1",
        profile_id="profile-A",
        collection_name="sub-1",
        top_k=20,
        max_iter=2,
    )

    assert retriever.calls >= 1
    assert any("python" in chunk.text.lower() for chunk in repaired)
    assert quality.quality in {"MEDIUM", "HIGH"}
    assert meta["iterations"]
