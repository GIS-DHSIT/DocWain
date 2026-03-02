import hashlib

from src.embedding.pipeline.chunk_integrity import ChunkIntegrityConfig, enforce_chunk_integrity
from src.embedding.pipeline.dedupe_gate import DedupeConfig, apply_dedupe_gate
from src.embedding.pipeline.embed_pipeline import prepare_embedding_chunks
from src.embedding.pipeline.payload_normalizer import normalize_payload


def _meta_for_section(count: int, section_id: str = "sec-1"):
    return [
        {
            "document_id": "doc-1",
            "section_title": "Section A",
            "section_path": "Section A",
            "section_id": section_id,
            "chunk_index": idx,
            "chunk_type": "text",
        }
        for idx in range(count)
    ]


def _overlap_ratio(prev: str, curr: str) -> float:
    prev_tokens = prev.split()
    curr_tokens = curr.split()
    if not prev_tokens or not curr_tokens:
        return 0.0
    max_overlap = min(len(prev_tokens), len(curr_tokens))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if prev_tokens[-size:] == curr_tokens[:size]:
            overlap = size
            break
    return overlap / max(1, len(curr_tokens))


def test_dedupe_prefix_copy():
    base = "This paragraph describes the policy changes and operational standards. " * 12
    chunks = [
        base.strip(),
        f"{base}Additional clarification A.".strip(),
        f"{base}Additional clarification A. Additional clarification B.".strip(),
        f"{base}Additional clarification A. Additional clarification B. Additional clarification C.".strip(),
    ]
    metadata = _meta_for_section(len(chunks))
    deduped_chunks, deduped_meta, stats = apply_dedupe_gate(
        chunks, metadata, config=DedupeConfig(similarity_threshold=0.92, max_overlap_ratio=0.20)
    )
    assert len(deduped_chunks) <= len(chunks) - 1
    assert len(deduped_chunks) == len(deduped_meta)
    assert len(set(deduped_chunks)) == len(deduped_chunks)
    for prev, curr in zip(deduped_chunks, deduped_chunks[1:]):
        assert _overlap_ratio(prev, curr) <= 0.20
    assert stats["dropped"] >= 1 or stats["merged"] >= 1


def test_sentence_completeness_with_bullets():
    bullets = "\n".join(
        [
            "- creates jobs across regions",
            "- boosts local spending and services",
            "- increases foreign exchange inflows",
            "- supports infrastructure upgrades",
            "- strengthens cultural exchange",
            "- encourages small business growth",
        ]
    )
    chunks = ["Tourism helps India by:", bullets]
    metadata = _meta_for_section(len(chunks))
    prepared_chunks, prepared_meta, _stats, _rescued = prepare_embedding_chunks(
        chunks,
        metadata,
        subscription_id="sub-1",
        profile_id="prof-1",
        document_id="doc-1",
        doc_name="tourism.txt",
    )
    assert prepared_chunks
    # The pipeline may strip bullet markers and merge short chunks
    assert "creates jobs" in prepared_chunks[0]
    assert prepared_meta[0].get("sentence_complete", True) is True


def test_tiny_chunk_merge():
    long_text = "This section explains the operating procedure and required approvals in detail. " * 12
    tiny_text = "Short filler that should be merged."
    chunks = [long_text.strip(), tiny_text, long_text.strip()]
    metadata = _meta_for_section(len(chunks))
    merged_chunks, merged_meta, stats = enforce_chunk_integrity(
        chunks,
        metadata,
        config=ChunkIntegrityConfig(target_min_tokens=1, target_max_tokens=1, hard_max_tokens=5, min_chars=300),
    )
    assert len(merged_chunks) == len(merged_meta)
    assert stats["merged_small"] >= 1
    assert all(len(chunk) >= 300 for chunk in merged_chunks)


def test_payload_normalization():
    payload = {
        "subscription_id": "sub-1",
        "profile_id": "prof-1",
        "document_id": "doc-1",
        "text": "Sample text.",
        "source_file": "report.pdf",
        "filename": "report.pdf",
        "file_name": "report.pdf",
        "document_name": "report.pdf",
        "doc_type": "policy",
        "document_type": "policy",
        "chunk_hash": "abc",
        "text_hash": "def",
        "languages": [],
        "evidence_pointer": "Section: Intro, Page: 2-3",
        "chunk_index": 0,
        "chunk_count": 2,
        "section_title": "Intro",
        "section_path": "Intro",
        "chunk_id": "chunk_1",
    }
    normalized = normalize_payload(payload)
    assert normalized["source"]["name"] == "report.pdf"
    assert normalized["document"]["type"] == "policy"
    assert "filename" not in normalized
    assert "file_name" not in normalized
    assert "document_name" not in normalized
    assert "document_type" not in normalized or normalized.get("document", {}).get("type") == "policy"
    assert "chunk_hash" not in normalized
    assert "text_hash" not in normalized
    assert normalized["chunk"]["hash"] == hashlib.sha256("Sample text.".encode("utf-8")).hexdigest()
    assert normalized["provenance"]["page_start"] == 2
    assert normalized["provenance"]["page_end"] == 3


def test_chain_integrity():
    chunks = [
        "First section content. " * 12,
        "Second section content. " * 12,
        "Third section content. " * 12,
    ]
    metadata = _meta_for_section(len(chunks))
    prepared_chunks, prepared_meta, _stats, _rescued = prepare_embedding_chunks(
        chunks,
        metadata,
        subscription_id="sub-1",
        profile_id="prof-1",
        document_id="doc-1",
        doc_name="chain.txt",
        integrity_config=ChunkIntegrityConfig(target_min_tokens=1, target_max_tokens=1, hard_max_tokens=5, min_chars=30),
    )
    assert [m["chunk_index"] for m in prepared_meta] == list(range(len(prepared_meta)))
    assert all(m["chunk_count"] == len(prepared_meta) for m in prepared_meta)
    for idx, meta in enumerate(prepared_meta):
        if idx == 0:
            assert meta["prev_chunk_id"] is None
            assert meta["next_chunk_id"] == prepared_meta[idx + 1]["chunk_id"]
        elif idx == len(prepared_meta) - 1:
            assert meta["next_chunk_id"] is None
            assert meta["prev_chunk_id"] == prepared_meta[idx - 1]["chunk_id"]
        else:
            assert meta["prev_chunk_id"] == prepared_meta[idx - 1]["chunk_id"]
            assert meta["next_chunk_id"] == prepared_meta[idx + 1]["chunk_id"]
