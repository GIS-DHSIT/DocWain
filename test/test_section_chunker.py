from src.api.pipeline_models import ChunkCandidate, ExtractedDocument, Figure, Section, Table
from src.embedding.chunking.section_chunker import SectionChunker


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _ends_with_bullet_or_table(text: str) -> bool:
    last = _last_line(text)
    if not last:
        return False
    bullet_markers = ("-", "*", "•")
    if last.lstrip().startswith(bullet_markers):
        return True
    if "|" in last:
        return True
    # Heuristic: multi-column table row with repeated separators.
    if last.count("  ") >= 2 or last.count(",") >= 3:
        return True
    return False


def test_section_chunker_sentence_safe_and_section_aware():
    sections = [
        Section(
            section_id="sec-1",
            title="INTRODUCTION",
            level=1,
            start_page=1,
            end_page=1,
            text="",
        ),
        Section(
            section_id="sec-2",
            title="",
            level=2,
            start_page=2,
            end_page=2,
            text="",
        ),
    ]

    intro_text = (
        "This is a complete introduction sentence. "
        "It explains the purpose clearly. "
        "The last sentence should remain complete."
    )
    bullets = (
        "- First bullet explains a key point\n"
        "  with a continuation line that should stay attached.\n"
        "- Second bullet ends cleanly."
    )
    table_rows = (
        "Header A | Header B | Header C\n"
        "Row 1A  | Row 1B  | Row 1C\n"
        "Row 2A  | Row 2B  | Row 2C"
    )

    chunk_candidates = [
        ChunkCandidate(text=intro_text, page=1, section_title="INTRODUCTION", section_id="sec-1", chunk_type="text"),
        ChunkCandidate(text=bullets, page=1, section_title="INTRODUCTION", section_id="sec-1", chunk_type="bullet"),
        ChunkCandidate(text=table_rows, page=2, section_title="", section_id="sec-2", chunk_type="table"),
    ]

    extracted = ExtractedDocument(
        full_text="\n\n".join([intro_text, bullets, table_rows]),
        sections=sections,
        tables=[Table(page=2, text=table_rows)],
        figures=[Figure(page=1, caption="")],
        chunk_candidates=chunk_candidates,
        doc_type="policy",
        errors=[],
        metrics={},
    )

    chunker = SectionChunker(target_chunk_chars=260, min_chunk_chars=80, max_chunk_chars=360, overlap_sentences=1)
    chunks = chunker.chunk_document(extracted, doc_internal_id="doc-xyz", source_filename="sample.pdf")

    assert chunks, "Expected non-empty chunks"

    min_chars = chunker.min_chunk_chars
    max_chars = chunker.max_chunk_chars

    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.doc_internal_id == "doc-xyz"
        assert (chunk.section_title or "").strip(), "section_title should never be empty"
        assert (chunk.section_path or "").strip(), "section_path should never be empty"

        clen = len(chunk.text)
        assert min_chars <= clen <= max_chars, f"chunk length out of bounds: {clen}"

        last_char = chunk.text.rstrip()[-1]
        assert last_char in ".?!" or _ends_with_bullet_or_table(chunk.text), (
            f"chunk ends mid-sentence: {chunk.text[-80:]}"
        )
